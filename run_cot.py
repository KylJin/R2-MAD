import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from src.agent import Agent, AgentId, Query
from src.memory import MemoryVectorDB
from src.models import build_language_model, EmbeddingModel
from src.chain_of_thought import ChainOfThought
from src.paths import PERSISTENT_DIR, LOG_DIR
from src.utils import LLMConfig, EmbeddingConfig, load_configs_from_yaml, load_env_file, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Run Chain-of-Thought / Self-Consistency evaluation")

    # self-consistency settings
    parser.add_argument("--task", type=str, default="MATH500", help="Task / dataset type")
    parser.add_argument("--self_consistency", action="store_true", help="Enable self-consistency with multi-path sampling")
    parser.add_argument("--num_reasoning_paths", type=int, default=9, help="Number of reasoning paths for self-consistency")
    
    # agent settings
    parser.add_argument("--model_name", type=str, default="qwen3-8b", help="The name of adopted LLM")
    parser.add_argument("--if_use_memory", action="store_true", help="Retrieve past cases as in-context examples")
    parser.add_argument("--memory_model", type=str, default="", help="Model name whose memory collection to use (default: same as --model_name)")
    parser.add_argument("--collection_name", type=str, default="", help="Optional suffix for the ChromaDB collection name")
    parser.add_argument("--embedding_model", type=str, default="bge-m3", help="The name of adopted embedding model")
    parser.add_argument("--n_retrieval", type=int, default=1, help="Number of retrieved examples")
    parser.add_argument("--retrieval_policy", type=str, default="similarity",
                        choices=["similarity", "random", "diversity", "positive_only"],
                        help="Memory retrieval policy (only active with --if_use_memory; 'similarity' is the natural default for CoT)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for stochastic retrieval policies (random/diversity)")
    parser.add_argument("--memory_agent_id", type=str, default="Debater-0", help="agent_id whose memory to retrieve from (memory is stored per debate agent)")

    # general settings
    parser.add_argument("--exp_name", type=str, default="test", help="Optional prefix for the output filename")
    parser.add_argument("--run_suffix", type=str, default="", help="Optional suffix appended before .json")
    parser.add_argument("--save_log", action="store_true", help="Whether to save evaluation log to disk")
    parser.add_argument("--train", action="store_true", help="Use training split instead of test split")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of queries per batch")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of questions (for quick tests)")

    # gpu settings
    parser.add_argument("--gpu_id", type=str, default="0", help="The GPU ID.")
    parser.add_argument("--embed_gpu_id", type=str, default=None, help="Dedicated GPU for the embedding model. Default None = share --gpu_id.")
    parser.add_argument("--parallel", action="store_true", default=False, help="Whether to use parallel mode.")
    
    args = parser.parse_args()
    return args


def build_agent(args, memory_db=None) -> Agent:
    agent_id = AgentId(agent_type="CoT", agent_key="0")
    agent = Agent(
        agent_id=agent_id,
        task_type=args.task,
        if_use_memory=args.if_use_memory,
        memory_db=memory_db,
        n_retrieval=args.n_retrieval,
        retrieval_policy=args.retrieval_policy,
        seed=args.seed,
        # CoT has no debate rounds; retrieve on round 0 and reuse a debate
        # agent's memory (memory is stored per Debater-{i}).
        retrieve_first_round=True,
        memory_agent_id=args.memory_agent_id,
    )
    return agent


def load_questions(task_type: str, limit: int | None, is_training: bool) -> List[Query]:
    if is_training:
        raw_data = DataLoader.load_dataset(task_type + "_TRAIN")
    else:
        raw_data = DataLoader.load_dataset(task_type)

    queries = []
    for idx, item in enumerate(raw_data):
        if limit is not None and idx >= limit:
            break
        content, answer, solution, category = DataLoader.format_question(item, task_type)
        queries.append(Query(
            content=content,
            query_id=idx,
            answer=answer,
            solution=solution,
            category=category,
        ))
    
    return queries


def compute_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("is_final_correct", False))
    accuracy = correct / total if total > 0 else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def main():
    args = parse_args()
    
    # Load API keys etc. from .env (no-op if the file is absent)
    load_env_file()

    # GPU settings
    gpus = args.gpu_id.split(",")
    tensor_parallel_size = len(gpus) if args.parallel else 1
    # Embedding device: shares the vLLM GPU(s) by default; --embed_gpu_id puts it
    # on a dedicated card, which is then added to CUDA_VISIBLE_DEVICES.
    visible_gpus = list(gpus) if args.parallel else [args.gpu_id]
    embed_device = None
    if args.embed_gpu_id is not None:
        if args.embed_gpu_id not in visible_gpus:
            visible_gpus.append(args.embed_gpu_id)
        embed_device = f"cuda:{visible_gpus.index(args.embed_gpu_id)}"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)

    # load configurations
    configs = load_configs_from_yaml("configs.yaml")
    llm_config = LLMConfig(configs["llm_configs"]["general_configs"], configs["llm_configs"][args.model_name], name=args.model_name)
    llm_config.tensor_parallel_size = tensor_parallel_size

    print("=" * 80)
    reasoning_mode = "Self-Consistency" if args.self_consistency else "CoT"
    print(f"Task: {args.task} | Reasoning: {reasoning_mode}")
    if args.self_consistency:
        print(f"Reasoning paths per query: {max(1, args.num_reasoning_paths)}")
    print(f"Model: {llm_config.model} | Backend: {llm_config.backend}")
    print("=" * 80)

    model = build_language_model(llm_config)

    memory_db = None
    if args.if_use_memory:
        # load embedding model
        embedding_config = EmbeddingConfig(configs["embedding_configs"][args.embedding_model])
        embedding_model = EmbeddingModel(embedding_config, device=embed_device)

        memory_model = args.memory_model if args.memory_model else args.model_name
        if args.collection_name:
            collection_name = f"{args.task}_{memory_model}_{args.collection_name}"
        else:
            collection_name = f"{args.task}_{memory_model}"
        print(f"Collection name: {collection_name} (memory from: {memory_model}, agent: {args.memory_agent_id})")

        # initialize memory database
        memory_db = MemoryVectorDB(
            persistent_dir=PERSISTENT_DIR,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

        npz_path = os.path.join(PERSISTENT_DIR, f"{collection_name}_doc_embeddings.npz")
        if os.path.exists(npz_path):
            memory_db.load_doc_embeddings(npz_path)
        else:
            print(f"Warning: doc embeddings not found at {npz_path}, MMR will fall back to on-the-fly encode.")

    start_time = datetime.now()

    agent = build_agent(args, memory_db=memory_db)
    cot = ChainOfThought(
        agent=agent,
        model=model,
        task_type=args.task,
        num_reasoning_paths=args.num_reasoning_paths,
        self_consistency=args.self_consistency,
    )
    
    queries = load_questions(args.task, args.limit, args.train)
    print(f"Loaded {len(queries)} questions")

    all_results = []
    for batch_start in range(0, len(queries), args.batch_size):
        batch_idx = batch_start // args.batch_size + 1
        batch = queries[batch_start: batch_start + args.batch_size]
        print(f"\n>>> Batch {batch_idx} "
              f"(questions {batch_start}-{batch_start + len(batch) - 1})")
        all_results.extend(cot.run(batch))
    
    metrics = compute_metrics(all_results)
    print("\n" + "=" * 80)
    print(f"Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.4f}")
    print("=" * 80)

    token_summary = cot.get_token_usage_summary()
    print(f"Token usage: {token_summary}")

    elapsed = datetime.now() - start_time
    print(f"\nTotal time: {elapsed}")

    if args.save_log:
        out_dir = os.path.join(LOG_DIR, args.task, args.model_name)
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        prefix = f"{args.exp_name}_" if args.exp_name else ""
        suffix = f"_{args.run_suffix}" if args.run_suffix else ""
        out_path = os.path.join(
            out_dir, 
            f"{prefix}{timestamp}{suffix}.json"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "args": vars(args),
                "metrics": metrics,
                "time_consumption": str(elapsed),
                "token_usage": token_summary,
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
