import os
import argparse
import chromadb
import numpy as np

from src.memory import MemoryVectorDB
from src.models import EmbeddingModel
from src.paths import MEMORY_DATA_DIR, PERSISTENT_DIR
from src.utils import (
    EmbeddingConfig,
    load_configs_from_yaml,
    read_memory_data,
    sample_memories,
    construct_memory,
    construct_response_memory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Memory DB")

    # data settings
    parser.add_argument("--task", type=str, required=True, help="Task subdirectory under MEMORY_DATA_DIR")
    parser.add_argument("--model_name", type=str, required=True, help="Model subdirectory under MEMORY_DATA_DIR/<task>")
    parser.add_argument("--collection_name", type=str, default="", help="Optional suffix for the ChromaDB collection name")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of records per batch when adding to ChromaDB")

    # embedding settings
    parser.add_argument("--embedding_model", type=str, default="bge-m3", help="SentenceTransformer model name or path")
    
    # sample settings
    parser.add_argument("--sample_size", type=int, default=-1, help="Number of unique questions to sample (-1 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load embedding configurations and model
    configs = load_configs_from_yaml("configs.yaml")
    embedding_config = EmbeddingConfig(configs["embedding_configs"][args.embedding_model])
    embedding_model = EmbeddingModel(embedding_config)

    if args.collection_name:
        collection_name = f"{args.task}_{args.model_name}_{args.collection_name}"
    else:
        collection_name = f"{args.task}_{args.model_name}"
    print(f"Main Collection name: {collection_name}")

    # Reset collections if they already exist
    client = chromadb.PersistentClient(path=str(PERSISTENT_DIR))
    collection_names = client.list_collections()
    for name in [collection_name, f"{collection_name}_responses"]:
        if name in collection_names:
            client.delete_collection(name)
            print(f"Deleted existing collection '{name}'")
    
    # initialize memory database
    memory_db = MemoryVectorDB(
        persistent_dir=str(PERSISTENT_DIR),
        collection_name=collection_name,
        embedding_model=embedding_model,
        verbose=True,
    )

    # load memory data from JSONL files
    data_dir = os.path.join(MEMORY_DATA_DIR, args.task, args.model_name)
    records = read_memory_data(data_dir)
    if not records:
        print("No memory data found. Exiting.")
        return
    print(f"Loaded {len(records)} memory cases from {data_dir}")

    # ------------------------------------------------------------------
    # 1. Create memory entries for debate state retrieval
    # ------------------------------------------------------------------
    
    records_sample = sample_memories(records, sample_size=args.sample_size, seed=args.seed)
    print(f"Sampled {len(records_sample)} memory cases.")

    all_db_ids = []
    all_value_texts = []

    batch_size = args.batch_size
    for batch_start in range(0, len(records_sample), batch_size):
        batch = records_sample[batch_start: batch_start + batch_size]
        db_ids, embedding_texts, value_texts, meta_datas = zip(*[construct_memory(row) for row in batch])
        print(f"Adding batch {batch_start // batch_size + 1} ({len(batch)} records)...")

        memory_db.add_batch_memories(
            db_ids=list(db_ids),
            embedding_texts=list(embedding_texts),
            document_texts=list(value_texts),
            add_meta_datas=list(meta_datas),
        )

        all_db_ids.extend(db_ids)
        all_value_texts.extend(value_texts)

    print(f"\nDone. Total memory entries in collection: {memory_db.get_memories_count()}")

    # Pre-compute and save document embeddings for fast MMR re-ranking at inference time.
    print("\nPre-computing document embeddings...")
    doc_embeddings = embedding_model.encode(all_value_texts, show_progress_bar=True)
    doc_embeddings = np.array(doc_embeddings, dtype=np.float16)

    npz_path = os.path.join(PERSISTENT_DIR, f"{collection_name}_doc_embeddings.npz")
    np.savez(npz_path, db_ids=np.array(all_db_ids), embeddings=doc_embeddings)
    print(f"Saved document embeddings to {npz_path}  (shape: {doc_embeddings.shape})")

    # ------------------------------------------------------------------
    # 2. Create response entries for confidence estimation
    # ------------------------------------------------------------------

    seen_ids = set()
    resp_ids, resp_texts, resp_metas = [], [], []
    for row in records:
        if not row.get("current_response"):
            continue
        db_id, text, meta = construct_response_memory(row)
        if db_id not in seen_ids:
            resp_ids.append(db_id)
            resp_texts.append(text)
            resp_metas.append(meta)
            seen_ids.add(db_id)

    for batch_start in range(0, len(resp_ids), batch_size):
        memory_db.add_batch_responses(
            db_ids=resp_ids[batch_start: batch_start + batch_size],
            texts=resp_texts[batch_start: batch_start + batch_size],
            meta_datas=resp_metas[batch_start: batch_start + batch_size],
        )
    print(f"\nDone. Total response entries in collection: {len(resp_ids)}")


if __name__ == "__main__":
    main()
