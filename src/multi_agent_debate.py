import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

import numpy as np

from src.agent import Query, Agent
from src.models import LanguageModel
from src.checker import ANSWER_CHECKER
from src.paths import MEMORY_DATA_DIR
from src.prompts import DEBATE_SUMMARY_PROMPT


def remove_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


class MultiAgentDebate:
    """
    Orchestrates a multi-round debate for a batch of queries.

    A fixed list of Agent instances is shared across all queries; each agent
    stores conversation history per query_id internally, so no duplication
    of agent objects is needed.

    Round 0 : every agent answers independently.
    Round 1+: every agent sees the other agents' previous responses and revises.

    All queries are processed in a single batched model call per round;
    the batch sent to LanguageModel has shape (num_active_queries * num_agents).
    """

    def __init__(
        self,
        agents: List[Agent],
        model: LanguageModel,
        task_type: str,
        max_round: int = 3,
        consensus_threshold: float = 0.8,
        if_summarize_state: bool = False,
        if_confidence_weight: bool = False,
    ) -> None:
        self.agents = agents
        self.model = model
        self.max_round = max_round
        self.consensus_threshold = consensus_threshold
        self.if_summarize_state = if_summarize_state
        self.if_confidence_weight = if_confidence_weight

        self.task_type = task_type
        self._checker = ANSWER_CHECKER[task_type]

        # Persistent debate state (reset each `run`)
        self.current_rounds = {}
        self.agent_responses = defaultdict(lambda: defaultdict(dict))
        self.consensus_flags = {}
        self.debate_log = {}
        # prev_resp_embeddings[qid][agent_id_str] = unit-norm float32 ndarray
        self._prev_resp_embeddings = defaultdict(lambda: dict())
    
    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, queries: List[Query], is_training: bool = True, batch_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Run a full debate for a batch of queries.

        Args:
            queries: list of Query objects

        Returns:
            list of debate_log entries (one per query), each with keys:
            query, rounds, final_answer, consensus
        """
        self._clear_debate()
        self._initialize_debate(queries)

        for round_idx in range(self.max_round):
            if is_training:
                # force all queries to run all rounds during training
                active_queries = queries
            else:
                active_queries = [q for q in queries if not self.consensus_flags[q.query_id]]
            if not active_queries:
                break
            print(f"\n>>>>>> Round {round_idx}\n")
            self._run_debate_round(active_queries)

        for query in queries:
            qid = query.query_id
            last_round = self.current_rounds[qid] - 1

            correct_count, error_count = 0, 0
            final_true_answer, final_false_answer = None, None

            for agent in self.agents:
                if self.agent_responses[qid][agent.agent_id][f"round_{last_round}"]["is_correct"]:
                    correct_count += 1
                    final_true_answer = self.agent_responses[qid][agent.agent_id][f"round_{last_round}"]["pred_answer"]
                else:
                    error_count += 1
                    final_false_answer = self.agent_responses[qid][agent.agent_id][f"round_{last_round}"]["pred_answer"]
            
            is_final_correct = (correct_count > error_count)
            final_answer = final_true_answer if is_final_correct else final_false_answer
            self.debate_log[qid]["final_answer"] = final_answer
            self.debate_log[qid]["is_final_correct"] = is_final_correct
            self.debate_log[qid]["consensus"] = self.consensus_flags[qid]
        
        if is_training:
            self._save_memory_data(queries, batch_idx=batch_idx)

        return [self.debate_log[q.query_id] for q in queries]
    
    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_debate_round(self, queries: List[Query]) -> None:
        """
        Run one debate round for the given active queries.

        For round 0, each agent receives no prior responses (initial answer).
        For round > 0, each agent receives the other agents' responses from the previous round, read from self.agent_responses.
        """
        # Build batched messages: [q0_a0, q0_a1, ..., q1_a0, q1_a1, ...]
        batch_messages = []
        flat_meta = []  # (query, agent_idx)

        # Precompute per-query context maps for batch memory retrieval
        prev_responses_map: Dict[int, Any] = {}
        prev_round_summaries: Dict[int, Any] = {}
        prev_consensus_ratios: Dict[int, Any] = {}
        for query in queries:
            qid = query.query_id
            round_idx = self.current_rounds[qid]
            rounds_log = self.debate_log[qid]["rounds"]

            if round_idx == 0:
                prev_responses_map[qid] = None
                prev_round_summaries[qid] = None
                prev_consensus_ratios[qid] = None
            else:
                emb_cache = self._prev_resp_embeddings.get(qid, {}) if self.if_confidence_weight else {}
                prev_responses_map[qid] = {
                    agent.agent_id: (
                        self.agent_responses[qid][agent.agent_id][f"round_{round_idx - 1}"].get("response", ""),
                        emb_cache.get(agent.agent_id, None),
                    ) for agent in self.agents
                }
                prev_round_summaries[qid] = rounds_log[-1].get("summary", None)
                prev_consensus_ratios[qid] = rounds_log[-1].get("consensus_ratio", None)
        
        # Batch memory retrieval: one DB round-trip per agent instead of per (agent, query)
        agent_prefetched = {}
        for agent in self.agents:
            agent_prefetched[agent.agent_id] = agent.batch_retrieve_all(
                queries=queries,
                prev_responses_map={
                    qid: resp for qid, resp in prev_responses_map.items() if resp is not None
                },
                prev_round_summaries=prev_round_summaries,
                prev_consensus_ratios=prev_consensus_ratios,
            )
        
        for query in queries:
            qid = query.query_id
            prev_responses = prev_responses_map[qid]
            prev_round_summary = prev_round_summaries[qid]
            prev_consensus_ratio = prev_consensus_ratios[qid]

            for a_idx, agent in enumerate(self.agents):
                prefetched = agent_prefetched[agent.agent_id].get(qid)
                messages = agent.build_messages(
                    query=query,
                    prev_responses=prev_responses,
                    prev_round_summary=prev_round_summary,
                    prev_consensus_ratio=prev_consensus_ratio,
                    prefetched=prefetched,
                )
                batch_messages.append(messages)
                flat_meta.append((query, a_idx))
        
        model_output = self.model(batch_messages, task_name="debate")
        responses = [remove_think_tags(r).strip() for r in model_output["responses"]]

        # Batch-encode all responses once for confidence weighting in the next round
        resp_embs = None
        if self.if_confidence_weight:
            embed_fn = next(
                (a.memory_db.embedding_model.encode for a in self.agents if a.memory_db is not None), None,
            )
            if embed_fn is not None:
                resp_embs = np.array(embed_fn(responses), dtype=np.float32)
        
        # Process and store per-agent results into agent_responses
        for i, (query, a_idx) in enumerate(flat_meta):
            qid = query.query_id
            round_idx = self.current_rounds[qid]

            pred_answer = self._checker.parse_answer(responses[i])
            is_correct = self._checker.check_answer(pred_answer, query.answer)

            agent = self.agents[a_idx]
            agent.record_response(query, responses[i])
            self.agent_responses[qid][agent.agent_id][f"round_{round_idx}"] = {
                "response": responses[i],
                "pred_answer": pred_answer,
                "is_correct": is_correct,
            }
            if resp_embs is not None:
                self._prev_resp_embeddings[qid][agent.agent_id] = resp_embs[i]
        
        # Check consensus and log each active query
        for query in queries:
            qid = query.query_id
            round_idx = self.current_rounds[qid]

            pred_answers = [
                self.agent_responses[qid][agent.agent_id][f"round_{round_idx}"]["pred_answer"]
                for agent in self.agents
            ]
            is_consensus, consensus_ratio, common_answer = self._check_consensus(query, pred_answers)

            agents_log = []
            for agent in self.agents:
                round_data = self.agent_responses[qid][agent.agent_id][f"round_{round_idx}"]
                entry = {
                    "agent_id": str(agent.agent_id),
                    "response": round_data["response"],
                    "pred_answer": round_data["pred_answer"],
                    "is_correct": round_data["is_correct"],
                }

                # prefetched is (retrieved_memories, other_confidences), or None at 0
                prefetched = agent_prefetched.get(agent.agent_id, {}).get(qid)
                retrieved, confidences = prefetched if prefetched else (None, None)

                # Memories this agent retrieved to produce this round's response.
                if agent.if_use_memory:
                    entry["retrieved_memories"] = [{
                        "content": rm.content,
                        "is_correct": rm.is_correct,
                        "similarity": rm.similarity,
                    } for rm in (retrieved or [])]
                
                # Confidence this agent assigned to each peer's previous response.
                if self.if_confidence_weight:
                    entry["peer_confidences"] = {
                        str(peer_id): float(conf) for peer_id, conf in (confidences or {}).items()
                    }
                
                agents_log.append(entry)
            
            self.debate_log[qid]["rounds"].append({
                "round": round_idx,
                "agents": agents_log,
                "common_answer": common_answer,
                "consensus": is_consensus,
                "consensus_ratio": consensus_ratio,
            })
            if is_consensus:
                self.consensus_flags[qid] = True
            
            self.current_rounds[qid] += 1
        
        if self.if_summarize_state:
            print(f"\n>>>>>> Summarize Debate State\n")
            self._summarize_debate_state(queries)
    
    def _summarize_debate_state(self, queries: List[Query]) -> None:
        """
        Generate debate summaries for the just-completed round across all queries
        in a single batched model call, then write results back to debate_log.
        """
        # Build one summary prompt per query, tracking which round each refers to
        batch_messages = []
        meta = []  # (qid, round_log_index)

        for query in queries:
            rounds = self.debate_log[query.query_id]["rounds"]
            round_log = rounds[-1]
            round_idx = len(rounds) - 1
            prev_summary = rounds[-2]["summary"] if round_idx > 0 else "None"

            agent_responses = []
            for i, entry in enumerate(round_log["agents"]):
                if round_idx > 0:
                    prev_entry = rounds[-2]["agents"][i]
                    stance_changed = str(entry["pred_answer"] != prev_entry["pred_answer"]).lower()
                    agent_responses.append(
                        f'<agent index="{i}">\n'
                        f'<response>\n{entry["response"]}\n</response>\n'
                        f'<answer>\n{entry["pred_answer"]}\n</answer>\n'
                        f'<previous_answer>\n{prev_entry["pred_answer"]}\n</previous_answer>\n'
                        f'<stance_changed>\n{stance_changed}\n</stance_changed>\n'
                        f'</agent>'
                    )
                else:
                    agent_responses.append(
                        f'<agent index="{i}">\n'
                        f'<response>\n{entry["response"]}\n</response>\n'
                        f'<answer>\n{entry["pred_answer"]}\n</answer>\n'
                        f'</agent>'
                    )
            agent_responses_text = "\n".join(agent_responses)

            prompt = DEBATE_SUMMARY_PROMPT.format(
                QUESTION=query.content,
                PREV_SUMMARY=prev_summary,
                AGENT_RESPONSES=agent_responses_text,
                CONSENSUS_RATIO=f"{round_log['consensus_ratio']:.2f}",
            )
            batch_messages.append([{"role": "user", "content": prompt}])
            meta.append((query.query_id, round_idx))

        if not batch_messages:
            return

        model_output = self.model(batch_messages, task_name="summary")
        for (qid, round_idx), summary in zip(meta, model_output["responses"]):
            self.debate_log[qid]["rounds"][round_idx]["summary"] = remove_think_tags(summary).strip()
    
    def _check_consensus(self, query: Query, pred_answers: List[str]) -> tuple[bool, float, str]:
        """
        Check if all agents have reached consensus on the same answer.

        Returns:
            is_consensus: bool
            consensus_ratio: float (fraction of agents that agreed on the most common answer)
            common_answer: str (the answer that the majority agreed on, or None if no consensus)
        """
        if not pred_answers:
            return False, 0.0, ""

        qid = query.query_id
        round_idx = self.current_rounds[qid]

        normalized_answers = [answer.lower().strip() for answer in pred_answers]

        correct_count = 0
        for agent in self.agents:
            if self.agent_responses[qid][agent.agent_id][f"round_{round_idx}"]["is_correct"]:
                correct_count += 1
        if correct_count == len(pred_answers):
            return True, 1.0, normalized_answers[0]
        
        answer_counts = Counter(normalized_answers)
        common_answer, max_count = answer_counts.most_common(1)[0]
        consensus_ratio = max_count / len(pred_answers)
        is_consensus = consensus_ratio >= self.consensus_threshold
        
        return is_consensus, consensus_ratio, common_answer
    
    # ------------------------------------------------------------------
    # Debate lifecycle
    # ------------------------------------------------------------------

    def _initialize_debate(self, queries: List[Query]) -> None:
        for query in queries:
            self.current_rounds[query.query_id] = 0
            self.debate_log[query.query_id] = {
                "query": query.content,
                "answer": query.answer,
                "rounds": []
            }
            self.consensus_flags[query.query_id] = False
            for agent in self.agents:
                agent.initialize_conversation(query)

    def _clear_debate(self) -> None:
        self.current_rounds.clear()
        self.agent_responses.clear()
        self.debate_log.clear()
        self.consensus_flags.clear()
        self._prev_resp_embeddings.clear()
        for agent in self.agents:
            agent.clear_all_conversations()
    
    # ------------------------------------------------------------------
    # data saving utilities
    # ------------------------------------------------------------------

    def _save_memory_data(self, queries: List[Query], batch_idx: int = 0) -> None:
        """
        After a completed debate, write one JSON line per (query, agent, round)
        to MEMORY_DATA_DIR/<task_type>/agent{n}_batch{idx:03d}_{timestamp}.jsonl.
        """
        task_dir = os.path.join(MEMORY_DATA_DIR, self.task_type, self.model.model_name)
        os.makedirs(task_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent{len(self.agents)}_batch{batch_idx:03d}_{timestamp}.jsonl"
        filepath = os.path.join(task_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            for query in queries:
                qid = query.query_id
                log = self.debate_log[qid]
                is_final_correct = log["is_final_correct"]
                rounds_log = log["rounds"]
                num_rounds = len(rounds_log)

                for agent in self.agents:
                    aid = agent.agent_id
                    for round_idx in range(num_rounds):
                        round_data = self.agent_responses[qid][aid][f"round_{round_idx}"]
                        current_response = round_data["response"]
                        if not current_response:
                            continue

                        is_current_correct = round_data["is_correct"]

                        if round_idx > 0:
                            prev_response = self.agent_responses[qid][aid][f"round_{round_idx - 1}"]["response"]
                            prev_pred = self.agent_responses[qid][aid][f"round_{round_idx - 1}"]["pred_answer"]
                            stance_changed = (prev_pred != round_data["pred_answer"])
                            prev_consensus_ratio = rounds_log[round_idx - 1]["consensus_ratio"]
                            prev_round_summary = rounds_log[round_idx - 1].get("summary", None)
                        else:
                            prev_response = None
                            stance_changed = None
                            prev_consensus_ratio = None
                            prev_round_summary = None
                        
                        memory_case = {
                            "db_id": f"Q{qid}_Round{round_idx}_{aid}",

                            "query_id": qid,
                            "agent_id": str(aid),
                            "round": round_idx,

                            "query": query.content,
                            "answer": query.answer,
                            "solution": query.solution,
                            "category": query.category,

                            "prev_response": prev_response,
                            "prev_round_summary": prev_round_summary,
                            "prev_consensus_ratio": prev_consensus_ratio,

                            "current_response": current_response,
                            "is_current_correct": is_current_correct,

                            "stance_changed": stance_changed,
                            "is_final_correct": is_final_correct,
                        }
                        
                        f.write(json.dumps(memory_case, ensure_ascii=False) + "\n")

    def save_debate_log(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.debate_log, f, ensure_ascii=False, indent=2)

    def get_token_usage_summary(self):
        return self.model.get_token_usage_summary()
