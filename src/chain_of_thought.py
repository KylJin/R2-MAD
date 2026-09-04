import os
import re
import json
from collections import Counter
from typing import Dict, List, Any

from src.agent import Query, Agent
from src.models import LanguageModel
from src.checker import ANSWER_CHECKER


def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


class ChainOfThought:
    """
    Chain-of-Thought / Self-Consistency inference wrapper.
    """

    def __init__(
        self,
        agent: Agent,
        model: LanguageModel,
        task_type: str,
        num_reasoning_paths: int = 1,
        self_consistency: bool = False,
    ) -> None:
        self.agent = agent
        self.model = model
        self.num_reasoning_paths = num_reasoning_paths
        self.self_consistency = self_consistency

        self.task_type = task_type
        self._checker = ANSWER_CHECKER[task_type]

        self.cot_log = None
    
    def run(self, queries: List[Query]) -> List[Dict[str, Any]]:
        for query in queries:
            self.agent.initialize_conversation(query)

        # Optional similarity-based retrieval: fetch in-context examples for the
        # whole batch in one shot. CoT has no debate state, so prev_responses is
        # empty and retrieval falls back to a plain semantic query on the question
        # (Agent is built with retrieve_first_round=True so round 0 retrieves).
        prefetched_map: Dict[int, Any] = {}
        if self.agent.if_use_memory:
            prefetched_map = self.agent.batch_retrieve_all(
                queries=queries,
                prev_responses_map={query.query_id: {} for query in queries},
                prev_round_summaries={query.query_id: None for query in queries},
                prev_consensus_ratios={query.query_id: None for query in queries},
            )

        formated_messages = [
            self._build_cot_messages(query, prefetched_map.get(query.query_id))
            for query in queries
        ]
        final_results = []

        if self.self_consistency and self.num_reasoning_paths > 1:
            flat_meta, sc_messages = [], []
            for query, messages in zip(queries, formated_messages):
                sc_messages.extend([messages] * self.num_reasoning_paths)
                flat_meta.extend([(query, path_idx) for path_idx in range(self.num_reasoning_paths)])
            
            model_output = self.model(sc_messages, task_name="self-consistency")
            responses = [remove_think_tags(r).strip() for r in model_output["responses"]]

            responses_by_qid = {query.query_id: [] for query in queries}
            for i, (query, path_idx) in enumerate(flat_meta):
                pred_answer = self._checker.parse_answer(responses[i])
                is_correct = self._checker.check_answer(pred_answer, query.answer)
                responses_by_qid[query.query_id].append({
                    "path_id": path_idx,
                    "response": responses[i],
                    "pred_answer": pred_answer,
                    "is_correct": is_correct,
                })
            
            for query in queries:
                reasoning_paths = sorted(responses_by_qid[query.query_id], key=lambda x: x["path_id"])

                _, common_answer = self._check_consensus(reasoning_paths)
                is_final_correct = bool(self._checker.check_answer(common_answer, query.answer))

                result = {
                    "query": query.content,
                    "answer": query.answer,
                    "reasoning_paths": reasoning_paths,
                    "final_answer": common_answer,
                    "is_final_correct": is_final_correct,
                }
                self._attach_retrieved_memories(result, prefetched_map.get(query.query_id))
                final_results.append(result)

        else:
            model_output = self.model(formated_messages, task_name="cot")
            responses = [remove_think_tags(r).strip() for r in model_output["responses"]]

            for query, response in zip(queries, responses):
                pred_answer = self._checker.parse_answer(response)
                is_final_correct = bool(self._checker.check_answer(pred_answer, query.answer))

                result = {
                    "query": query.content,
                    "answer": query.answer,
                    "response": response,
                    "final_answer": pred_answer,
                    "is_final_correct": is_final_correct,
                }
                self._attach_retrieved_memories(result, prefetched_map.get(query.query_id))
                final_results.append(result)
        
        self.cot_log = final_results
        for query in queries:
            self.agent.clear_conversation(query)
        
        return final_results
    
    def _attach_retrieved_memories(self, result: Dict[str, Any], prefetched) -> None:
        """
        Record the in-context memory cases retrieved for this query, mirroring the
        per-round `retrieved_memories` format in multi_agent_debate.py. Only added
        when memory is enabled; prefetched is (retrieved_memories, other_confidences).
        """
        if not self.agent.if_use_memory:
            return
        retrieved, _ = prefetched if prefetched else (None, None)
        result["retrieved_memories"] = [{
            "content": rm.content,
            "is_correct": rm.is_correct,
            "similarity": rm.similarity,
        } for rm in (retrieved or [])]

    def _build_cot_messages(self, query: Query, prefetched=None) -> List[Dict[str, str]]:
        messages = self.agent.build_messages(query=query, prev_responses=None, prefetched=prefetched)
        # Keep the original task-specific answer format while nudging CoT style.
        messages[-1]["content"] += "\nPlease solve the problem step by step."
        return messages
    
    def _check_consensus(self, reasoning_paths: List[str], consensus_threshold: float = 0.8) -> tuple[bool, str]:
        """
        Check if all agents have reached consensus on the same answer.

        Returns:
            is_consensus: bool
            common_answer: str (the answer that the majority agreed on, or None if no consensus)
        """
        if not reasoning_paths:
            return False, ""

        normalized_answers = [output["pred_answer"].lower().strip() for output in reasoning_paths]
        
        answer_correctness = [output["is_correct"] for output in reasoning_paths]
        if sum(answer_correctness) == len(reasoning_paths):
            return True, normalized_answers[0]
        
        answer_counts = Counter(normalized_answers)
        common_answer, max_count = answer_counts.most_common(1)[0]
        consensus_ratio = max_count / len(reasoning_paths)
        is_consensus = consensus_ratio >= consensus_threshold
        
        return is_consensus, common_answer
    
    def save_cot_log(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.cot_log, f, ensure_ascii=False, indent=2)
    
    def get_token_usage_summary(self):
        return self.model.get_token_usage_summary()
