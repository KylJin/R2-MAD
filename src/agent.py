import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from src.memory import MemoryVectorDB
from src.retrieval_policy import build_retrieval_policy
from src.prompts import MAD_SYS_PROMPTS, TASK_PROMPT, DEBATE_PROMPT


@dataclass
class Query:
    """Encapsulates a query and its expected answer."""
    content: str
    query_id: int
    answer: str
    solution: str
    category: str

    def __str__(self):
        return f"The Query {self.query_id} is :: {self.content}"

    def __repr__(self):
        return self.__str__()


@dataclass(frozen=True)
class AgentId:
    agent_type: str
    agent_key: str

    def __str__(self) -> str:
        return f"{self.agent_type}-{self.agent_key}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class RetrievedMemory:
    content: str
    query_id: int
    agent_id: AgentId
    is_correct: bool = False
    similarity: float = 0.0

    def __str__(self):
        return f"Memory of {self.agent_id} for Q{self.query_id}: {self.content}"

    def __repr__(self):
        return self.__str__()


class Agent:
    """
    Manages prompt construction and memory retrieval for a single debate agent.

    Conversation histories are stored per query_id so that one Agent instance
    can participate in debates on multiple querys simultaneously (batch mode).
    Inference is delegated to the shared LanguageModel instance.
    """
    def __init__(
        self,
        agent_id: AgentId,
        task_type: str,
        if_use_memory: bool = False,
        memory_db: Optional[MemoryVectorDB] = None,
        n_retrieval: int = 3,
        if_confidence_weight: bool = False,
        retrieval_policy: str = "debate_state_aware",
        seed: Optional[int] = None,
        ref_lambda: Optional[float] = None,
        retrieve_first_round: bool = False,
        memory_agent_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.task_type = task_type

        self.if_use_memory = if_use_memory
        self.memory_db = memory_db
        self.n_retrieval = n_retrieval
        self.if_confidence_weight = if_confidence_weight

        # Retrieve on round 0 too (used by CoT, which has no debate rounds).
        self._retrieve_first_round = retrieve_first_round
        # agent_id string used to filter the memory collection; defaults to this
        # agent's own id. CoT overrides it (e.g. "Debater-0") to reuse memory
        # collected under the debate agent ids.
        self._memory_agent_id = memory_agent_id or str(agent_id)

        # Offset the base seed by the agent index
        policy_seed = None if seed is None else seed + int(agent_id.agent_key)
        self._policy = build_retrieval_policy(retrieval_policy, seed=policy_seed, ref_lambda=ref_lambda)
        
        self._system_prompt = MAD_SYS_PROMPTS[task_type][int(agent_id.agent_key)]

        # Per-query conversation state
        # _histories[query_id] = list of {"role": ..., "content": ...}
        self._histories: Dict[int, List[Dict[str, str]]] = {}
        # _rounds[query_id] = number of completed rounds
        self._rounds: Dict[int, int] = {}
    
    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def initialize_conversation(self, query: Query) -> None:
        """Register a new query and reset its conversation state."""
        self._histories[query.query_id] = []
        self._rounds[query.query_id] = 0

    def clear_conversation(self, query: Query) -> None:
        self._histories.pop(query.query_id, None)
        self._rounds.pop(query.query_id, None)

    def clear_all_conversations(self) -> None:
        self._histories.clear()
        self._rounds.clear()
    
    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    
    def _build_prompt(
        self,
        query: Query,
        prev_responses: Optional[Dict[AgentId, Tuple[str, Optional[np.ndarray]]]],
        retrieved_memories: Optional[List[RetrievedMemory]] = None,
        other_confidences: Optional[Dict[AgentId, float]] = None,
    ) -> str:
        task_prompt = ""

        if retrieved_memories:
            task_prompt += "Here are some examples:\n\n<examples>\n"
            for idx, memory in enumerate(retrieved_memories):
                task_prompt += f'<example index="{idx}">\n{memory.content}\n</example>\n'
            task_prompt += "</examples>\n\n"

        if prev_responses is not None:
            other_agents = [
                (aid, resp) for aid, (resp, _) in prev_responses.items()
                if aid != self.agent_id and resp
            ]
            
            if self.if_confidence_weight and other_confidences:
                task_prompt += "These are the solutions to the problem from other agents with your confidences:\n\n<other_solutions>\n"
            else:
                task_prompt += "These are the solutions to the problem from other agents:\n\n<other_solutions>\n"
            
            for idx, (aid, response) in enumerate(other_agents):
                confidence_str = ""
                if other_confidences and aid in other_confidences:
                    if other_confidences[aid] > 0.55:
                        confidence_str = f'<confidence>\nHigh, historically reliable on relative problems.\n</confidence>\n'
                    elif other_confidences[aid] < 0.45:
                        confidence_str = f'<confidence>\nLow, low historical accuracy on relative problems.\n</confidence>\n'
                task_prompt += f'<other_solution index="{idx}">\n{response}\n{confidence_str}</other_solution>\n'
            
            task_prompt += "</other_solutions>\n\n"
            task_prompt += DEBATE_PROMPT[self.task_type].format(QUESTION=query.content)
        
        else:
            task_prompt += TASK_PROMPT[self.task_type].format(QUESTION=query.content)

        return task_prompt

    def build_messages(
        self,
        query: Query,
        prev_responses: Optional[Dict[AgentId, Tuple[str, Optional[np.ndarray]]]] = None,
        prev_round_summary: Optional[str] = None,
        prev_consensus_ratio: Optional[float] = None,
        prefetched: Optional[Tuple[List[RetrievedMemory], Optional[Dict[AgentId, float]]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Assemble the full message list (system + history + new user turn).
        Memory retrieval is handled internally when if_use_memory is True.

        prev_responses=None                     → initial round (round 0)
        prev_responses={aid: (resp, emb)}       → debate round; includes all agents (self + others);
                                                  emb is a pre-computed unit-norm embedding or None
        prev_round_summary                      → summary of the previous round (for memory retrieval)
        prev_consensus_ratio                    → consensus ratio of the previous round (for memory re-ranking)
        prefetched                              → (retrieved_memories, other_confidences) pre-computed by
                                                  batch_retrieve_all; skips internal DB queries when provided
        """
        round_idx = self._rounds.get(query.query_id, 0)

        retrieved_memories = None
        other_confidences = None
        if (round_idx > 0 or self._retrieve_first_round) and (self.if_use_memory or self.if_confidence_weight) and self.memory_db:
            if prefetched is not None:
                all_retrieved, other_confidences = prefetched
                retrieved_memories = all_retrieved if self.if_use_memory else None
            else:
                # No prefetched: run the same retrieval policy for this single query
                result = self.batch_retrieve_all(
                    queries=[query],
                    prev_responses_map={query.query_id: prev_responses},
                    prev_round_summaries={query.query_id: prev_round_summary},
                    prev_consensus_ratios={query.query_id: prev_consensus_ratio},
                )
                all_retrieved, other_confidences = result.get(query.query_id, ([], None))
                retrieved_memories = all_retrieved if self.if_use_memory else None

        task_prompt = self._build_prompt(query, prev_responses, retrieved_memories, other_confidences)

        self._histories[query.query_id].append({"role": "user", "content": task_prompt})
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._histories[query.query_id])

        return messages
    
    def record_response(self, query: Query, response: str) -> None:
        """Append a completed turn to the conversation history for query_id."""
        self._histories[query.query_id].append({"role": "assistant", "content": response})
        self._rounds[query.query_id] = self._rounds.get(query.query_id, 0) + 1
    
    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def batch_retrieve_all(
        self,
        queries: List[Query],
        prev_responses_map: Dict[int, Dict[AgentId, Tuple[str, Optional[np.ndarray]]]],
        prev_round_summaries: Dict[int, Optional[str]],
        prev_consensus_ratios: Dict[int, Optional[float]],
    ) -> Dict[int, Tuple[List[RetrievedMemory], Optional[Dict[AgentId, float]]]]:
        """
        Batch memory retrieval for a list of queries in a single round.

        For each query that needs memory (round > 0, if_use_memory, memory_db exists),
        builds all query texts and delegates the two-stage retrieval to the active
        retrieval policy, then estimates per-peer confidence from each query's
        candidate pool.

        Returns:
            {query_id: (retrieved_memories, other_confidences)}
            Only includes entries for queries where retrieval was performed.
        """
        if not (self.if_use_memory or self.if_confidence_weight) or self.memory_db is None:
            return {}

        # Collect queries that actually need retrieval (round > 0)
        active: List[Tuple[int, str]] = []  # (query_id, embedding_text)
        for query in queries:
            round_idx = self._rounds.get(query.query_id, 0)
            if round_idx == 0 and not self._retrieve_first_round:
                continue
            prev_responses = prev_responses_map.get(query.query_id, {})
            prev_agent_response, _ = prev_responses.get(self.agent_id, (None, None))
            prev_round_summary = prev_round_summaries.get(query.query_id)
            embedding_text = self._build_query_text(query, prev_agent_response, prev_round_summary)
            active.append((query.query_id, embedding_text))

        if not active:
            return {}

        # Candidate generation + selection.
        # When memory is on, the retrieval policy owns the whole two-stage retrieval
        # and returns (retrieved, candidates) per query. When only confidence
        # weighting is on there is no active memory policy, so fall back to the
        # plain semantic candidate pool (no retrieval) for confidence.
        if self.if_use_memory:
            policy_out = self._policy.batch_retrieve(
                memory_db=self.memory_db,
                agent_id=self._memory_agent_id,
                queries=active,
                n_retrieval=self.n_retrieval,
                prev_consensus_ratios=prev_consensus_ratios,
            )
        else:
            batch_candidates = self.memory_db.batch_query_similar(
                query_texts=[emb_text for _, emb_text in active],
                n_results=self.n_retrieval * 3,
                filter_metadata={"agent_id": {"$eq": self._memory_agent_id}},
            )
            policy_out = {qid: ([], candidates) for (qid, _), candidates in zip(active, batch_candidates)}

        # If confidence_weight is on, collect all (query_id, round, agent_id) keys needed
        # for get_response_embeddings across all candidates in one batch call
        resp_emb_cache: Dict[Tuple, Optional[np.ndarray]] = {}
        if self.if_confidence_weight:
            other_agent_strs = [
                str(aid) for aid in (
                    next(iter(prev_responses_map.values()), {}).keys()
                ) if aid != self.agent_id
            ]
            if other_agent_strs:
                keys_needed: List[Tuple] = []
                for _, candidates in policy_out.values():
                    for case_meta, _, _, _ in candidates:
                        hist_query_id = case_meta.get("question_id")
                        hist_round = int(case_meta.get("current_round", 1)) - 1
                        if hist_round < 0:
                            continue
                        for aid_str in other_agent_strs:
                            key = (hist_query_id, hist_round, aid_str)
                            if key not in resp_emb_cache:
                                keys_needed.append(key)
                                resp_emb_cache[key] = None  # placeholder

                if keys_needed:
                    fetched = self.memory_db.batch_get_response_embeddings(keys_needed)
                    resp_emb_cache.update(fetched)
        
        # Per-query: build RetrievedMemory from the retrieved cases + estimate confidences 
        # from the candidates using cached response embeddings
        results: Dict[int, Tuple[List[RetrievedMemory], Optional[Dict[AgentId, float]]]] = {}
        for qid, (retrieved, candidates) in policy_out.items():
            retrieved_memories = self._to_retrieved_memories(qid, retrieved) if self.if_use_memory else []

            other_confidences = None
            if self.if_confidence_weight:
                prev_responses = prev_responses_map.get(qid, {})
                other_confidences = self._estimate_confidences_cached(
                    prev_responses, candidates, resp_emb_cache
                )
            
            results[qid] = (retrieved_memories, other_confidences)

        return results

    def _estimate_confidences_cached(
        self,
        prev_responses: Dict[AgentId, Tuple[str, Optional[np.ndarray]]],
        candidate_memories: List[Tuple[Dict, str, float, Optional[np.ndarray]]],
        resp_emb_cache: Dict[Tuple, Optional[np.ndarray]],
    ) -> Dict[AgentId, float]:
        """
        Estimate a confidence score for each other agent's previous response from
        its similarity to historical responses in the candidate pool.

        For each other agent o and each candidate case c (which records what other
        agents answered at round c["current_round"] - 1):
            sim(o, c)  = cosine(current response of o, historical response of o in c)
            reward(c)  = response_correct of c (correctness at that round)
            confidence(o) = sum_c [ sim(o, c) * reward(c) ] / (sum_c sim(o, c) + eps)

        Response embeddings are read from a pre-fetched cache (resp_emb_cache) rather
        than via individual DB calls; missing entries are skipped (sim treated as 0).
        """
        if not candidate_memories or self.memory_db is None:
            return {}

        other_agents = [aid for aid in prev_responses if aid != self.agent_id]
        if not other_agents:
            return {}
        other_agent_strs = [str(aid) for aid in other_agents]

        precomputed = [prev_responses[aid][1] for aid in other_agents]
        if all(e is not None for e in precomputed):
            current_embs = np.stack(precomputed).astype(np.float32)
        else:
            embed_fn = self.memory_db.embedding_model.encode
            current_texts = [prev_responses[aid][0] for aid in other_agents]
            current_embs = np.array(embed_fn(current_texts), dtype=np.float32)

        weighted_sim_sum = np.zeros(len(other_agents))
        sim_sum = np.zeros(len(other_agents))

        for case_meta, _, _, _ in candidate_memories:
            hist_query_id = case_meta.get("question_id")
            hist_round = int(case_meta.get("current_round", 1)) - 1
            if hist_round < 0:
                continue
            round_correct = 1.0 if case_meta.get("response_correct", False) else 0.0

            for i, aid_str in enumerate(other_agent_strs):
                hist_vec = resp_emb_cache.get((hist_query_id, hist_round, aid_str))
                if hist_vec is None:
                    continue
                sim = float(np.dot(current_embs[i], hist_vec))
                sim = max(sim, 0.0)
                weighted_sim_sum[i] += sim * round_correct
                sim_sum[i] += sim

        eps = 1e-8
        confidences = weighted_sim_sum / (sim_sum + eps)

        return {aid: float(confidences[i]) for i, aid in enumerate(other_agents)}
    
    def _build_query_text(
        self,
        query: Query,
        prev_agent_response: Optional[str] = None,
        prev_round_summary: Optional[str] = None,
    ) -> str:
        """
        Mirror _build_embedding_text from memory_utils: constructs the debate-state
        string used as the retrieval key.

          1. task (query content)
          2. agent's previous response (from prev_responses, None at round 0)
          3. previous round summary (None at round 0 or when summarization is off)
        """
        embedding_text = ""

        # 1. task
        embedding_text += f"{query.content}\n"

        # 2. agent previous response (None for round 0)
        if prev_agent_response is not None:
            embedding_text += f"{prev_agent_response}\n"
        
        # 3. previous debate summary (None for round 0)
        if prev_round_summary is not None:
            embedding_text += f"{prev_round_summary}\n"
        
        return embedding_text.strip()
    
    def _to_retrieved_memories(
        self,
        query_id: int,
        retrieved: List[Tuple[Dict, str, float, Optional[np.ndarray]]],
    ) -> List[RetrievedMemory]:
        """Wrap the raw candidate tuples retrieved by a policy as RetrievedMemory."""
        return [
            RetrievedMemory(
                content=doc,
                query_id=query_id,
                agent_id=self.agent_id,
                is_correct=bool(meta.get("response_correct", False)),
                similarity=float(max(0.0, 1.0 - distance)),
            )
            for meta, doc, distance, _ in retrieved
        ]
