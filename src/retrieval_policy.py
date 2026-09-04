"""
Pluggable memory-retrieval policies.

A retrieval policy owns the *whole* two-stage retrieval — candidate generation
(the ChromaDB query) and selection — and returns, per query, both the retrieved
cases (which get formatted into the prompt) and the full candidate pool (which the
Agent reuses for memory-estimated confidence). Keeping the candidates in the
return value is what keeps confidence coupled to whatever cases the policy
actually retrieved (see Agent.batch_retrieve_all).

Policies operate purely on the raw ChromaDB candidate tuples

    Candidate = (metadata: Dict, document: str, distance: float, doc_embedding: Optional[np.ndarray])

and primitives, so this module does not import from `agent.py` (avoids a circular
import). The Agent converts selected candidates into RetrievedMemory objects.

The debate-state-aware policy is the method described in the paper; the other
three (similarity / random / diversity) exist as retrieval-policy ablations.
"""

import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple

Candidate = Tuple[Dict, str, float, Optional[np.ndarray]]

# Sentinel distance used for cases retrieved without a similarity notion (random
# policy). 1.0 keeps the logged similarity (1 - distance) at 0.0.
_NO_SIMILARITY_DISTANCE = 1.0


# ----------------------------------------------------------------------
# Shared selection helpers
# ----------------------------------------------------------------------

def _doc_embedding_matrix(candidates: List[Candidate], embedding_model=None) -> np.ndarray:
    """
    Stack candidate document embeddings into an (n, dim) matrix. Uses the
    pre-computed embeddings loaded from the .npz sidecar when all are present;
    otherwise falls back to encoding the documents on the fly.
    """
    precomputed = [c[3] for c in candidates]
    if all(e is not None for e in precomputed):
        return np.stack(precomputed)
    if embedding_model is None:
        raise ValueError("Document embeddings missing and no embedding_model provided for fallback")
    doc_texts = [c[1] for c in candidates]
    return np.array(embedding_model.encode(doc_texts))


def mmr_select(
    candidates: List[Candidate],
    n: int,
    lam: float,
    embedding_model=None,
) -> List[Candidate]:
    """
    Maximal Marginal Relevance greedy selection (debate-state-aware policy).

    Greedily pick n cases maximizing
        lambda * sim(e, q) * r(e) - (1 - lambda) * max_{e' in selected} sim(e', e)
    where sim(e, q) = 1 - distance, r(e) is case correctness (0/1), and the
    inter-candidate similarity is cosine of the document embeddings.

    Returns the selected candidate tuples in selection order.
    """
    if not candidates:
        return []

    sim_to_query = np.array([1.0 - c[2] for c in candidates])
    case_correct = np.array([
        1.0 if c[0].get("response_correct", False) else 0.0 
        for c in candidates
    ])

    doc_embeddings = _doc_embedding_matrix(candidates, embedding_model)
    sim_matrix = doc_embeddings @ doc_embeddings.T  # (n_candidates, n_candidates)

    n = min(n, len(candidates))
    selected_indices = []
    remaining = list(range(len(candidates)))
    for _ in range(n):
        if not selected_indices:
            max_sim_to_selected = np.zeros(len(candidates))
        else:
            max_sim_to_selected = sim_matrix[:, selected_indices].max(axis=1)

        scores = np.full(len(candidates), -np.inf)
        for i in remaining:
            scores[i] = (
                lam * sim_to_query[i] * case_correct[i]
                - (1.0 - lam) * max_sim_to_selected[i]
            )

        best = int(np.argmax(scores))
        selected_indices.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_indices]


def diversity_select(
    candidates: List[Candidate],
    n: int,
    embedding_model=None,
    seed: Optional[int] = None,
) -> List[Candidate]:
    """
    Pure diversity selection: anchor on a random candidate, then greedily add the candidate 
    that minimizes its maximum cosine similarity to those already selected. Ignores query relevance and case correctness.
    """
    if not candidates:
        return []

    doc_embeddings = _doc_embedding_matrix(candidates, embedding_model)
    sim_matrix = doc_embeddings @ doc_embeddings.T

    # Anchor the diverse set on a randomly chosen candidate, so the selection carries no query-similarity bias.
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(len(candidates)))

    n = min(n, len(candidates))
    selected_indices = [anchor]
    remaining = [i for i in range(len(candidates)) if i != anchor]
    while len(selected_indices) < n and remaining:
        best = min(remaining, key=lambda i: float(sim_matrix[i, selected_indices].max()))
        selected_indices.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_indices]


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------

class RetrievalPolicy:
    """Base class. Subclasses implement the full two-stage retrieval."""

    name: str = "base"

    def __init__(self, seed: Optional[int] = None):
        # Stateful RNG seeded once from the base seed.
        self._rng = np.random.default_rng(seed)

    def batch_retrieve(
        self,
        memory_db,
        agent_id: str,
        queries: List[Tuple[int, str]],
        n_retrieval: int,
        prev_consensus_ratios: Dict[int, Optional[float]],
    ) -> Dict[int, Tuple[List[Candidate], List[Candidate]]]:
        """
        Args:
            memory_db:             MemoryVectorDB instance.
            agent_id:              str(agent_id) used to filter the collection.
            queries:               [(query_id, embedding_text)] for queries needing retrieval.
            n_retrieval:           number of cases to select per query.
            prev_consensus_ratios: {query_id: prev-round consensus ratio or None}.

        Returns:
            {query_id: (retrieved, candidates)}
            where `retrieved` are the cases selected for the prompt and
            `candidates` is the full candidate pool (retrieved ⊆ candidates).
        """
        raise NotImplementedError
    
    def _next_seed(self) -> int:
        """Draw a fresh child seed from the policy's RNG."""
        return int(self._rng.integers(np.iinfo(np.int64).max))

    @staticmethod
    def _filter(agent_id: str, correct_only: bool = False) -> Dict:
        base = {"agent_id": {"$eq": agent_id}}
        if correct_only:
            return {"$and": [base, {"response_correct": {"$eq": True}}]}
        return base


class DebateStateAwarePolicy(RetrievalPolicy):
    """Semantic top-3n candidates + adaptive MMR re-ranking.

    With adaptive_lambda=True (paper method), the MMR relevance/diversity tradeoff
    lambda = 1 - gamma * consensus_ratio adapts to the previous round's consensus.
    With adaptive_lambda=False (fixed_lambda ablation), lambda is held constant at
    ref_lambda regardless of consensus.
    """
    name = "debate_state_aware"

    def __init__(self, gamma: float = 0.9, ref_lambda: float = 0.7,
                 adaptive_lambda: bool = True, seed: Optional[int] = None):
        super().__init__(seed)
        self.gamma = gamma
        self.ref_lambda = ref_lambda
        self.adaptive_lambda = adaptive_lambda

    def _lambda_for(self, consensus_ratio: Optional[float]) -> float:
        if self.adaptive_lambda and consensus_ratio is not None:
            return 1.0 - self.gamma * consensus_ratio
        return self.ref_lambda

    def batch_retrieve(self, memory_db, agent_id, queries, n_retrieval, prev_consensus_ratios):
        embedding_texts = [emb_text for _, emb_text in queries]
        batch_candidates = memory_db.batch_query_similar(
            query_texts=embedding_texts,
            n_results=n_retrieval * 3,
            filter_metadata=self._filter(agent_id),
        )

        out = {}
        for (qid, _), candidates in zip(queries, batch_candidates):
            lam = self._lambda_for(prev_consensus_ratios.get(qid))
            retrieved = mmr_select(candidates, n_retrieval, lam, memory_db.embedding_model)
            out[qid] = (retrieved, candidates)

        return out


class FixedLambdaPolicy(DebateStateAwarePolicy):
    """Debate_state_aware MMR with a constant lambda (no consensus adaptation)."""
    name = "fixed_lambda"

    def __init__(self, ref_lambda: float = 0.7, seed: Optional[int] = None):
        super().__init__(ref_lambda=ref_lambda, adaptive_lambda=False, seed=seed)


class SimilarityPolicy(RetrievalPolicy):
    """Plain semantic top-n retrieval (no re-ranking, no correctness)."""
    name = "similarity"

    def batch_retrieve(self, memory_db, agent_id, queries, n_retrieval, prev_consensus_ratios):
        embedding_texts = [emb_text for _, emb_text in queries]
        batch_candidates = memory_db.batch_query_similar(
            query_texts=embedding_texts,
            n_results=n_retrieval * 3,
            filter_metadata=self._filter(agent_id),
        )
        return {
            qid: (candidates[:n_retrieval], candidates)
            for (qid, _), candidates in zip(queries, batch_candidates)
        }


class DiversityPolicy(RetrievalPolicy):
    """Semantic top-3n candidates + pure diversity selection."""
    name = "diversity"

    def batch_retrieve(self, memory_db, agent_id, queries, n_retrieval, prev_consensus_ratios):
        embedding_texts = [emb_text for _, emb_text in queries]
        batch_candidates = memory_db.batch_query_similar(
            query_texts=embedding_texts,
            n_results=n_retrieval * 3,
            filter_metadata=self._filter(agent_id),
        )

        out = {}
        for (qid, _), candidates in zip(queries, batch_candidates):
            retrieved = diversity_select(candidates, n_retrieval, memory_db.embedding_model, seed=self._next_seed())
            out[qid] = (retrieved, candidates)
        
        return out


class RandomPolicy(RetrievalPolicy):
    """Uniformly random n cases from the whole collection (ignores query)."""
    name = "random"

    def batch_retrieve(self, memory_db, agent_id, queries, n_retrieval, prev_consensus_ratios):
        batch_candidates = memory_db.batch_get_random(
            n_queries=len(queries),
            n_results=n_retrieval * 3,
            filter_metadata=self._filter(agent_id),
            distance=_NO_SIMILARITY_DISTANCE,
            seed=self._next_seed(),
        )
        return {
            qid: (candidates[:n_retrieval], candidates)
            for (qid, _), candidates in zip(queries, batch_candidates)
        }


class PositiveOnlyPolicy(RetrievalPolicy):
    """Semantic retrieval restricted to correct (positive) cases only."""
    name = "positive_only"

    def batch_retrieve(self, memory_db, agent_id, queries, n_retrieval, prev_consensus_ratios):
        embedding_texts = [emb_text for _, emb_text in queries]
        batch_candidates = memory_db.batch_query_similar(
            query_texts=embedding_texts,
            n_results=n_retrieval * 3,
            filter_metadata=self._filter(agent_id, correct_only=True),
        )
        return {
            qid: (candidates[:n_retrieval], candidates)
            for (qid, _), candidates in zip(queries, batch_candidates)
        }


POLICY_REGISTRY = {
    DebateStateAwarePolicy.name: DebateStateAwarePolicy,
    SimilarityPolicy.name: SimilarityPolicy,
    DiversityPolicy.name: DiversityPolicy,
    RandomPolicy.name: RandomPolicy,
    FixedLambdaPolicy.name: FixedLambdaPolicy,
    PositiveOnlyPolicy.name: PositiveOnlyPolicy,
}


def build_retrieval_policy(name: str, seed: Optional[int] = None, **policy_kwargs) -> RetrievalPolicy:
    """
    Instantiate a policy by name. `seed` is passed to every policy; any extra
    policy_kwargs (e.g. ref_lambda, gamma) are forwarded only to policies whose
    constructor accepts them, and only when not None — so callers can pass a
    superset of knobs regardless of which policy is selected.
    """
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown retrieval_policy '{name}'. Available: {sorted(POLICY_REGISTRY)}"
        )
    cls = POLICY_REGISTRY[name]
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {"seed": seed}
    for key, value in policy_kwargs.items():
        if value is not None and key in accepted:
            kwargs[key] = value
    return cls(**kwargs)
