import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def read_memory_data(data_dir: str) -> List[Dict]:
    """
    Read all JSONL files under data_dir into a list of dicts, sorted by db_id.
    """
    rows = []

    for jsonl_file in sorted(Path(data_dir).glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    rows.sort(key=lambda r: r["db_id"])
    return rows


def sample_memories(records: List[Dict], sample_size: int = -1, seed: int = 42) -> List[Dict]:
    """
    Sample up to sample_size unique questions (all records for each sampled
    question are retained).  sample_size == -1 means keep all.
    Round-0 cases are excluded before sampling since memory is only used from round 1.
    """
    random.seed(seed)

    records = [r for r in records if int(r.get("round", 0)) > 0]

    question_ids = sorted({r["query_id"] for r in records})
    random.shuffle(question_ids)

    if sample_size == -1 or sample_size >= len(question_ids):
        print("All memory questions sampled.")
        sample_size = len(question_ids)

    selected_query_ids = set(question_ids[:sample_size])
    records_sample = [r for r in records if r["query_id"] in selected_query_ids]

    return records_sample


# ---------------------------------------------------------------------------
# Memory construction
# ---------------------------------------------------------------------------

def _build_embedding_text(row: Dict) -> str:
    """
    Construct the retrieval key (debate state) from:
      1. task x  (query)
      2. agent previous response z_i^{t-1}
      3. previous debate summary
    """
    embedding_text = ""

    # 1. task
    embedding_text += f"{row['query']}\n"

    # 2. agent previous response (None for round 0)
    prev_response = row.get("prev_response")
    if prev_response:
        embedding_text += f"{prev_response}\n"
    
    # 3. previous debate summary (None for round 0)
    prev_summary = row.get("prev_round_summary")
    if prev_summary:
        embedding_text += f"{prev_summary}\n"
    
    return embedding_text.strip()


def _build_value_text(row: Dict) -> str:
    """
    Construct the retrieval value (prompt context) from:
      1. task x  (query)
      2. correct solution z^*
      3. agent current response z_i^t
      4. whether agent changed response
      5. agent current correctness
    All parts are wrapped in XML tags.
    """
    value_text = ""

    # 1. task
    value_text += f"<example_question>\n{row['query']}\n</example_question>\n\n"

    # 2. correct solution
    value_text += f"<example_solution>\n{row['solution']}\n</example_solution>\n\n"

    # 3. agent current response
    value_text += f"<example_agent_response>\n{row['current_response']}\n</example_agent_response>\n\n"

    # 4. whether agent changed response (N/A for round 0)
    stance_changed = row.get("stance_changed")
    if stance_changed is None:
        changed_text = "N/A (initial round)"
    else:
        changed_text = "The agent changed its stance." if stance_changed else "The agent did not change its stance."
    value_text += f"<example_agent_stance_changed>\n{changed_text}\n</example_agent_stance_changed>\n\n"

    # 5. agent current correctness
    is_correct = row.get("is_current_correct", False)
    if is_correct:
        correct_text = "The response is correct."
    else:
        correct_text = "The response is incorrect."
    value_text += f"<example_agent_response_correctness>\n{correct_text}\n</example_agent_response_correctness>"

    return value_text


def _build_meta_data(row: Dict) -> Dict:
    """
    Construct the ChromaDB metadata dict.
    All values must be ChromaDB-compatible scalars (str / int / float / bool).
    """
    prev_consensus = row.get("prev_consensus_ratio")

    return {
        "category": str(row.get("category", "")),
        "question_id": str(row.get("query_id", "")),
        "agent_id": str(row.get("agent_id", "")),
        "current_round": int(row.get("round", 0)),
        "previous_consensus_ratio": float(prev_consensus) if prev_consensus is not None else -1.0,
        "response_correct": bool(row.get("is_current_correct", False)),
        "final_correct": bool(row.get("is_final_correct", False)),
    }


def construct_memory(row: Dict) -> Tuple[str, str, str, Dict]:
    """
    Build all four components needed to store one memory entry:
      db_id, embedding_text, value_text, meta_data
    """
    db_id          = row["db_id"]
    embedding_text = _build_embedding_text(row)
    value_text     = _build_value_text(row)
    meta_data      = _build_meta_data(row)

    return db_id, embedding_text, value_text, meta_data


def construct_response_memory(row: Dict) -> Tuple[str, str, Dict]:
    """
    Build the (db_id, response_text, metadata) triple for a single memory row's
    current_response, to be stored in the response-embedding collection.
    """
    response_text = row["current_response"]
    meta_data = {
        "category": str(row.get("category", "")),
        "question_id": str(row.get("query_id", "")),
        "agent_id": str(row.get("agent_id", "")),
        "round": int(row.get("round", 0)),
        "response_correct": bool(row.get("is_current_correct", False)),
        "final_correct": bool(row.get("is_final_correct", False)),
    }
    return row["db_id"], response_text, meta_data
