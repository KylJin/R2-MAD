import os

# Project root (this file lives at src/paths.py, so root is one level up)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw datasets downloaded from HuggingFace
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data")

# Processed / formatted datasets ready for DataLoader
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "processed_data")

# Debate result JSON logs
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# JSONL files written during training for memory construction
MEMORY_DATA_DIR = os.path.join(ROOT_DIR, "memory_data")

# ChromaDB persistent vector store for the memory module; also holds pre-computed
# document embedding .npz files (one per collection, named {collection}_doc_embeddings.npz)
PERSISTENT_DIR = os.path.join(ROOT_DIR, "persistent_memory")
