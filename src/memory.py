import os
import chromadb
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from src.models import EmbeddingModel


class MemoryVectorDB:
    def __init__(self, persistent_dir: str, collection_name: str, embedding_model: EmbeddingModel,
                 tokenizer: Optional[Callable[[str], List[str]]] = word_tokenize,
                 stemmer: Optional[Callable[[str], List[str]]] = SnowballStemmer("english"), verbose: bool = True):

        self.client = chromadb.PersistentClient(path=persistent_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )
        self.resp_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_responses",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.verbose = verbose

        # Pre-computed document embeddings: db_id -> float32 unit vector (1-D np.ndarray)
        # Loaded lazily via load_doc_embeddings(); None means not yet loaded.
        self._doc_embeddings: Optional[Dict[str, np.ndarray]] = None
    
    # ------------------------------------------------------------------
    # Agent memory collection
    # ------------------------------------------------------------------

    def load_doc_embeddings(self, npz_path: str) -> None:
        """Load pre-computed document embeddings from a .npz file into memory."""
        data = np.load(npz_path, allow_pickle=False)
        db_ids = data["db_ids"].astype(str)
        embeddings = data["embeddings"].astype(np.float32)  # (N, dim)

        # Ensure unit norm (float16 storage may introduce tiny drift)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        self._doc_embeddings = {db_id: embeddings[i] for i, db_id in enumerate(db_ids)}
        if self.verbose:
            print(f"Loaded {len(self._doc_embeddings)} document embeddings from {npz_path}")
    
    def add_memory(self, db_id: str, embedding_text: str, document_text: str,
                   add_meta_datas: Dict = None) -> None:
        try:
            assert isinstance(embedding_text, str), "embedding_text must be str"
            embedding = self.embedding_model.encode(embedding_text)

            meta_datas = dict()
            if add_meta_datas is not None:
                assert isinstance(add_meta_datas, dict), "meta_datas must be Dict"
                for key, val in add_meta_datas.items():
                    meta_datas[key] = val

            tokens = " ".join([self.stemmer.stem(token) for token in self.tokenizer(embedding_text.lower())])
            meta_datas["key_tokens"] = tokens

            self.collection.upsert(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[meta_datas],
                ids=[db_id]
            )
            if self.verbose:
                print(f"Successfully added memory ID.: {db_id}")
        except Exception as e:
            raise Exception(f"Failed to add memory.: {str(e)}")
    
    def add_batch_memories(self, db_ids: List[str], embedding_texts: List[str],
                           document_texts: List[str], add_meta_datas: List[Dict] = None) -> None:
        try:
            assert len(db_ids) == len(embedding_texts) == len(document_texts), \
                "db_ids, embedding_texts, and document_texts must have the same length"

            embeddings = self.embedding_model.encode(embedding_texts, show_progress_bar=self.verbose)

            meta_datas_list = []
            for i, emb_text in enumerate(embedding_texts):
                meta_datas = dict()
                if add_meta_datas is not None:
                    assert isinstance(add_meta_datas[i], dict), "each element of add_meta_datas must be Dict"
                    meta_datas.update(add_meta_datas[i])
                tokens = " ".join([self.stemmer.stem(token) for token in self.tokenizer(emb_text.lower())])
                meta_datas["key_tokens"] = tokens
                meta_datas_list.append(meta_datas)

            self.collection.upsert(
                embeddings=embeddings,
                documents=document_texts,
                metadatas=meta_datas_list,
                ids=db_ids,
            )
            if self.verbose:
                print(f"Successfully added {len(db_ids)} memories in batch.")
        except Exception as e:
            raise Exception(f"Failed to add memories in batch: {str(e)}")
    
    def query_similar(self, query_text: str, n_results: int = 3, filter_metadata: Dict = None) -> List[Tuple[Dict, str, float, Optional[np.ndarray]]]:
        """
        Returns a list of (metadata, document, distance, doc_embedding) tuples.
        doc_embedding is a float32 unit-norm np.ndarray when pre-computed embeddings
        are loaded via load_doc_embeddings(); otherwise None.
        """
        try:
            query_embedding = self.embedding_model.encode(query_text)

            if filter_metadata is not None:
                assert isinstance(filter_metadata, dict), "filter_metadata must be Dict"

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
                where=filter_metadata
            )

            similar_pairs = []
            for metadata, document, distance, db_id in zip(
                results['metadatas'][0], results['documents'][0],
                results['distances'][0], results['ids'][0],
            ):
                doc_emb = (
                    self._doc_embeddings.get(db_id)
                    if self._doc_embeddings is not None
                    else None
                )
                similar_pairs.append((metadata, document, distance, doc_emb))

            return similar_pairs
        except Exception as e:
            raise Exception(f"Query failed.: {str(e)}")
    
    def batch_query_similar(
        self,
        query_texts: List[str],
        n_results: int = 3,
        filter_metadata: Dict = None,
    ) -> List[List[Tuple[Dict, str, float, Optional[np.ndarray]]]]:
        """
        Batch version of query_similar: encode all query_texts in one call,
        then issue a single ChromaDB query with multiple query embeddings.
        Returns one result list per query_text (same order).
        """
        try:
            query_embeddings = self.embedding_model.encode(query_texts)

            if filter_metadata is not None:
                assert isinstance(filter_metadata, dict), "filter_metadata must be Dict"

            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
                where=filter_metadata,
            )

            all_results = []
            for i in range(len(query_texts)):
                similar_pairs = []
                for metadata, document, distance, db_id in zip(
                    results["metadatas"][i],
                    results["documents"][i],
                    results["distances"][i],
                    results["ids"][i],
                ):
                    doc_emb = (
                        self._doc_embeddings.get(db_id)
                        if self._doc_embeddings is not None
                        else None
                    )
                    similar_pairs.append((metadata, document, distance, doc_emb))
                all_results.append(similar_pairs)

            return all_results
        except Exception as e:
            raise Exception(f"Batch query failed: {str(e)}")
    
    def batch_get_random(
        self,
        n_queries: int,
        n_results: int = 3,
        filter_metadata: Dict = None,
        distance: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[List[Tuple[Dict, str, float, Optional[np.ndarray]]]]:
        """
        Draw n_queries independent random samples (without replacement) of
        n_results cases each from the collection, optionally restricted by
        filter_metadata. Fetches the matching pool once, then samples in Python.

        Returns one candidate list per query in the same 4-tuple shape as
        batch_query_similar: (metadata, document, distance, doc_embedding).
        `distance` is a constant placeholder (random cases have no similarity to
        a query).
        """
        try:
            # Independent per-query RNG streams: sequential draws from one
            # Generator can collide for some seeds, so spawn a child stream per
            # query (reproducible given `seed`, distinct across queries).
            child_seeds = np.random.SeedSequence(seed).spawn(n_queries)

            pool = self.collection.get(
                where=filter_metadata,
                include=["metadatas", "documents"],
            )
            ids = pool["ids"]
            metas = pool["metadatas"]
            docs = pool["documents"]

            all_results = []
            for q in range(n_queries):
                candidates = []
                if ids:
                    rng = np.random.default_rng(child_seeds[q])
                    k = min(n_results, len(ids))
                    idxs = rng.choice(len(ids), size=k, replace=False)
                    for j in idxs:
                        doc_emb = (
                            self._doc_embeddings.get(ids[j])
                            if self._doc_embeddings is not None
                            else None
                        )
                        candidates.append((metas[j], docs[j], distance, doc_emb))
                all_results.append(candidates)
            
            return all_results
        except Exception as e:
            raise Exception(f"Random fetch failed: {str(e)}")

    def get_by_ids(self, select_ids: List[str]) -> List[Tuple[Dict, str]]:
        try:
            results = self.collection.get(ids=select_ids, include=['metadatas', 'documents'])

            similar_pairs = []
            for metadata, document in zip(results['metadatas'], results['documents']):
                similar_pairs.append((metadata, document))

            return similar_pairs
        except Exception as e:
            raise Exception(f"Query by IDs failed: {str(e)}")
    
    def update_memory(self, db_id: str, embedding_text: str, document_text: str,
                      add_meta_datas: Dict = None) -> None:

        try:
            assert isinstance(embedding_text, str), "embedding_text must be str"
            embedding = self.embedding_model.encode(embedding_text)

            meta_datas = dict()
            if add_meta_datas is not None:
                assert isinstance(add_meta_datas, dict), "meta_datas must be Dict"
                for key, val in add_meta_datas.items():
                    meta_datas[key] = val

            tokens = " ".join([self.stemmer.stem(token) for token in self.tokenizer(embedding_text.lower())])
            meta_datas["key_tokens"] = tokens

            self.collection.update(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[meta_datas],
                ids=[db_id]
            )

            if self.verbose:
                print(f"Successfully updated memory ID: {db_id}")
        except Exception as e:
            raise Exception(f"Failed to update memory: {str(e)}")
    
    def delete_memory(self, db_id: str) -> None:
        try:
            self.collection.delete(ids=[db_id])
            if self.verbose:
                print(f"Successfully deleted memory ID: {db_id}")
        except Exception as e:
            raise Exception(f"Failed to delete memory: {str(e)}")
    
    def get_memories_count(self) -> int:
        return self.collection.count()
    
    def get_all_ids(self, filter_metadata=None):
        data = self.collection.get(include=["metadatas"], where=filter_metadata)
        ids = data['ids']
        return ids
    
    # ------------------------------------------------------------------
    # Response embedding collection
    # ------------------------------------------------------------------
    
    @staticmethod
    def _make_resp_id(query_id, round_idx: int, agent_id: str) -> str:
        return f"Q{query_id}_Round{round_idx}_{agent_id}"

    def add_batch_responses(self, db_ids: List[str], texts: List[str],
                            meta_datas: List[Dict]) -> None:
        embeddings = self.embedding_model.encode(texts, show_progress_bar=self.verbose)
        self.resp_collection.upsert(
            embeddings=embeddings,
            documents=texts,
            metadatas=meta_datas,
            ids=db_ids,
        )
        if self.verbose:
            print(f"Successfully added {len(db_ids)} response embeddings in batch.")

    def get_response_embeddings(self, query_id, round_idx: int,
                                agent_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Return {agent_id: unit-norm embedding} for the given agents at a specific
        (query_id, round_idx). Missing entries map to None.
        """
        ids = [self._make_resp_id(query_id, round_idx, aid) for aid in agent_ids]
        try:
            results = self.resp_collection.get(ids=ids, include=["embeddings"])
        except Exception:
            return {aid: None for aid in agent_ids}

        id_to_emb: Dict[str, np.ndarray] = {
            db_id: np.array(emb, dtype=np.float32)
            for db_id, emb in zip(results["ids"], results["embeddings"])
        }

        out = {}
        for aid, db_id in zip(agent_ids, ids):
            out[aid] = id_to_emb.get(db_id)
        return out
    
    def batch_get_response_embeddings(
        self,
        keys: List[Tuple],  # list of (query_id, round_idx, agent_id_str)
    ) -> Dict[Tuple, Optional[np.ndarray]]:
        """
        Fetch response embeddings for many (query_id, round_idx, agent_id) triples
        in a single ChromaDB get call.  Returns {(query_id, round_idx, agent_id): vec}.
        """
        ids = [self._make_resp_id(qid, r, aid) for qid, r, aid in keys]
        try:
            results = self.resp_collection.get(ids=ids, include=["embeddings"])
        except Exception:
            return {k: None for k in keys}

        id_to_emb: Dict[str, np.ndarray] = {
            db_id: np.array(emb, dtype=np.float32)
            for db_id, emb in zip(results["ids"], results["embeddings"])
        }

        out = {}
        for key, db_id in zip(keys, ids):
            out[key] = id_to_emb.get(db_id)
        return out
