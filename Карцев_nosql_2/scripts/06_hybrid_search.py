import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "arxiv-papers"
TOP_K = 10

pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df    = pd.read_parquet("Карцев_nosql_2/data/arxiv_subset.parquet").reset_index(drop=True)

# BM25 по заголовку + абстракту
corpus_texts = (df["title"] + " " + df["abstract"]).tolist()
tokenized    = [t.lower().split() for t in corpus_texts]
bm25         = BM25Okapi(tokenized)


def bm25_search(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def vector_search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    q_emb = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return [(m["id"], m["score"], m["metadata"]) for m in res["matches"]]


def rrf_fusion(bm25_results, vector_results, k: int = 60, top_k: int = 5) -> list:
    """Reciprocal Rank Fusion."""
    rrf: dict[str, dict] = {}
    
    # BM25 ranks: ключ — локальный индекс датафрейма
    for rank, (local_idx, score) in enumerate(bm25_results):
        key = f"bm25_{local_idx}"
        rrf[key] = rrf.get(key, {"score": 0.0, "source": "bm25", "local_idx": local_idx})
        rrf[key]["score"] += 1.0 / (k + rank + 1)
    
    # Vector ranks: ключ — pinecone id вида "paper_N"
    for rank, (pid, score, meta) in enumerate(vector_results):
        rrf[pid] = rrf.get(pid, {"score": 0.0, "source": "vector", "pid": pid, "meta": meta})
        rrf[pid]["score"] += 1.0 / (k + rank + 1)
    
    sorted_docs = sorted(rrf.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return sorted_docs


def hybrid_search(query: str, k_rrf: int = 60, top_k: int = 5):
    bm25_res   = bm25_search(query, top_k=TOP_K)
    vector_res = vector_search(query, top_k=TOP_K)
    return rrf_fusion(bm25_res, vector_res, k=k_rrf, top_k=top_k)


def print_bm25(query: str):
    print(f"\n--- BM25: '{query}' ---")
    for local_idx, score in bm25_search(query, top_k=5):
        title = df.iloc[local_idx]["title"][:60]
        print(f"  [{score:.4f}] {title}")


def print_vector(query: str):
    print(f"\n--- Векторний: '{query}' ---")
    for pid, score, meta in vector_search(query, top_k=5):
        print(f"  [{score:.4f}] {meta.get('title', '')[:60]}")


def print_hybrid(query: str, k_rrf: int = 60):
    print(f"\n--- Гібридний RRF (k={k_rrf}): '{query}' ---")
    for doc in hybrid_search(query, k_rrf=k_rrf):
        score = doc["score"]
        if "meta" in doc:
            title = doc["meta"].get("title", "")[:60]
        elif "local_idx" in doc:
            title = df.iloc[doc["local_idx"]]["title"][:60]
        else:
            title = "N/A"
        print(f"  [RRF={score:.5f}] {title}")


if __name__ == "__main__":
    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text",
    ]
    
    for q in queries:
        print(f"\n{'='*60}\nЗапит: {q}\n{'='*60}")
        print_bm25(q)
        print_vector(q)
        print_hybrid(q, k_rrf=60)
        print_hybrid(q, k_rrf=1)   # для сравнения влияния k