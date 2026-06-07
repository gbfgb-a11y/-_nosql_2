import numpy as np
from rank_bm25 import BM25Okapi
import pandas as pd
from sentence_transformers import SentenceTransformer

# Завантажуємо корпус та модель
df = pd.read_parquet("data/arxiv_subset.parquet")
tokenized_corpus = [doc.split() for doc in df['abstract']]
bm25 = BM25Okapi(tokenized_corpus)
model = SentenceTransformer("allenai/specter2")

def hybrid_search(query: str, top_k: int = 10, k_rrf: int = 60):
    """Гібридний пошук з Reciprocal Rank Fusion."""
    # 1. Векторний пошук (через Pinecone або локально)
    q_emb = model.encode(f"TITLE: {query}\nABSTRACT: {query}", normalize_embeddings=True)
    # Імітуємо векторний пошук за косинусною схожістю всього корпусу (для простоти, в реальності – індекс)
    doc_embs = np.load("embeddings/embeddings.npy")
    vec_scores = np.dot(doc_embs, q_emb)  # косинусна (вектори нормовані)
    vec_ranks = np.argsort(-vec_scores)[:top_k * 2]
    
    # 2. BM25 пошук
    bm25_scores = bm25.get_scores(query.split())
    bm25_ranks = np.argsort(-bm25_scores)[:top_k * 2]
    
    # 3. RRF
    rrf_scores = {}
    for rank, idx in enumerate(vec_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
    
    # Сортуємо за RRF
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    results = [(idx, rrf_scores[idx]) for idx in sorted_ids]
    return results

if __name__ == "__main__":
    query = "neural network optimization"
    results = hybrid_search(query)
    for doc_id, score in results:
        print(f"{doc_id}: {df.iloc[doc_id]['title']} (RRF={score:.4f})")