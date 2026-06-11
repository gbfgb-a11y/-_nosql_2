import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_NAME = "arxiv-papers"
TOP_K = 5

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("Карцев_nosql_2/data/arxiv_subset.parquet")


def encode_query(query: str) -> list:
    # specter2 не требует префиксов — просто текст запроса
    emb = model.encode(query, normalize_embeddings=True)
    return emb.tolist()


def semantic_search(query: str, top_k: int = TOP_K):
    q_emb = encode_query(query)
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return results["matches"]


def filtered_search(query: str, year_from: int = None, year_to: int = None,
                    category: str = None, top_k: int = TOP_K):
    q_emb = encode_query(query)
    
    filter_cond = {}
    year_filter = {}
    if year_from:
        year_filter["$gte"] = year_from
    if year_to:
        year_filter["$lte"] = year_to
    if year_filter:
        filter_cond["year"] = year_filter
    if category:
        filter_cond["category"] = {"$eq": category}
    
    results = index.query(
        vector=q_emb,
        top_k=top_k,
        filter=filter_cond if filter_cond else None,
        include_metadata=True
    )
    return results["matches"]


def print_results(matches, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not matches:
        print("  Результатів не знайдено")
        return
    for i, m in enumerate(matches, 1):
        meta = m["metadata"]
        print(f"\n{i}. {meta.get('title', 'N/A')}")
        print(f"   Категорія: {meta.get('category', '?')}  |  Рік: {meta.get('year', '?')}")
        print(f"   Score: {m['score']:.4f}")
        print(f"   Abstract: {meta.get('abstract', '')[:120]}...")


def compare_metrics_local(query: str):
    """Порівняння cosine, dot product і L2 на локальних ембеддингах."""
    print(f"\n{'='*60}")
    print(f"  Порівняння метрик для: '{query}'")
    print(f"{'='*60}")
    
    embeddings = np.load("Карцев_nosql_2/embeddings/embeddings.npy")
    q_emb = np.array(encode_query(query))
    
    # Cosine similarity (для нормалізованих = dot product)
    cosine_scores = embeddings @ q_emb
    cosine_top5 = np.argsort(-cosine_scores)[:5]
    
    # Dot product (той самий результат при нормалізації)
    dot_scores = embeddings @ q_emb
    dot_top5 = np.argsort(-dot_scores)[:5]
    
    # L2 distance (менше = краще)
    l2_scores = np.linalg.norm(embeddings - q_emb, axis=1)
    l2_top5 = np.argsort(l2_scores)[:5]
    
    for label, top5, scores, reverse in [
        ("Cosine similarity", cosine_top5, cosine_scores, True),
        ("Dot product",       dot_top5,    dot_scores,    True),
        ("L2 distance",       l2_top5,     l2_scores,     False),
    ]:
        print(f"\n--- {label} ---")
        for rank, idx in enumerate(top5, 1):
            title = df.iloc[idx]["title"][:60]
            score = scores[idx]
            print(f"  {rank}. [{score:.4f}] {title}")


if __name__ == "__main__":
    # Діагностика
    stats = index.describe_index_stats()
    print(f"Векторів в індексі: {stats['total_vector_count']}")
    
    # 1. Чистий семантичний пошук
    query1 = "teaching machines to recognize objects in pictures"
    matches1 = semantic_search(query1)
    print_results(matches1, f"Семантичний пошук: '{query1}'")
    
    # 2A. Фільтр: reinforcement learning, останні 5 років, cs.LG
    query2 = "reinforcement learning"
    matches2a = filtered_search(query2, year_from=2000, category="cs.LG")
    print_results(matches2a, f"Фільтр A: RL після 2000, cs.LG")
    
    # 2B. Фільтр: старі статті до 2015
    matches2b = filtered_search(query2, year_to=2010)
    print_results(matches2b, f"Фільтр B: до 2010, будь-яка категорія")
    
    # 3. Порівняння метрик локально
    compare_metrics_local("attention mechanism in neural networks")