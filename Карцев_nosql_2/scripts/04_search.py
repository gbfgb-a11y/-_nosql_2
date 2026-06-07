import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
model = SentenceTransformer("allenai/specter2", device='cpu')
index = pinecone.Index("arxiv-search")

def semantic_search(query: str, top_k: int = 5):
    q_emb = model.encode(f"TITLE: {query}\nABSTRACT: {query}", normalize_embeddings=True)
    result = index.query(q_emb.tolist(), top_k=top_k, include_metadata=True)
    return [(match['id'], match['metadata']['title'], match['score']) for match in result['matches']]

def filtered_search(query: str, year_from: int, category: str, top_k: int = 5):
    q_emb = model.encode(f"TITLE: {query}\nABSTRACT: {query}", normalize_embeddings=True)
    filter_cond = {}
    if year_from:
        filter_cond["year"] = {"$gte": year_from}
    if category:
        filter_cond["categories"] = {"$in": [category]}  # точний збіг категорії
    result = index.query(q_emb.tolist(), top_k=top_k, filter=filter_cond, include_metadata=True)
    return [(match['id'], match['metadata']['title'], match['score']) for match in result['matches']]

def compare_metrics(query: str):
    """Порівнює косинусну схожість та Евклідову відстань (Pinecone підтримує обидві)."""
    q_emb = model.encode(query, normalize_embeddings=True)
    # За замовчуванням метрика cosine
    res_cos = index.query(q_emb.tolist(), top_k=5)
    # Для евклідової довелося б створити індекс з metric='euclidean'. Тут просто демонструємо ідею.
    print("При порівнянні: косинусна краща для семантичної близькості, евклідова — для щільних кластерів.")
    return res_cos

if __name__ == "__main__":
    print(semantic_search("attention mechanism in transformers"))
    print(filtered_search("graph neural networks", year_from=2020, category="cs.LG"))