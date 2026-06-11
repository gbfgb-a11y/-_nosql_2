import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384
INDEX_FIXED    = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
model = SentenceTransformer(MODEL_NAME)
df    = pd.read_parquet("Карцев_nosql_2/data/arxiv_subset.parquet")


# ── Chunking strategies ──────────────────────────────────────────────────────

def fixed_size_chunking(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Фіксований розмір у словах з перекриттям."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def semantic_chunking(text: str, max_words: int = 100) -> list[str]:
    """Об'єднання речень до досягнення max_words слів."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_words = len(sent.split())
        if current_len + sent_words > max_words and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_words
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ── Pinecone helpers ─────────────────────────────────────────────────────────

def get_or_create_index(name: str) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    return pc.Index(name)


def upload_chunks(index, chunks_data: list, batch_size: int = 100):
    for i in tqdm(range(0, len(chunks_data), batch_size), desc=f"Завантаження {index}"):
        batch = chunks_data[i: i + batch_size]
        texts = [c["text"] for c in batch]
        embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        
        vectors = []
        for k, (c, emb) in enumerate(zip(batch, embs)):
            vectors.append((
                c["id"],
                emb.tolist(),
                {
                    "arxiv_id":    c["arxiv_id"],
                    "title":       c["title"][:200],
                    "chunk_text":  c["text"][:500],
                    "chunk_index": c["chunk_index"],
                    "year":        c["year"],
                    "category":    c["category"],
                }
            ))
        index.upsert(vectors=vectors)


# ── Build chunk lists ────────────────────────────────────────────────────────

df["abstract_len"] = df["abstract"].str.len()

top30 = (
    df.sort_values("abstract_len", ascending=False)
      .head(30)
      .copy()
)

fixed_chunks, semantic_chunks = [], []

for _, row in top30.iterrows():
    text     = f"{row['title']} [SEP] {row['abstract']}"
    arxiv_id = str(row["id"])
    title    = str(row["title"])
    year_val = int(row["year"]) if pd.notna(row.get("year")) else 0
    cat      = str(row.get("categories", row.get("category", "unknown"))).split()[0]
    
    for i, chunk in enumerate(fixed_size_chunking(text)):
        fixed_chunks.append({
            "id": f"fixed_{arxiv_id}_{i}",
            "text": chunk, "arxiv_id": arxiv_id,
            "title": title, "chunk_index": i,
            "year": year_val, "category": cat,
        })
    
    for i, chunk in enumerate(semantic_chunking(text)):
        semantic_chunks.append({
            "id": f"sem_{arxiv_id}_{i}",
            "text": chunk, "arxiv_id": arxiv_id,
            "title": title, "chunk_index": i,
            "year": year_val, "category": cat,
        })

print(f"Fixed chunks: {len(fixed_chunks)}")
print(f"Semantic chunks: {len(semantic_chunks)}")


# ── Upload ───────────────────────────────────────────────────────────────────

idx_fixed    = get_or_create_index(INDEX_FIXED)
idx_semantic = get_or_create_index(INDEX_SEMANTIC)

upload_chunks(idx_fixed,    fixed_chunks)
upload_chunks(idx_semantic, semantic_chunks)


# ── Search demo ──────────────────────────────────────────────────────────────

def search_chunks(index, query: str, top_k: int = 5):
    q_emb = model.encode(query, normalize_embeddings=True).tolist()
    res   = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return res["matches"]


test_queries = [
    "neural network training optimization",
    "natural language processing text classification",
]

for q in test_queries:
    print(f"\n{'='*60}\nЗапит: {q}\n{'='*60}")
    
    print("\n-- Fixed chunking --")
    for m in search_chunks(idx_fixed, q):
        print(f"  [{m['score']:.4f}] {m['metadata']['title'][:50]}")
        print(f"           chunk: {m['metadata']['chunk_text'][:100]}...")
    
    print("\n-- Semantic chunking --")
    for m in search_chunks(idx_semantic, q):
        print(f"  [{m['score']:.4f}] {m['metadata']['title'][:50]}")
        print(f"           chunk: {m['metadata']['chunk_text'][:100]}...")