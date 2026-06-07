import os
import pinecone
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def init_pinecone(index_name: str = "arxiv-search"):
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
    # Якщо індекс існує – видаляємо (для чистоти експерименту)
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    # Створюємо індекс: розмірність 768 (specter2), метрика cosine
    pinecone.create_index(index_name, dimension=768, metric="cosine")
    return pinecone.Index(index_name)

def upload_to_pinecone(df: pd.DataFrame, embeddings: np.ndarray, batch_size: int = 100):
    index = init_pinecone()
    vectors = []
    for i, (_, row) in enumerate(df.iterrows()):
        metadata = {
            "title": row['title'],
            "abstract": row['abstract'][:500],  # обмежуємо для економії
            "authors": row['authors'],
            "year": int(row['year']) if pd.notna(row['year']) else 0,
            "categories": row['categories']
        }
        vectors.append((str(row['id']), embeddings[i].tolist(), metadata))
        
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            vectors = []
    if vectors:
        index.upsert(vectors)
    print("Вектори завантажено в Pinecone")

if __name__ == "__main__":
    df = pd.read_parquet("data/arxiv_subset.parquet")
    embeddings = np.load("embeddings/embeddings.npy")
    upload_to_pinecone(df, embeddings)