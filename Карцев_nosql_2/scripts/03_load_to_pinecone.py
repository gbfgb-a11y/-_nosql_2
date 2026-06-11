import os
import time
import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 384
BATCH_SIZE = 200

def init_pinecone(index_name: str):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing:
        print(f"Створюємо індекс {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Ждём пока индекс будет готов
        while not pc.describe_index(index_name).status["ready"]:
            print("Чекаємо готовності індексу...")
            time.sleep(3)
    else:
        print(f"Індекс {index_name} вже існує")
    
    return pc.Index(index_name)

def upload_to_pinecone(df: pd.DataFrame, embeddings: np.ndarray):
    index = init_pinecone(INDEX_NAME)
    total = len(df)
    
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Завантаження в Pinecone"):
        batch_vectors = []
        for j in range(i, min(i + BATCH_SIZE, total)):
            row = df.iloc[j]
            
            year_val = row.get("year", 0)
            try:
                year_int = int(year_val) if year_val and str(year_val) != "nan" else 0
            except (ValueError, TypeError):
                year_int = 0
            
            # Берём первую категорию если их несколько
            categories_raw = str(row.get("categories", row.get("category", "unknown")))
            primary_category = categories_raw.split()[0]
            
            metadata = {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"])[:500],
                "abstract": str(row["abstract"])[:500],
                "authors": str(row.get("authors", ""))[:200],
                "year": year_int,
                "category": primary_category,
            }
            batch_vectors.append((
                f"paper_{j}",
                embeddings[j].tolist(),
                metadata
            ))
        
        index.upsert(vectors=batch_vectors)
    
    time.sleep(5)  # даём время на синхронизацию
    stats = index.describe_index_stats()
    print(f"\nВсього векторів в індексі: {stats['total_vector_count']}")

if __name__ == "__main__":
    df = pd.read_parquet("Карцев_nosql_2/data/arxiv_subset.parquet")
    embeddings = np.load("Карцев_nosql_2/embeddings/embeddings.npy")
    print(f"Датафрейм: {len(df)} записів, ембеддинги: {embeddings.shape}")
    upload_to_pinecone(df, embeddings)