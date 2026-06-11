import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_corpus(df: pd.DataFrame):
    os.makedirs("Карцев_nosql_2/embeddings", exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        f"{row['title']} [SEP] {row['abstract']}"
        for _, row in df.iterrows()
    ]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(f"Кількість текстів: {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

    np.save(
        "Карцев_nosql_2/embeddings/embeddings.npy",
        embeddings
    )

    print(f"Збережено {embeddings.shape}")

if __name__ == "__main__":
    df = pd.read_parquet(
        "Карцев_nosql_2/data/arxiv_subset.parquet"
    )

    embed_corpus(df)