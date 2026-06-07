import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def embed_corpus(df: pd.DataFrame, model_name: str = "allenai/specter2"):
    """Генерація векторних представлень для заголовка + анотації."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    # Створюємо текст для кодування: "TITLE: {title}\nABSTRACT: {abstract}"
    texts = [f"TITLE: {row['title']}\nABSTRACT: {row['abstract']}" 
             for _, row in df.iterrows()]
    
    # Кодуємо батчами
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    np.save("embeddings/embeddings.npy", embeddings)
    print(f"Збережено {embeddings.shape} вектори в embeddings/embeddings.npy")
    return embeddings

if __name__ == "__main__":
    df = pd.read_parquet("data/arxiv_subset.parquet")
    embed_corpus(df)