from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50, strategy: str = "recursive"):
    """Різні стратегії розбиття тексту."""
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "])
    elif strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=[" "])
    else:
        raise ValueError("Unknown strategy")
    return splitter.split_text(text)

def evaluate_chunking(df: pd.DataFrame, model, index):
    """Оцінка: для одного запиту порівнюємо релевантність при різних стратегіях."""
    query = "transformer attention"
    q_emb = model.encode(query, normalize_embeddings=True)
    
    for strategy in ["recursive", "fixed"]:
        all_chunks = []
        chunk_to_doc = []
        for _, row in df.iterrows():
            chunks = chunk_text(row['abstract'], strategy=strategy)
            for ch in chunks:
                all_chunks.append(ch)
                chunk_to_doc.append(row['id'])
        # Кодуємо чанки (в реальності краще зберігати окремі вектори)
        chunk_embs = model.encode(all_chunks, normalize_embeddings=True)
        # Шукаємо найближчі чанки -> потім унікальні документи
        # (спрощено)
        print(f"Strategy {strategy}: created {len(all_chunks)} chunks")