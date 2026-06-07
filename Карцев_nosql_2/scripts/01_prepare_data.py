import json
import pandas as pd
from pathlib import Path

def prepare_arxiv_subset(input_path: str, output_path: str, n_rows: int = 10000):
    "Читання JSONL, відбір перших n_rows, збереження в Parquet."
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            try:
                paper = json.loads(line)
                # Витягуємо потрібні поля
                record = {
                    'id': paper.get('id', ''),
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'authors': ', '.join(paper.get('authors', [])),
                    'year': paper.get('year', None),
                    'categories': paper.get('categories', '')
                }
                data.append(record)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    # Очищення від пустих анотацій
    df = df[df['abstract'].str.strip().astype(bool)]
    df.to_parquet(output_path, index=False)
    print(f"Збережено {len(df)} записів у {output_path}")

if __name__ == "__main__":
    prepare_arxiv_subset("arxiv-metadata-oai-snapshot.jsonl", "data/arxiv_subset.parquet", n_rows=10000)