import json

with open(r"Карцев_nosql_2\arxiv-metadata-oai-snapshot.json", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        paper = json.loads(line)
        print("Keys:", paper.keys())
        print("year:", paper.get('year'))
        print("versions:", paper.get('versions'))
        print("---")