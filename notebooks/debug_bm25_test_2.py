import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from vectorstore.sparse import BM25Index
from langchain_core.documents import Document

idx = BM25Index()
docs = [
    Document(page_content="Python is a programming language"),
    Document(page_content="Java is another programming language"),
    Document(page_content="Cats are wonderful pets"),
]
idx.build(docs)

for doc in docs:
    print(doc.page_content)

print("tokenized programming language:", idx._tokenize("programming language"))
scores = idx.bm25.get_scores(idx._tokenize("programming language"))
print("scores for 'programming language':", scores)
results = idx.search("programming language", top_k=2)
print("results:", results)
