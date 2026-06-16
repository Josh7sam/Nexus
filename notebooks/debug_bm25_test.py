import os
import sys
import tempfile
import pathlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import config
from vectorstore.sparse import BM25Index
from langchain_core.documents import Document

tmp_dir = tempfile.mkdtemp()
tmp_path = pathlib.Path(tmp_dir)

config.BM25_INDEX_PATH = str(tmp_path / "test_bm25.pkl")
config.BM25_CORPUS_PATH = str(tmp_path / "test_corpus.pkl")

idx = BM25Index()
docs = [Document(page_content="Quantum computing research")]
idx.build(docs)
idx.save()

idx2 = BM25Index()
idx2.load()

print("idx2.documents:", idx2.documents)
print("tokenized_query:", idx2._tokenize("quantum"))
if idx2.bm25:
    print("bm25 corpus size:", idx2.bm25.corpus_size)
    # Check what the internal tokenized corpus is
    # In rank-bm25, it depends on the library, but let's check
    scores = idx2.bm25.get_scores(idx2._tokenize("quantum"))
    print("scores for 'quantum':", scores)
    
    # Let's see how rank-bm25 is initialized and its internal parameters
    print("IDF:", getattr(idx2.bm25, "idf", None))
    print("Doc freqs:", getattr(idx2.bm25, "doc_freqs", None))
