import inspect
from rank_bm25 import BM25Okapi
print(inspect.getsource(BM25Okapi._calc_idf))
