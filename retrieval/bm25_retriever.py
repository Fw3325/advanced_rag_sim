from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks):
        self.texts = [c.page_content for c in chunks]
        self.chunks = chunks
        self.bm25 = BM25Okapi([t.split() for t in self.texts])

    def search(self, query, k=10):
        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]
