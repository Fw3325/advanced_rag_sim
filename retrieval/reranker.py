from FlagEmbedding import FlagReranker
import torch

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = FlagReranker(model_name, use_fp16=True, device=device)

    def rerank(self, query, chunks, top_n=5):
        """
        chunks: list[Document]
        return: list[Document] reranked by cross-encoder
        """
        if len(chunks) <= top_n:
            return chunks

        pairs = [(query, c.page_content) for c in chunks]
        scores = self.reranker.compute_score(pairs, normalize=True)

        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_n]]
