"""Retriever (design.md §1.2, module retrieval) — bọc quanh BaseVectorStore.similarity_search.

Triển khai: S3-PE-02 (cấu hình top_k, dùng trong RAGPipeline.query và
notebook 02_retrieval_strategies).
"""

from typing import List, Optional

from src.interfaces import BaseVectorStore
from src.models import ScoredChunk


class Retriever:
    """Truy xuất các `ScoredChunk` liên quan nhất tới một query vector.

    Bọc quanh `BaseVectorStore.similarity_search()` (xem sequence diagram
    design.md §1.3: `RT->>VS: similarity_search(query_vector, k)`), đồng thời
    cung cấp `top_k` mặc định để `RAGPipeline`/notebook không phải lặp lại
    cấu hình này ở mỗi lần gọi.
    """

    def __init__(self, vector_store: BaseVectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query_vector: List[float], k: Optional[int] = None) -> List[ScoredChunk]:
        """Truy xuất tối đa `k` (hoặc `self.top_k` nếu không truyền) chunk
        liên quan nhất tới `query_vector`, sắp xếp giảm dần theo `score`
        (Property 6, 7, 8 — được `similarity_search` đảm bảo)."""
        return self.vector_store.similarity_search(query_vector, k=k if k is not None else self.top_k)
