"""ChromaVectorStore (design.md §2.3) — kế thừa `src.interfaces.BaseVectorStore`.

Triển khai: S2-DE-03 (add, _init_client, get_collection_stats,
delete_collection) và S3-DE-01 (similarity_search — Property 6, 7, 8).
"""

from typing import List, Optional, Dict, Any
from src.interfaces import BaseVectorStore
from src.models import Chunk, ScoredChunk


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store sử dụng ChromaDB.
    Hỗ trợ persistent storage và in-memory mode cho notebook.
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_dir: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """Thêm chunks và vectors vào vector store."""
        raise NotImplementedError(
            "ChromaVectorStore.add() sẽ được triển khai đầy đủ ở Sprint 2"
        )

    def similarity_search(
        self, query_vector: List[float], k: int = 5
    ) -> List[ScoredChunk]:
        """Tìm k chunks tương đồng nhất với query_vector."""
        raise NotImplementedError(
            "ChromaVectorStore.similarity_search() sẽ được triển khai đầy đủ ở Sprint 3"
        )

    def delete_collection(self, collection_name: str) -> bool:
        """Xóa collection khỏi vector store."""
        raise NotImplementedError(
            "ChromaVectorStore.delete_collection() sẽ được triển khai đầy đủ ở Sprint 2"
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Trả về thống kê của collection."""
        raise NotImplementedError(
            "ChromaVectorStore.get_collection_stats() sẽ được triển khai đầy đủ ở Sprint 2"
        )
