"""ChromaVectorStore (design.md §2.3) — kế thừa `src.interfaces.BaseVectorStore`.

Triển khai: S2-DE-03 (add, _init_client, get_collection_stats,
delete_collection) và S3-DE-01 (similarity_search — Property 6, 7, 8).
"""

import logging
from typing import List, Optional, Dict, Any

import chromadb

from src.interfaces import BaseVectorStore
from src.models import Chunk, ScoredChunk

_logger = logging.getLogger(__name__)


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
        self._client = None    # chromadb.EphemeralClient hoặc chromadb.PersistentClient
        self._collection = None

    

    def _make_collection(self):
        """Tạo hoặc lấy collection từ client hiện tại (single source of truth)."""
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_client(self) -> None:
        """Khởi tạo ChromaDB client (in-memory khi persist_dir=None, persistent khi có persist_dir)."""
        if self.persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._make_collection()

    def _ensure_client(self) -> None:
        """Lazy-init: gọi _init_client lần đầu, hoặc tái tạo collection nếu đã bị xóa."""
        if self._client is None:
            self._init_client()
        elif self._collection is None:
            # Xảy ra sau delete_collection(self.collection_name)
            self._collection = self._make_collection()

   

    @staticmethod
    def _to_chroma_metadata(chunk: Chunk) -> Dict[str, Any]:
        """Chuyển Chunk sang metadata dict mà ChromaDB chấp nhận (chỉ str/int/float/bool).

        Fixed keys (doc_id, start_index, end_index) được ghi CUỐI CÙNG để tránh bị
        user metadata ghi đè, đảm bảo chunk identity luôn chính xác.
        """
        meta: Dict[str, Any] = {}
        for k, v in chunk.metadata.items():
            meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        # Fixed keys written last — always authoritative, cannot be overwritten
        meta["doc_id"] = chunk.doc_id
        meta["start_index"] = chunk.start_index
        meta["end_index"] = chunk.end_index
        return meta


    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """Thêm chunks và vectors vào ChromaDB collection.

        Precondition: len(chunks) == len(vectors).
        Dùng upsert để idempotent với chunk_id trùng.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Precondition violated: len(chunks)={len(chunks)} != len(vectors)={len(vectors)}"
            )
        if not chunks:
            return True

        self._ensure_client()
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=vectors,
                documents=[chunk.content for chunk in chunks],
                metadatas=[self._to_chroma_metadata(c) for c in chunks],
            )
            return True
        except Exception as exc:
            _logger.error("ChromaDB upsert failed: %s", exc, exc_info=True)
            return False

    def similarity_search(
        self, query_vector: List[float], k: int = 5
    ) -> List[ScoredChunk]:
        """Tìm k chunks tương đồng nhất với query_vector."""
        raise NotImplementedError(
            "ChromaVectorStore.similarity_search() sẽ được triển khai ở S3-DE-01"
        )

    def delete_collection(self, collection_name: str) -> bool:
        """Xóa collection khỏi ChromaDB; trả về False nếu collection không tồn tại."""
        self._ensure_client()
        try:
            self._client.delete_collection(collection_name)
            if collection_name == self.collection_name:
                # Đánh dấu để _ensure_client tái tạo khi cần
                self._collection = None
            return True
        except Exception as exc:
            _logger.debug("delete_collection(%r) failed: %s", collection_name, exc)
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Trả về thống kê của collection: tên, số chunk, chế độ lưu trữ."""
        self._ensure_client()
        return {
            "collection_name": self.collection_name,
            "num_chunks": self._collection.count(),
            "persist_dir": self.persist_dir,
        }
