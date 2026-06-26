"""ChromaVectorStore (design.md §2.3) — kế thừa `src.interfaces.BaseVectorStore`.

Triển khai: S2-DE-03 (add, _init_client, get_collection_stats,
delete_collection) và S3-DE-01 (similarity_search — Property 6, 7, 8).
"""

import os
import logging
from typing import List, Optional, Dict, Any

from src.interfaces import BaseVectorStore
from src.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store sử dụng ChromaDB.
    Hỗ trợ persistent storage và in-memory mode cho notebook.

    Preconditions:
      - collection_name không rỗng
      - persist_dir (nếu có) là đường dẫn hợp lệ, ChromaDB tự tạo nếu chưa có

    Postconditions:
      - Sau khi add() thành công, chunks có thể truy xuất qua similarity_search()
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_dir: Optional[str] = None,
    ):
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name không được để trống")

        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None      # chromadb.Client hoặc chromadb.PersistentClient
        self._collection = None  # chromadb.Collection

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """
        Khởi tạo ChromaDB client (in-memory hoặc persistent) — lazy init.

        Chỉ khởi tạo một lần duy nhất. Các lần gọi sau khi đã có
        self._collection sẽ trả về ngay.

        Postconditions:
          - self._client không còn None
          - self._collection trỏ đến đúng collection_name
        """
        if self._collection is not None:
            return  # đã khởi tạo rồi, bỏ qua

        try:
            import chromadb  # lazy import để tránh lỗi nếu chưa cài
        except ImportError as e:
            raise ImportError(
                "chromadb chưa được cài — chạy: pip install chromadb"
            ) from e

        if self.persist_dir is None:
            # In-memory mode: phù hợp cho notebook, dữ liệu mất khi restart
            self._client = chromadb.Client()
            logger.info("ChromaVectorStore: khởi tạo in-memory client")
        else:
            # Persistent mode: lưu ra disk, khôi phục được sau restart
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(
                "ChromaVectorStore: khởi tạo persistent client tại %s",
                self.persist_dir,
            )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # Dùng cosine distance để score ∈ [0, 1] sau khi normalize
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore: collection '%s' sẵn sàng", self.collection_name
        )

    @staticmethod
    def _build_metadata(chunk: Chunk) -> Dict[str, Any]:
        """
        Xây dựng metadata dict để lưu vào ChromaDB.

        Merge các field cốt lõi của Chunk (chunk_id, doc_id, start_index,
        end_index) vào trước, rồi overlay chunk.metadata lên sau.
        Điều này đảm bảo:
          1. ChromaDB không nhận dict rỗng {} (chromadb==0.4.22 reject empty metadata)
          2. Đủ thông tin để reconstruct Chunk từ kết quả similarity_search sau này
          3. Mọi giá trị đều là kiểu ChromaDB chấp nhận: str | int | float | bool

        Lưu ý: chunk.metadata value có thể là Enum, None, list, dict —
        tất cả đều được convert sang str.
        """
        # Field cốt lõi — luôn có giá trị hợp lệ
        base: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
        }

        # Overlay metadata tùy chỉnh từ chunk, convert sang kiểu ChromaDB hợp lệ
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                base[key] = value
            elif value is None:
                base[key] = ""
            else:
                # Enum, list, dict, ... → chuyển sang string
                base[key] = str(value)

        return base

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """
        Thêm chunks và vectors vào ChromaDB collection.

        Preconditions:
          - len(chunks) == len(vectors)
          - Mỗi vector có cùng dimension

        Postconditions:
          - Tất cả chunks có thể truy xuất được bằng similarity_search
          - Trả về True nếu thành công, False nếu có lỗi

        Yêu cầu 4.1
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Precondition thất bại: len(chunks)={len(chunks)} "
                f"!= len(vectors)={len(vectors)}"
            )
        if not chunks:
            logger.warning("ChromaVectorStore.add() được gọi với danh sách rỗng")
            return True  # không có gì để lưu — coi như thành công

        self._init_client()

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        # _build_metadata đảm bảo: không rỗng, có đủ field cốt lõi, type hợp lệ
        metadatas = [self._build_metadata(chunk) for chunk in chunks]

        try:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas,
            )
            logger.info(
                "ChromaVectorStore.add(): đã lưu %d chunks vào collection '%s'",
                len(chunks),
                self.collection_name,
            )
            return True
        except Exception as e:
            logger.error("ChromaVectorStore.add() thất bại: %s", e)
            return False

    def similarity_search(
        self, query_vector: List[float], k: int = 5
    ) -> List[ScoredChunk]:
        """Tìm k chunks tương đồng nhất với query_vector."""
        raise NotImplementedError(
            "ChromaVectorStore.similarity_search() sẽ được triển khai ở S3-DE-01"
        )

    def delete_collection(self, collection_name: str) -> bool:
        """
        Xóa collection khỏi ChromaDB.

        Nếu collection_name trùng với collection đang dùng,
        reset self._collection về None để buộc _init_client() tạo lại.

        Postconditions:
          - Collection đã bị xóa khỏi ChromaDB
          - Trả về True nếu thành công

        Yêu cầu 4.7
        """
        self._init_client()

        try:
            self._client.delete_collection(collection_name)
            logger.info(
                "ChromaVectorStore.delete_collection(): đã xóa collection '%s'",
                collection_name,
            )
            # Nếu xóa đúng collection đang dùng → reset để tránh dùng object cũ
            if collection_name == self.collection_name:
                self._collection = None
            return True
        except Exception as e:
            logger.error(
                "ChromaVectorStore.delete_collection('%s') thất bại: %s",
                collection_name,
                e,
            )
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Trả về thống kê của collection hiện tại.

        Returns:
            dict gồm:
              - collection_name: str
              - count: số chunk đang lưu trong collection
              - persist_dir: str | None
              - mode: "persistent" | "in-memory"
        """
        self._init_client()

        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "count": count,
            "persist_dir": self.persist_dir,
            "mode": "persistent" if self.persist_dir else "in-memory",
        }
