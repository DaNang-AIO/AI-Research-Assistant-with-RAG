"""ChromaVectorStore (design.md §2.3) — kế thừa `src.interfaces.BaseVectorStore`.

Triển khai: S2-DE-03 (add, _init_client, get_collection_stats,
delete_collection) và S3-DE-01 (similarity_search — Property 6, 7, 8).
"""

import os
from typing import Any, Dict, List, Optional

import chromadb

from src.interfaces import BaseVectorStore
from src.models import Chunk, ScoredChunk

# Dùng không gian "cosine" để khoảng cách ∈ [0, 2] (1 - cosine similarity),
# từ đó quy đổi tuyến tính sang score ∈ [0.0, 1.0] (Yêu cầu 4.4).
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class ChromaVectorStore(BaseVectorStore):
    """Vector store sử dụng ChromaDB.

    Hỗ trợ persistent storage và in-memory mode cho notebook.
    """

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_dir: Optional[str] = None,  # None = in-memory
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None    # chromadb.Client hoặc chromadb.PersistentClient
        self._collection = None

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """Thêm chunks và vectors vào ChromaDB collection (Yêu cầu 4.1).

        `metadata` của mỗi entry lưu `doc_id`/`start_index`/`end_index` để
        `similarity_search` có thể tái tạo lại đúng `Chunk` gốc.
        """
        if not chunks:
            return True

        collection = self._get_collection()
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                }
                for chunk in chunks
            ],
        )
        return True

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[ScoredChunk]:
        """Query ChromaDB và trả về `List[ScoredChunk]` có rank và score.

        Đảm bảo (Yêu cầu 4.2-4.4 / Property 6-8):
          - `len(result) <= k`
          - sắp xếp giảm dần theo `score`
          - mỗi `score` thuộc `[0.0, 1.0]`
        """
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, count),
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        scored_chunks: List[ScoredChunk] = []
        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=metadata.get("doc_id", ""),
                content=content,
                start_index=metadata.get("start_index", 0),
                end_index=metadata.get("end_index", 0),
            )
            # cosine distance ∈ [0, 2] → score = 1 - distance/2 ∈ [0, 1]
            score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            scored_chunks.append(ScoredChunk(chunk=chunk, score=score, rank=rank))

        return scored_chunks

    def delete_collection(self, collection_name: str) -> bool:
        """Xóa collection khỏi ChromaDB (Yêu cầu 4.7). Trả về `False` nếu
        collection không tồn tại hoặc không thể xóa."""
        self._init_client()
        try:
            self._client.delete_collection(name=collection_name)
        except Exception:
            return False

        if collection_name == self.collection_name:
            self._collection = None
        return True

    def get_collection_stats(self) -> Dict[str, Any]:
        """Trả về thống kê: tên collection, số lượng chunks, chế độ lưu trữ."""
        collection = self._get_collection()
        return {
            "collection_name": self.collection_name,
            "num_chunks": collection.count(),
            "persist_dir": self.persist_dir,
            "in_memory": self.persist_dir is None,
        }

    def _init_client(self) -> None:
        """Khởi tạo ChromaDB client — persistent (khi có `persist_dir`) hoặc
        in-memory (Yêu cầu 4.5, 4.6), chỉ khởi tạo một lần."""
        if self._client is not None:
            return

        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self._client = chromadb.Client()

    def _get_collection(self):
        """Đảm bảo client và collection đã sẵn sàng, trả về collection hiện tại."""
        self._init_client()
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=_COLLECTION_METADATA,
            )
        return self._collection
