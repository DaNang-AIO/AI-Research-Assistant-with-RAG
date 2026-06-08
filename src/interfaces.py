"""Core interfaces (ABC) cho các thành phần RAG (design.md §2.2).

Mỗi interface quy định pre/postcondition và loop invariant — đây là cơ sở
cho các correctness property tương ứng (design.md Phần 3). Các lớp triển khai
cụ thể (DocumentLoader, TextChunker, OllamaEmbeddingModel, ChromaVectorStore,
OllamaClient, ...) PHẢI kế thừa đúng interface tương ứng tại đây, không tự ý
đổi tên method hay bỏ qua interface.
"""

from abc import ABC, abstractmethod
from typing import List

from src.models import Chunk, Document, ScoredChunk


class BaseLoader(ABC):
    """Interface cho các Document Loader."""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Tải một tài liệu từ đường dẫn file.

        Preconditions:
          - file_path trỏ đến file tồn tại
          - file_path có extension được hỗ trợ (.pdf, .txt, .md)

        Postconditions:
          - Trả về Document với content không rỗng
          - Document.doc_id là duy nhất
        """
        raise NotImplementedError

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Kiểm tra loader có hỗ trợ loại file này không."""
        raise NotImplementedError


class BaseChunker(ABC):
    """Interface cho các Text Chunker."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Chia nhỏ document thành danh sách các chunk.

        Preconditions:
          - document.content không rỗng

        Postconditions:
          - Trả về list ít nhất 1 Chunk
          - Mỗi Chunk.doc_id == document.doc_id
          - Nối tất cả Chunk.content phải bao phủ document.content gốc

        Loop Invariant (trong vòng lặp tạo chunks):
          - Tất cả chunks đã tạo đều có doc_id hợp lệ
          - Không có nội dung bị mất giữa các lần lặp
        """
        raise NotImplementedError


class BaseEmbeddingModel(ABC):
    """Interface cho các Embedding Model."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Tạo embedding vector cho một đoạn văn bản.

        Preconditions:
          - text không rỗng
          - text có độ dài <= max_token_limit của model

        Postconditions:
          - Trả về list[float] có độ dài == self.dimension
          - Cùng một text luôn trả về cùng một vector (deterministic)
        """
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Tạo embedding cho nhiều văn bản cùng lúc.

        Postconditions:
          - len(result) == len(texts)
          - result[i] == embed_text(texts[i]) với mọi i
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều của embedding vector."""
        raise NotImplementedError


class BaseVectorStore(ABC):
    """Interface cho Vector Store / Database."""

    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """Lưu trữ chunks cùng với embedding vectors.

        Preconditions:
          - len(chunks) == len(vectors)
          - Mỗi vector có cùng dimension

        Postconditions:
          - Tất cả chunks có thể truy xuất được bằng similarity_search
          - Trả về True nếu thành công
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[ScoredChunk]:
        """Tìm k chunks gần nhất với query_vector.

        Preconditions:
          - len(query_vector) == dimension của store
          - k >= 1

        Postconditions:
          - len(result) <= k
          - result được sắp xếp theo score giảm dần
          - Mỗi ScoredChunk.score thuộc [0.0, 1.0]
        """
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Xóa toàn bộ collection."""
        raise NotImplementedError


class BaseLLMClient(ABC):
    """Interface cho LLM Client."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Gọi LLM để sinh văn bản từ prompt.

        Preconditions:
          - prompt không rỗng
          - LLM server (OLLAMA) đang chạy

        Postconditions:
          - Trả về string không rỗng
          - Không có side effect ngoài network call
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Kiểm tra LLM server có sẵn sàng không. Không bao giờ ném exception."""
        raise NotImplementedError
