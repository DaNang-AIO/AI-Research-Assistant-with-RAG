"""Data models & enums dùng chung cho toàn bộ hệ thống RAG (design.md §2.1).

Mọi module trong src/ và app/ import các kiểu dữ liệu từ đây để tránh
định nghĩa trùng lặp và import vòng (circular import) giữa các package con.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import datetime


class ChunkStrategy(Enum):
    """Chiến lược chia nhỏ văn bản."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentType(Enum):
    """Loại tài liệu được hỗ trợ."""

    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class Document:
    """Đại diện một tài liệu gốc sau khi tải."""

    doc_id: str
    file_path: str
    doc_type: DocumentType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Một đoạn văn bản nhỏ được tách từ Document."""

    chunk_id: str
    doc_id: str
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingVector:
    """Vector embedding của một Chunk."""

    chunk_id: str
    vector: List[float]
    model_name: str
    dimension: int


@dataclass
class ScoredChunk:
    """Chunk kèm điểm similarity và thứ hạng từ kết quả retrieval."""

    chunk: Chunk
    score: float
    rank: int


@dataclass
class RAGResponse:
    """Kết quả trả về từ toàn bộ RAG pipeline cho một câu hỏi."""

    question: str
    answer: str
    contexts: List[ScoredChunk]
    model_name: str
    latency_ms: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class IndexingResult:
    """Kết quả sau khi index một tài liệu."""

    doc_id: str
    num_chunks: int
    collection_name: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ExperimentLog:
    """Bản ghi một sự kiện thực nghiệm (indexing hoặc query)."""

    experiment_id: str
    event_type: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
