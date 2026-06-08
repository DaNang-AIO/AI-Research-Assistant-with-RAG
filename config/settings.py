"""Cấu hình hệ thống (design.md §2.4).

Khung cấu hình Sprint 0: định nghĩa các dataclass cấu hình với giá trị mặc
định hợp lý. `AppConfig.from_env()` sẽ được hoàn thiện ở Sprint 5 (S5-ME-01,
Yêu cầu 11.1, 11.3) để đọc `.env`/biến môi trường qua `python-dotenv`.
"""

from dataclasses import dataclass, field


@dataclass
class OllamaConfig:
    """Cấu hình kết nối OLLAMA server."""

    base_url: str = "http://localhost:11434"
    default_llm_model: str = "llama3"
    default_embedding_model: str = "nomic-embed-text"
    timeout_seconds: int = 120


@dataclass
class ChromaConfig:
    """Cấu hình ChromaDB."""

    persist_dir: str = "data/vector_db"
    collection_name: str = "rag_collection"
    in_memory: bool = False


@dataclass
class ChunkerConfig:
    """Cấu hình Text Chunker."""

    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class AppConfig:
    """Cấu hình tổng hợp của ứng dụng."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    top_k: int = 5
    log_experiments: bool = True
    experiment_log_dir: str = "experiments/logs"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Tạo config từ biến môi trường / file `.env`.

        Triển khai đầy đủ ở Sprint 5 — S5-ME-01 (Yêu cầu 11.1, 11.3):
        đọc `.env` qua `python-dotenv`, dùng giá trị mặc định ở trên khi
        biến môi trường không được định nghĩa.
        """
        return cls()
