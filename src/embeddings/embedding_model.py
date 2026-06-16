"""OllamaEmbeddingModel (design.md §2.3) — kế thừa `src.interfaces.BaseEmbeddingModel`.

Triển khai: S2-ME-01 (embed_text, _call_ollama_api, dimension) và
S2-ME-02 (embed_batch nhất quán với embed_text — Property 5).
"""

from typing import List
from src.interfaces import BaseEmbeddingModel


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Tạo embedding vector sử dụng OLLAMA embedding endpoint.
    Mặc định dùng model 'nomic-embed-text'.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = ollama_base_url
        self._dimension = 768  # Kích thước vector mặc định giả lập

    def embed_text(self, text: str) -> List[float]:
        """Tạo embedding cho một đoạn văn bản."""
        raise NotImplementedError(
            "OllamaEmbeddingModel.embed_text() sẽ được triển khai đầy đủ ở Sprint 2"
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Tạo embedding cho nhiều văn bản cùng lúc."""
        raise NotImplementedError(
            "OllamaEmbeddingModel.embed_batch() sẽ được triển khai đầy đủ ở Sprint 2"
        )

    @property
    def dimension(self) -> int:
        """Số chiều của embedding vector."""
        return self._dimension
