"""OllamaEmbeddingModel (design.md §2.3) — kế thừa `src.interfaces.BaseEmbeddingModel`.

Triển khai: S2-ME-01 (embed_text, _call_ollama_api, dimension) và
S2-ME-02 (embed_batch nhất quán với embed_text — Property 5).
"""

import json
import urllib.request
import urllib.error
from typing import List, Optional
from src.interfaces import BaseEmbeddingModel


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Tạo embedding vector sử dụng OLLAMA embedding endpoint.
    Mặc định dùng model 'nomic-embed-text'.
    """

    def __init__(
        self,
        model_name: str = "bge-m3:latest",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = ollama_base_url
        self._dimension: Optional[int] = None
        # Khởi tạo none cho lazy-init

    def _call_ollama_api(self, text: str) -> List[float]:
        """
        HTTP POST đến OLLAMA embedding endpoint.
        Xử lý lỗi kết nối theo Yêu cầu 3.5.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": text
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("embedding", [])
        except urllib.error.URLError as e:
            # Yêu cầu 3.5: Trả về lỗi mô tả rõ địa chỉ server và nguyên nhân
            raise ConnectionError(
                f"Không thể kết nối đến OLLAMA server tại {url}. Nguyên nhân: {e.reason}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Lỗi phân tích phản hồi từ OLLAMA: {e}") from e

    def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một đoạn văn bản.
        Thỏa mãn tính Deterministic (Yêu cầu 3.2).
        """
        # Gọi API lấy vector
        vector = self._call_ollama_api(text)

        # Lazy-init dimension (Gán _dimension ở lần gọi đầu tiên)
        if self._dimension is None:
            self._dimension = len(vector)

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Tạo embedding cho nhiều văn bản cùng lúc."""
        raise NotImplementedError(
            "OllamaEmbeddingModel.embed_batch() sẽ được triển khai đầy đủ ở Sprint 2"
        )

    @property
    def dimension(self) -> int:
        """
        Số chiều của embedding vector.
        Nếu chưa được gọi lần nào, tự động triggers 1 lần để lấy chiều. 
        """

        if self._dimension is None:
            # Gọi hàm với text rỗng hoặc text mồi để OLLAMA trả về vector
            self.embed_text("init")
        return self._dimension
