"""OllamaEmbeddingModel (design.md §2.3) — kế thừa `src.interfaces.BaseEmbeddingModel`.

Triển khai: S2-ME-01 (embed_text, _call_ollama_api, dimension) và
S2-ME-02 (embed_batch nhất quán với embed_text — Property 5).
"""

import os
import json
import urllib.request
import urllib.error
from urllib.parse import urlsplit
from typing import List, Optional
from src.interfaces import BaseEmbeddingModel


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Tạo embedding vector sử dụng OLLAMA embedding endpoint.
    Hỗ trợ bảo mật endpoint cục bộ và validate dữ liệu nghiêm ngặt.
    """

    def __init__(
        self,
        model_name: str = "bge-m3:latest",
        ollama_base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        # Lấy URL cấu hình từ tham số hoặc biến môi trường, dự phòng endpoint mặc định
        configured_base_url = ollama_base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434")

        # Kiểm tra bảo mật: Chỉ cho phép local endpoints (localhost, 127.0.0.1, ::1) qua HTTP
        parsed = urlsplit(configured_base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError(
                f"URL cơ sở OLLAMA phải là điểm cuối cục bộ (local endpoint), nhận được: {configured_base_url}"
            )
        self.base_url = configured_base_url.rstrip("/")
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
                # Validate cấu trúc Schema phản hồi của OLLAMA để tránh lỗi ngầm
                embedding = result.get("embedding") if isinstance(
                    result, dict) else None
                if (
                    not isinstance(embedding, list)
                    or len(embedding) == 0
                    or not all(isinstance(x, (int, float)) for x in embedding)
                ):
                    raise ValueError(
                        "Phản hồi OLLAMA không chứa cấu trúc embedding hợp lệ")

                return [float(x) for x in embedding]
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

        if not text or not text.strip():
            raise ValueError(
                "Văn bản đầu vào (text) không được để rỗng hoặc None")

        # Gọi API lấy vector
        vector = self._call_ollama_api(text)

        # Lazy-init dimension (Gán _dimension ở lần gọi đầu tiên)
        if self._dimension is None:
            self._dimension = len(vector)

        # Kiểm tra tính nhất quán số chiều ở các lần gọi sau (Dimension Guard)
        elif len(vector) != self._dimension:
            raise ValueError(
                f"Kích thước embedding không nhất quán: kỳ vọng {self._dimension} chiều, nhận được {len(vector)} chiều"
            )

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
            self.embed_text("init_demension_guard")
        return self._dimension
