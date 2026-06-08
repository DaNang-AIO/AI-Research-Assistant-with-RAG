"""OllamaEmbeddingModel (design.md §2.3) — kế thừa `src.interfaces.BaseEmbeddingModel`.

Triển khai: S2-ME-01 (embed_text, _call_ollama_api, dimension) và
S2-ME-02 (embed_batch nhất quán với embed_text — Property 5).
"""

from typing import List, Optional

import requests

from src.interfaces import BaseEmbeddingModel


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """Tạo embedding vector sử dụng OLLAMA embedding endpoint.

    Mặc định dùng model 'nomic-embed-text'.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = ollama_base_url
        self._dimension: Optional[int] = None

    def embed_text(self, text: str) -> List[float]:
        """Tạo embedding vector cho một đoạn văn bản qua OLLAMA.

        Tất định: cùng `text` luôn trả về cùng vector vì OLLAMA embedding
        suy luận không lấy mẫu ngẫu nhiên (Property 3, 4 — Yêu cầu 3.1, 3.2).
        Lazy-init `self._dimension` từ độ dài vector trả về đầu tiên.
        """
        vector = self._call_ollama_api(text)
        if self._dimension is None:
            self._dimension = len(vector)
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Gọi `embed_text` tuần tự cho từng văn bản và gom kết quả.

        Đảm bảo `embed_batch(texts)[i] == embed_text(texts[i])` với mọi `i`
        (Property 5 — Yêu cầu 3.3, 3.4).
        """
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Số chiều của embedding vector — lazy-init từ lần `embed_text` đầu
        tiên. Nếu chưa từng embed, gọi thử với một văn bản mẫu để xác định.
        """
        if self._dimension is None:
            self.embed_text("dimension probe")
        return self._dimension

    def _call_ollama_api(self, text: str) -> List[float]:
        """HTTP POST đến OLLAMA embedding endpoint (`/api/embeddings`).

        Yêu cầu 3.5: nếu OLLAMA server không khả dụng, ném lỗi mô tả rõ địa
        chỉ server (`self.base_url`) và nguyên nhân kết nối.
        """
        url = f"{self.base_url}/api/embeddings"
        try:
            response = requests.post(
                url,
                json={"model": self.model_name, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(
                f"Không thể kết nối tới OLLAMA server tại '{self.base_url}' để tạo "
                f"embedding (model='{self.model_name}'). Hãy kiểm tra OLLAMA đã "
                f"chạy chưa (`ollama serve`) và đã pull model này chưa "
                f"(`ollama pull {self.model_name}`). Chi tiết lỗi: {exc}"
            ) from exc

        embedding = data.get("embedding")
        if not embedding:
            raise ConnectionError(
                f"OLLAMA server tại '{self.base_url}' trả về phản hồi không hợp lệ "
                f"(thiếu trường 'embedding') cho model '{self.model_name}'."
            )
        return embedding
