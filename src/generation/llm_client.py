"""OllamaClient (design.md §2.3) — kế thừa `src.interfaces.BaseLLMClient`.

Triển khai: S1-ME-01 (is_available, list_models), S3-ME-01 (generate,
_build_request_body) và S3-ME-02 (generate_stream).
"""

from typing import Any, Dict, List

import requests

from src.interfaces import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Client giao tiếp với OLLAMA server để sinh văn bản.

    Hỗ trợ streaming và non-streaming response.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        """Gọi OLLAMA /api/generate endpoint, trả về text hoàn chỉnh.

        Triển khai: S3-ME-01 (Yêu cầu 5.1, 5.5).
        """
        raise NotImplementedError("OllamaClient.generate() sẽ được triển khai ở S3-ME-01")

    def generate_stream(self, prompt: str):
        """Generator — yield từng token khi OLLAMA stream response.

        Triển khai: S3-ME-02 (Yêu cầu 5.3).
        """
        raise NotImplementedError("OllamaClient.generate_stream() sẽ được triển khai ở S3-ME-02")

    def is_available(self) -> bool:
        """Ping OLLAMA server, trả về True nếu đang chạy.

        Yêu cầu 5.2: không bao giờ ném exception — mọi lỗi kết nối/timeout
        đều được coi là server không khả dụng và trả về False.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """Lấy danh sách tên các model đã pull về OLLAMA server (Yêu cầu 5.4).

        Trả về danh sách rỗng nếu server không khả dụng hoặc phản hồi bất thường.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
        except (requests.exceptions.RequestException, ValueError):
            return []

        return [model.get("name", "") for model in data.get("models", [])]

    def _build_request_body(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Xây dựng JSON body cho API request.

        Triển khai: S3-ME-01.
        """
        raise NotImplementedError("OllamaClient._build_request_body() sẽ được triển khai ở S3-ME-01")
