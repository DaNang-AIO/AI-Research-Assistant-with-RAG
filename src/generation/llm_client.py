"""OllamaClient (design.md §2.3) — kế thừa `src.interfaces.BaseLLMClient`.

Triển khai: S1-ME-01 (is_available, list_models), S3-ME-01 (generate,
_build_request_body) và S3-ME-02 (generate_stream).
"""

import urllib
import json
from typing import List, Any, Dict
from src.interfaces import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """
    Client giao tiếp với OLLAMA server để sinh văn bản.
    Hỗ trợ streaming và non-streaming response.
    """

    def __init__(
        self,
        model_name: str = "vicuna:7b-v1.5-q5_1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gọi OLLAMA /api/generate endpoint, trả về text hoàn chỉnh
        Waitting for Sprint 3
        """
        pass

    def generate_stream(self, prompt: str):
        """
        Generator — yield từng token khi OLLAMA stream response
        Waitting for Sprint 3
        """
        pass

    def is_available(self) -> bool:
        """
        Ping OLLAMA server, trả về True nếu đang chạy
        """

        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=5) as reponse:
                return reponse.status == 200

        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        Lấy danh sách models đã pull về OLLAMA
        """

        url = f"{self.base_url}/api/tags"

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as reponse:
                raw = reponse.read().decode("utf-8")

        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Không kết nối được với OLLAMA server tại {self.base_url}: {e}"
            ) from e

        try:
            data: Dict[str, Any] = json.loads(raw)
            return [m["name"] for m in data.get("models", [])]

        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Reponse từ OLLAMA không hợp lệ: {e}\nRaw: {raw[:200]} "
            ) from e

    def _build_request_body(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Xây dựng JSON body cho API request"""
        ...
