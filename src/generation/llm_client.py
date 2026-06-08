"""OllamaClient (design.md §2.3) — kế thừa `src.interfaces.BaseLLMClient`.

Triển khai: S1-ME-01 (is_available, list_models), S3-ME-01 (generate,
_build_request_body) và S3-ME-02 (generate_stream).
"""

import json
from typing import Any, Dict, List

import requests

from src.interfaces import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Client giao tiếp với OLLAMA server để sinh văn bản.

    Hỗ trợ streaming và non-streaming response.
    """

    # LLM cục bộ chạy trên CPU có thể mất hàng chục giây chỉ để "đọc" prompt
    # (prefill) trước khi sinh token đầu tiên — 120s từng gây ReadTimeout giả
    # (server vẫn đang xử lý bình thường, chỉ là chậm hơn ngưỡng chờ).
    REQUEST_TIMEOUT_SECONDS = 300

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
        """Gọi OLLAMA `/api/generate` endpoint (non-streaming), trả về text
        hoàn chỉnh (Yêu cầu 5.1).

        Lỗi kết nối được bọc lại thành `ConnectionError` mô tả rõ `base_url`
        đang dùng, để người dùng biết chính xác cần kiểm tra OLLAMA ở đâu
        (Yêu cầu 5.5).
        """
        assert prompt, "prompt không được rỗng"

        body = self._build_request_body(prompt, stream=False, **kwargs)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=body, timeout=self.REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(
                f"Không thể kết nối tới OLLAMA server tại '{self.base_url}' "
                f"— hãy chắc chắn 'ollama serve' đang chạy. Chi tiết lỗi: {exc}"
            ) from exc

        return response.json().get("response", "")

    def generate_stream(self, prompt: str):
        """Generator — yield từng token khi OLLAMA stream response (Yêu cầu 5.3).

        Mỗi dòng JSON trả về từ OLLAMA (`/api/generate` với `stream=True`)
        chứa một mẩu `response`; generator yield tuần tự từng mẩu cho tới khi
        OLLAMA báo `done=True`.
        """
        assert prompt, "prompt không được rỗng"

        body = self._build_request_body(prompt, stream=True)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=body,
                stream=True,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(
                f"Không thể kết nối tới OLLAMA server tại '{self.base_url}' "
                f"— hãy chắc chắn 'ollama serve' đang chạy. Chi tiết lỗi: {exc}"
            ) from exc

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break

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
        """Xây dựng JSON body cho `/api/generate`.

        `temperature`/`max_tokens` mặc định lấy từ cấu hình của client, nhưng
        có thể bị ghi đè qua `**kwargs` (ví dụ khi thực nghiệm trong notebook).
        `stream` mặc định `False` — `generate_stream()` truyền `stream=True`.
        """
        stream = kwargs.pop("stream", False)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            **kwargs,
        }
