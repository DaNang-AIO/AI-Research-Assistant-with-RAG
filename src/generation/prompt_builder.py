"""PromptBuilder (design.md §2.3).

Triển khai: S3-PE-01 (DEFAULT_SYSTEM_PROMPT, build, format_context,
set_system_prompt — Property 9).
"""

from typing import List
from src.models import ScoredChunk


class PromptBuilder:
    """
    Xây dựng prompt cho RAG với system instruction và context injection.
    Hỗ trợ nhiều template khác nhau để thực nghiệm.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Bạn là trợ lý nghiên cứu. Hãy trả lời câu hỏi DỰA TRÊN "
        "ngữ cảnh được cung cấp. Nếu không đủ thông tin, hãy nói rõ."
    )

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build(self, question: str, contexts: List[ScoredChunk]) -> str:
        """Ghép system_prompt + context chunks + câu hỏi thành một prompt hoàn chỉnh."""
        raise NotImplementedError(
            "PromptBuilder.build() sẽ được triển khai đầy đủ ở Sprint 3"
        )

    def format_context(self, contexts: List[ScoredChunk]) -> str:
        """Định dạng danh sách ScoredChunk thành đoạn text context."""
        raise NotImplementedError(
            "PromptBuilder.format_context() sẽ được triển khai đầy đủ ở Sprint 3"
        )

    def set_system_prompt(self, new_prompt: str) -> None:
        """Thay đổi system prompt."""
        self.system_prompt = new_prompt
