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

    Triển khai đầy đủ: S3-PE-01.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Bạn là trợ lý nghiên cứu. Hãy trả lời câu hỏi DỰA TRÊN "
        "ngữ cảnh được cung cấp. Nếu không đủ thông tin, hãy nói rõ."
    )

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def build(self, question: str, contexts: List[ScoredChunk]) -> str:
        """
        Ghép system_prompt + context chunks + câu hỏi thành prompt hoàn chỉnh.

        Preconditions:
          - question không rỗng
          - contexts là list (có thể rỗng)

        Postconditions:
          - Trả về prompt string hợp lệ
          - prompt chứa nội dung của tất cả contexts
          - prompt chứa question (Property 9)
        """
        ctx_text = self.format_context(contexts)
        return (
            f"{self.system_prompt}\n\n"
            f"NGỮ CẢNH:\n{ctx_text}\n\n"
            f"CÂU HỎI: {question}\n\n"
            f"TRẢ LỜI:"
        )

    def format_context(self, contexts: List[ScoredChunk]) -> str:
        """Định dạng danh sách ScoredChunk thành đoạn text context."""
        if not contexts:
            return "(Không có ngữ cảnh liên quan.)"
        parts = []
        for i, sc in enumerate(contexts, 1):
            parts.append(f"[Đoạn {i}] (score: {sc.score:.3f})\n{sc.chunk.content}")
        return "\n\n".join(parts)

    def set_system_prompt(self, new_prompt: str) -> None:
        """Thay đổi system prompt (dùng khi thực nghiệm prompt engineering)."""
        self.system_prompt = new_prompt
