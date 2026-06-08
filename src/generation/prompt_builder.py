"""PromptBuilder (design.md §2.3).

Triển khai: S3-PE-01 (DEFAULT_SYSTEM_PROMPT, build, format_context,
set_system_prompt — Property 9).
"""

from typing import List

from src.models import ScoredChunk


class PromptBuilder:
    """Xây dựng prompt cho RAG với system instruction và context injection.

    Hỗ trợ nhiều template khác nhau để thực nghiệm.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Bạn là trợ lý nghiên cứu. Hãy trả lời câu hỏi DỰA TRÊN "
        "ngữ cảnh được cung cấp. Nếu không đủ thông tin, hãy nói rõ."
    )

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build(self, question: str, contexts: List[ScoredChunk]) -> str:
        """Ghép system_prompt + context chunks + câu hỏi thành một prompt hoàn chỉnh.

        Preconditions:
          - question không rỗng
          - contexts là list (có thể rỗng)

        Postconditions (Yêu cầu 6.1, 6.2, 6.4 / Property 9):
          - Trả về prompt string hợp lệ
          - prompt chứa nội dung của tất cả contexts
          - prompt chứa question
          - prompt luôn chứa system_prompt, kể cả khi contexts rỗng
        """
        assert question, "question không được rỗng"

        return (
            f"{self.system_prompt}\n\n"
            f"### Ngữ cảnh\n{self.format_context(contexts)}\n\n"
            f"### Câu hỏi\n{question}\n\n"
            f"### Trả lời"
        )

    def format_context(self, contexts: List[ScoredChunk]) -> str:
        """Định dạng danh sách ScoredChunk thành đoạn text context.

        Trả về một thông báo "không có ngữ cảnh" rõ ràng khi `contexts` rỗng,
        để `build()` vẫn tạo ra prompt hợp lệ (Yêu cầu 6.4).
        """
        if not contexts:
            return "(Không có đoạn tài liệu nào liên quan được tìm thấy.)"

        return "\n\n".join(
            f"[Đoạn {scored.rank}] (score={scored.score:.3f}, "
            f"nguồn: {scored.chunk.doc_id})\n{scored.chunk.content}"
            for scored in contexts
        )

    def set_system_prompt(self, new_prompt: str) -> None:
        """Thay đổi system prompt (dùng khi thực nghiệm prompt engineering)."""
        self.system_prompt = new_prompt
