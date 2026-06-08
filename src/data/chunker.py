"""TextChunker (design.md §2.3) — kế thừa `src.interfaces.BaseChunker`.

Triển khai: S2-DE-01 (chunk_by_fixed_size, _create_chunk) và S2-DE-02
(chunk_by_recursive, chunk_by_semantic, điều phối qua `chunk()` —
Property 1, 2).
"""

import re
from typing import List, Tuple

from src.interfaces import BaseChunker
from src.models import Chunk, ChunkStrategy, Document

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
_WORD_BOUNDARY = re.compile(r"\s+")

_RECURSIVE_PATTERNS = (_PARAGRAPH_BOUNDARY, _SENTENCE_BOUNDARY, _WORD_BOUNDARY)

Span = Tuple[int, int]


class TextChunker(BaseChunker):
    """Chia nhỏ document theo nhiều chiến lược khác nhau.

    Cho phép thực nghiệm chunk_size và chunk_overlap.
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size      # Số ký tự mỗi chunk
        self.chunk_overlap = chunk_overlap  # Số ký tự chồng lấp giữa chunks

    def chunk(self, document: Document) -> List[Chunk]:
        """Chia document theo chiến lược đã cấu hình (`self.strategy`).

        Document.content rỗng → trả về danh sách rỗng (Yêu cầu 2.5).
        """
        if not document.content:
            return []

        if self.strategy == ChunkStrategy.FIXED_SIZE:
            return self.chunk_by_fixed_size(document)
        if self.strategy == ChunkStrategy.SEMANTIC:
            return self.chunk_by_semantic(document)
        return self.chunk_by_recursive(document)

    def chunk_by_fixed_size(self, document: Document) -> List[Chunk]:
        """Cắt theo `chunk_size` ký tự cố định bằng cửa sổ trượt — mỗi chunk
        dài tối đa `chunk_size` ký tự và hai chunk liên tiếp chồng lấp đúng
        `chunk_overlap` ký tự (Yêu cầu 2.1, 2.6), bao phủ toàn bộ nội dung
        gốc (Property 1).
        """
        content = document.content
        if not content:
            return []

        total = len(content)
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: List[Chunk] = []
        start = 0
        index = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            chunks.append(
                self._create_chunk(document.doc_id, content[start:end], start, end, index)
            )
            index += 1
            if end == total:
                break
            start += step
        return chunks

    def chunk_by_recursive(self, document: Document) -> List[Chunk]:
        """Chia đệ quy theo thứ tự ưu tiên: đoạn văn (paragraph) → câu
        (sentence) → từ (word). Trước tiên cắt nội dung thành các đoạn con
        liên tiếp (bao phủ trọn vẹn, không chồng lấp) bằng ranh giới tự
        nhiên, sau đó gộp lại thành các chunk có độ dài gần `chunk_size`
        nhất có thể và chồng lấp `chunk_overlap` ký tự giữa các chunk liền kề.
        """
        content = document.content
        if not content:
            return []

        spans = self._split_recursive((0, len(content)), content, level=0)
        return self._pack_spans(document, spans)

    def chunk_by_semantic(self, document: Document) -> List[Chunk]:
        """Chia theo ranh giới câu — mỗi chunk gồm một hoặc nhiều câu liên
        tiếp được gộp lại sao cho độ dài gần với `chunk_size`, các chunk
        liền kề chồng lấp `chunk_overlap` ký tự.
        """
        content = document.content
        if not content:
            return []

        spans = self._split_by_pattern((0, len(content)), content, _SENTENCE_BOUNDARY)
        return self._pack_spans(document, spans)

    def _create_chunk(
        self, doc_id: str, content: str, start: int, end: int, index: int
    ) -> Chunk:
        """Tạo Chunk object với `chunk_id` duy nhất trong phạm vi document."""
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            content=content,
            start_index=start,
            end_index=end,
        )

    # -- Helpers cho chunk_by_recursive / chunk_by_semantic -------------------

    def _split_by_pattern(self, span: Span, content: str, pattern: "re.Pattern") -> List[Span]:
        """Cắt `span` thành các đoạn con liên tiếp ngay sau mỗi vị trí khớp
        `pattern`. Luôn bao phủ trọn vẹn `span`, không mất ký tự nào — kể cả
        phần đuôi sau dấu phân cách cuối cùng.
        """
        start, end = span
        text = content[start:end]
        spans: List[Span] = []
        last = 0
        for match in pattern.finditer(text):
            cut = match.end()
            if cut > last:
                spans.append((start + last, start + cut))
                last = cut
        if last < len(text):
            spans.append((start + last, end))
        return spans or [span]

    def _split_recursive(self, span: Span, content: str, level: int) -> List[Span]:
        """Chia `span` đệ quy bằng các separator theo thứ tự ưu tiên
        (`_RECURSIVE_PATTERNS`) cho đến khi mỗi đoạn con dài <= `chunk_size`
        hoặc đã hết separator (lúc đó cắt cứng theo ký tự)."""
        start, end = span
        if end - start <= self.chunk_size:
            return [span]
        if level >= len(_RECURSIVE_PATTERNS):
            return self._split_fixed(span)

        sub_spans = self._split_by_pattern(span, content, _RECURSIVE_PATTERNS[level])
        if len(sub_spans) <= 1:
            return self._split_recursive(span, content, level + 1)

        result: List[Span] = []
        for sub in sub_spans:
            result.extend(self._split_recursive(sub, content, level + 1))
        return result

    def _split_fixed(self, span: Span) -> List[Span]:
        """Cắt cứng theo `chunk_size` ký tự — chặn cuối khi không còn
        separator nào chia nhỏ thêm được (ví dụ một "từ" quá dài, liền mạch).
        """
        start, end = span
        spans: List[Span] = []
        cursor = start
        while cursor < end:
            cut = min(cursor + self.chunk_size, end)
            spans.append((cursor, cut))
            cursor = cut
        return spans

    def _pack_spans(self, document: Document, spans: List[Span]) -> List[Chunk]:
        """Gộp các đoạn con liên tiếp (đã bao phủ trọn vẹn nội dung, không
        chồng lấp) thành các chunk có độ dài gần `chunk_size`, đảm bảo hai
        chunk liền kề chồng lấp tối thiểu `chunk_overlap` ký tự khi có thể —
        và toàn bộ nội dung gốc luôn được bao phủ (Property 1).
        """
        content = document.content
        n = len(spans)

        # Bước 1: gộp các đoạn con liên tiếp thành các "phân vùng" liền kề,
        # không chồng lấp, mỗi phân vùng dài tối đa chunk_size — phủ trọn
        # vẹn nội dung đúng một lần (không thiếu, không trùng).
        partitions: List[Span] = []  # (chỉ số span đầu, chỉ số span cuối) — bao gồm cả hai đầu
        i = 0
        while i < n:
            j = i
            while j + 1 < n and (spans[j + 1][1] - spans[i][0]) <= self.chunk_size:
                j += 1
            partitions.append((i, j))
            i = j + 1

        # Bước 2: với mỗi phân vùng (trừ phân vùng đầu tiên), kéo lùi điểm
        # bắt đầu về phía trước để chồng lấp với chunk liền trước tối thiểu
        # `chunk_overlap` ký tự — miễn là chunk vẫn không vượt quá chunk_size.
        # Vì `end` luôn là điểm cuối của phân vùng (tăng dần, không trùng),
        # chunk sau luôn mở rộng ra ngoài chunk trước — không tạo chunk thừa.
        chunks: List[Chunk] = []
        for index, (start_idx, end_idx) in enumerate(partitions):
            end = spans[end_idx][1]
            actual_start_idx = start_idx
            if index > 0 and self.chunk_overlap > 0:
                k = start_idx
                while k > 0:
                    candidate = k - 1
                    length = end - spans[candidate][0]
                    if length > self.chunk_size:
                        break
                    k = candidate
                    overlap = spans[start_idx][0] - spans[candidate][0]
                    if overlap >= self.chunk_overlap:
                        break
                actual_start_idx = k

            start = spans[actual_start_idx][0]
            chunks.append(
                self._create_chunk(document.doc_id, content[start:end], start, end, index)
            )
        return chunks
