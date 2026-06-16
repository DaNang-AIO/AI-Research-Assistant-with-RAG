"""TextChunker (design.md §2.3) — kế thừa `src.interfaces.BaseChunker`.

Triển khai: S2-DE-01 (chunk_by_fixed_size) và S2-DE-02 (chunk_by_recursive,
chunk_by_semantic, điều phối qua `chunk()`).
"""

from typing import List
from src.interfaces import BaseChunker
from src.models import Chunk, Document, ChunkStrategy


class TextChunker(BaseChunker):
    """
    Chia nhỏ document theo nhiều chiến lược khác nhau.
    Cho phép thực nghiệm chunk_size và chunk_overlap.
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        """Chia document theo chiến lược đã cấu hình."""
        raise NotImplementedError(
            "TextChunker.chunk() sẽ được triển khai đầy đủ ở Sprint 2"
        )
