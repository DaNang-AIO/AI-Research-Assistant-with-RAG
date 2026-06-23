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
        strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap không được âm")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
        

        self.strategy = strategy
        self.chunk_size = chunk_size        # Số ký tự mỗi chunk
        self.chunk_overlap = chunk_overlap  # Số ký tự chồng lấp giữa chunks

    def chunk(self, document: Document) -> List[Chunk]:
        """Chia document theo chiến lược đã cấu hình"""
        if not document.content:
            return []
        
        strategy_map = {
            ChunkStrategy.FIXED_SIZE: self.chunk_by_fixed_size,
            ChunkStrategy.RECURSIVE: self.chunk_by_recursive,
            ChunkStrategy.SEMANTIC: self.chunk_by_semantic
        }

        strategy_fn = strategy_map.get(self.strategy)
        if strategy_fn is None:
            raise ValueError(f"Chiến lược {self.strategy} không hộp lệ hoặc chưa được hỗ trợ")
        
        chunks = strategy_fn(document)
        return chunks

    def chunk_by_fixed_size(self, document: Document) -> List[Chunk]:
        """Cắt theo chunk_size cố định với overlap"""
        text = document.content
        if text == "":
            return []
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        start = 0 
        chunk_index = 0

        while start < len(text): 
            end = min(start + self.chunk_size, len(text)) 
            chunk_content = text[start:end] 
            chunks.append( 
                self._create_chunk( 
                    doc_id=document.doc_id, 
                    content=chunk_content, 
                    start=start, end=end, 
                    index=chunk_index, 
                )) 
            chunk_index += 1 
            if end == len(text):
                break
            start += step 
        return chunks

    def chunk_by_recursive(self, document: Document) -> List[Chunk]:
        """Chia theo thứ tự: paragraph → sentence → word"""
        raise NotImplementedError("Recursive chunking chưa được triển khai")

    def chunk_by_semantic(self, document: Document) -> List[Chunk]:
        """Chia theo ranh giới câu (sentence boundary)"""
        raise NotImplementedError("Semantic chunking chưa được triển khai")

    def _create_chunk(
        self, doc_id: str, content: str, start: int, end: int, index: int
    ) -> Chunk:
        """Tạo Chunk object với chunk_id duy nhất"""
        if content is None:
            raise ValueError("Không có nội dung để tạo chunk")
        
        unique_chunk_id = f"{doc_id}_ch_{index}"

        return Chunk(
            chunk_id = unique_chunk_id,
            doc_id = doc_id,
            content = content,
            start_index = start,
            end_index = end,
            metadata={ 
                "chunk_index": index, 
                "chunk_length": len(content), 
                "strategy": self.strategy.value
            }
        )

        
