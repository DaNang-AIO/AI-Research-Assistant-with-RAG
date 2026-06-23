"""TextChunker (design.md §2.3) — kế thừa `src.interfaces.BaseChunker`.

Triển khai: S2-DE-01 (chunk_by_fixed_size) và S2-DE-02 (chunk_by_recursive,
chunk_by_semantic, điều phối qua `chunk()`).
"""

from typing import List, Optional
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

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Hàm đệ quy lõi xử lý cắt và gom chuỗi (str -> List[str])"""
        if len(text) <= self.chunk_size:
            return [text]
        
        appropriate_separator = None
        remaining_separators = []

        for i, sep in enumerate(separators):
            if sep in text:
                appropriate_separator = sep
                remaining_separators = separators[i+1:]
                break
        
        # Fallback: Nếu không còn separator nào phù hợp, cắt cứng theo chuỗi ký tự
        if appropriate_separator is None:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        splits = text.split(appropriate_separator)
        parts = [
            part + (appropriate_separator if i < len(splits) - 1 else "")
            for i, part in enumerate(splits)
        ]
        final_chunks = []
        current_chunk = ""
        
        for part in parts:
            # Nếu phần split nhỏ vẫn lớn hơn chunk_size -> Đệ quy sâu xuống tiếp
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                # Đệ quy xuống cấp thấp hơn với phần text con
                sub_chunks = self._recursive_split(part, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                # Gom các cụm text nhỏ lại để tối ưu hóa kích thước chunk gần với chunk_size nhất
                separator_to_append = appropriate_separator if current_chunk else ""
                if len(current_chunk) + len(part) <= self.chunk_size:
                    current_chunk += part
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    current_chunk = part
                    
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    def chunk_by_recursive(self, document: Document, separators: Optional[List[str]] = None) -> List[Chunk]:
        """Chia theo thứ tự mặc định hoặc cấu hình: paragraph → sentence → word"""
        text = document.content
        if text == "":
            return []
            
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]
            
        # 1. Thực hiện bổ nhỏ đệ quy để lấy danh sách nội dung dạng text chuỗi phẳng
        text_chunks = self._recursive_split(text, separators)
        
        # 2. Đóng gói danh sách chuỗi thành List[Chunk] chuẩn chỉnh, tìm vị trí start/end chính xác
        chunks = []
        current_search_idx = 0
        
        for index, content in enumerate(text_chunks):
            if not content.strip():
                continue
            # Tìm vị trí xuất hiện của chunk này trong văn bản gốc để gán start/end_index chuẩn xác
            start_idx = text.find(content, current_search_idx)
            if start_idx == -1:  # Dự phòng trường hợp strip làm lệch chuỗi
                raise ValueError(f"Không tìm thấy chunk trong văn bản gốc: {content[:30]}...")
            end_idx = start_idx + len(content)
            current_search_idx = end_idx
            
            chunks.append(
                self._create_chunk(
                    doc_id=document.doc_id,
                    content=content,
                    start=start_idx,
                    end=end_idx,
                    index=index
                )
            )
        return chunks

    def chunk_by_semantic(self, document: Document) -> List[Chunk]:
        """Chia dựa trên ranh giới ngữ nghĩa của câu (sentence boundaries)"""
        # Tận dụng cấu trúc đệ quy nhưng ưu tiên thêm các dấu kết thúc câu chuyên sâu
        raise NotImplementedError("chunk_by_semantic chưa được triển khai. Vui lòng sử dụng chunk_by_recursive hoặc chunk_by_fixed_size.")
    
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