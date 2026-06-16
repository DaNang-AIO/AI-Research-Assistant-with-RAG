"""RAGPipeline (design.md §2.3, §2.7) — orchestrator trung tâm của hệ thống.

Triển khai: S1-PE-02 (khung + stub index_document/query để dashboard demo
luồng giả lập), S2-PE-01 (index_document/index_directory thật), và
S3-PE-03 (query thật — Property 10).
"""

import os
import hashlib
import time
from typing import List
import datetime

from src.interfaces import (
    BaseLoader,
    BaseChunker,
    BaseEmbeddingModel,
    BaseVectorStore,
    BaseLLMClient,
)
from src.generation.prompt_builder import PromptBuilder
from src.models import IndexingResult, RAGResponse, ScoredChunk, Chunk


class RAGPipeline:
    """
    Orchestrator điều phối toàn bộ luồng RAG:
    Document → Chunk → Embed → Store → Retrieve → Generate.
    
    Đây là class trung tâm được sử dụng trong cả notebooks và Streamlit app.
    """

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient,
        prompt_builder: PromptBuilder,
        top_k: int = 5,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k

    def index_document(self, file_path: str) -> IndexingResult:
        """
        Luồng indexing: chạy loader thật để xác thực tài liệu, các bước sau giả lập ở Sprint 1.
        Sprint 2: thay bằng luồng indexing thật đầy đủ.
        """
        try:
            # 1. Gọi loader thật để kiểm tra định dạng và nạp tài liệu (S1-DE-01 / S1-DE-02)
            # Preconditions: file phải tồn tại
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

            doc = self.loader.load(file_path)

            # 2. Sinh mock IndexingResult dựa trên độ dài nội dung tài liệu
            time.sleep(0.5)  # mô phỏng độ trễ xử lý
            
            # Đọc chunk_size cấu hình từ chunker (hoặc dùng mặc định 512)
            chunk_size = getattr(self.chunker, "chunk_size", 512)
            num_chunks = max(1, len(doc.content) // chunk_size + 1)
            
            collection_name = getattr(self.vector_store, "collection_name", "rag_collection")

            return IndexingResult(
                doc_id=doc.doc_id,
                num_chunks=num_chunks,
                collection_name=collection_name,
                success=True,
                error_message=None,
            )
        except Exception as e:
            # Yêu cầu 7.6: bắt ngoại lệ từ loader và trả về IndexingResult với success=False
            return IndexingResult(
                doc_id="",
                num_chunks=0,
                collection_name="",
                success=False,
                error_message=str(e),
            )

    def query(self, question: str) -> RAGResponse:
        """
        Mô phỏng kết quả sinh câu trả lời RAG ở Sprint 1.
        Sprint 3: thay bằng luồng query thật đầy đủ.
        """
        time.sleep(1.0)  # mô phỏng độ trễ sinh từ LLM

        # Tạo chunk giả lập làm ngữ cảnh truy xuất
        fake_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=f"chunk_{i}",
                    doc_id="demo_doc_001",
                    content=f"[Demo Sprint 1] Đoạn ngữ cảnh mẫu số {i+1} liên quan đến câu hỏi: '{question}'",
                    start_index=i * 200,
                    end_index=(i + 1) * 200,
                    metadata={"source": "demo_document.pdf", "page": i + 1},
                ),
                score=round(0.95 - i * 0.08, 2),
                rank=i + 1,
            )
            for i in range(min(self.top_k, 3))
        ]

        model_name = getattr(self.llm_client, "model_name", "llama3")

        fake_answer = (
            f"**[Sprint 1 — Demo Mode]** Đây là câu trả lời giả lập cho câu hỏi: "
            f'"{question}"\n\n'
            f"Hệ thống đang chạy ở chế độ stub. Câu trả lời thật từ LLM cục bộ ({model_name}) "
            "sẽ được kích hoạt từ Sprint 3 khi `RAGPipeline.query()` được triển khai đầy đủ (task S3-PE-03)."
        )

        return RAGResponse(
            question=question,
            answer=fake_answer,
            contexts=fake_chunks,
            model_name=model_name,
            latency_ms=1000.0,
            timestamp=datetime.datetime.now(),
        )

    def index_directory(self, dir_path: str) -> List[IndexingResult]:
        """Tải và index tất cả tài liệu hỗ trợ trong thư mục."""
        results = []
        if not os.path.isdir(dir_path):
            return results
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.loader.supports(file_path):
                    results.append(self.index_document(file_path))
        return results
