"""RAGPipeline (design.md §2.3, §2.7) — orchestrator trung tâm của hệ thống.

Triển khai: S1-PE-02 (khung + stub index_document/query để dashboard demo
luồng giả lập), S2-PE-01 (index_document/index_directory thật), và
S3-PE-03 (query thật — Property 10).
"""

import math
import time
from typing import List

from src.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseLoader,
    BaseVectorStore,
)
from src.models import IndexingResult, RAGResponse


class RAGPipeline:
    """Orchestrator điều phối toàn bộ luồng RAG:
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
        prompt_builder: "PromptBuilder",
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
        """Stub Sprint 1 — tải tài liệu thật bằng `loader`, giả lập phần
        chunk/embed/store để dashboard có thể demo luồng "Mocked End-to-End".

        Luồng thật (Load → Chunk → Embed → Store, theo loop invariant
        `len(vectors) == i`) sẽ thay thế phần giả lập này ở S2-PE-01
        (pseudocode design.md §2.7).
        """
        collection_name = getattr(self.vector_store, "collection_name", "rag_collection")
        try:
            document = self.loader.load(file_path)
        except Exception as exc:
            return IndexingResult(
                doc_id="",
                num_chunks=0,
                collection_name=collection_name,
                success=False,
                error_message=str(exc),
            )

        # Giả lập số chunk dựa trên chunk_size cấu hình (sẽ thay bằng
        # chunker.chunk(document) thật ở S2-PE-01)
        chunk_size = getattr(self.chunker, "chunk_size", 512)
        num_chunks = max(1, math.ceil(len(document.content) / chunk_size))

        return IndexingResult(
            doc_id=document.doc_id,
            num_chunks=num_chunks,
            collection_name=collection_name,
            success=True,
        )

    def query(self, question: str) -> RAGResponse:
        """Stub Sprint 1 — trả về RAGResponse giả lập để dashboard demo
        Chat Interface trước khi luồng thật hoàn thiện.

        Luồng thật (Embed câu hỏi → Retrieve → Build Prompt → Generate,
        đo `latency_ms` — Property 10) sẽ thay thế ở S3-PE-03.
        """

        def _mock_answer() -> str:
            return (
                f"[Câu trả lời giả lập] Đây là phản hồi mô phỏng cho câu hỏi: "
                f"'{question}'. RAGPipeline.query() sẽ sinh câu trả lời thật từ "
                f"LLM cục bộ qua OLLAMA ở Sprint 3 (S3-PE-03)."
            )

        answer, latency_ms = self._measure_latency(_mock_answer)
        model_name = getattr(self.llm_client, "model_name", "llama3")

        return RAGResponse(
            question=question,
            answer=answer,
            contexts=[],
            model_name=model_name,
            latency_ms=latency_ms,
        )

    def index_directory(self, dir_path: str) -> List[IndexingResult]:
        """Index tất cả tài liệu trong một thư mục."""
        return [
            self.index_document(document.file_path)
            for document in self.loader.load_directory(dir_path)
        ]

    def _measure_latency(self, func, *args, **kwargs):
        """Wrapper đo thời gian thực thi của một hàm.

        Trả về tuple `(kết quả của func, latency_ms)`.
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return result, latency_ms
