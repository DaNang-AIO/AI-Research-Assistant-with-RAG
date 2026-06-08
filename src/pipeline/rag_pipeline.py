"""RAGPipeline (design.md §2.3, §2.7) — orchestrator trung tâm của hệ thống.

Triển khai: S1-PE-02 (khung + stub index_document/query để dashboard demo
luồng giả lập), S2-PE-01 (index_document/index_directory thật), và
S3-PE-03 (query thật — Property 10).
"""

import time
from typing import List

from src.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseLoader,
    BaseVectorStore,
)
from src.generation.prompt_builder import PromptBuilder
from src.models import IndexingResult, RAGResponse
from src.retrieval.retriever import Retriever


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
        self.retriever = Retriever(vector_store=vector_store, top_k=top_k)

    def index_document(self, file_path: str) -> IndexingResult:
        """Luồng indexing đầy đủ: Load → Chunk → Embed → Store
        (pseudocode design.md §2.7, S2-PE-01).

        Loop Invariant của vòng lặp embed: `len(vectors) == i` — số vector
        đã tạo luôn khớp số chunk đã xử lý, không chunk nào bị bỏ sót.
        """
        collection_name = getattr(self.vector_store, "collection_name", "rag_collection")
        try:
            document = self.loader.load(file_path)
            assert document.content, (
                "Tài liệu không có nội dung văn bản trích xuất được "
                "(ví dụ: PDF dạng ảnh/đồ hoạ không chứa text có thể chọn)"
            )

            chunks = self.chunker.chunk(document)
            assert len(chunks) >= 1, "Không tạo được chunk nào từ nội dung tài liệu"

            vectors: List[List[float]] = []
            for i, chunk in enumerate(chunks):
                assert len(vectors) == i  # invariant: đã embed đúng i chunks
                vectors.append(self.embedding_model.embed_text(chunk.content))

            success = self.vector_store.add(chunks, vectors)
        except Exception as exc:
            return IndexingResult(
                doc_id="",
                num_chunks=0,
                collection_name=collection_name,
                success=False,
                error_message=str(exc),
            )

        return IndexingResult(
            doc_id=document.doc_id,
            num_chunks=len(chunks),
            collection_name=collection_name,
            success=success,
        )

    def query(self, question: str) -> RAGResponse:
        """Luồng query đầy đủ: Embed(question) → Retrieve → Build Prompt →
        Generate (pseudocode design.md §2.7, S3-PE-03 — Property 10).

        Preconditions:
          - question không rỗng
          - llm_client.is_available() == True

        `latency_ms` đo trọn vẹn 4 bước (Yêu cầu 7.4) bằng `_measure_latency`.
        """
        assert question, "question không được rỗng"
        assert self.llm_client.is_available(), (
            f"OLLAMA server không khả dụng tại "
            f"'{getattr(self.llm_client, 'base_url', '?')}' — hãy chạy 'ollama serve'"
        )

        def _answer_with_contexts():
            # Bước 1: Embed câu hỏi
            query_vector = self.embedding_model.embed_text(question)
            assert len(query_vector) == self.embedding_model.dimension

            # Bước 2: Truy xuất context liên quan (có thể rỗng nếu chưa index)
            contexts = self.retriever.retrieve(query_vector, k=self.top_k)

            # Bước 3: Xây dựng prompt
            prompt = self.prompt_builder.build(question, contexts)
            assert question in prompt

            # Bước 4: Sinh câu trả lời
            answer = self.llm_client.generate(prompt)
            assert answer, "LLM trả về câu trả lời rỗng"

            return answer, contexts

        (answer, contexts), latency_ms = self._measure_latency(_answer_with_contexts)

        return RAGResponse(
            question=question,
            answer=answer,
            contexts=contexts,
            model_name=getattr(self.llm_client, "model_name", "llama3"),
            latency_ms=latency_ms,
        )

    def query_stream(self, question: str):
        """Biến thể streaming của `query()` — dùng cho UI cần hiển thị câu trả
        lời ngay khi từng token được LLM sinh ra (vd. `st.write_stream`), thay
        vì đợi toàn bộ `generate()` hoàn tất rồi mới hiển thị một lần.

        Preconditions giống `query()`. Thực hiện 3 bước đầu (Embed → Retrieve
        → Build Prompt) ngay lập tức rồi trả về `(contexts, model_name,
        token_stream)` — `token_stream` là generator sinh từng đoạn văn bản từ
        `llm_client.generate_stream()`. Caller tự tiêu thụ `token_stream` để
        hiển thị realtime, ráp `answer` từ các token và tự đo `latency_ms`
        (vì thời gian sinh phụ thuộc vào tốc độ caller tiêu thụ generator).
        """
        assert question, "question không được rỗng"
        assert self.llm_client.is_available(), (
            f"OLLAMA server không khả dụng tại "
            f"'{getattr(self.llm_client, 'base_url', '?')}' — hãy chạy 'ollama serve'"
        )

        query_vector = self.embedding_model.embed_text(question)
        assert len(query_vector) == self.embedding_model.dimension

        contexts = self.retriever.retrieve(query_vector, k=self.top_k)

        prompt = self.prompt_builder.build(question, contexts)
        assert question in prompt

        model_name = getattr(self.llm_client, "model_name", "llama3")
        return contexts, model_name, self.llm_client.generate_stream(prompt)

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
