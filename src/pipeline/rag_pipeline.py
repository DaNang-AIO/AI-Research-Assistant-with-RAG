"""RAGPipeline (design.md §2.3, §2.7) — orchestrator trung tâm của hệ thống.

Triển khai: S1-PE-02 (khung + stub index_document/query để dashboard demo
luồng giả lập), S2-PE-01 (index_document/index_directory thật), và
S3-PE-03 (query thật — Property 10).
"""

import os
import hashlib
import time
import logging
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

logger = logging.getLogger(__name__)

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
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        self.loader = loader
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k

    def index_document(self, file_path: str) -> IndexingResult:
        """
        Luồng indexing đầy đủ: Load → Chunk → Embed → Store.
        
        Preconditions:
          - file_path trỏ đến file tồn tại
          - embedding_model và vector_store đang sẵn sàng
        
        Postconditions:
          - Document có thể được truy xuất qua similarity_search
          - Trả về IndexingResult với success=True nếu thành công
        
        Loop Invariant (vòng lặp embed từng chunk):
          - Số chunks đã embed == số chunks đã xử lý
          - Không có chunk nào bị mất
        """
        # Precondition checks
        if file_path is None:
            raise ValueError("file_path không được để trống (None)")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"file_path phải trỏ đến file tồn tại: {file_path}")
        if self.embedding_model is None:
            raise RuntimeError("Embedding_model chưa sẵn sàng")
        if self.vector_store is None:
            raise RuntimeError("Vector_store chưa sẵn sàng")

        doc = self.loader.load(file_path)
        if doc is None or doc.content is None:
            raise ValueError(f"File không có nội dung: {file_path}")

        chunks = self.chunker.chunk(doc)
        if not chunks or len(chunks) < 1:
            raise ValueError(f"File không đủ nội dung để tạo chunk: {file_path}")

        texts = [c.content for c in chunks]
        vectors = self.embedding_model.embed_batch(texts)

        # Postconditions
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding batch returned unexpected number of vectors")
        for i, v in enumerate(vectors):
            if len(v) != self.embedding_model.dimension:
                raise ValueError(f"Embedding dimension mismatch for chunk index {i}")

        # All vectors validated — store into vector DB
        success = self.vector_store.add(chunks, vectors)
        result = IndexingResult(
            doc_id = doc.doc_id,
            num_chunks = len(chunks),
            success = success
        )

        return result


    def query(self, question: str) -> RAGResponse:
        """
        Luồng query đầy đủ: Embed(question) → Retrieve → Build Prompt → Generate.
        
        Preconditions:
          - question không rỗng
          - vector_store đã có ít nhất một document được index
          - llm_client.is_available() == True
        
        Postconditions:
          - Trả về RAGResponse với answer không rỗng
          - response.contexts chứa ít nhất 1 ScoredChunk (nếu có dữ liệu)
        """
        #Precondition
        if not question:
            raise ValueError("Không có nội dung câu hỏi")
        if not self.llm_client.is_available():
            raise RuntimeError("LLM client hiện không sẵn sàng (is_available() == False)")

        query_vector = self.embedding_model.embed_text(question)
        assert len(query_vector) == self.embedding_model.dimension

        contexts = self.vector_store.similarity_search(query_vector, k = self.top_k)

        if contexts:
            context_block = "\n\n".join(
                # TODO: đổi ctx.content / ctx.score thành đúng field thật của ScoredChunk
                f"[Đoạn {i + 1}] (độ liên quan: {getattr(ctx, 'score', 0):.3f})\n{getattr(ctx, 'content', ctx)}"
                for i, ctx in enumerate(contexts)
            )
        else:
            context_block = "(Không tìm thấy đoạn văn bản liên quan nào trong cơ sở dữ liệu.)"  

        # chưa định nghĩa nên sẽ để một prompt theo mẫu
        # prompt = self.prompt_builder.build(question, contexts)
        prompt = f"""Bạn là một trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.

QUY TẮC:
- Chỉ sử dụng thông tin trong phần "NGỮ CẢNH" dưới đây để trả lời, không dùng kiến thức ngoài tài liệu.
- Nếu ngữ cảnh không chứa thông tin liên quan, hãy nói rõ là không tìm thấy thông tin trong tài liệu, KHÔNG tự suy diễn hoặc bịa đặt.
- Trả lời ngắn gọn, chính xác, bằng tiếng Việt.
- Nếu có thể, chỉ rõ câu trả lời dựa vào đoạn nào (ví dụ: "Theo [Đoạn 2]...").

NGỮ CẢNH:
{context_block}

CÂU HỎI:
{question}

TRẢ LỜI:"""
        assert question in prompt

        answer, latency_ms = self._measure_latency(self.llm_client.generate, prompt)
        response = RAGResponse(
            question = question,
            answer = answer,
            contexts = contexts,
            model_name = self.llm_client.model_name,
            latency_ms = latency_ms
        )
        return response

    def index_directory(self, dir_path: str) -> List[IndexingResult]:
        """Index tất cả tài liệu trong một thư mục"""
        if not os.path.isdir(dir_path):
            raise ValueError(f"{dir_path} phải là thư mục")
        
        list_index = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    idx_doc = self.index_document(file_path)
                except Exception as e:
                    logger.warning("Bỏ qua file lỗi khi index: %s — %s", file_path, e)
                    continue
                if not idx_doc.success:
                    logger.warning("Index thất bại (vector_store.add trả False): %s", file_path)
                    continue
                list_index.append(idx_doc)
        return list_index

    def _measure_latency(self, func, *args, **kwargs):
        """Wrapper đo thời gian thực thi của một hàm (ms)."""
        start_time = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        return result, latency_ms
