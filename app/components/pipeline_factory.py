"""Factory tạo `RAGPipeline` dùng chung giữa các trang dashboard.

Sprint 2: `TextChunker`, `OllamaEmbeddingModel`, `ChromaVectorStore` đã có
triển khai thật (S2-DE-01/02, S2-ME-01/02, S2-DE-03), nên được khởi tạo theo
cấu hình sidebar — `RAGPipeline.index_document()` giờ chạy luồng thật
Load → Chunk → Embed → Store và lưu vào ChromaDB persistent storage.
`PromptBuilder` vẫn là placeholder `None` cho tới khi S3-PE-01 hoàn thiện —
`RAGPipeline.query()` ở giai đoạn này vẫn là stub và không gọi tới nó.
"""

from config.settings import ChromaConfig, ChunkerConfig
from src.data.chunker import TextChunker
from src.data.loader import DocumentLoader
from src.embeddings.embedding_model import OllamaEmbeddingModel
from src.embeddings.vector_store import ChromaVectorStore
from src.generation.llm_client import OllamaClient
from src.models import ChunkStrategy
from src.pipeline.rag_pipeline import RAGPipeline

_chunker_defaults = ChunkerConfig()
_chroma_defaults = ChromaConfig()


def build_pipeline(config: dict) -> RAGPipeline:
    """Khởi tạo `RAGPipeline` theo cấu hình hiện tại của sidebar.

    Sidebar chỉ điều khiển `chunk_size`/`top_k`/tên model — các tham số còn
    lại (chunk_overlap, persist_dir, collection_name) lấy từ giá trị mặc định
    hợp lý trong `config/settings.py` (Sprint 5 sẽ đọc đầy đủ qua `.env`).
    """
    return RAGPipeline(
        loader=DocumentLoader(),
        chunker=TextChunker(
            strategy=ChunkStrategy(_chunker_defaults.strategy),
            chunk_size=config["chunk_size"],
            chunk_overlap=_chunker_defaults.chunk_overlap,
        ),
        embedding_model=OllamaEmbeddingModel(model_name=config["embedding_model"]),
        vector_store=ChromaVectorStore(
            collection_name=_chroma_defaults.collection_name,
            persist_dir=_chroma_defaults.persist_dir,
        ),
        llm_client=OllamaClient(model_name=config["ollama_model"]),
        prompt_builder=None,
        top_k=config["top_k"],
    )
