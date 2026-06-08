"""Factory tạo `RAGPipeline` dùng chung giữa các trang dashboard.

Sprint 3: toàn bộ thành phần (`TextChunker`, `OllamaEmbeddingModel`,
`ChromaVectorStore`, `PromptBuilder`, `OllamaClient`) đã có triển khai thật
(S2-*, S3-DE-01, S3-PE-01, S3-ME-01), nên `RAGPipeline.query()` giờ chạy luồng
thật Embed → Retrieve → Build Prompt → Generate với LLM cục bộ qua OLLAMA.
"""

from config.settings import ChromaConfig, ChunkerConfig
from src.data.chunker import TextChunker
from src.data.loader import DocumentLoader
from src.embeddings.embedding_model import OllamaEmbeddingModel
from src.embeddings.vector_store import ChromaVectorStore
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import PromptBuilder
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
        prompt_builder=PromptBuilder(),
        top_k=config["top_k"],
    )
