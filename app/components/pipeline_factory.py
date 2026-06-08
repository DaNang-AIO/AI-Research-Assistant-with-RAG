"""Factory tạo `RAGPipeline` dùng chung giữa các trang dashboard.

Sprint 1: `TextChunker`, `OllamaEmbeddingModel`, `ChromaVectorStore` và
`PromptBuilder` chưa có triển khai thật (xem S2-DE-*, S2-ME-*, S3-PE-01) nên
được truyền vào dưới dạng placeholder `None` — `RAGPipeline.index_document`/
`query` ở giai đoạn này là stub và không gọi tới các thành phần đó. Khi các
sprint sau hoàn thiện những class này, factory sẽ được cập nhật để khởi tạo
chúng thật theo cấu hình sidebar.
"""

from src.data.loader import DocumentLoader
from src.generation.llm_client import OllamaClient
from src.pipeline.rag_pipeline import RAGPipeline


def build_pipeline(config: dict) -> RAGPipeline:
    """Khởi tạo `RAGPipeline` theo cấu hình hiện tại của sidebar."""
    return RAGPipeline(
        loader=DocumentLoader(),
        chunker=None,
        embedding_model=None,
        vector_store=None,
        llm_client=OllamaClient(model_name=config["ollama_model"]),
        prompt_builder=None,
        top_k=config["top_k"],
    )
