# CLAUDE.md

File này cung cấp hướng dẫn cho Claude Code (claude.ai/code) khi làm việc với mã nguồn trong repository này.

## Trạng Thái Dự Án

Repository này hiện **chỉ mới có spec** — chưa có mã nguồn nào được viết (chưa có `src/`, `app/`, `notebooks/`, `tests/`, hay `requirements.txt`). Nội dung hiện có chỉ gồm:

- [README.md](README.md) — tiêu đề placeholder
- [specs/rag-research-assistant/requirements.md](specs/rag-research-assistant/requirements.md) — tài liệu yêu cầu chức năng/phi chức năng (định dạng EARS/INCOSE, 12 nhóm yêu cầu)
- [specs/rag-research-assistant/design.md](specs/rag-research-assistant/design.md) — tài liệu thiết kế đầy đủ: kiến trúc, data models, interfaces, function signatures, thuật toán, correctness properties và chiến lược testing

**Trước khi viết bất kỳ đoạn code nào, hãy đọc cả hai file spec.** Tài liệu thiết kế (design.md) là bản thiết kế chuẩn (authoritative blueprint): nó quy định chính xác tên class, method signature, các trường trong dataclass, cấu trúc thư mục, và thậm chí cả pseudocode cho các thuật toán cốt lõi (`index_document`, `query`). Khi triển khai, hãy bám sát các signature này thay vì tự ý thay đổi, vì các acceptance criteria trong requirements.md và các correctness properties (Phần 3 của design.md) đều được viết dựa trên các signature đó.

## Dự Án Này Là Gì

Một hệ thống RAG (Retrieval-Augmented Generation) mang tính học tập/nghiên cứu viết bằng Python, giúp người học khám phá tương tác từng thành phần của RAG pipeline (tải tài liệu → chia nhỏ văn bản → tạo embedding → vector store → retrieval → sinh câu trả lời) sử dụng LLM **cục bộ** thông qua **OLLAMA** (không phụ thuộc dịch vụ đám mây). Dự án được tổ chức theo 3 "vai trò kỹ sư AI" — Data Engineer, Pipeline Engineer, Model Engineer — mỗi vai trò có bộ Jupyter notebook riêng, cùng với một Streamlit research dashboard.

## Kiến Trúc Dự Kiến (theo design.md)

### Cấu trúc thư mục
```
notebooks/
  data_engineer/        # 01_document_loading, 02_text_chunking, 03_embedding_exploration, 04_vector_db_indexing
  pipeline_engineer/    # 01_rag_pipeline_basics, 02_retrieval_strategies, 03_prompt_engineering, 04_pipeline_evaluation
  model_engineer/       # 01_ollama_setup, 02_model_comparison, 03_embedding_models, 04_inference_optimization
src/
  data/                 # loader.py (DocumentLoader), chunker.py (TextChunker), preprocessor.py
  embeddings/           # embedding_model.py (OllamaEmbeddingModel), vector_store.py (ChromaVectorStore)
  retrieval/            # retriever.py, reranker.py
  generation/           # llm_client.py (OllamaClient), prompt_builder.py (PromptBuilder), response_generator.py
  pipeline/             # rag_pipeline.py (RAGPipeline), experiment_tracker.py (ExperimentTracker)
app/                    # Streamlit dashboard: main.py + pages/{01_document_upload,02_chat_interface,03_retrieval_debug,04_experiment_log}.py
data/                   # raw/, processed/, vector_db/ (ChromaDB persistent storage)
experiments/            # logs/, results/
config/                 # settings.py (AppConfig/OllamaConfig/ChromaConfig/ChunkerConfig), models.yaml
tests/                  # test_loader.py, test_chunker.py, test_retriever.py, test_pipeline.py
```

### Luồng dữ liệu cốt lõi (RAGPipeline điều phối toàn bộ)

**Indexing:** `DocumentLoader.load()` → `TextChunker.chunk()` → `OllamaEmbeddingModel.embed_batch()` → `ChromaVectorStore.add()` → `ExperimentTracker.log_indexing()`

**Querying:** `OllamaEmbeddingModel.embed_text(question)` → `ChromaVectorStore.similarity_search()` → `PromptBuilder.build()` → `OllamaClient.generate()` → `ExperimentTracker.log_query()`

`RAGPipeline` là class điều phối trung tâm, được dùng chung bởi cả notebooks và Streamlit app — không nên triển khai trùng lặp logic này ở hai nơi. Xem design.md §2.7 để biết pseudocode chi tiết (kèm assertion/invariant) cho `index_document()` và `query()`.

### Các interface cốt lõi (design.md §2.2 — triển khai dựa trên các ABC này)
`BaseLoader`, `BaseChunker`, `BaseEmbeddingModel`, `BaseVectorStore`, `BaseLLMClient` — mỗi interface đều có pre/postcondition và loop invariant rõ ràng, là cơ sở cho các correctness property tương ứng (design.md §2.5 / Phần 3).

### Data models (design.md §2.1)
`Document`, `Chunk`, `EmbeddingVector`, `ScoredChunk`, `RAGResponse`, `IndexingResult`, `ExperimentLog`, cùng với enum `ChunkStrategy` (FIXED_SIZE/RECURSIVE/SEMANTIC) và `DocumentType` (PDF/TXT/MARKDOWN/HTML).

## Các Ràng Buộc Thiết Kế Cần Tuân Thủ

- **LLM chỉ chạy cục bộ**: mọi lời gọi sinh câu trả lời/embedding đều đi qua OLLAMA server (`http://localhost:11434` mặc định) — không gọi LLM trên cloud. `OllamaClient.is_available()` không bao giờ được ném exception, chỉ trả về bool.
- **Embedding tất định (deterministic)**: `embed_text()` phải trả về cùng một vector cho cùng một input, và `embed_batch(texts)[i] == embed_text(texts[i])`.
- **Chunking phải bảo toàn nội dung**: mọi chiến lược chunking phải bao phủ toàn bộ `Document.content` gốc, và mỗi `Chunk.doc_id` phải khớp với `Document.doc_id` nguồn.
- **Vector store hỗ trợ song song hai chế độ**: `ChromaVectorStore` phải hỗ trợ cả persistent (khi có `persist_dir`) và in-memory (cho notebook); kết quả `similarity_search()` phải được sắp xếp theo `score` giảm dần trong khoảng `[0.0, 1.0]` và `len(results) <= k`.
- **ExperimentTracker round-trip**: `save_session()` → `load_session()` phải khôi phục đúng danh sách `ExperimentLog` ban đầu.
- Có 12 correctness property được liệt kê trong design.md Phần 3 (ví dụ: "chunking bảo toàn nội dung", "embedding có chiều nhất quán", "retrieval sắp xếp theo score giảm dần") — các property này ánh xạ trực tiếp tới các property-based test (Hypothesis) được yêu cầu trong Yêu Cầu 12.

## Các Lệnh Thường Dùng (sau khi dự án được triển khai theo spec)

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Khởi động OLLAMA server cục bộ và pull các model cần thiết
ollama serve
ollama pull llama3
ollama pull nomic-embed-text

# Chạy Streamlit research dashboard
streamlit run app/main.py

# Mở Jupyter cho các notebook theo vai trò
jupyter notebook notebooks/

# Chạy test suite (unit test + property-based test bằng Hypothesis, mục tiêu coverage ≥70% cho src/)
pytest tests/ -v
```

## Lưu Ý Khi Triển Khai

- Cấu hình được nạp qua biến môi trường: `AppConfig.from_env()` đọc từ `.env` (xem `.env.example`, sẽ cần được tạo) và dùng giá trị mặc định hợp lý khi biến môi trường không được định nghĩa; danh sách model OLLAMA được hỗ trợ nằm trong `config/models.yaml`.
- Các notebook phải chạy được từ đầu đến cuối mà không gặp lỗi ngoại lệ chưa xử lý, và được kỳ vọng sử dụng `ExperimentTracker` để ghi lại kết quả thực nghiệm mà không làm gián đoạn luồng notebook.
- Streamlit dashboard có đúng 4 trang (Document Upload, Chat Interface, Retrieval Debug, Experiment Log), được điều khiển bởi cấu hình sidebar dùng chung (LLM model, embedding model, chunk size, top-k) lưu trong `st.session_state`.
