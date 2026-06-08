# AI Research Assistant with RAG

Hệ thống RAG (Retrieval-Augmented Generation) mang tính học tập/nghiên cứu, giúp khám phá tương tác từng thành phần của RAG pipeline (tải tài liệu → chia nhỏ văn bản → tạo embedding → vector store → retrieval → sinh câu trả lời) bằng LLM **cục bộ** thông qua **OLLAMA** — không phụ thuộc dịch vụ đám mây.

Dự án được tổ chức theo 3 vai trò kỹ sư AI — **Data Engineer**, **Pipeline Engineer**, **Model Engineer** — mỗi vai trò có bộ Jupyter notebook riêng, cùng một Streamlit research dashboard dùng chung.

> 📋 Tài liệu đặc tả đầy đủ nằm trong [specs/rag-research-assistant/](specs/rag-research-assistant/):
> [requirements.md](specs/rag-research-assistant/requirements.md) (yêu cầu chức năng/phi chức năng),
> [design.md](specs/rag-research-assistant/design.md) (kiến trúc & thiết kế chi tiết),
> [tasks.md](specs/rag-research-assistant/tasks.md) (kế hoạch sprint theo Agile/Scrum).

## Bắt Đầu Nhanh

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Sao chép cấu hình mẫu
cp .env.example .env

# 3. Khởi động OLLAMA server cục bộ và pull các model cần thiết
ollama serve
ollama pull llama3
ollama pull nomic-embed-text

# 4. Chạy Streamlit research dashboard
streamlit run app/main.py

# 5. Hoặc mở Jupyter để khám phá theo từng vai trò
jupyter notebook notebooks/

# 6. Chạy test suite
pytest tests/ -v
```

## Cấu Trúc Dự Án

```
notebooks/        # Jupyter notebooks theo 3 vai trò: data_engineer, pipeline_engineer, model_engineer
src/              # Source code lõi: data, embeddings, retrieval, generation, pipeline
app/              # Streamlit research dashboard (4 trang: Upload, Chat, Retrieval Debug, Experiment Log)
data/             # Tài liệu gốc (raw), đã xử lý (processed), và ChromaDB persistent storage (vector_db)
experiments/      # Logs & kết quả các phiên thực nghiệm (ExperimentTracker)
config/           # Cấu hình hệ thống (settings.py, models.yaml) và .env
tests/            # Unit test & property-based test (Hypothesis)
```

Xem [design.md §1.2](specs/rag-research-assistant/design.md) để biết chi tiết đầy đủ cấu trúc thư mục và các class cốt lõi.

## Trạng Thái

Dự án đang ở **Sprint 2 — Kết Nối Ollama & Pipeline Cục Bộ**: đã hoàn thiện cấu hình Ollama, triển khai pipeline index/retrieval cục bộ với nomic-embed-text và Llama 3, tích hợp vào Streamlit và thêm trang debug. Xem [tasks.md](specs/rag-research-assistant/tasks.md) để theo dõi lộ trình các sprint tiếp theo.

## Cách kiểm tra Sprint 2 trên giao diện Streamlit

Sprint 2 làm cho trang Document Upload chạy luồng index thật (Load → Chunk → Embed → Store), nên để thấy nó hoạt động bạn cần OLLAMA đang chạy thật sự (mình vừa kiểm tra — hiện OLLAMA chưa chạy trên máy bạn).

### Bước 1 — Khởi động OLLAMA và tải model embedding (mở một terminal riêng):

ollama serve
ollama pull nomic-embed-text # model embedding mặc định trong config/settings.py
Có thể cần ollama pull llama3 nếu trang chính (main.py) báo "chưa kết nối" — trang đó cũng kiểm tra trạng thái OLLAMA cho model sinh câu trả lời.

### Bước 2 — Chạy dashboard:

streamlit run app/main.py

### Bước 3 — Vào trang "📄 Document Upload" ở thanh điều hướng bên trái:

Tải lên một file .txt/.md/.pdf — bạn có thể dùng luôn 2 file mẫu vừa được notebook Sprint 2 tạo ra: data/raw/sample_rag_overview.txt và data/raw/sample_chunking_notes.md

Sau khi index xong, bạn sẽ thấy 3 chỉ số thật: Số chunks, Doc ID, Collection — đây chính là kết quả từ RAGPipeline.index_document() chạy luồng Sprint 2 (đọc file → TextChunker chia nhỏ → OllamaEmbeddingModel tạo vector → ChromaVectorStore lưu trữ)

Phần "Lịch sử upload trong phiên này" bên dưới ghi lại mọi lần index trong session

Kiểm chứng thêm (tuỳ chọn): dữ liệu vector sẽ được lưu persistent tại data/vector_db/ (theo ChromaConfig.persist_dir mặc định) — bạn có thể kiểm tra thư mục này có file mới sau khi upload thành công.

Link web: https://ai-research-assistant-with-rag-tnieqozgdhfzsrhujiz9bu.streamlit.app/
