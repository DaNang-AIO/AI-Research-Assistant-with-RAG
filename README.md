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

Dự án đang ở **Sprint 0 — Khởi Tạo & Nền Tảng**: đã có khung thư mục, data models, interfaces (ABC) và cấu hình dùng chung. Xem [tasks.md](specs/rag-research-assistant/tasks.md) để theo dõi lộ trình các sprint tiếp theo.
