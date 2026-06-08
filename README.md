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

Dự án đang ở **Sprint 3 — Thử Nghiệm & Theo Dõi**: Đã hoàn thiện giao diện người dùng với trang Chat Interface (tích hợp OLLAMA) và trang Retrieval Debug. Hệ thống hiện đã có thể thực hiện end-to-end RAG pipeline: nạp tài liệu, tìm kiếm ngữ nghĩa, sinh câu trả lời và hiển thị nguồn tham chiếu chi tiết. Các tính năng experiment tracking và evaluation đều đã sẵn sàng. Xem [tasks.md](specs/rag-research-assistant/tasks.md) để theo dõi lộ trình các sprint tiếp theo.

### Cách kiểm tra trên UI

Làm cụ thể từng bước với máy của bạn
Hiện bạn đã có sẵn 1 phiên tên chunk_512_top_k_5. Bạn cần thêm một phiên thứ hai để có cái để so sánh:

Bước 1 — Tạo dữ liệu cho phiên thứ 2:

Kéo thanh trượt Chunk Size sang một giá trị khác (vd. 256)
Vào trang Document Upload hoặc Chat Interface, upload tài liệu / đặt câu hỏi vài lần (để có sự kiện mới được ghi log với cấu hình mới này)
Bước 2 — Lưu phiên thứ 2:

Quay lại trang Experiment Log
Trong ô "Tên phiên" (cột trái, "Lưu phiên hiện tại"), gõ một tên mới, vd: chunk_256_top_k_6
Bấm nút "💾 Lưu phiên"
→ Bây giờ bạn có 2 file: chunk_512_top_k_5.json và chunk_256_top_k_6.json
Bước 3 — So sánh (cột phải):

Ô "Phiên A": gõ chunk_512_top_k_5 (gõ y hệt, không có .json)
Ô "Phiên B": gõ chunk_256_top_k_6
Bấm "🔍 So sánh"

Link web: https://ai-research-assistant-with-rag-5zbwnhwqirbkmdui4ejijh.streamlit.app/
