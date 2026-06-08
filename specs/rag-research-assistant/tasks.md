# Kế Hoạch Công Việc (Tasks): AI Research Assistant with RAG

## 0. Mục Đích Tài Liệu

Tài liệu này chia nhỏ [requirements.md](requirements.md) và [design.md](design.md) thành các **sprint** và **task** cụ thể, tổ chức theo mô hình **Agile/Scrum**, để 3 thành viên trong vai trò **AI Engineer (Data / Pipeline / Model)** có thể làm việc song song một cách chuyên nghiệp.

Nguyên tắc tổ chức:

1. **Demo-driven**: mỗi sprint kết thúc bằng một tính năng *xem được, bấm được* (ưu tiên hoàn thiện UI/UX — Streamlit dashboard và notebook — trước hoặc song song với phần lõi), để cả team thấy kết quả ngay sau Sprint Review thay vì chỉ có code chạy ngầm.
2. **Hai loại task** (xem mục 2.3) để mỗi kỹ sư biết rõ: việc nào mình có thể làm ngay trong vai trò của mình, việc nào phải phối hợp/chờ vai trò khác.
3. **Bám sát design.md**: mọi task đều trỏ về đúng class, method signature, function, hoặc requirement đã được đặc tả — không tự ý đổi tên/API.

---

## 1. Vai Trò & Ký Hiệu

| Ký hiệu | Vai trò | Phạm vi sở hữu chính (theo design.md) |
|---|---|---|
| `DE` | AI Engineer — Data | `src/data/` (loader, chunker, preprocessor), `src/embeddings/vector_store.py`, `notebooks/data_engineer/` |
| `PE` | AI Engineer — Pipeline | `src/retrieval/`, `src/generation/` (trừ phần model), `src/pipeline/`, `app/`, `notebooks/pipeline_engineer/` |
| `ME` | AI Engineer — Model | `src/embeddings/embedding_model.py`, `src/generation/llm_client.py`, `config/`, `notebooks/model_engineer/` |
| `SHARED` | Cả nhóm / điều phối chung | Cấu trúc thư mục, data models & interfaces dùng chung, README, kiểm thử tích hợp |

---

## 2. Quy Ước Chung

### 2.1 Định dạng Task ID

`S<số sprint>-<vai trò>-<số thứ tự>`, ví dụ `S2-DE-03` = Sprint 2, AI Engineer Data, task thứ 3.

### 2.2 Sprint Backlog Board

Mỗi sprint dùng board 4 cột: `To Do → In Progress → Review/Demo Prep → Done`. Task chỉ chuyển sang `Done` khi đáp ứng **Definition of Done** (mục 2.4) *và* đã được trình diễn trong Sprint Review.

### 2.3 Hai Loại Task

| Loại | Ký hiệu | Định nghĩa | Cách làm việc |
|---|---|---|---|
| **Loại 1 — Độc lập theo vai trò** | 🔹 Độc lập | Task nằm hoàn toàn trong chuyên môn của một vai trò, không cần chờ output từ người khác để bắt đầu. | Người phụ trách chủ động lấy task ngay khi sprint bắt đầu (hoặc khi Definition of Ready của task đó thoả). |
| **Loại 2 — Phụ thuộc liên vai trò/nội bộ** | 🔗 Phụ thuộc | Task cần một hoặc nhiều task khác (có thể của vai trò khác, ví dụ: DE chờ output của ME hoặc PE) hoàn thành trước, vì input/API của nó dựa trên class hoặc dữ liệu do task kia tạo ra. | Người phụ trách task phụ thuộc nên: (a) thống nhất trước **interface/contract** (theo `BaseLoader/BaseChunker/BaseEmbeddingModel/BaseVectorStore/BaseLLMClient` trong design.md §2.2) với người phụ trách task nguồn ngay từ đầu sprint, (b) dùng stub/mock để code song song, (c) tích hợp thật khi task nguồn chuyển `Done`. |

> Cột **Phụ thuộc** trong các bảng task ở mục 4 liệt kê chính xác Task ID phải hoàn thành trước.

### 2.4 Definition of Done (DoD) — áp dụng cho mọi task code

- [ ] Tên class/method/field khớp chính xác với signature trong design.md (không tự đổi tên).
- [ ] Thoả mãn các Acceptance Criteria liên quan trong requirements.md (đối chiếu theo cột "Tham chiếu").
- [ ] Nếu task ánh xạ tới một **Correctness Property** (design.md Phần 3): có ít nhất 1 test (unit hoặc property-based) kiểm chứng property đó.
- [ ] Không có exception chưa xử lý khi chạy qua golden path lẫn ít nhất 1 edge case (input rỗng/không hợp lệ/server không khả dụng...).
- [ ] Đã demo được trực tiếp (qua notebook chạy từ đầu-đến-cuối, hoặc qua dashboard) trong Sprint Review.
- [ ] Code đã được ít nhất 1 thành viên khác review (PR/code review chéo giữa 3 vai trò).

### 2.5 Story Points

Ước lượng theo thang Fibonacci rút gọn: `1` (vài giờ) · `2` (nửa ngày) · `3` (1 ngày) · `5` (2-3 ngày) · `8` (gần hết sprint, nên cân nhắc tách nhỏ).

### 2.6 Nhịp Sprint (Scrum Ceremonies)

| Nghi thức | Tần suất | Nội dung |
|---|---|---|
| Sprint Planning | Đầu mỗi sprint | Chọn task từ backlog, thống nhất *contract* cho các task 🔗 Phụ thuộc liên vai trò trước khi code |
| Daily Standup | Hàng ngày (15') | Hôm qua làm gì / hôm nay làm gì / đang bị block bởi task nào (đặc biệt task 🔗) |
| Sprint Review & Demo | Cuối sprint | Trình diễn **"Kết quả Demo"** của sprint (mục 4) trên dashboard/notebook thật, không dùng slide |
| Retrospective | Sau Review | Rút kinh nghiệm phối hợp giữa 3 vai trò — đặc biệt các điểm nghẽn ở task phụ thuộc |

---

## 3. Lộ Trình Sprint (Roadmap)

| Sprint | Chủ đề | Mục tiêu chính | Kết quả Demo cuối sprint |
|---|---|---|---|
| **0** | Khởi Tạo & Nền Tảng | Dựng khung thư mục, data models, interfaces, config dùng chung | `import src` chạy được; cấu trúc thư mục khớp design.md §1.2 |
| **1** | Khung Giao Diện & Luồng Demo (mock) | Dashboard 4 trang điều hướng được + RAGPipeline khung (stub) + DocumentLoader thật + kiểm tra OLLAMA | Bấm chạy dashboard, đi qua 4 trang, đổi cấu hình sidebar, "upload" thấy phản hồi; notebook 01_ollama_setup chạy được |
| **2** | Indexing Thật (Load→Chunk→Embed→Store) | Pipeline index thật, lưu vào ChromaDB | Upload tài liệu thật trên dashboard → thấy số chunk thật được tạo & lưu trữ |
| **3** | Truy Vấn Thật (Retrieval→Prompt→Generation) | Hỏi-đáp end-to-end với LLM cục bộ | Đặt câu hỏi trên Chat Interface → nhận câu trả lời thật + xem nguồn trên Retrieval Debug |
| **4** | Theo Dõi Thực Nghiệm & Notebook Nâng Cao | ExperimentTracker, trang Experiment Log, các notebook so sánh/trực quan hoá | Xem lịch sử thực nghiệm thật trên dashboard; demo so sánh 2 chiến lược/2 model qua notebook |
| **5** | Kiểm Thử, Cấu Hình & Hoàn Thiện | Test suite (unit + property-based, coverage ≥70%), AppConfig/.env, polish | Chạy `pytest tests/ -v` xanh toàn bộ + demo trọn vẹn hệ thống từ đầu đến cuối |

---

## 4. Chi Tiết Từng Sprint

### Sprint 0 — Khởi Tạo & Nền Tảng

**Mục tiêu sprint:** Dựng "bộ khung" dùng chung để cả 3 vai trò có thể bắt đầu code song song ngay từ Sprint 1 mà không giẫm chân nhau.

**Kết quả Demo cuối sprint:** Repo có đủ cấu trúc thư mục theo design.md §1.2, `requirements.txt` cài được, `from src.data.loader import Document` (và các data model khác) import thành công, `.env.example` + `config/models.yaml` tồn tại.

| Task ID | Vai trò | Mô tả | Loại | Phụ thuộc | Điểm | Tham chiếu |
|---|---|---|---|---|---|---|
| S0-PE-01 | PE | Khởi tạo cấu trúc thư mục đầy đủ (`notebooks/`, `src/`, `app/`, `data/`, `experiments/`, `config/`, `tests/`), `requirements.txt`, cập nhật `README.md`, `.gitignore` | 🔹 Độc lập | — | 3 | design.md §1.2, §2.8 |
| S0-DE-01 | DE | Định nghĩa data models & enums dùng chung: `Document`, `Chunk`, `EmbeddingVector`, `ScoredChunk`, `RAGResponse`, `IndexingResult`, `ExperimentLog`, `ChunkStrategy`, `DocumentType` | 🔹 Độc lập | — | 3 | design.md §2.1 |
| S0-PE-02 | PE | Định nghĩa core interfaces (ABC): `BaseLoader`, `BaseChunker`, `BaseEmbeddingModel`, `BaseVectorStore`, `BaseLLMClient` kèm docstring pre/postcondition | 🔗 Phụ thuộc | S0-DE-01 | 3 | design.md §2.2 |
| S0-ME-01 | ME | Khởi tạo khung cấu hình: `OllamaConfig`, `ChromaConfig`, `ChunkerConfig`, `AppConfig` (chưa cần `from_env()` đầy đủ — để Sprint 5), file `.env.example`, `config/models.yaml` | 🔹 Độc lập | — | 3 | design.md §2.4, Yêu cầu 11.2, 11.4 |

**DoD riêng của sprint:** `pytest --collect-only` không lỗi import; cả 3 thành viên đã thống nhất *contract* của 5 interface trong S0-PE-02 trước khi đóng sprint (vì toàn bộ Sprint 1-3 phụ thuộc vào đây).

---

### Sprint 1 — Khung Giao Diện & Luồng Demo (Mocked End-to-End)

**Mục tiêu sprint:** Ưu tiên dựng UI/UX trước — một dashboard Streamlit *có thể bấm được* (dù dữ liệu còn giả lập), song song với việc hoàn thiện các thành phần thật đầu tiên (DocumentLoader, kiểm tra kết nối OLLAMA), để cả team nhìn thấy hình hài sản phẩm ngay từ tuần đầu.

**Kết quả Demo cuối sprint:** Mở `streamlit run app/main.py`, điều hướng qua đủ 4 trang, chỉnh sidebar (LLM model / embedding model / chunk size / top-k) và thấy cấu hình được lưu; trên trang Document Upload, "tải" một file lên và thấy `IndexingResult` (giả lập) hiển thị; chạy `notebooks/model_engineer/01_ollama_setup.ipynb` thấy `is_available()` trả về đúng trạng thái server.

| Task ID | Vai trò | Mô tả | Loại | Phụ thuộc | Điểm | Tham chiếu |
|---|---|---|---|---|---|---|
| S1-PE-01 | PE | Dựng `app/main.py`: 4 trang điều hướng (Document Upload, Chat Interface, Retrieval Debug, Experiment Log) + sidebar cấu hình lưu vào `st.session_state["config"]` | 🔹 Độc lập | — | 5 | design.md §2.6, Yêu cầu 10.1, 10.2 |
| S1-DE-01 | DE | Triển khai `DocumentLoader` thật cho `.txt` và `.md`: `load()`, `supports()`, `_get_extension()`, `_generate_doc_id()`, `load_directory()` | 🔹 Độc lập | S0-PE-02 | 5 | design.md §2.3 (DocumentLoader), Yêu cầu 1.1, 1.2, 1.5, Property 12 |
| S1-DE-02 | DE | Bổ sung loader PDF (PyMuPDF) + xử lý lỗi: định dạng không hỗ trợ (`.docx`...) và file không tồn tại trả lỗi mô tả rõ nguyên nhân | 🔗 Phụ thuộc | S1-DE-01 | 3 | Yêu cầu 1.3, 1.4 |
| S1-ME-01 | ME | Triển khai `OllamaClient.is_available()` và `list_models()` (không bao giờ ném exception ở `is_available`) | 🔹 Độc lập | S0-PE-02 | 3 | design.md §2.3 (OllamaClient), Yêu cầu 5.2, 5.4 |
| S1-ME-02 | ME | Notebook `model_engineer/01_ollama_setup.ipynb`: cài đặt, pull model, kiểm tra `is_available()`/`list_models()`, chạy hết cell không lỗi | 🔗 Phụ thuộc | S1-ME-01 | 3 | Yêu cầu 9.1, 9.4 |
| S1-PE-02 | PE | Dựng khung `RAGPipeline` (constructor đúng signature design.md §2.3 + `index_document()`/`query()` dạng stub trả `IndexingResult`/`RAGResponse` giả lập) để dashboard có thể gọi được ngay | 🔗 Phụ thuộc | S0-DE-01, S0-PE-02 | 5 | design.md §2.3 (RAGPipeline) |
| S1-PE-03 | PE | Trang "Document Upload": form upload gọi `pipeline.index_document()` (đang là stub) và hiển thị kết quả (số chunk, trạng thái) | 🔗 Phụ thuộc | S1-PE-01, S1-PE-02, S1-DE-01 | 3 | Yêu cầu 10.3 |

> **Ví dụ minh hoạ "task phụ thuộc liên vai trò"**: S1-PE-03 (của Pipeline Engineer) không thể demo đầy đủ nếu S1-DE-01 (DocumentLoader thật của Data Engineer) chưa xong — vì input thật để upload cần loader thật. PE có thể bắt đầu code UI với dữ liệu giả, nhưng phải tích hợp lại khi DE hoàn thành.

---

### Sprint 2 — Indexing Thật: Load → Chunk → Embed → Store

**Mục tiêu sprint:** Thay thế phần "giả lập" của `index_document()` bằng luồng thật theo pseudocode design.md §2.7, để upload tài liệu thật trên dashboard ra số chunk thật và lưu vào ChromaDB.

**Kết quả Demo cuối sprint:** Upload một file PDF/TXT/MD thật trên trang Document Upload → thấy số chunk thật + trạng thái thành công, dữ liệu tồn tại trong ChromaDB (kiểm chứng qua `get_collection_stats()`); chạy notebook `02_text_chunking.ipynb` so sánh trực quan ≥ 2 chiến lược chunking.

| Task ID | Vai trò | Mô tả | Loại | Phụ thuộc | Điểm | Tham chiếu |
|---|---|---|---|---|---|---|
| S2-DE-01 | DE | Triển khai `TextChunker.chunk_by_fixed_size()` với `chunk_size`/`chunk_overlap`, `_create_chunk()` | 🔹 Độc lập | S0-DE-01 | 3 | design.md §2.3 (TextChunker), Yêu cầu 2.1, 2.6 |
| S2-DE-02 | DE | Triển khai `chunk_by_recursive()` và `chunk_by_semantic()` + `chunk()` điều phối theo `strategy`; đảm bảo bao phủ toàn bộ `document.content` và `chunk.doc_id == document.doc_id` | 🔗 Phụ thuộc | S2-DE-01 | 5 | Yêu cầu 2.2-2.5, Property 1, 2 |
| S2-ME-01 | ME | Triển khai `OllamaEmbeddingModel.embed_text()`, `_call_ollama_api()`, property `dimension` (lazy-init); xử lý lỗi khi OLLAMA server không khả dụng | 🔹 Độc lập | S0-PE-02, S0-ME-01 | 5 | design.md §2.3 (OllamaEmbeddingModel), Yêu cầu 3.1, 3.2, 3.5, Property 3, 4 |
| S2-ME-02 | ME | Triển khai `embed_batch()` đảm bảo `embed_batch(texts)[i] == embed_text(texts[i])` | 🔗 Phụ thuộc | S2-ME-01 | 2 | Yêu cầu 3.3, 3.4, Property 5 |
| S2-DE-03 | DE | Triển khai `ChromaVectorStore.add()`, `_init_client()` (in-memory & persistent), `get_collection_stats()`, `delete_collection()` | 🔗 Phụ thuộc | S2-ME-01 | 5 | design.md §2.3 (ChromaVectorStore), Yêu cầu 4.1, 4.5, 4.6, 4.7 |
| S2-PE-01 | PE | Thay stub bằng triển khai thật `RAGPipeline.index_document()` + `index_directory()` đúng pseudocode §2.7 (load→chunk→embed→store, loop invariant `len(vectors)==i`) | 🔗 Phụ thuộc | S2-DE-01, S2-DE-02, S2-ME-01, S2-ME-02, S2-DE-03 | 5 | Yêu cầu 7.1, 7.5, 7.6 |
| S2-PE-02 | PE | Cập nhật trang "Document Upload" để dùng `index_document()` thật, hiển thị số chunk & trạng thái thật từ `IndexingResult` | 🔗 Phụ thuộc | S2-PE-01 | 2 | Yêu cầu 10.3 |
| S2-DE-04 | DE | Notebook `data_engineer/01_document_loading.ipynb` | 🔗 Phụ thuộc | S1-DE-02 | 2 | Yêu cầu 9.1, 9.2 |
| S2-DE-05 | DE | Notebook `data_engineer/02_text_chunking.ipynb`: so sánh ≥ 2 chiến lược chunking trên cùng tài liệu (theo cấu trúc cell mẫu design.md §2.5) | 🔗 Phụ thuộc | S2-DE-02 | 3 | Yêu cầu 9.5 |

> **Ví dụ minh hoạ task phụ thuộc khác vai trò:** S2-DE-03 (`ChromaVectorStore.add`, của Data Engineer) phụ thuộc vào S2-ME-01 (`embed_text`, của Model Engineer) vì cần vector thật để kiểm thử việc lưu trữ — đúng tình huống "DE phụ thuộc vào ME" mà nhóm cần lường trước khi lập kế hoạch song song.

---

### Sprint 3 — Truy Vấn Thật: Retrieval → Prompt → Generation

**Mục tiêu sprint:** Hoàn thiện luồng `query()` thật theo pseudocode §2.7 và đưa lên dashboard — đây là tính năng "lõi" nhất của sản phẩm, nên ưu tiên demo được Chat Interface hoạt động thật với LLM cục bộ.

**Kết quả Demo cuối sprint:** Gõ một câu hỏi trên Chat Interface → nhận câu trả lời thật từ LLM cục bộ (qua OLLAMA) kèm danh sách nguồn tham chiếu; chuyển sang trang Retrieval Debug thấy danh sách chunk truy xuất với điểm similarity và vị trí trong tài liệu gốc.

| Task ID | Vai trò | Mô tả | Loại | Phụ thuộc | Điểm | Tham chiếu |
|---|---|---|---|---|---|---|
| S3-DE-01 | DE | Triển khai thật `ChromaVectorStore.similarity_search()`: trả về `List[ScoredChunk]` sắp xếp giảm dần theo `score`, `score ∈ [0.0, 1.0]`, `len(kết quả) <= k` | 🔹 Độc lập | S2-DE-03 | 5 | Yêu cầu 4.2, 4.3, 4.4, Property 6, 7, 8 |
| S3-PE-01 | PE | Triển khai `PromptBuilder`: `DEFAULT_SYSTEM_PROMPT`, `build()`, `format_context()`, `set_system_prompt()` | 🔹 Độc lập | S0-DE-01 | 3 | design.md §2.3 (PromptBuilder), Yêu cầu 6.1-6.4, Property 9 |
| S3-ME-01 | ME | Triển khai `OllamaClient.generate()` + `_build_request_body()`; trả lỗi kết nối mô tả rõ URL khi server không khả dụng | 🔹 Độc lập | S1-ME-01 | 5 | Yêu cầu 5.1, 5.5 |
| S3-ME-02 | ME | Triển khai `OllamaClient.generate_stream()` (yield từng token tuần tự) | 🔗 Phụ thuộc | S3-ME-01 | 3 | Yêu cầu 5.3 |
| S3-PE-02 | PE | Triển khai `Retriever` (`src/retrieval/retriever.py`) bọc `vector_store.similarity_search()`, hỗ trợ cấu hình `top_k` | 🔗 Phụ thuộc | S3-DE-01 | 2 | design.md §1.2 (retrieval module) |
| S3-PE-03 | PE | Thay stub bằng triển khai thật `RAGPipeline.query()` đúng pseudocode §2.7 (embed câu hỏi → retrieve → build prompt → generate → đo `latency_ms`) | 🔗 Phụ thuộc | S3-PE-02, S3-PE-01, S3-ME-01 | 5 | Yêu cầu 7.3, 7.4, Property 10 |
| S3-PE-04 | PE | Component `app/components/chat_widget.py` + trang "Chat Interface": hiển thị câu trả lời và danh sách nguồn tham chiếu | 🔗 Phụ thuộc | S3-PE-03 | 3 | Yêu cầu 10.4 |
| S3-PE-05 | PE | Trang "Retrieval Debug": hiển thị chunk truy xuất kèm điểm similarity và vị trí (`start_index`/`end_index`) trong tài liệu gốc | 🔗 Phụ thuộc | S3-PE-03 | 3 | Yêu cầu 10.5 |
| S3-PE-06 | PE | Notebook `pipeline_engineer/03_prompt_engineering.ipynb` | 🔗 Phụ thuộc | S3-PE-01 | 2 | Yêu cầu 9.3 |
| S3-PE-07 | PE | Notebook `pipeline_engineer/01_rag_pipeline_basics.ipynb` | 🔗 Phụ thuộc | S3-PE-03 | 2 | Yêu cầu 9.3 |
| S3-PE-08 | PE | Notebook `pipeline_engineer/02_retrieval_strategies.ipynb` (so sánh dense/sparse/hybrid) | 🔗 Phụ thuộc | S3-PE-02 | 3 | Yêu cầu 9.3 |
| S3-DE-02 | DE | Notebook `data_engineer/04_vector_db_indexing.ipynb` | 🔗 Phụ thuộc | S3-DE-01 | 2 | Yêu cầu 9.2 |

> **Lưu ý phối hợp:** S3-PE-03 là điểm hội tụ của cả 3 vai trò (cần `similarity_search` thật của DE, `PromptBuilder`/`Retriever` của chính PE, và `generate()` thật của ME). Nên lên lịch tích hợp ("integration day") giữa sprint thay vì dồn vào cuối.

---

### Sprint 4 — Theo Dõi Thực Nghiệm & Notebook Nâng Cao

**Mục tiêu sprint:** Khép vòng "học tập có thể đo lường được" — `ExperimentTracker` ghi/so sánh thực nghiệm thật, hiển thị trên dashboard, và các notebook khám phá còn lại (so sánh model, trực quan hoá embedding, đánh giá pipeline).

**Kết quả Demo cuối sprint:** Trang "Experiment Log" hiển thị danh sách sự kiện indexing/query thật theo thời gian; demo `compare_sessions()` so sánh 2 phiên làm việc; trình chiếu biểu đồ PCA/t-SNE không gian embedding từ notebook `03_embedding_exploration.ipynb`; notebook `04_pipeline_evaluation.ipynb` in ra hit rate & latency trung bình.

| Task ID | Vai trò | Mô tả | Loại | Phụ thuộc | Điểm | Tham chiếu |
|---|---|---|---|---|---|---|
| S4-PE-01 | PE | Triển khai `ExperimentTracker.log_indexing()`, `log_query()`, `get_summary()` | 🔹 Độc lập | S0-DE-01 | 3 | design.md §2.3 (ExperimentTracker), Yêu cầu 8.1, 8.2, 8.6 |
| S4-PE-02 | PE | Triển khai `save_session()`/`load_session()` đảm bảo round-trip dữ liệu | 🔗 Phụ thuộc | S4-PE-01 | 3 | Yêu cầu 8.3, 8.4, Property 11 |
| S4-PE-03 | PE | Triển khai `compare_sessions()` trả về dictionary metrics so sánh | 🔗 Phụ thuộc | S4-PE-02 | 2 | Yêu cầu 8.5 |
| S4-PE-04 | PE | Tích hợp `ExperimentTracker` vào `RAGPipeline.index_document()`/`query()` (gọi `log_indexing`/`log_query` không làm gián đoạn luồng chính) | 🔗 Phụ thuộc | S2-PE-01, S3-PE-03, S4-PE-01 | 3 | Yêu cầu 8.1, 8.2, design.md §2.7 |
| S4-PE-05 | PE | Component `app/components/metrics_widget.py` + trang "Experiment Log": liệt kê sự kiện theo thứ tự thời gian | 🔗 Phụ thuộc | S4-PE-04 | 3 | Yêu cầu 10.6 |
| S4-DE-01 | DE | Notebook `data_engineer/03_embedding_exploration.ipynb`: trực quan hoá không gian embedding bằng PCA/t-SNE (plotly) | 🔗 Phụ thuộc | S2-ME-02 | 5 | Yêu cầu 9.6 |
| S4-ME-01 | ME | Notebook `model_engineer/02_model_comparison.ipynb`: benchmark các LLM (`llama3`, `mistral`, `gemma`...) | 🔗 Phụ thuộc | S1-ME-01 | 3 | Yêu cầu 9.4 |
| S4-ME-02 | ME | Notebook `model_engineer/03_embedding_models.ipynb`: so sánh chất lượng các embedding model | 🔗 Phụ thuộc | S2-ME-02 | 3 | Yêu cầu 9.4 |
| S4-ME-03 | ME | Notebook `model_engineer/04_inference_optimization.ipynb`: tối ưu batch/cache/GPU | 🔗 Phụ thuộc | S3-ME-02 | 3 | Yêu cầu 9.4 |
| S4-PE-06 | PE | Notebook `pipeline_engineer/04_pipeline_evaluation.ipynb`: hàm `hit_rate()`, chạy tập câu hỏi đánh giá, in hit rate & latency trung bình | 🔗 Phụ thuộc | S3-PE-03, S4-PE-04 | 3 | Yêu cầu 9.7 |

> **Ví dụ "task cùng vai trò nhưng phụ thuộc theo trình tự":** S4-PE-02 → S4-PE-03 → S4-PE-05 là một chuỗi phụ thuộc nội bộ của Pipeline Engineer — không thể làm `compare_sessions` trước khi `save/load_session` xong, và không thể demo trang Experiment Log nếu chưa có dữ liệu thật để hiển thị.

---

### Sprint 5 — Kiểm Thử, Cấu Hình & Hoàn Thiện

**Mục tiêu sprint:** Khoá chất lượng (test suite + coverage ≥ 70%), hoàn thiện cấu hình qua môi trường (`AppConfig.from_env()`), và polish toàn bộ trải nghiệm trước khi "release" phiên bản học tập đầu tiên.

**Kết quả Demo cuối sprint:** Chạy `pytest tests/ -v` toàn bộ pass + báo cáo coverage ≥ 70% cho `src/`; trình diễn lại toàn bộ hành trình end-to-end (đổi `.env` → khởi động lại → upload → hỏi đáp → xem debug → so sánh thực nghiệm) không gặp lỗi.

| Task ID | Vai trò | Mô tả | Loại | Phụ thuộc | Điểm | Tham chiếu |
|---|---|---|---|---|---|---|
| S5-ME-01 | ME | Hoàn thiện `AppConfig.from_env()` (đọc `.env`/biến môi trường, dùng giá trị mặc định hợp lệ khi thiếu), hoàn thiện `.env.example` & `config/models.yaml` | 🔹 Độc lập | S0-ME-01 | 3 | Yêu cầu 11.1, 11.2, 11.3, 11.4 |
| S5-DE-01 | DE | Unit test + property-based test (Hypothesis) cho `TextChunker`: `test_chunk_coverage`, `test_chunk_overlap`, property "bao phủ nội dung" & "doc_id đúng" | 🔹 Độc lập | S2-DE-02 | 5 | Yêu cầu 12.1, 12.3, Property 1, 2 |
| S5-DE-02 | DE | Unit test cho `DocumentLoader`: `test_load_pdf`, `test_unsupported_format`, kiểm chứng `doc_id` duy nhất giữa nhiều file | 🔗 Phụ thuộc | S1-DE-02 | 3 | Yêu cầu 12.1, Property 12 |
| S5-DE-03 | DE | Unit + property test cho `ChromaVectorStore`: `test_add_and_retrieve`, property "score giảm dần", "score ∈ [0,1]", "không vượt quá k" | 🔗 Phụ thuộc | S3-DE-01 | 5 | Yêu cầu 12.1, 12.3, Property 6, 7, 8 |
| S5-ME-02 | ME | Unit + property test cho `OllamaEmbeddingModel`: tính nhất quán chiều vector, tính tất định, `embed_batch == embed_text` từng phần tử; test `is_available` không ném exception | 🔗 Phụ thuộc | S2-ME-02, S3-ME-01 | 5 | Yêu cầu 12.1, Property 3, 4, 5 |
| S5-PE-01 | PE | Test cho `RAGPipeline`: `test_index_then_query` (index xong rồi query → answer không rỗng và chứa context từ đúng tài liệu — Property 10) | 🔗 Phụ thuộc | S3-PE-03 | 5 | Yêu cầu 12.1, Property 10 |
| S5-PE-02 | PE | Test round-trip cho `ExperimentTracker`: `save_session` → `load_session` khôi phục đúng danh sách `ExperimentLog` ban đầu | 🔗 Phụ thuộc | S4-PE-02 | 3 | Yêu cầu 12.1, Property 11 |
| S5-PE-03 | PE | Đo coverage tổng thể (`pytest --cov=src`), bổ sung test còn thiếu để đạt ≥ 70%, đảm bảo `pytest tests/ -v` không lỗi setup môi trường | 🔗 Phụ thuộc | S5-DE-01, S5-DE-02, S5-DE-03, S5-ME-02, S5-PE-01, S5-PE-02 | 5 | Yêu cầu 12.2, 12.4 |
| S5-SHARED-01 | SHARED | Hoàn thiện `README.md` (hướng dẫn cài đặt & chạy), kiểm thử thủ công golden path + edge case trên dashboard, chuẩn bị kịch bản demo cuối dự án | 🔗 Phụ thuộc | toàn bộ Sprint 1-4 | 3 | — |

---

## 5. Ma Trận Phụ Thuộc Liên Vai Trò (Tổng Hợp)

Bảng dưới chỉ liệt kê các cặp **phụ thuộc khác vai trò** (loại 2 "đáng chú ý nhất" mà user đã nêu) để Scrum Master/PO theo dõi rủi ro nghẽn:

| Task phụ thuộc (chờ) | Vai trò | ⟶ phụ thuộc vào | Vai trò nguồn | Vì sao |
|---|---|---|---|---|
| S1-PE-02/03 | PE | S0-DE-01, S1-DE-01 | DE | RAGPipeline & trang Upload cần `Document`/`DocumentLoader` thật |
| S2-DE-03 | DE | S2-ME-01 | ME | `ChromaVectorStore.add` cần vector thật từ `embed_text` để lưu/kiểm thử |
| S2-PE-01 | PE | S2-DE-01/02/03, S2-ME-01/02 | DE + ME | `index_document()` điều phối load→chunk→embed→store, cần cả 3 thành phần thật |
| S3-PE-02/03 | PE | S3-DE-01 | DE | `Retriever`/`RAGPipeline.query` cần `similarity_search` thật để truy xuất context |
| S3-PE-03 | PE | S3-ME-01 | ME | `query()` cần `OllamaClient.generate()` thật để sinh câu trả lời |
| S4-PE-04 | PE | S2-PE-01, S3-PE-03 | PE (nội bộ, xuyên sprint) | Phải có cả index & query thật trước khi gắn log thực nghiệm |
| S4-DE-01, S4-ME-02 | DE, ME | S2-ME-02 | ME | Cả notebook trực quan hoá embedding (DE) lẫn so sánh embedding model (ME) đều cần `embed_batch` thật |
| S5-PE-03 | PE | tất cả task test S5 (DE/ME/PE) | DE + ME + PE | Coverage tổng thể chỉ tính được khi mọi module đã có test |

**Khuyến nghị vận hành:** Ở đầu mỗi sprint có chứa các cặp trong bảng trên, hai vai trò liên quan nên dành 15-30 phút thống nhất *interface thật sự sẽ trông như thế nào* (input/output, tên field, kiểu dữ liệu) **trước khi** bắt tay code song song bằng stub — đây là cách một team chuyên nghiệp tránh "tích hợp địa ngục" vào cuối sprint.
