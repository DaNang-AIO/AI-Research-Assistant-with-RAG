# Tài Liệu Yêu Cầu: AI Research Assistant with RAG

## Giới Thiệu

AI Research Assistant with RAG là một hệ thống nghiên cứu và học tập hướng đến việc khám phá từng thành phần của kiến trúc RAG (Retrieval-Augmented Generation) theo cách tương tác và có thể tái lặp. Hệ thống phục vụ người học Python ở trình độ cơ bản đến trung cấp muốn hiểu sâu về cách hoạt động của RAG thông qua thực nghiệm thực tế.

Tài liệu này mô tả các yêu cầu chức năng và phi chức năng được rút ra từ tài liệu thiết kế, tổ chức theo 6 nhóm chức năng chính: tải tài liệu, chia nhỏ văn bản, embedding, vector store, pipeline RAG, và giao diện người dùng. Mỗi nhóm đi kèm với user story và acceptance criteria chuẩn EARS/INCOSE.

---

## Bảng Thuật Ngữ

- **DocumentLoader**: Thành phần tải và phân tích tài liệu từ file vào cấu trúc dữ liệu Document
- **TextChunker**: Thành phần chia nhỏ Document thành danh sách Chunk theo nhiều chiến lược
- **Chunk**: Một đoạn văn bản nhỏ được tách từ Document, dùng làm đơn vị lưu trữ và truy xuất
- **OllamaEmbeddingModel**: Thành phần tạo embedding vector bằng cách gọi OLLAMA embedding API
- **ChromaVectorStore**: Thành phần lưu trữ và tìm kiếm vector sử dụng ChromaDB
- **OllamaClient**: Thành phần giao tiếp với OLLAMA server để sinh văn bản từ prompt
- **PromptBuilder**: Thành phần xây dựng prompt hoàn chỉnh từ câu hỏi và danh sách context chunks
- **RAGPipeline**: Thành phần điều phối toàn bộ luồng RAG từ indexing đến query
- **ExperimentTracker**: Thành phần ghi lại, lưu trữ và so sánh các phiên thực nghiệm
- **IndexingResult**: Cấu trúc dữ liệu chứa kết quả của quá trình index một tài liệu
- **RAGResponse**: Cấu trúc dữ liệu chứa câu hỏi, câu trả lời, context và metadata của một lần query
- **ScoredChunk**: Chunk kèm điểm similarity và thứ hạng từ kết quả retrieval
- **ExperimentLog**: Bản ghi một sự kiện thực nghiệm gồm tham số, kết quả và timestamp
- **OLLAMA**: Nền tảng chạy LLM cục bộ không cần kết nối internet
- **ChunkStrategy**: Chiến lược chia nhỏ văn bản: FIXED_SIZE, RECURSIVE, hoặc SEMANTIC
- **Notebook**: Jupyter Notebook dùng cho học tập và thực nghiệm tương tác
- **Dashboard**: Giao diện web Streamlit để sử dụng và quan sát hệ thống RAG

---

## Yêu Cầu

### Yêu Cầu 1: Tải Tài Liệu

**User Story:** Là một Data Engineer, tôi muốn tải tài liệu từ nhiều định dạng file khác nhau vào hệ thống, để có thể xử lý và index nội dung tài liệu đó.

#### Acceptance Criteria

1. WHEN DocumentLoader nhận đường dẫn đến file PDF, TXT, Markdown, hoặc HTML hợp lệ, THE DocumentLoader SHALL tải và trả về một đối tượng Document có trường `content` không rỗng
2. WHEN DocumentLoader tải thành công bất kỳ tài liệu nào, THE DocumentLoader SHALL gán cho Document một `doc_id` duy nhất và không trùng lặp với các Document khác trong cùng phiên làm việc
3. IF đường dẫn file trỏ đến định dạng không được hỗ trợ (ví dụ: .docx, .xlsx), THEN THE DocumentLoader SHALL trả về lỗi mô tả rõ định dạng không hỗ trợ
4. IF đường dẫn file không tồn tại hoặc không thể đọc được, THEN THE DocumentLoader SHALL trả về lỗi mô tả nguyên nhân cụ thể
5. THE DocumentLoader SHALL hỗ trợ tải toàn bộ tài liệu hợp lệ trong một thư mục thông qua phương thức `load_directory()`

---

### Yêu Cầu 2: Chia Nhỏ Văn Bản (Text Chunking)

**User Story:** Là một Data Engineer, tôi muốn chia nhỏ tài liệu thành các đoạn văn bản (chunk) bằng nhiều chiến lược khác nhau, để tìm ra cách tổ chức nội dung phù hợp nhất với bài toán RAG.

#### Acceptance Criteria

1. WHEN TextChunker được gọi với chiến lược FIXED_SIZE, THE TextChunker SHALL tạo ra các chunk có độ dài không vượt quá `chunk_size` ký tự đã cấu hình
2. WHEN TextChunker được gọi với bất kỳ chiến lược nào, THE TextChunker SHALL đảm bảo toàn bộ nội dung của Document gốc được bao phủ bởi tập hợp các chunk tạo ra
3. THE TextChunker SHALL gán `doc_id` của Document gốc cho tất cả các Chunk được tạo ra từ Document đó
4. THE TextChunker SHALL hỗ trợ ba chiến lược chunking: FIXED_SIZE (cắt theo số ký tự cố định), RECURSIVE (đệ quy theo dấu phân cách), và SEMANTIC (theo ranh giới câu)
5. WHEN TextChunker xử lý Document có `content` rỗng, THE TextChunker SHALL trả về danh sách rỗng mà không gây ra lỗi ngoại lệ không mong muốn
6. WHERE tham số `chunk_overlap` được cấu hình lớn hơn 0, THE TextChunker SHALL tạo ra các chunk liên tiếp có phần nội dung chồng lấp với nhau đúng bằng số ký tự `chunk_overlap`

---

### Yêu Cầu 3: Tạo Embedding Vector

**User Story:** Là một Model Engineer, tôi muốn tạo embedding vector cho các đoạn văn bản sử dụng mô hình embedding cục bộ qua OLLAMA, để biểu diễn ngữ nghĩa nội dung dưới dạng vector số học.

#### Acceptance Criteria

1. WHEN OllamaEmbeddingModel nhận một đoạn văn bản không rỗng, THE OllamaEmbeddingModel SHALL trả về một vector số thực có số chiều bằng đúng giá trị `dimension` của model
2. WHEN OllamaEmbeddingModel được gọi hai lần với cùng một đoạn văn bản, THE OllamaEmbeddingModel SHALL trả về hai vector giống hệt nhau (hành vi tất định)
3. WHEN OllamaEmbeddingModel nhận danh sách nhiều đoạn văn bản qua `embed_batch()`, THE OllamaEmbeddingModel SHALL trả về danh sách vector có số lượng bằng đúng số đoạn văn bản đầu vào
4. WHEN OllamaEmbeddingModel xử lý batch, THE OllamaEmbeddingModel SHALL đảm bảo vector tại vị trí `i` trong kết quả tương đương với gọi `embed_text()` riêng lẻ cho văn bản thứ `i`
5. IF OLLAMA server không khả dụng, THEN THE OllamaEmbeddingModel SHALL trả về lỗi kết nối mô tả rõ địa chỉ server và nguyên nhân

---

### Yêu Cầu 4: Lưu Trữ và Tìm Kiếm Vector

**User Story:** Là một Data Engineer, tôi muốn lưu trữ các chunk và vector embedding vào cơ sở dữ liệu vector, và tìm kiếm các chunk ngữ nghĩa gần nhất với một câu hỏi, để cung cấp context liên quan cho hệ thống sinh câu trả lời.

#### Acceptance Criteria

1. WHEN ChromaVectorStore nhận danh sách chunks và vectors có cùng kích thước qua `add()`, THE ChromaVectorStore SHALL lưu trữ tất cả chunks và trả về `True` khi thành công
2. WHEN ChromaVectorStore thực hiện `similarity_search()` sau khi đã thêm ít nhất một chunk, THE ChromaVectorStore SHALL trả về danh sách ScoredChunk có số lượng không vượt quá tham số `k`
3. THE ChromaVectorStore SHALL sắp xếp kết quả `similarity_search()` theo chiều giảm dần của `score` (chunk liên quan nhất đứng đầu)
4. THE ChromaVectorStore SHALL đảm bảo mỗi `ScoredChunk.score` trong kết quả `similarity_search()` có giá trị thuộc khoảng `[0.0, 1.0]`
5. WHERE tham số `persist_dir` được cấu hình, THE ChromaVectorStore SHALL lưu dữ liệu lâu dài vào thư mục đó và khôi phục được sau khi khởi động lại
6. WHERE tham số `persist_dir` không được cấu hình (in-memory mode), THE ChromaVectorStore SHALL hoạt động hoàn toàn trong bộ nhớ, phù hợp cho môi trường Jupyter Notebook
7. WHEN `delete_collection()` được gọi với tên collection hợp lệ, THE ChromaVectorStore SHALL xóa toàn bộ dữ liệu của collection đó

---

### Yêu Cầu 5: Sinh Câu Trả Lời qua OLLAMA

**User Story:** Là một Pipeline Engineer, tôi muốn gọi mô hình LLM cục bộ thông qua OLLAMA để sinh câu trả lời từ prompt, để xây dựng năng lực sinh văn bản trong hệ thống RAG mà không phụ thuộc vào dịch vụ đám mây.

#### Acceptance Criteria

1. WHEN OllamaClient nhận một prompt không rỗng và OLLAMA server đang chạy, THE OllamaClient SHALL trả về chuỗi văn bản câu trả lời không rỗng
2. THE OllamaClient SHALL cung cấp phương thức `is_available()` trả về giá trị boolean mà không ném ra ngoại lệ trong bất kỳ trường hợp nào
3. WHEN `generate_stream()` được gọi, THE OllamaClient SHALL yield từng đoạn token tuần tự cho đến khi hoàn thành câu trả lời
4. THE OllamaClient SHALL cung cấp phương thức `list_models()` trả về danh sách tên các model đã pull về OLLAMA server
5. IF OLLAMA server không khả dụng khi `generate()` được gọi, THEN THE OllamaClient SHALL trả về lỗi kết nối mô tả rõ URL server

---

### Yêu Cầu 6: Xây Dựng Prompt

**User Story:** Là một Pipeline Engineer, tôi muốn xây dựng prompt hoàn chỉnh bằng cách kết hợp system instruction, context chunks và câu hỏi người dùng, và tôi muốn thay đổi được system prompt để thực nghiệm, để tối ưu hóa chất lượng câu trả lời từ LLM.

#### Acceptance Criteria

1. WHEN PromptBuilder nhận một câu hỏi và danh sách ScoredChunk, THE PromptBuilder SHALL tạo ra một prompt string chứa nội dung của câu hỏi và toàn bộ nội dung các context chunks
2. THE PromptBuilder SHALL bao gồm system prompt trong mọi prompt được tạo ra
3. WHEN `set_system_prompt()` được gọi với nội dung mới, THE PromptBuilder SHALL sử dụng system prompt mới cho tất cả các lần gọi `build()` tiếp theo
4. WHEN PromptBuilder nhận danh sách context rỗng, THE PromptBuilder SHALL vẫn tạo ra prompt hợp lệ chứa câu hỏi và system prompt

---

### Yêu Cầu 7: Điều Phối RAG Pipeline

**User Story:** Là một Pipeline Engineer, tôi muốn có một thành phần điều phối thống nhất toàn bộ luồng RAG từ tải tài liệu đến sinh câu trả lời, để có thể sử dụng trong cả Jupyter Notebook và Streamlit Dashboard mà không cần lặp lại code.

#### Acceptance Criteria

1. WHEN RAGPipeline.index_document() được gọi với đường dẫn file hợp lệ, THE RAGPipeline SHALL thực hiện tuần tự các bước: load → chunk → embed → store và trả về IndexingResult với `success=True`
2. WHEN RAGPipeline.index_document() thành công, THE RAGPipeline SHALL đảm bảo Document đó có thể được truy xuất thông qua `query()` với câu hỏi liên quan
3. WHEN RAGPipeline.query() được gọi với câu hỏi không rỗng và vector store có dữ liệu, THE RAGPipeline SHALL trả về RAGResponse với trường `answer` không rỗng
4. THE RAGPipeline SHALL ghi nhận thời gian thực thi trong RAGResponse.latency_ms với giá trị lớn hơn 0
5. WHEN RAGPipeline.index_directory() được gọi với đường dẫn thư mục hợp lệ, THE RAGPipeline SHALL index tất cả tài liệu được hỗ trợ trong thư mục đó và trả về danh sách IndexingResult tương ứng
6. IF DocumentLoader gặp lỗi trong quá trình index_document(), THEN THE RAGPipeline SHALL trả về IndexingResult với `success=False` và mô tả lỗi trong `error_message`

---

### Yêu Cầu 8: Theo Dõi Thực Nghiệm

**User Story:** Là một người học RAG, tôi muốn ghi lại, lưu trữ và so sánh các phiên thực nghiệm với các tham số khác nhau, để theo dõi sự tiến triển trong học tập và tìm ra cấu hình pipeline tối ưu.

#### Acceptance Criteria

1. WHEN ExperimentTracker.log_indexing() được gọi, THE ExperimentTracker SHALL thêm một ExperimentLog vào phiên hiện tại với đầy đủ thông tin: doc_id, chiến lược chunking, chunk_size, số chunks, và latency
2. WHEN ExperimentTracker.log_query() được gọi, THE ExperimentTracker SHALL thêm một ExperimentLog vào phiên hiện tại với đầy đủ thông tin: câu hỏi, top_k, danh sách context, câu trả lời, và latency
3. WHEN ExperimentTracker.save_session() được gọi với tên phiên hợp lệ, THE ExperimentTracker SHALL lưu toàn bộ danh sách ExperimentLog hiện tại ra file JSON và trả về đường dẫn file
4. WHEN ExperimentTracker.load_session() được gọi với tên phiên đã lưu, THE ExperimentTracker SHALL tải lại đúng danh sách ExperimentLog ban đầu (round-trip)
5. WHEN ExperimentTracker.compare_sessions() được gọi với tên hai phiên hợp lệ, THE ExperimentTracker SHALL trả về dictionary chứa các metrics so sánh giữa hai phiên
6. THE ExperimentTracker SHALL cung cấp phương thức `get_summary()` trả về thống kê tổng hợp của phiên hiện tại bao gồm tổng số sự kiện, tổng số lần indexing, và tổng số lần query

---

### Yêu Cầu 9: Jupyter Notebooks Theo Vai Trò

**User Story:** Là một người học Python muốn tìm hiểu RAG, tôi muốn có Jupyter Notebooks được tổ chức theo 3 vai trò kỹ sư AI rõ ràng và có thứ tự, để học từng thành phần RAG một cách có hệ thống và tương tác.

#### Acceptance Criteria

1. THE hệ thống SHALL cung cấp 12 notebooks tổ chức trong 3 thư mục: `notebooks/data_engineer/` (4 notebooks), `notebooks/pipeline_engineer/` (4 notebooks), và `notebooks/model_engineer/` (4 notebooks)
2. WHEN người dùng chạy các notebook trong thư mục `data_engineer/` theo thứ tự từ 01 đến 04, THE Notebook SHALL thực thi tất cả các cell từ đầu đến cuối mà không có lỗi ngoại lệ không được xử lý
3. WHEN người dùng chạy các notebook trong thư mục `pipeline_engineer/` theo thứ tự từ 01 đến 04, THE Notebook SHALL thực thi tất cả các cell từ đầu đến cuối mà không có lỗi ngoại lệ không được xử lý
4. WHEN người dùng chạy các notebook trong thư mục `model_engineer/` theo thứ tự từ 01 đến 04, THE Notebook SHALL thực thi tất cả các cell từ đầu đến cuối mà không có lỗi ngoại lệ không được xử lý
5. THE Notebook `02_text_chunking.ipynb` SHALL cho phép so sánh trực tiếp kết quả của ít nhất hai chiến lược chunking khác nhau với cùng một tài liệu đầu vào
6. THE Notebook `03_embedding_exploration.ipynb` SHALL trực quan hóa không gian vector embedding qua biểu đồ PCA hoặc t-SNE bằng thư viện plotly
7. THE Notebook `04_pipeline_evaluation.ipynb` SHALL tính toán và hiển thị hit rate trung bình và latency trung bình trên tập câu hỏi đánh giá định nghĩa sẵn
8. WHILE thực nghiệm trong bất kỳ notebook nào, THE ExperimentTracker SHALL có thể ghi lại kết quả thực nghiệm vào phiên hiện tại mà không làm gián đoạn luồng notebook

---

### Yêu Cầu 10: Streamlit Research Dashboard

**User Story:** Là một người học RAG, tôi muốn có một giao diện web tương tác để upload tài liệu, đặt câu hỏi, debug kết quả retrieval và xem lịch sử thực nghiệm, để quan sát toàn bộ hệ thống RAG hoạt động mà không cần dùng Jupyter Notebook.

#### Acceptance Criteria

1. WHEN Dashboard khởi động, THE Dashboard SHALL hiển thị đầy đủ 4 trang: Document Upload, Chat Interface, Retrieval Debug, và Experiment Log
2. WHEN người dùng cấu hình LLM model, embedding model, chunk size và top-k trên sidebar, THE Dashboard SHALL áp dụng cấu hình đó cho tất cả các thao tác tiếp theo trong phiên làm việc
3. WHEN người dùng upload file PDF, TXT hoặc Markdown qua trang Document Upload, THE Dashboard SHALL index tài liệu và hiển thị kết quả gồm: số chunks đã tạo và trạng thái thành công hoặc lỗi
4. WHEN người dùng nhập câu hỏi trên trang Chat Interface và nhấn gửi, THE Dashboard SHALL hiển thị câu trả lời từ LLM cùng với danh sách các nguồn tài liệu được tham chiếu
5. WHEN người dùng truy cập trang Retrieval Debug sau khi đặt câu hỏi, THE Dashboard SHALL hiển thị danh sách chunks được truy xuất kèm điểm similarity và vị trí trong tài liệu gốc
6. WHEN người dùng truy cập trang Experiment Log, THE Dashboard SHALL hiển thị danh sách các sự kiện thực nghiệm trong phiên hiện tại theo thứ tự thời gian

---

### Yêu Cầu 11: Cấu Hình và Môi Trường

**User Story:** Là một nhà phát triển, tôi muốn cấu hình hệ thống thông qua biến môi trường và file cấu hình, để có thể dễ dàng thay đổi địa chỉ OLLAMA, model mặc định và các tham số pipeline mà không cần sửa source code.

#### Acceptance Criteria

1. THE AppConfig SHALL cung cấp phương thức `from_env()` tải cấu hình từ file `.env` và biến môi trường hệ thống
2. THE hệ thống SHALL cung cấp file `.env.example` mô tả tất cả các biến môi trường được hỗ trợ
3. WHEN biến môi trường không được định nghĩa, THE AppConfig SHALL sử dụng giá trị mặc định hợp lệ cho tất cả tham số
4. THE hệ thống SHALL cung cấp file `config/models.yaml` liệt kê các model OLLAMA được hỗ trợ và thông tin cấu hình của từng model

---

### Yêu Cầu 12: Kiểm Thử Tự Động

**User Story:** Là một nhà phát triển, tôi muốn có bộ test tự động bao gồm unit tests và property-based tests cho các thành phần cốt lõi, để đảm bảo tính đúng đắn của hệ thống khi thực hiện thay đổi.

#### Acceptance Criteria

1. THE hệ thống SHALL cung cấp test suite trong thư mục `tests/` bao gồm test cho: DocumentLoader, TextChunker, ChromaVectorStore, và RAGPipeline
2. WHEN chạy `pytest tests/ -v`, THE hệ thống SHALL thực thi tất cả test và báo cáo kết quả mà không có lỗi setup môi trường
3. THE test suite SHALL bao gồm property-based tests sử dụng thư viện `hypothesis` để kiểm tra các thuộc tính đúng đắn cho TextChunker và ChromaVectorStore
4. THE test suite SHALL đạt tỷ lệ coverage ít nhất 70% cho các module trong thư mục `src/`
