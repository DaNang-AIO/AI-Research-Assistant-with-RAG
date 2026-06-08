"""Trang Document Upload — tải tài liệu lên và xem kết quả indexing.

Triển khai: S1-PE-03 (Yêu cầu 10.3) — gọi `pipeline.index_document()` (đang là
stub giả lập ở Sprint 1, sẽ thay bằng luồng thật ở S2-PE-02).
"""

import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.pipeline_factory import build_pipeline
from app.components.sidebar import render_sidebar_config

st.set_page_config(page_title="Document Upload", page_icon="📄", layout="wide")

config = render_sidebar_config()
pipeline = build_pipeline(config)

st.title("📄 Document Upload")
st.markdown(
    "Tải lên một tài liệu (`.pdf`, `.txt`, `.md`) để **\"dạy\"** cho hệ thống "
    "RAG biết về nội dung đó — sau bước này, bạn có thể đặt câu hỏi về tài "
    "liệu ở trang **Chat Interface**."
)

with st.expander("💡 \"Index tài liệu\" thực ra là làm những gì? (bấm để xem)"):
    st.markdown(
        """
Khi bạn tải một tài liệu lên, hệ thống thực hiện 4 bước nhỏ phía sau:

1. **Đọc (Load)** — mở file và trích xuất toàn bộ nội dung văn bản.
2. **Cắt nhỏ (Chunk)** — chia tài liệu dài thành nhiều đoạn nhỏ (chunk) dựa
   trên **Chunk Size** bạn chọn ở sidebar, để dễ xử lý và tra cứu sau này.
3. **Tạo embedding** — "dịch" mỗi đoạn thành một dãy số (vector) đại diện
   cho ý nghĩa của nó, dùng **Embedding Model** đã chọn ở sidebar.
4. **Lưu trữ (Store)** — lưu các vector này vào kho lưu trữ (vector store)
   để có thể tìm kiếm theo độ liên quan khi bạn đặt câu hỏi.

Kết quả trả về cho biết tài liệu đã được chia thành **bao nhiêu chunk** —
con số này phụ thuộc vào độ dài tài liệu và Chunk Size bạn đã chọn.
        """
    )

st.caption(
    "ℹ️ Ở Sprint 1, bước \"đọc tài liệu\" đã chạy thật — các bước "
    "chunk/embed/lưu trữ vẫn đang ở chế độ **minh hoạ (giả lập)** và sẽ trở "
    "thành luồng thật ở Sprint 2 (S2-PE-02)."
)

if "indexing_history" not in st.session_state:
    st.session_state["indexing_history"] = []

uploaded_file = st.file_uploader(
    "Chọn tài liệu", type=["pdf", "txt", "md", "markdown"]
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner(f"Đang index '{uploaded_file.name}'..."):
        result = pipeline.index_document(tmp_path)
    st.session_state["indexing_history"].append((uploaded_file.name, result))

    if result.success:
        st.success(f"Đã index '{uploaded_file.name}' thành công.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Số chunks", result.num_chunks)
        col2.metric("Doc ID", result.doc_id[:12])
        col3.metric("Collection", result.collection_name)
    else:
        st.error(f"Index '{uploaded_file.name}' thất bại: {result.error_message}")

if st.session_state["indexing_history"]:
    st.subheader("Lịch sử upload trong phiên này")
    for file_name, result in reversed(st.session_state["indexing_history"]):
        status_icon = "✅" if result.success else "❌"
        st.write(
            f"{status_icon} **{file_name}** — `{result.doc_id or 'n/a'}` — "
            f"{result.num_chunks} chunks — collection `{result.collection_name}`"
        )
