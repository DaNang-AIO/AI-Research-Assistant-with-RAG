"""Trang Chat Interface — đặt câu hỏi và xem câu trả lời từ RAG pipeline.

Khung điều hướng + cấu hình: S1-PE-01. Từ Sprint 3 (S3-PE-04), `RAGPipeline.
query()` chạy luồng thật Embed → Retrieve → Build Prompt → Generate với LLM
cục bộ qua OLLAMA — câu trả lời và danh sách nguồn tham chiếu được hiển thị
qua component `app.components.chat_widget` (Yêu cầu 10.4).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components import chat_widget
from app.components.pipeline_factory import build_pipeline
from app.components.sidebar import render_sidebar_config

st.set_page_config(page_title="Chat Interface", page_icon="💬", layout="wide")

config = render_sidebar_config()
pipeline = build_pipeline(config)

st.title("💬 Chat Interface")
st.markdown(
    "Đặt câu hỏi về **tài liệu bạn đã tải lên** ở trang Document Upload — "
    "hệ thống sẽ tìm những đoạn liên quan nhất và nhờ AI soạn câu trả lời "
    "dựa trên chúng (giống kiểu \"thi mở sách\")."
)

with st.expander("💡 RAG trả lời câu hỏi của bạn như thế nào? (bấm để xem)"):
    st.markdown(
        """
1. Câu hỏi của bạn cũng được "dịch" thành vector (embedding) để so sánh.
2. Hệ thống tìm trong kho lưu trữ những đoạn tài liệu (chunk) có vector
   **gần giống** với câu hỏi nhất — đây gọi là bước **truy xuất (retrieval)**,
   số lượng đoạn lấy ra phụ thuộc vào **Top-K** bạn chọn ở sidebar.
3. Các đoạn này được ghép cùng câu hỏi thành một **prompt** hoàn chỉnh, gửi
   cho LLM (mục **LLM Model** ở sidebar).
4. LLM đọc prompt và soạn câu trả lời — *dựa trên nội dung tài liệu thật*
   thay vì chỉ "đoán" từ kiến thức có sẵn.

👉 Muốn biết chính xác đoạn nào đã được dùng? Ghé trang **🔍 Retrieval Debug**
ngay sau khi nhận được câu trả lời.
        """
    )

if not pipeline.llm_client.is_available():
    st.error(
        f"🔌 Không kết nối được tới OLLAMA server tại "
        f"`{pipeline.llm_client.base_url}`. Hãy mở terminal và chạy "
        f"`ollama serve`, đảm bảo model `{pipeline.llm_client.model_name}` đã "
        f"được `ollama pull` về máy, rồi tải lại trang này."
    )
else:
    st.caption(
        "ℹ️ Câu trả lời được sinh **thật** từ LLM cục bộ qua OLLAMA — luồng "
        "Embed → Retrieve → Build Prompt → Generate (Sprint 3)."
    )

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []  # list[(question, RAGResponse)]

for question, response in st.session_state["chat_messages"]:
    chat_widget.render_message(question, response)

question = st.chat_input("Đặt câu hỏi về tài liệu đã index...")
if question:
    if not pipeline.llm_client.is_available():
        st.warning(
            "Không thể gửi câu hỏi vì OLLAMA server hiện không khả dụng — "
            "hãy khởi động `ollama serve` rồi thử lại."
        )
    else:
        try:
            with st.spinner("Đang xử lý câu hỏi (embed → truy xuất → soạn câu trả lời)..."):
                response = pipeline.query(question)
        except Exception as exc:
            st.error(f"Có lỗi khi xử lý câu hỏi: {exc}")
        else:
            st.session_state["chat_messages"].append((question, response))
            st.rerun()
