"""Trang Chat Interface — đặt câu hỏi và xem câu trả lời từ RAG pipeline.

Khung điều hướng + cấu hình: S1-PE-01. Câu trả lời ở Sprint 1 vẫn là **giả
lập** (RAGPipeline.query() đang là stub) — component `chat_widget` và luồng
hỏi-đáp thật sẽ hoàn thiện ở S3-PE-04 (Yêu cầu 10.4).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

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

st.caption(
    "ℹ️ Sprint 1: câu trả lời vẫn đang ở chế độ **minh hoạ (giả lập)** vì "
    "`RAGPipeline.query()` còn là stub — luồng hỏi-đáp thật với LLM cục bộ "
    "qua OLLAMA sẽ hoàn thiện ở Sprint 3."
)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []  # list[(question, RAGResponse)]

for question, response in st.session_state["chat_messages"]:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(response.answer)
        st.caption(
            f"Model: {response.model_name} · Latency: {response.latency_ms:.1f} ms · "
            f"Nguồn tham chiếu: {len(response.contexts)}"
        )

question = st.chat_input("Đặt câu hỏi về tài liệu đã index...")
if question:
    with st.spinner("Đang xử lý câu hỏi..."):
        response = pipeline.query(question)
    st.session_state["chat_messages"].append((question, response))
    st.rerun()
