"""Trang Chat Interface — đặt câu hỏi và xem câu trả lời từ RAG pipeline.

Khung điều hướng + cấu hình: S1-PE-01. Từ Sprint 3 (S3-PE-04), `RAGPipeline.
query()` chạy luồng thật Embed → Retrieve → Build Prompt → Generate với LLM
cục bộ qua OLLAMA. Câu trả lời cho câu hỏi mới được hiển thị **streaming**
qua `RAGPipeline.query_stream()` + `st.write_stream()` — người dùng thấy chữ
xuất hiện dần ngay khi LLM sinh ra, thay vì chờ cả khối văn bản; lịch sử các
lượt hỏi-đáp trước đó hiển thị qua component `app.components.chat_widget`
(Yêu cầu 10.4).
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components import chat_widget
from app.components.pipeline_factory import build_pipeline
from app.components.sidebar import render_sidebar_config
from src.models import RAGResponse

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
        "Embed → Retrieve → Build Prompt → Generate (Sprint 3), hiển thị "
        "**streaming** (chữ xuất hiện dần khi LLM đang sinh)."
    )
    st.caption(
        "⏳ **Lưu ý:** trước khi hiện chữ đầu tiên, mô hình cần thời gian "
        "\"đọc\" toàn bộ prompt (câu hỏi + các đoạn ngữ cảnh truy xuất được) — "
        "với LLM chạy cục bộ trên CPU, bước này có thể mất **30-90 giây** tuỳ "
        "độ dài ngữ cảnh, *trước khi* streaming thực sự bắt đầu. Muốn rút "
        "ngắn: giảm **Top-K Retrieval** và/hoặc **Chunk Size** ở sidebar để "
        "prompt ngắn hơn."
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
        with st.chat_message("user"):
            st.write(question)

        try:
            with st.spinner("Đang embed câu hỏi và truy xuất context liên quan..."):
                contexts, model_name, token_stream = pipeline.query_stream(question)

            start_time = time.perf_counter()
            with st.chat_message("assistant"):
                st.caption(
                    "🧠 Mô hình đang đọc prompt (câu hỏi + ngữ cảnh) — chữ đầu "
                    "tiên có thể cần khoảng nửa phút mới xuất hiện, sau đó sẽ "
                    "hiện dần liên tục cho tới khi trả lời xong..."
                )
                answer = st.write_stream(token_stream)
                latency_ms = (time.perf_counter() - start_time) * 1000
                st.caption(
                    f"Model: `{model_name}` · Latency: {latency_ms:.1f} ms · "
                    f"Nguồn tham chiếu: {len(contexts)}"
                )
                chat_widget.render_sources(contexts)
        except Exception as exc:
            st.error(f"Có lỗi khi xử lý câu hỏi: {exc}")
        else:
            response = RAGResponse(
                question=question,
                answer=answer,
                contexts=contexts,
                model_name=model_name,
                latency_ms=latency_ms,
            )
            st.session_state["chat_messages"].append((question, response))
