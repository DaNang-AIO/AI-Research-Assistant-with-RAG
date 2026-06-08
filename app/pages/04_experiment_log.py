"""Trang Experiment Log — lịch sử các sự kiện thực nghiệm (indexing/query).

Khung điều hướng + cấu hình: S1-PE-01. Component `metrics_widget` và việc
hiển thị dữ liệu thật từ `ExperimentTracker` theo thứ tự thời gian sẽ hoàn
thiện ở S4-PE-05 (Yêu cầu 10.6), sau khi `ExperimentTracker` được tích hợp
vào `RAGPipeline` (S4-PE-04). Ở Sprint 1, trang chỉ tổng hợp tạm thời các sự
kiện đã xảy ra trong phiên (từ Document Upload và Chat Interface) để minh hoạ
luồng điều hướng.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_config

st.set_page_config(page_title="Experiment Log", page_icon="📊", layout="wide")

config = render_sidebar_config()

st.title("📊 Experiment Log")
st.markdown(
    "Đây là \"sổ nhật ký\" ghi lại những gì đã xảy ra trong phiên làm việc của "
    "bạn — giúp bạn **so sánh** các lần thử khác nhau (đổi model, đổi chunk "
    "size, đổi top-k...) để hiểu rõ hơn từng tuỳ chọn ảnh hưởng thế nào tới "
    "kết quả RAG."
)

with st.expander("💡 Tại sao cần theo dõi thực nghiệm (experiment tracking)? (bấm để xem)"):
    st.markdown(
        """
RAG có rất nhiều "núm vặn" có thể chỉnh — model nào, chunk size bao nhiêu,
lấy bao nhiêu đoạn liên quan (top-k)... Mỗi lựa chọn có thể cho ra kết quả
khác nhau. Việc ghi lại từng lần thử giúp bạn:

- **So sánh khách quan** — ví dụ: chunk size `256` hay `512` cho câu trả lời
  tốt hơn với loại tài liệu của bạn?
- **Không lặp lại sai lầm** — biết được cấu hình nào đã thử và kết quả ra sao.
- **Hiểu sâu hơn về RAG** — đây chính là cách các kỹ sư AI thực thụ làm việc:
  thử nghiệm có hệ thống thay vì đoán mò.

👉 Hãy thử upload cùng một tài liệu hoặc đặt cùng một câu hỏi với các cấu
hình sidebar khác nhau, rồi quay lại đây để so sánh!
        """
    )

st.info(
    "ℹ️ Sprint 1: trang chỉ tổng hợp tạm thời các sự kiện trong phiên hiện "
    "tại (từ Document Upload và Chat Interface). Việc hiển thị lịch sử đầy đủ "
    "từ `ExperimentTracker` (lưu lại giữa các phiên) sẽ hoàn thiện ở Sprint 4 (S4-PE-05)."
)

indexing_history = st.session_state.get("indexing_history", [])
chat_messages = st.session_state.get("chat_messages", [])

col1, col2 = st.columns(2)
col1.metric("Tài liệu đã index trong phiên", len(indexing_history))
col2.metric("Câu hỏi đã đặt trong phiên", len(chat_messages))

if not indexing_history and not chat_messages:
    st.write(
        "Chưa có sự kiện nào trong phiên này. Hãy thử upload tài liệu ở "
        "**Document Upload** hoặc đặt câu hỏi ở **Chat Interface**."
    )
else:
    if indexing_history:
        st.subheader("📥 Sự kiện indexing")
        for file_name, result in indexing_history:
            status = "thành công" if result.success else f"lỗi: {result.error_message}"
            st.write(f"- **{file_name}** → {result.num_chunks} chunks ({status})")

    if chat_messages:
        st.subheader("💬 Sự kiện query")
        for question, response in chat_messages:
            st.write(
                f"- `{response.timestamp:%H:%M:%S}` — “{question}” → "
                f"{response.latency_ms:.1f} ms · {len(response.contexts)} nguồn tham chiếu"
            )
