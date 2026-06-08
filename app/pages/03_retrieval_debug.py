"""Trang Retrieval Debug — xem chunk truy xuất kèm điểm similarity và vị trí.

Khung điều hướng + cấu hình: S1-PE-01. Từ Sprint 3 (S3-PE-05), `RAGPipeline.
query()` truy xuất context thật từ `ChromaVectorStore.similarity_search()` —
trang này hiển thị các `ScoredChunk` đó kèm điểm similarity (Property 6-8) và
vị trí (`start_index`/`end_index`) trong tài liệu gốc (Yêu cầu 10.5).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_config

st.set_page_config(page_title="Retrieval Debug", page_icon="🔍", layout="wide")

config = render_sidebar_config()

st.title("🔍 Retrieval Debug")
st.markdown(
    "Đây là nơi bạn \"nhìn vào hộp đen\" của RAG — xem chính xác **những đoạn "
    "tài liệu nào** đã được hệ thống chọn ra để tham khảo trước khi trả lời "
    "câu hỏi gần nhất của bạn ở trang **Chat Interface**."
)

with st.expander("💡 \"Similarity score\" và \"vị trí\" nghĩa là gì? (bấm để xem)"):
    st.markdown(
        """
- **Similarity score (điểm tương đồng)** — một con số trong khoảng
  `0.0 → 1.0` cho biết đoạn tài liệu đó *liên quan đến câu hỏi của bạn* đến
  mức nào. Điểm càng gần `1.0`, đoạn đó càng liên quan. Danh sách luôn được
  sắp xếp theo điểm **giảm dần** — đoạn liên quan nhất hiện ở trên cùng.
- **Vị trí (`start_index` / `end_index`)** — cho biết đoạn này nằm ở đâu
  trong tài liệu gốc (tính theo ký tự), giúp bạn truy ngược lại đúng chỗ
  trong tài liệu ban đầu.
- **Rank (#1, #2, ...)** — thứ hạng liên quan, dựa trên **Top-K** bạn đã
  chọn ở sidebar (số lượng đoạn tối đa được lấy ra).

👉 Trang này giúp bạn **kiểm chứng** xem AI có đang trả lời dựa trên đúng
nội dung tài liệu hay không — một phần quan trọng để tin tưởng kết quả của RAG.
        """
    )

messages = st.session_state.get("chat_messages", [])
if not messages:
    st.write("Chưa có câu hỏi nào được đặt trong phiên này. Hãy thử ở trang **Chat Interface**.")
else:
    last_question, last_response = messages[-1]
    st.subheader(f"Context cho câu hỏi gần nhất: “{last_question}”")

    if not last_response.contexts:
        st.write(
            "Câu trả lời gần nhất không kèm context nào — có thể vì kho lưu "
            "trữ vector chưa có tài liệu nào được index (hãy thử ở trang "
            "**Document Upload** trước), hoặc không tìm thấy đoạn nào đủ liên quan."
        )
    else:
        for scored in last_response.contexts:
            with st.expander(
                f"#{scored.rank} · score={scored.score:.3f} · chunk_id={scored.chunk.chunk_id}"
            ):
                st.write(scored.chunk.content)
                st.caption(
                    f"doc_id={scored.chunk.doc_id} · "
                    f"vị trí [{scored.chunk.start_index}:{scored.chunk.end_index}]"
                )
