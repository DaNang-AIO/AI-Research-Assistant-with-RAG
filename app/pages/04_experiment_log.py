"""Trang Experiment Log — lịch sử các sự kiện thực nghiệm (indexing/query).

Khung điều hướng + cấu hình: S1-PE-01. Triển khai S4-PE-05 (Yêu cầu 10.6):
hiển thị dữ liệu thật từ `ExperimentTracker` (được `RAGPipeline` ghi lại tự
động qua `log_indexing`/`log_query` ở S4-PE-04) — gồm thống kê tổng hợp
(`get_summary()`), dòng thời gian sự kiện, và so sánh hai phiên đã lưu
(`save_session`/`load_session`/`compare_sessions`).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.metrics_widget import render_comparison, render_summary, render_timeline
from app.components.pipeline_factory import get_experiment_tracker
from app.components.sidebar import render_sidebar_config

st.set_page_config(page_title="Experiment Log", page_icon="📊", layout="wide")

config = render_sidebar_config()
tracker = get_experiment_tracker()

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

st.subheader("📈 Thống kê phiên hiện tại")
render_summary(tracker.get_summary())

st.divider()
st.subheader("🕒 Dòng thời gian sự kiện")
st.caption(
    "Mỗi lần bạn upload tài liệu (Document Upload) hoặc đặt câu hỏi (Chat "
    "Interface), `RAGPipeline` tự động gọi `ExperimentTracker.log_indexing()` "
    "/ `log_query()` — không cần thao tác gì thêm ở đây."
)
render_timeline(tracker._current_session)

st.divider()
st.subheader("💾 Lưu & so sánh phiên (`save_session` / `compare_sessions`)")
st.caption(
    "Lưu phiên hiện tại ra file JSON (round-trip qua `save_session`/"
    "`load_session` — Property 11), rồi so sánh hai phiên đã lưu để xem "
    "cấu hình nào cho kết quả tốt hơn."
)

col_save, col_compare = st.columns(2)

with col_save:
    st.markdown("**Lưu phiên hiện tại**")
    session_name = st.text_input(
        "Tên phiên",
        key="experiment_session_name",
        placeholder="vd: chunk_512_top_k_5",
    )
    if st.button("💾 Lưu phiên", disabled=not session_name):
        try:
            saved_path = tracker.save_session(session_name)
            st.success(f"Đã lưu phiên `{session_name}` → `{saved_path}`")
        except Exception as exc:
            st.error(f"Không lưu được phiên: {exc}")

with col_compare:
    st.markdown("**So sánh hai phiên đã lưu**")
    saved_session_names = sorted(
        path.stem for path in Path(tracker.log_dir).glob("*.json")
    )
    if saved_session_names:
        st.caption("Các phiên đã lưu (gõ đúng tên để so sánh): " + ", ".join(f"`{name}`" for name in saved_session_names))
    else:
        st.caption("Chưa có phiên nào được lưu — hãy lưu ít nhất 2 phiên ở cột bên trái trước khi so sánh.")
    session_a = st.text_input("Phiên A", key="experiment_session_a", placeholder="vd: chunk_256")
    session_b = st.text_input("Phiên B", key="experiment_session_b", placeholder="vd: chunk_512")
    if st.button("🔍 So sánh", disabled=not (session_a and session_b)):
        try:
            comparison = tracker.compare_sessions(session_a, session_b)
            st.session_state["experiment_comparison"] = (session_a, session_b, comparison)
        except Exception as exc:
            st.error(
                f"Không so sánh được — kiểm tra lại tên phiên đã được lưu "
                f"trong `{tracker.log_dir}` chưa. Lỗi: {exc}"
            )

if "experiment_comparison" in st.session_state:
    saved_a, saved_b, comparison = st.session_state["experiment_comparison"]
    st.markdown(f"**Kết quả so sánh `{saved_a}` ↔ `{saved_b}`**")
    render_comparison(comparison, saved_a, saved_b)
