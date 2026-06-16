"""Trang 3 — Retrieval Debug (design.md §2.6, Yêu cầu 10.5).

Triển khai: S3-PE-05 — hiển thị chunk truy xuất kèm điểm similarity
và vị trí (start_index/end_index) trong tài liệu gốc.
Sprint 1: đọc kết quả từ st.session_state["last_retrieval_result"] (stub).
Sprint 3: dữ liệu thật khi RAGPipeline.query() được triển khai.
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="Retrieval Debug — RAG Research",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .chunk-card {
        background: #0f1e2e;
        border-left: 4px solid #f59e0b;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .chunk-card-high {
        border-left-color: #22c55e;
        background: #0a1f13;
    }
    .chunk-card-low {
        border-left-color: #ef4444;
        background: #1f0a0a;
    }
    .score-bar-wrap {
        background: #1e293b;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 6px 0;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 6px;
        background: linear-gradient(90deg, #f59e0b, #22c55e);
        transition: width 0.4s ease;
    }
    .meta-tag {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        color: #94a3b8;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _score_color_class(score: float) -> str:
    """Chọn class CSS dựa trên ngưỡng score."""
    if score >= 0.8:
        return "chunk-card chunk-card-high"
    if score >= 0.5:
        return "chunk-card"
    return "chunk-card chunk-card-low"


def _render_chunk_card(sc, index: int) -> None:
    """Render một ScoredChunk dưới dạng card có đầy đủ thông tin debug."""
    css_class = _score_color_class(sc.score)
    score_pct = int(sc.score * 100)

    # Score bar
    bar_html = (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="width:{score_pct}%"></div>'
        f"</div>"
    )

    source = sc.chunk.metadata.get("source", sc.chunk.doc_id)
    page = sc.chunk.metadata.get("page", "—")

    st.markdown(
        f"""
        <div class="{css_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
                <strong style="color:#e2e8f0">#{sc.rank} — Chunk {index}</strong>
                <strong style="color:#fbbf24; font-size:18px">{sc.score:.3f}</strong>
            </div>
            {bar_html}
            <div style="margin:8px 0">
                <span class="meta-tag">📄 {source}</span>
                <span class="meta-tag">📄 Trang {page}</span>
                <span class="meta-tag">🆔 {sc.chunk.chunk_id}</span>
                <span class="meta-tag">📍 [{sc.chunk.start_index} → {sc.chunk.end_index}]</span>
                <span class="meta-tag">📏 {sc.chunk.end_index - sc.chunk.start_index} ký tự</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Nội dung chunk
    with st.expander(f"📝 Nội dung chunk #{sc.rank}", expanded=(index == 0)):
        st.markdown(sc.chunk.content)
        st.caption(
            f"Doc ID: `{sc.chunk.doc_id}` | "
            f"Chunk ID: `{sc.chunk.chunk_id}` | "
            f"Vị trí: [{sc.chunk.start_index} : {sc.chunk.end_index}]"
        )


def main() -> None:
    # ── Khởi tạo session_state ───────────────────────────────────────────────
    if "config" not in st.session_state:
        st.session_state["config"] = {
            "ollama_model": "llama3",
            "embedding_model": "nomic-embed-text",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
        }

    st.title("🔍 Retrieval Debug")
    st.markdown(
        "Xem chi tiết các chunk được truy xuất cho câu hỏi gần nhất: "
        "**điểm similarity, vị trí trong tài liệu gốc, nội dung đầy đủ.**"
    )

    st.info(
        "⚠️ **Sprint 1 — Demo mode:** Kết quả retrieval đọc từ "
        "`st.session_state[\"last_retrieval_result\"]` (dữ liệu stub). "
        "Dữ liệu thật sẽ có từ Sprint 3 (S3-PE-05).",
        icon="🔧",
    )

    st.markdown("---")

    # ── Đọc kết quả retrieval từ session_state ───────────────────────────────
    result = st.session_state.get("last_retrieval_result")

    if result is None:
        st.warning(
            "⚠️ Chưa có kết quả retrieval nào. "
            "Hãy đặt ít nhất **1 câu hỏi** trên trang **Chat Interface** trước.",
            icon="💬",
        )
        return

    # ── Thông tin câu hỏi ───────────────────────────────────────────────────
    st.subheader("❓ Câu hỏi đã truy vấn")
    st.markdown(
        f'<div style="background:#1e293b; border-radius:10px; padding:12px 16px; '
        f'color:#e2e8f0; font-size:15px">{result["question"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("📦 Số chunk truy xuất", len(result["contexts"]))
    m2.metric("🤖 Model", result["model"])
    m3.metric("⏱ Latency", f"{result['latency_ms']:.0f} ms")

    st.markdown("---")

    # ── Score summary ────────────────────────────────────────────────────────
    contexts = result["contexts"]
    if contexts:
        scores = [sc.score for sc in contexts]
        st.subheader(
            f"📊 Kết quả Retrieval — {len(contexts)} chunks "
            f"(avg score: {sum(scores)/len(scores):.3f})"
        )

        # Biểu đồ score
        score_data = {f"#{sc.rank} {sc.chunk.chunk_id[:8]}": sc.score for sc in contexts}
        st.bar_chart(score_data, height=180)

        st.markdown("---")

        # ── Render từng chunk ────────────────────────────────────────────────
        st.subheader("🗂️ Chi tiết từng Chunk")

        # Filter theo ngưỡng score
        min_score = st.slider(
            "Lọc theo ngưỡng score tối thiểu",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="debug_score_filter",
        )

        filtered = [sc for sc in contexts if sc.score >= min_score]
        st.caption(f"Hiển thị {len(filtered)}/{len(contexts)} chunks (score ≥ {min_score:.2f})")

        for i, sc in enumerate(filtered):
            _render_chunk_card(sc, i)
    else:
        st.info("[INFO] Không có chunk nào được truy xuất cho câu hỏi này.")

    # ── Phân tích phân bố score ──────────────────────────────────────────────
    if contexts:
        st.markdown("---")
        st.subheader("📈 Phân tích Score")
        cols = st.columns(4)
        cols[0].metric("🏆 Max", f"{max(scores):.3f}")
        cols[1].metric("📉 Min", f"{min(scores):.3f}")
        cols[2].metric("📊 Mean", f"{sum(scores)/len(scores):.3f}")
        cols[3].metric(
            "✅ Score ≥ 0.7",
            f"{sum(1 for s in scores if s >= 0.7)}/{len(scores)}",
        )


if __name__ == "__main__":
    main()
