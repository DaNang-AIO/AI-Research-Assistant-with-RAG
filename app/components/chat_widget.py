"""Component chat_widget.py — widget chat tái sử dụng (design.md §1.2).

Triển khai: S3-PE-04 — component hiển thị câu trả lời và danh sách nguồn tham chiếu.
Có thể import và dùng tại nhiều trang Streamlit khác nhau.
"""

import streamlit as st
from typing import List, Optional

from src.models import RAGResponse, ScoredChunk


def render_rag_response(response: RAGResponse, show_contexts: bool = True) -> None:
    """
    Render đầy đủ một RAGResponse: câu trả lời + nguồn tham chiếu.

    Args:
        response: RAGResponse object từ pipeline.query()
        show_contexts: Có hiển thị phần nguồn tham chiếu hay không
    """
    # ── Câu trả lời ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background:#1e293b; border:1px solid #334155;
            border-radius:16px 16px 16px 4px; padding:16px 20px; color:#e2e8f0;
        ">
            <div style="font-size:12px; color:#64748b; margin-bottom:6px">
                🤖 {response.model_name} · ⏱ {response.latency_ms:.0f} ms
            </div>
            {response.answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not show_contexts or not response.contexts:
        return

    # ── Nguồn tham chiếu ────────────────────────────────────────────────────
    st.markdown("")
    with st.expander(f"📎 {len(response.contexts)} nguồn tham chiếu", expanded=False):
        render_context_list(response.contexts)


def render_context_list(contexts: List[ScoredChunk]) -> None:
    """
    Render danh sách ScoredChunk làm nguồn tham chiếu.

    Args:
        contexts: Danh sách ScoredChunk từ retrieval
    """
    for sc in contexts:
        source = sc.chunk.metadata.get("source", sc.chunk.doc_id)
        page = sc.chunk.metadata.get("page", "—")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(
                f"**#{sc.rank}** · `{source}` · trang {page} "
                f"· [{sc.chunk.start_index}:{sc.chunk.end_index}]"
            )
            st.caption(sc.chunk.content[:250] + "…" if len(sc.chunk.content) > 250 else sc.chunk.content)

        with col2:
            # Score hiển thị dưới dạng progress bar
            st.metric(label="Score", value=f"{sc.score:.3f}")
            st.progress(sc.score)

        st.markdown("---")


def render_empty_chat_placeholder() -> None:
    """Hiển thị placeholder khi chưa có tin nhắn nào."""
    st.markdown(
        """
        <div style="text-align:center; padding:40px; color:#475569">
            <div style="font-size:56px">💬</div>
            <h3 style="color:#64748b; margin: 12px 0 6px">Bắt đầu trò chuyện</h3>
            <p style="font-size:14px">
                Nhập câu hỏi bên dưới. LLM sẽ trả lời dựa trên
                nội dung tài liệu bạn đã upload và index.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str, timestamp: Optional[str] = None) -> None:
    """
    Render một tin nhắn chat (user hoặc assistant).

    Args:
        role: "user" hoặc "assistant"
        content: Nội dung tin nhắn
        timestamp: Thời gian (tuỳ chọn)
    """
    if role == "user":
        bubble_style = (
            "background:linear-gradient(135deg,#1e40af,#1e3a8a);"
            "border-radius:16px 16px 4px 16px;"
            "margin:4px 60px 4px 0;"
        )
        prefix = "👤 <strong>Bạn</strong>"
    else:
        bubble_style = (
            "background:#1e293b;border:1px solid #334155;"
            "border-radius:16px 16px 16px 4px;"
            "margin:4px 0 4px 60px;"
        )
        prefix = "🤖 <strong>Assistant</strong>"

    ts_html = (
        f'<div style="font-size:11px;color:#64748b;margin-top:4px">{timestamp}</div>'
        if timestamp
        else ""
    )

    st.markdown(
        f"""
        <div style="{bubble_style} padding:12px 16px; color:#e2e8f0;">
            <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">{prefix}</div>
            {content}
            {ts_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
