"""Chat widget dùng chung cho trang Chat Interface (design.md §2.6, Yêu cầu 10.4).

Triển khai: S3-PE-04 — hiển thị một lượt hỏi-đáp (câu hỏi, câu trả lời từ LLM
cục bộ, và danh sách nguồn tài liệu tham chiếu kèm điểm similarity).
"""

from typing import List

import streamlit as st

from src.models import RAGResponse, ScoredChunk


def render_message(question: str, response: RAGResponse) -> None:
    """Hiển thị một lượt hỏi-đáp hoàn chỉnh: câu hỏi của người dùng, câu trả
    lời từ LLM kèm thông tin model/latency, và danh sách nguồn tham chiếu."""
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(response.answer)
        st.caption(
            f"Model: `{response.model_name}` · Latency: {response.latency_ms:.1f} ms · "
            f"Nguồn tham chiếu: {len(response.contexts)}"
        )
        render_sources(response.contexts)


def render_sources(contexts: List[ScoredChunk]) -> None:
    """Hiển thị danh sách nguồn tài liệu được tham chiếu (Yêu cầu 10.4) — mỗi
    nguồn kèm điểm similarity, doc_id và một đoạn trích nội dung để người dùng
    nhanh chóng kiểm chứng câu trả lời dựa trên đúng tài liệu nào."""
    if not contexts:
        st.caption("📚 Không có nguồn tham chiếu nào (chưa index tài liệu hoặc không tìm thấy đoạn liên quan).")
        return

    with st.expander(f"📚 Nguồn tham chiếu ({len(contexts)})"):
        for scored in contexts:
            preview = scored.chunk.content.strip().replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "…"
            st.markdown(
                f"**#{scored.rank}** · score=`{scored.score:.3f}` · "
                f"doc_id=`{scored.chunk.doc_id}` · "
                f"vị trí [{scored.chunk.start_index}:{scored.chunk.end_index}]"
            )
            st.caption(preview)
