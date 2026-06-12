"""Trang 2 — Chat Interface (design.md §2.6, Yêu cầu 10.4).

Triển khai: S3-PE-04 (hiển thị câu trả lời + nguồn tham chiếu).
Sprint 1: giao diện chat stub — nhận câu hỏi, trả lời giả lập.
Sprint 3: thay bằng RAGPipeline.query() thật (S3-PE-03).
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import datetime
import streamlit as st

from src.models import RAGResponse, ScoredChunk, Chunk, DocumentType

st.set_page_config(
    page_title="Chat Interface — RAG Research",
    page_icon="💬",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .chat-user {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 4px 60px 4px 0;
        color: #e2e8f0;
    }
    .chat-assistant {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 4px 0 4px 60px;
        color: #e2e8f0;
    }
    .source-badge {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #3b82f644;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 12px;
        color: #93c5fd;
        margin: 2px;
    }
    .latency-tag {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _stub_rag_query(question: str, config: dict) -> RAGResponse:
    """
    STUB cho Sprint 1: Giả lập RAGResponse.
    Sprint 3 (S3-PE-03): thay bằng RAGPipeline.query() thật.
    """
    time.sleep(1.2)  # giả lập độ trễ LLM

    # Tạo chunk giả lập để mô phỏng retrieval context
    fake_chunks = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"chunk_{i}",
                doc_id="demo_doc_001",
                content=f"[Demo Sprint 1] Đây là đoạn văn bản mô phỏng số {i+1} "
                        f"được truy xuất liên quan đến câu hỏi của bạn. "
                        f"Nội dung thật sẽ đến từ tài liệu đã index ở Sprint 2.",
                start_index=i * 200,
                end_index=(i + 1) * 200,
                metadata={"source": "demo_document.pdf", "page": i + 1},
            ),
            score=round(0.95 - i * 0.08, 2),
            rank=i + 1,
        )
        for i in range(min(config.get("top_k", 3), 3))
    ]

    fake_answer = (
        f"**[Sprint 1 — Demo Mode]** Đây là câu trả lời giả lập cho câu hỏi: "
        f'"{question}"\n\n'
        "Hệ thống đang chạy ở chế độ stub. Câu trả lời thật từ LLM cục bộ "
        f"({config.get('ollama_model', 'llama3')}) sẽ được kích hoạt từ Sprint 3 "
        "khi `RAGPipeline.query()` được triển khai đầy đủ (task S3-PE-03)."
    )

    return RAGResponse(
        question=question,
        answer=fake_answer,
        contexts=fake_chunks,
        model_name=config.get("ollama_model", "llama3"),
        latency_ms=1234.5,
        timestamp=datetime.datetime.now(),
    )


def _render_message(role: str, content: str) -> None:
    """Render một tin nhắn trong khung chat."""
    if role == "user":
        st.markdown(
            f'<div class="chat-user">👤 <strong>Bạn</strong><br/>{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-assistant">🤖 <strong>Assistant</strong><br/>{content}</div>',
            unsafe_allow_html=True,
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
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_retrieval_result" not in st.session_state:
        st.session_state["last_retrieval_result"] = None

    cfg = st.session_state["config"]

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("💬 Chat Interface")
    st.markdown(
        "Đặt câu hỏi về tài liệu đã index. "
        "LLM sẽ trả lời **dựa trên nội dung tài liệu** được truy xuất."
    )

    # ── Thông báo Sprint 1 (stub) ────────────────────────────────────────────
    st.info(
        "⚠️ **Sprint 1 — Demo mode:** `RAGPipeline.query()` đang dùng stub giả lập. "
        "Hỏi đáp thật với LLM sẽ hoạt động từ Sprint 3 (S3-PE-03).",
        icon="🔧",
    )

    # ── Config hiện tại ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("🤖 LLM", cfg["ollama_model"])
    c2.metric("🧮 Embedding", cfg["embedding_model"])
    c3.metric("🔍 Top-K", cfg["top_k"])

    # Cảnh báo nếu chưa index tài liệu nào
    indexed_docs = st.session_state.get("indexed_docs", [])
    if not indexed_docs:
        st.warning(
            "📄 Chưa có tài liệu nào được index. "
            "Hãy đến trang **Document Upload** để upload tài liệu trước.",
            icon="⚠️",
        )

    st.markdown("---")

    # ── Lịch sử chat ────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state["chat_history"]:
            st.markdown(
                "_Chưa có tin nhắn nào. Nhập câu hỏi bên dưới để bắt đầu._"
            )
        else:
            for turn in st.session_state["chat_history"]:
                _render_message("user", turn["question"])
                _render_message("assistant", turn["answer"])

                # Hiển thị nguồn tham chiếu (contexts)
                if turn.get("contexts"):
                    with st.expander(
                        f"📎 Nguồn tham chiếu ({len(turn['contexts'])} chunks)",
                        expanded=False,
                    ):
                        for sc in turn["contexts"]:
                            st.markdown(
                                f'<span class="source-badge">'
                                f"#{sc.rank} · score={sc.score:.2f} · "
                                f"{sc.chunk.metadata.get('source', sc.chunk.doc_id)}"
                                f"</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(sc.chunk.content[:200] + "...")

                st.markdown(
                    f'<div class="latency-tag">⏱ {turn["latency_ms"]:.0f} ms · '
                    f'{turn["timestamp"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

    # ── Input câu hỏi ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area(
            "✏️ Câu hỏi của bạn",
            placeholder="Ví dụ: RAG là gì? Giải thích quá trình indexing...",
            height=80,
            key="chat_input",
        )
        col_btn1, col_btn2 = st.columns([1, 5])
        submitted = col_btn1.form_submit_button("📤 Gửi", type="primary")
        col_btn2.form_submit_button("🗑️ Xóa lịch sử", on_click=lambda: st.session_state.update({"chat_history": []}))

    if submitted and question.strip():
        with st.spinner("🤔 Đang xử lý câu hỏi..."):
            response: RAGResponse = _stub_rag_query(question.strip(), cfg)

        # Lưu vào lịch sử chat
        st.session_state["chat_history"].append({
            "question": question.strip(),
            "answer": response.answer,
            "contexts": response.contexts,
            "latency_ms": response.latency_ms,
            "timestamp": response.timestamp.strftime("%H:%M:%S"),
        })

        # Lưu kết quả retrieval để trang Retrieval Debug đọc
        st.session_state["last_retrieval_result"] = {
            "question": question.strip(),
            "contexts": response.contexts,
            "model": response.model_name,
            "latency_ms": response.latency_ms,
        }

        st.rerun()

    elif submitted and not question.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi trước khi gửi.")


if __name__ == "__main__":
    main()
