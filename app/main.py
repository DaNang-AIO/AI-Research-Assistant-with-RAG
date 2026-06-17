"""Streamlit Research Dashboard — entry point.
Triển khai: S1-PE-01 (4 trang điều hướng + sidebar cấu hình lưu vào
st.session_state["config"] — Yêu cầu 10.1, 10.2).
"""

import sys
import os

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
# dù Streamlit được chạy từ đâu.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from config.settings import AppConfig

# ── Phải gọi set_page_config() ở đầu tiên, trước mọi lệnh st khác ──────────
st.set_page_config(
    page_title="RAG Research Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tuỳ chỉnh ────────────────────────────────────────────────────────────
# st.markdown(
#     """
#     <style>
#     /* Màu nền sidebar */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
#     }
#     [data-testid="stSidebar"] * {
#         color: #e2e8f0 !important;
#     }
#     /* Badge trạng thái */
#     .status-badge {
#         display: inline-block;
#         padding: 2px 10px;
#         border-radius: 12px;
#         font-size: 12px;
#         font-weight: 600;
#     }
#     .status-ok   { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }
#     .status-warn { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
#     .status-err  { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


def _init_session_defaults() -> None:
    """Khởi tạo session_state với giá trị mặc định từ AppConfig nếu chưa có."""
    if "config" not in st.session_state:
        default = AppConfig.from_env()
        st.session_state["config"] = {
            "ollama_model": default.ollama.default_llm_model,
            "embedding_model": default.ollama.default_embedding_model,
            "chunk_size": default.chunker.chunk_size,
            "chunk_overlap": default.chunker.chunk_overlap,
            "top_k": default.top_k,
        }
    # Lịch sử chat và kết quả retrieval debug — dùng chung giữa các page
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_retrieval_result" not in st.session_state:
        st.session_state["last_retrieval_result"] = None
    if "indexed_docs" not in st.session_state:
        st.session_state["indexed_docs"] = []


def _render_sidebar() -> None:
    """Vẽ sidebar cấu hình pipeline; cập nhật st.session_state["config"]."""
    with st.sidebar:
        st.markdown("## 🔬 RAG Research")
        st.markdown("---")

        st.header("⚙️ Cấu hình Pipeline")

        # ── LLM Model ──────────────────────────────────────────────────────
        _llm_presets = ["llama3", "mistral", "gemma"]
        current_llm = st.session_state["config"].get("ollama_model", "llama3")
        llm_options = _llm_presets if current_llm in _llm_presets else [current_llm] + _llm_presets
        ollama_model = st.selectbox(
            "🤖 LLM Model",
            options=llm_options,
            index=llm_options.index(current_llm),
            key="sidebar_llm_model",
            help="Model ngôn ngữ dùng để sinh câu trả lời (chạy cục bộ qua OLLAMA).",
        )

        # ── Embedding Model ─────────────────────────────────────────────────
        _emb_presets = ["nomic-embed-text", "mxbai-embed-large"]
        current_emb = st.session_state["config"].get("embedding_model", "nomic-embed-text")
        emb_options = _emb_presets if current_emb in _emb_presets else [current_emb] + _emb_presets
        embedding_model = st.selectbox(
            "🧮 Embedding Model",
            options=emb_options,
            index=emb_options.index(current_emb),
            key="sidebar_emb_model",
            help="Model tạo vector embedding cho tài liệu và câu hỏi.",
        )

        st.markdown("---")
        st.subheader("📄 Chunking")

        # ── Chunk Size ──────────────────────────────────────────────────────
        chunk_size = st.slider(
            "Chunk Size (ký tự)",
            min_value=128,
            max_value=1024,
            value=st.session_state["config"].get("chunk_size", 512),
            step=64,
            key="sidebar_chunk_size",
            help="Số ký tự tối đa mỗi đoạn văn bản (chunk).",
        )

        # ── Chunk Overlap ───────────────────────────────────────────────────
        chunk_overlap = st.slider(
            "Chunk Overlap (ký tự)",
            min_value=0,
            max_value=200,
            value=st.session_state["config"].get("chunk_overlap", 50),
            step=10,
            key="sidebar_chunk_overlap",
            help="Số ký tự chồng lấp giữa hai chunk liên tiếp để tránh mất ngữ cảnh.",
        )

        st.markdown("---")
        st.subheader("🔍 Retrieval")

        # ── Top-K ───────────────────────────────────────────────────────────
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=10,
            value=st.session_state["config"].get("top_k", 5),
            key="sidebar_top_k",
            help="Số chunk liên quan nhất được truy xuất để làm ngữ cảnh cho LLM.",
        )

        # ── Cập nhật config vào session_state ──────────────────────────────
        st.session_state["config"] = {
            "ollama_model": ollama_model,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
        }

        # ── Hiển thị config hiện tại (compact) ─────────────────────────────
        st.markdown("---")
        st.caption("📋 Config hiện tại")
        cfg = st.session_state["config"]
        st.code(
            f"llm     : {cfg['ollama_model']}\n"
            f"embed   : {cfg['embedding_model']}\n"
            f"chunk   : {cfg['chunk_size']} / overlap {cfg['chunk_overlap']}\n"
            f"top_k   : {cfg['top_k']}",
            language="text",
        )

        # ── Trạng thái tài liệu đã index ───────────────────────────────────
        docs = st.session_state.get("indexed_docs", [])
        st.markdown("---")
        st.caption(f"📚 Tài liệu đã index: **{len(docs)}**")
        if docs:
            for d in docs[-3:]:  # chỉ hiện 3 file gần nhất
                st.markdown(f"• `{d}`")
            if len(docs) > 3:
                st.caption(f"… và {len(docs) - 3} tài liệu khác")


def main() -> None:
    """Entry point của Streamlit dashboard."""
    _init_session_defaults()
    _render_sidebar()

    # ── Trang chủ (Home) ────────────────────────────────────────────────────
    st.title("🔬 AI Research Assistant with RAG")
    st.markdown(
        "**Research dashboard** để khám phá và thực nghiệm các thành phần RAG pipeline. "
        "Chọn trang từ menu bên trái (sidebar) để bắt đầu."
    )

    st.markdown("---")

    # ── Thẻ điều hướng nhanh ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1e40af22, #3b82f622);
                border: 1px solid #3b82f644;
                border-radius: 12px; padding: 20px; text-align: center;
            ">
                <div style="font-size:2rem">📄</div>
                <h4 style="margin:8px 0 4px; color:#60a5fa">Document Upload</h4>
                <p style="font-size:13px; color:#94a3b8; margin:0">
                    Upload và index tài liệu PDF, TXT, MD vào ChromaDB
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #16543022, #22c55e22);
                border: 1px solid #22c55e44;
                border-radius: 12px; padding: 20px; text-align: center;
            ">
                <div style="font-size:2rem">💬</div>
                <h4 style="margin:8px 0 4px; color:#4ade80">Chat Interface</h4>
                <p style="font-size:13px; color:#94a3b8; margin:0">
                    Hỏi đáp với tài liệu đã index, nhận câu trả lời từ LLM
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #78350f22, #f59e0b22);
                border: 1px solid #f59e0b44;
                border-radius: 12px; padding: 20px; text-align: center;
            ">
                <div style="font-size:2rem">🔍</div>
                <h4 style="margin:8px 0 4px; color:#fbbf24">Retrieval Debug</h4>
                <p style="font-size:13px; color:#94a3b8; margin:0">
                    Xem chunk nào được truy xuất và điểm similarity
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #4c1d9522, #a855f722);
                border: 1px solid #a855f744;
                border-radius: 12px; padding: 20px; text-align: center;
            ">
                <div style="font-size:2rem">📊</div>
                <h4 style="margin:8px 0 4px; color:#c084fc">Experiment Log</h4>
                <p style="font-size:13px; color:#94a3b8; margin:0">
                    Theo dõi lịch sử thực nghiệm indexing và querying
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Hướng dẫn bắt đầu nhanh ────────────────────────────────────────────
    st.subheader("🚀 Bắt đầu nhanh")
    with st.expander("📋 Hướng dẫn sử dụng", expanded=True):
        st.markdown(
            """
            1. **Cấu hình sidebar** bên trái — chọn LLM model, embedding model, chunk size và top-k.
            2. **Document Upload** — upload file PDF/TXT/MD để index vào ChromaDB.
            3. **Chat Interface** — đặt câu hỏi và nhận câu trả lời từ LLM cục bộ.
            4. **Retrieval Debug** — xem chi tiết các chunk được truy xuất và điểm similarity.
            5. **Experiment Log** — xem lại lịch sử các phiên thực nghiệm.

            > 💡 **Lưu ý:** Cần khởi động OLLAMA server trước khi sử dụng:
            > ```bash
            > ollama serve
            > ollama pull llama3
            > ollama pull nomic-embed-text
            > ```
            """
        )

    # ── RAG Pipeline flow diagram ───────────────────────────────────────────
    st.subheader("🔄 RAG Pipeline")
    st.markdown(
        """
        ```
        Tài liệu  →  [DocumentLoader]  →  Document
                                              ↓
                                       [TextChunker]  →  List[Chunk]
                                              ↓
                                  [OllamaEmbeddingModel]  →  vectors
                                              ↓
                                      [ChromaVectorStore]  ←── lưu trữ

        Câu hỏi  →  [OllamaEmbeddingModel]  →  query_vector
                                              ↓
                               [ChromaVectorStore.similarity_search]
                                              ↓
                                    [PromptBuilder.build()]
                                              ↓
                                   [OllamaClient.generate()]
                                              ↓
                                        💬 Câu trả lời
        ```
        """
    )

    # ── Trạng thái hiện tại ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Trạng thái hệ thống")
    docs = st.session_state.get("indexed_docs", [])
    chats = st.session_state.get("chat_history", [])

    m1, m2, m3 = st.columns(3)
    m1.metric("📚 Tài liệu đã index", len(docs))
    m2.metric("💬 Lượt hỏi đáp", len(chats))
    m3.metric("⚙️ Sprint hiện tại", "S1-PE-01 ✅")


if __name__ == "__main__":
    main()
