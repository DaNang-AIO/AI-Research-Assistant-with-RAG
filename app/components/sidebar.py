"""Sidebar cấu hình pipeline dùng chung cho mọi trang dashboard (design.md §2.6).

Mỗi trang gọi `render_sidebar_config()` để vừa hiển thị sidebar cấu hình vừa
đảm bảo `st.session_state["config"]` luôn tồn tại — kể cả khi người dùng điều
hướng thẳng tới một trang con mà chưa đi qua trang chính (Yêu cầu 10.1, 10.2).
"""

import streamlit as st


def render_sidebar_config() -> dict:
    """Hiển thị sidebar cấu hình pipeline và lưu lựa chọn vào `st.session_state["config"]`."""
    with st.sidebar:
        st.header("⚙️ Cấu hình Pipeline")
        st.caption(
            "Mới dùng RAG lần đầu? Ghé trang 🎓 **Hướng dẫn sử dụng** ở "
            "đầu danh sách điều hướng để hiểu rõ từng tuỳ chọn bên dưới."
        )

        ollama_model = st.selectbox(
            "LLM Model",
            options=["llama3", "mistral", "gemma"],
            key="sidebar_ollama_model",
            help=(
                "🧠 'Bộ não' soạn câu trả lời cho bạn — một mô hình ngôn ngữ "
                "lớn (LLM) chạy **cục bộ trên máy bạn** thông qua OLLAMA "
                "(không gửi dữ liệu lên cloud). Hãy thử đổi model để so sánh "
                "chất lượng câu trả lời."
            ),
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["nomic-embed-text", "mxbai-embed-large"],
            key="sidebar_embedding_model",
            help=(
                "🧭 Mô hình 'dịch' văn bản thành dãy số (vector) để hệ thống "
                "so sánh được *ý nghĩa* giữa các đoạn văn bản — đây là cách "
                "hệ thống tìm ra đoạn tài liệu nào liên quan tới câu hỏi của bạn."
            ),
        )
        chunk_size = st.slider(
            "Chunk Size",
            min_value=128,
            max_value=1024,
            value=512,
            key="sidebar_chunk_size",
            help=(
                "✂️ Tài liệu dài được cắt thành từng đoạn nhỏ (chunk) trước khi "
                "xử lý — giống như chia một quyển sách thành từng trang. Số "
                "**lớn** → đoạn dài hơn, nhiều ngữ cảnh hơn; số **nhỏ** → đoạn "
                "ngắn, tập trung hơn nhưng dễ mất ngữ cảnh xung quanh."
            ),
        )
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=10,
            value=5,
            key="sidebar_top_k",
            help=(
                "🔝 Số lượng đoạn tài liệu liên quan nhất sẽ được lấy ra để "
                "'đưa' cho AI tham khảo trước khi trả lời — giống như đánh dấu "
                "trước K trang sách hữu ích nhất trước khi vào phòng thi."
            ),
        )

    config = {
        "ollama_model": ollama_model,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "top_k": top_k,
    }
    st.session_state["config"] = config
    return config
