"""Streamlit Research Dashboard — entry point (design.md §2.6).

Triển khai: S1-PE-01 (4 trang điều hướng + sidebar cấu hình lưu vào
st.session_state — Yêu cầu 10.1, 10.2).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_config
from src.generation.llm_client import OllamaClient

st.set_page_config(
    page_title="RAG Research Dashboard",
    page_icon="🔬",
    layout="wide",
)


def main():
    st.title("🔬 AI Research Assistant with RAG")
    st.markdown(
        "Chào mừng bạn! Đây là một **phòng thí nghiệm RAG** — nơi bạn có thể "
        "tải tài liệu của riêng mình lên, đặt câu hỏi, và *nhìn thấy tận mắt* "
        "cách một trợ lý AI tìm thông tin rồi soạn câu trả lời dựa trên "
        "**chính tài liệu đó**, thay vì chỉ \"đoán\" từ kiến thức có sẵn."
    )

    st.info(
        "🆕 **Lần đầu sử dụng?** Hãy ghé trang **🎓 Hướng dẫn sử dụng** ở "
        "đầu thanh điều hướng bên trái — nơi giải thích RAG bằng ví dụ dễ "
        "hiểu, hướng dẫn từng bước và giải nghĩa mọi tuỳ chọn cấu hình."
    )

    config = render_sidebar_config()

    st.subheader("🧭 Hành trình 4 bước của bạn")
    cols = st.columns(4)
    journey = [
        ("📄", "Tải tài liệu", "Document Upload", "Đưa tài liệu của bạn vào hệ thống để \"học\"."),
        ("💬", "Đặt câu hỏi", "Chat Interface", "Hỏi đáp dựa trên nội dung tài liệu vừa tải."),
        ("🔍", "Xem nguồn trả lời", "Retrieval Debug", "Kiểm chứng AI đã \"đọc\" đoạn nào để trả lời."),
        ("📊", "Theo dõi lịch sử", "Experiment Log", "Xem lại và so sánh các lần thử nghiệm."),
    ]
    for col, (icon, title, page, desc) in zip(cols, journey):
        with col:
            st.markdown(f"### {icon} {title}")
            st.caption(f"**Trang:** {page}")
            st.caption(desc)

    st.divider()

    st.subheader("⚙️ Cấu hình hiện tại của phiên làm việc")
    st.caption(
        "Các giá trị này được chọn ở sidebar bên trái và áp dụng cho mọi "
        "trang trong phiên làm việc của bạn — xem trang Hướng dẫn nếu bạn "
        "chưa rõ ý nghĩa của từng mục."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧠 LLM Model", config["ollama_model"])
    col2.metric("🧭 Embedding Model", config["embedding_model"])
    col3.metric("✂️ Chunk Size", config["chunk_size"])
    col4.metric("🔝 Top-K", config["top_k"])

    st.divider()

    st.subheader("🔌 Trạng thái OLLAMA (LLM cục bộ)")
    client = OllamaClient(model_name=config["ollama_model"])
    if client.is_available():
        st.success(f"✅ Đã kết nối tới OLLAMA tại `{client.base_url}` — sẵn sàng sử dụng.")
    else:
        st.warning(
            f"⚠️ Chưa kết nối được tới OLLAMA tại `{client.base_url}`. Hãy mở "
            "terminal và chạy `ollama serve` (xem chi tiết ở trang **🎓 Hướng "
            "dẫn sử dụng** → tab Hỏi đáp)."
        )

    st.info(
        "👈 Chọn một trang ở thanh điều hướng bên trái để bắt đầu: "
        "**🎓 Hướng dẫn sử dụng** (giải thích RAG & cách dùng dashboard), "
        "**Document Upload** (tải & index tài liệu), "
        "**Chat Interface** (hỏi đáp với RAG), "
        "**Retrieval Debug** (xem chunk truy xuất), "
        "hoặc **Experiment Log** (lịch sử thực nghiệm)."
    )


if __name__ == "__main__":
    main()
