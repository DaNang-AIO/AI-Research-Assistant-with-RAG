"""Trang Hướng Dẫn Sử Dụng — "tutor" giúp người mới làm quen với RAG và dashboard.

Bổ sung cho UI/UX: giải thích RAG bằng ngôn ngữ đơn giản, hướng dẫn từng bước,
giải nghĩa các tuỳ chọn cấu hình ở sidebar và từ điển thuật ngữ — để một
người hoàn toàn mới vẫn có thể tự tin sử dụng dashboard.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_config
from src.generation.llm_client import OllamaClient

st.set_page_config(page_title="Hướng dẫn sử dụng", page_icon="🎓", layout="wide")

config = render_sidebar_config()

st.title("🎓 Hướng Dẫn Sử Dụng — Bắt Đầu Từ Đây")
st.markdown(
    "Chào mừng bạn đến với **AI Research Assistant with RAG** — một phòng "
    "thí nghiệm nhỏ để bạn *nhìn thấy* cách một trợ lý AI trả lời câu hỏi "
    "dựa trên **chính tài liệu của bạn**, thay vì chỉ dựa vào kiến thức có sẵn."
)

tab_overview, tab_quickstart, tab_config, tab_glossary, tab_faq = st.tabs(
    [
        "🧭 RAG là gì?",
        "🚀 Bắt đầu nhanh",
        "⚙️ Giải thích cấu hình",
        "📚 Từ điển thuật ngữ",
        "❓ Hỏi đáp",
    ]
)

# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("RAG (Retrieval-Augmented Generation) là gì?")
    st.markdown(
        """
Hãy tưởng tượng bạn đi thi theo kiểu **"thi mở sách"**:

- Nếu chỉ dựa vào trí nhớ, bạn có thể trả lời sai hoặc thiếu chi tiết.
- Nhưng nếu được **mở đúng trang sách có liên quan** trước khi viết câu trả lời,
  bạn sẽ trả lời chính xác và có dẫn chứng hơn rất nhiều.

**RAG hoạt động đúng theo tinh thần đó với AI:** trước khi mô hình ngôn ngữ
(LLM) trả lời câu hỏi của bạn, hệ thống sẽ tự động **tìm những đoạn tài liệu
liên quan nhất** và "đưa" chúng cho LLM tham khảo. Nhờ vậy, AI có thể trả lời
dựa trên tài liệu thật của bạn — kể cả những thông tin nó chưa từng được
huấn luyện.
        """
    )

    st.subheader("Hệ thống này hoạt động theo 4 bước")
    cols = st.columns(4)
    steps = [
        ("📄", "1. Nạp tài liệu", "Bạn tải lên file (.pdf/.txt/.md). Hệ thống đọc và "
                                   "cắt tài liệu thành các đoạn nhỏ (chunk) rồi lưu lại."),
        ("💬", "2. Đặt câu hỏi", "Bạn gõ câu hỏi. Hệ thống tìm các đoạn tài liệu liên quan "
                                   "nhất và nhờ AI soạn câu trả lời dựa trên chúng."),
        ("🔍", "3. Kiểm chứng nguồn", "Bạn xem chính xác đoạn nào đã được dùng để trả lời, "
                                        "kèm điểm liên quan (similarity score)."),
        ("📊", "4. Theo dõi & so sánh", "Bạn xem lại lịch sử các lần thử — đổi cấu hình, "
                                          "so sánh kết quả để hiểu RAG rõ hơn."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.info(
        "💡 4 bước trên tương ứng với 4 trang ở thanh điều hướng bên trái: "
        "**Document Upload → Chat Interface → Retrieval Debug → Experiment Log**."
    )

# ---------------------------------------------------------------------------
with tab_quickstart:
    st.subheader("Thử ngay trong 3 bước")

    with st.expander("**Bước 1 — Tải một tài liệu lên** (trang 📄 Document Upload)", expanded=True):
        st.markdown(
            """
1. Mở trang **Document Upload** ở thanh điều hướng bên trái.
2. Bấm vào ô tải file và chọn một file `.txt`, `.md` hoặc `.pdf` bất kỳ
   (ví dụ: một bài viết, ghi chú, tài liệu môn học...).
3. Chờ vài giây — hệ thống sẽ báo số lượng đoạn (chunk) đã tạo ra và trạng thái thành công.
            """
        )

    with st.expander("**Bước 2 — Đặt câu hỏi về tài liệu** (trang 💬 Chat Interface)"):
        st.markdown(
            """
1. Mở trang **Chat Interface**.
2. Gõ một câu hỏi liên quan đến nội dung tài liệu bạn vừa tải lên, ví dụ:
   *"Tài liệu này nói về điều gì?"*
3. AI sẽ trả lời, kèm thông tin về model đã dùng và số nguồn tham khảo.
            """
        )

    with st.expander("**Bước 3 — Xem AI đã \"đọc\" gì để trả lời** (trang 🔍 Retrieval Debug)"):
        st.markdown(
            """
1. Mở trang **Retrieval Debug** ngay sau khi đặt câu hỏi.
2. Bạn sẽ thấy chính xác những đoạn tài liệu mà hệ thống đã chọn ra để
   "đưa" cho AI tham khảo, kèm **điểm liên quan (similarity score)** và
   **vị trí của đoạn đó trong tài liệu gốc**.
            """
        )

    st.success(
        "🎉 Vậy là xong! Bạn vừa trải nghiệm trọn vẹn một vòng RAG: "
        "Nạp tài liệu → Hỏi đáp → Kiểm chứng nguồn."
    )

    st.warning(
        "⚠️ **Lưu ý cho phiên bản hiện tại (Sprint 1):** phần \"đọc tài liệu\" đã "
        "chạy thật, nhưng phần chia nhỏ / tạo embedding / trả lời vẫn đang ở chế độ "
        "**minh hoạ (giả lập)** — đây là một dự án học tập được xây dựng dần theo "
        "từng sprint. Các phần còn lại sẽ trở thành thật ở Sprint 2 và Sprint 3."
    )

# ---------------------------------------------------------------------------
with tab_config:
    st.subheader("Sidebar \"⚙️ Cấu hình Pipeline\" có ý nghĩa gì?")
    st.markdown(
        "Mọi trang đều có chung một sidebar cấu hình bên trái — thay đổi ở đây "
        "sẽ áp dụng cho toàn bộ phiên làm việc của bạn:"
    )

    st.markdown("##### 🧠 LLM Model")
    st.caption(
        "**LLM (Large Language Model)** là \"bộ não\" sinh ra câu trả lời — giống "
        "như người viết bài thi. Đây là mô hình ngôn ngữ chạy **cục bộ trên máy "
        "bạn** thông qua OLLAMA (không gửi dữ liệu lên cloud). Đổi model để so "
        "sánh xem model nào trả lời tốt hơn với tài liệu của bạn."
    )

    st.markdown("##### 🧭 Embedding Model")
    st.caption(
        "**Embedding** là quá trình \"dịch\" văn bản thành dãy số (vector) để máy "
        "tính so sánh được mức độ giống nhau về *ý nghĩa* giữa các đoạn văn bản — "
        "giống như một chiếc la bàn giúp hệ thống tìm đúng đoạn liên quan đến câu hỏi."
    )

    st.markdown("##### ✂️ Chunk Size")
    st.caption(
        "Tài liệu dài được cắt thành từng **đoạn nhỏ (chunk)** trước khi xử lý — "
        "giống như chia một quyển sách thành từng trang để dễ tra cứu. "
        "**Số lớn** → mỗi đoạn dài hơn, chứa nhiều ngữ cảnh hơn nhưng có thể lẫn "
        "thông tin không liên quan. **Số nhỏ** → đoạn ngắn, tập trung hơn nhưng dễ "
        "mất ngữ cảnh xung quanh."
    )

    st.markdown("##### 🔝 Top-K Retrieval")
    st.caption(
        "Số lượng đoạn tài liệu **liên quan nhất** mà hệ thống sẽ lấy ra để đưa "
        "cho AI tham khảo trước khi trả lời — giống như việc bạn đánh dấu trước "
        "*K trang sách* hữu ích nhất trước khi vào phòng thi."
    )

# ---------------------------------------------------------------------------
with tab_glossary:
    st.subheader("Từ điển thuật ngữ RAG (giải thích ngắn gọn)")

    glossary = [
        ("Document (Tài liệu)", "File gốc bạn tải lên — ví dụ một file PDF, ghi chú Markdown hay văn bản thuần."),
        ("Chunk (Đoạn)", "Một mẩu nhỏ được cắt ra từ tài liệu gốc, đủ ngắn để xử lý nhưng vẫn giữ được ngữ cảnh."),
        ("Embedding (Vector nhúng)", "Một dãy số đại diện cho *ý nghĩa* của một đoạn văn bản — hai đoạn có ý nghĩa giống nhau sẽ có vector \"gần nhau\"."),
        ("Vector Store (Kho lưu vector)", "Cơ sở dữ liệu chuyên lưu các embedding để có thể tìm kiếm theo độ tương đồng ngữ nghĩa một cách nhanh chóng."),
        ("Retrieval (Truy xuất)", "Bước tìm ra những đoạn tài liệu liên quan nhất tới câu hỏi, dựa trên việc so sánh embedding."),
        ("Similarity Score (Điểm tương đồng)", "Một con số trong khoảng 0.0–1.0 cho biết một đoạn tài liệu *liên quan đến câu hỏi* đến mức nào — càng gần 1.0 càng liên quan."),
        ("Prompt", "Đoạn văn bản hoàn chỉnh (gồm câu hỏi + các đoạn tài liệu liên quan + chỉ dẫn) được gửi cho LLM để nó soạn câu trả lời."),
        ("LLM (Large Language Model)", "Mô hình ngôn ngữ lớn — \"bộ não\" sinh văn bản, đọc prompt và viết ra câu trả lời."),
        ("OLLAMA", "Phần mềm giúp chạy các LLM **ngay trên máy tính của bạn**, không cần gửi dữ liệu lên dịch vụ đám mây."),
        ("Latency (Độ trễ)", "Thời gian (tính bằng mili-giây) mà hệ thống cần để xử lý và trả lời một yêu cầu."),
    ]
    for term, definition in glossary:
        with st.expander(term):
            st.write(definition)

# ---------------------------------------------------------------------------
with tab_faq:
    st.subheader("Câu hỏi thường gặp")

    with st.expander("Tại sao câu trả lời ở Chat Interface có vẻ chưa \"thông minh\"?"):
        st.write(
            "Dự án này được xây dựng dần theo từng **sprint** để bạn có thể quan "
            "sát từng thành phần của RAG. Ở Sprint 1, phần \"sinh câu trả lời "
            "thật\" chưa được nối — bạn đang thấy câu trả lời **minh hoạ (giả "
            "lập)** để hình dung luồng hoạt động. Câu trả lời thật từ LLM cục bộ "
            "sẽ xuất hiện từ Sprint 3."
        )

    with st.expander("OLLAMA là gì và tại sao tôi cần nó?"):
        st.write(
            "OLLAMA là phần mềm chạy các mô hình AI **ngay trên máy tính của bạn** "
            "— giúp toàn bộ hệ thống hoạt động mà không cần gửi tài liệu hay câu "
            "hỏi của bạn lên Internet. Hãy chắc chắn bạn đã cài đặt và khởi động "
            "OLLAMA (`ollama serve`) cũng như tải về các model cần thiết "
            "(`ollama pull llama3`, `ollama pull nomic-embed-text`)."
        )

    with st.expander("Tôi nên thử với loại tài liệu nào trước?"):
        st.write(
            "Hãy bắt đầu với một file `.txt` hoặc `.md` ngắn (vài đoạn văn) mà "
            "bạn đã quen thuộc nội dung — như vậy bạn sẽ dễ dàng nhận ra liệu "
            "câu trả lời và các đoạn được truy xuất có hợp lý hay không."
        )

    st.markdown("---")
    st.subheader("Kiểm tra nhanh: OLLAMA đã sẵn sàng chưa?")
    if st.button("🔄 Kiểm tra kết nối OLLAMA", key="guide_check_ollama"):
        client = OllamaClient(model_name=config["ollama_model"])
        if client.is_available():
            models = client.list_models()
            st.success(f"✅ Đã kết nối tới OLLAMA tại `{client.base_url}`.")
            if models:
                st.caption("Các model đã sẵn sàng: " + ", ".join(f"`{m}`" for m in models))
            else:
                st.caption(
                    "Chưa thấy model nào — hãy chạy `ollama pull llama3` và "
                    "`ollama pull nomic-embed-text`."
                )
        else:
            st.error(
                "❌ Chưa kết nối được tới OLLAMA. Hãy mở terminal và chạy "
                "`ollama serve`, rồi bấm kiểm tra lại."
            )
