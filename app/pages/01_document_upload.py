"""Trang 1 — Document Upload (design.md §2.6, Yêu cầu 10.3).

Triển khai: S1-PE-03 (phụ thuộc S1-PE-01, S1-PE-02, S1-DE-01).
Sprint 1: gọi pipeline.index_document() — hiện là STUB trả IndexingResult giả lập.
Sprint 2: thay bằng indexing thật khi S2-PE-01 hoàn thành.
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import streamlit as st

from src.models import IndexingResult, ChunkStrategy
from src.data.loader import DocumentLoader
from src.data.chunker import TextChunker
from src.embeddings.embedding_model import OllamaEmbeddingModel
from src.embeddings.vector_store import ChromaVectorStore
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import PromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Document Upload — RAG Research",
    page_icon="📄",
    layout="wide",
)


# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .upload-card {
        background: linear-gradient(135deg, #1e3a5f22, #3b82f622);
        border: 1px dashed #3b82f688;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-card {
        background: #0f2a1a;
        border: 1px solid #22c55e44;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .error-card {
        background: #2a0f0f;
        border: 1px solid #ef444444;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    # ── Đảm bảo config đã được khởi tạo ────────────────────────────────────
    if "config" not in st.session_state:
        st.session_state["config"] = {
            "ollama_model": "llama3",
            "embedding_model": "nomic-embed-text",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
        }
    if "indexed_docs" not in st.session_state:
        st.session_state["indexed_docs"] = []
    if "indexing_results" not in st.session_state:
        st.session_state["indexing_results"] = []

    cfg = st.session_state["config"]

    # ── Khởi tạo RAGPipeline và các component ──────────────────────────────
    loader = DocumentLoader()
    chunker = TextChunker(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 50),
    )
    embedding_model = OllamaEmbeddingModel(
        model_name=cfg.get("embedding_model", "nomic-embed-text")
    )
    vector_store = ChromaVectorStore(
        collection_name="rag_collection"
    )
    llm_client = OllamaClient(
        model_name=cfg.get("ollama_model", "llama3")
    )
    prompt_builder = PromptBuilder()

    pipeline = RAGPipeline(
        loader=loader,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        top_k=cfg.get("top_k", 5),
    )

    st.title("📄 Document Upload")
    st.markdown(
        "Upload tài liệu **(PDF, TXT, Markdown)** để index vào ChromaDB. "
        "Sau khi index xong, bạn có thể đặt câu hỏi trên trang **Chat Interface**."
    )

    # ── Thông báo Sprint 1 (stub) ────────────────────────────────────────────
    st.info(
        "ℹ️ **Sprint 1 — RAGPipeline integrated:** `index_document()` được gọi qua pipeline thật tích hợp với `DocumentLoader` chạy thật. "
        "Các công đoạn chia nhỏ, vector hóa và lưu trữ (ChromaDB) đang ở chế độ stub và sẽ được hoàn thiện ở Sprint 2.",
        icon="ℹ️",
    )

    st.markdown("---")

    # ── Cấu hình chunking hiện tại ───────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    col_cfg1.metric("⚙️ Chunk Size", f"{cfg['chunk_size']} ký tự")
    col_cfg2.metric("↔️ Chunk Overlap", f"{cfg['chunk_overlap']} ký tự")
    col_cfg3.metric("🤖 Embedding", cfg["embedding_model"])

    st.markdown("---")

    # ── File uploader ────────────────────────────────────────────────────────
    st.subheader("📂 Chọn tài liệu để upload")
    uploaded_files = st.file_uploader(
        label="Kéo và thả file vào đây, hoặc click để duyệt",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="document_uploader",
        help="Hỗ trợ: PDF (PyMuPDF), TXT, Markdown (.md). Tối đa 200 MB mỗi file.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file đã chọn:**")
        for f in uploaded_files:
            size_kb = f.size / 1024
            st.markdown(f"• `{f.name}` — {size_kb:.1f} KB")

        st.markdown("")

        if st.button(
            f"🚀 Index {len(uploaded_files)} tài liệu",
            type="primary",
            key="btn_index",
        ):
            results = []
            progress = st.progress(0, text="Đang khởi tạo...")

            # Đảm bảo thư mục lưu trữ file gốc vật lý tồn tại (design.md §1.2)
            os.makedirs("data/raw", exist_ok=True)

            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Đang xử lý: `{uploaded_file.name}` ({i+1}/{len(uploaded_files)})",
                )

                # Lưu file vật lý vào thư mục data/raw/
                file_path = os.path.join("data/raw", uploaded_file.name)
                try:
                    with open(file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                except Exception as e:
                    st.error(f"Không thể ghi file {uploaded_file.name} lên ổ đĩa: {e}")
                    continue

                with st.spinner(f"Indexing `{uploaded_file.name}`..."):
                    result = pipeline.index_document(file_path)
                    results.append((uploaded_file.name, result))

                    # Thêm vào danh sách tài liệu đã index
                    if result.success and uploaded_file.name not in st.session_state["indexed_docs"]:
                        st.session_state["indexed_docs"].append(uploaded_file.name)

            progress.progress(1.0, text="✅ Hoàn thành!")
            st.session_state["indexing_results"].extend(results)

            # ── Hiển thị kết quả ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Kết quả Indexing")

            success_count = sum(1 for _, r in results if r.success)
            fail_count = len(results) - success_count

            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Thành công", success_count)
            c2.metric("❌ Thất bại", fail_count)
            c3.metric(
                "📦 Tổng số chunk",
                sum(r.num_chunks for _, r in results if r.success),
            )

            for file_name, result in results:
                if result.success:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <strong>✅ {file_name}</strong><br/>
                            <small>
                                Doc ID: <code>{result.doc_id}</code> &nbsp;|&nbsp;
                                Chunks: <strong>{result.num_chunks}</strong> &nbsp;|&nbsp;
                                Collection: <code>{result.collection_name}</code>
                            </small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="error-card">
                            <strong>❌ {file_name}</strong><br/>
                            <small>Lỗi: {result.error_message}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ── Lịch sử tài liệu đã index ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Tài liệu đã index trong phiên này")

    indexed = st.session_state.get("indexed_docs", [])
    if indexed:
        for idx, doc_name in enumerate(indexed, 1):
            st.markdown(f"**{idx}.** `{doc_name}`")
    else:
        st.markdown(
            "_Chưa có tài liệu nào được index. Upload file để bắt đầu._"
        )

    if indexed and st.button("🗑️ Xóa danh sách (reset session)", key="btn_clear"):
        st.session_state["indexed_docs"] = []
        st.session_state["indexing_results"] = []
        st.rerun()


if __name__ == "__main__":
    main()
