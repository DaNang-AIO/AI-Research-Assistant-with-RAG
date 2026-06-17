"""Component metrics_widget.py — widget hiển thị metrics & stats (design.md §1.2).

Triển khai: S4-PE-05 — component trang Experiment Log.
Tái sử dụng được ở nhiều trang Streamlit.
"""

import streamlit as st
from typing import Dict, Any, List, Optional


def render_pipeline_status(
    ollama_available: bool = False,
    indexed_doc_count: int = 0,
    collection_name: str = "rag_collection",
) -> None:
    """
    Render trạng thái hệ thống (OLLAMA, ChromaDB, số tài liệu đã index).

    Args:
        ollama_available: OLLAMA server có đang chạy không
        indexed_doc_count: Số tài liệu đã index
        collection_name: Tên ChromaDB collection đang dùng
    """
    st.subheader("🔌 Trạng thái hệ thống")

    col1, col2, col3 = st.columns(3)

    with col1:
        if ollama_available:
            st.markdown(
                '<div style="background:#0a1f13;border:1px solid #22c55e44;'
                'border-radius:10px;padding:12px;text-align:center">'
                '<span style="font-size:24px">✅</span><br/>'
                '<strong style="color:#4ade80">OLLAMA</strong><br/>'
                '<small style="color:#64748b">Online</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#1f0a0a;border:1px solid #ef444444;'
                'border-radius:10px;padding:12px;text-align:center">'
                '<span style="font-size:24px">❌</span><br/>'
                '<strong style="color:#f87171">OLLAMA</strong><br/>'
                '<small style="color:#64748b">Offline</small></div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #334155;'
            f'border-radius:10px;padding:12px;text-align:center">'
            f'<span style="font-size:24px">📚</span><br/>'
            f'<strong style="color:#60a5fa">{indexed_doc_count}</strong><br/>'
            f'<small style="color:#64748b">Tài liệu đã index</small></div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #334155;'
            f'border-radius:10px;padding:12px;text-align:center">'
            f'<span style="font-size:24px">🗄️</span><br/>'
            f'<strong style="color:#c084fc">{collection_name}</strong><br/>'
            f'<small style="color:#64748b">ChromaDB collection</small></div>',
            unsafe_allow_html=True,
        )


def render_indexing_metrics(results: List[tuple]) -> None:
    """
    Render bảng thống kê kết quả indexing.

    Args:
        results: List of (file_name, IndexingResult) tuples
    """
    if not results:
        st.info("Chưa có tài liệu nào được index trong phiên này.")
        return

    success_list = [(n, r) for n, r in results if r.success]
    fail_list = [(n, r) for n, r in results if not r.success]
    total_chunks = sum(r.num_chunks for _, r in success_list)

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Thành công", len(success_list))
    col2.metric("❌ Thất bại", len(fail_list))
    col3.metric("📦 Tổng chunks", total_chunks)

    if success_list:
        st.markdown("**Chi tiết indexing:**")
        rows = [
            {
                "File": name,
                "Doc ID": r.doc_id,
                "Chunks": r.num_chunks,
                "Collection": r.collection_name,
                "Status": "✅" if r.success else "❌",
            }
            for name, r in results
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_query_metrics(chat_history: List[Dict[str, Any]]) -> None:
    """
    Render thống kê các lần query: latency histogram, context count.

    Args:
        chat_history: Danh sách turn chat từ st.session_state["chat_history"]
    """
    if not chat_history:
        st.info("Chưa có lượt truy vấn nào trong phiên này.")
        return

    latencies = [t.get("latency_ms", 0) for t in chat_history]
    ctx_counts = [len(t.get("contexts", [])) for t in chat_history]
    avg_lat = sum(latencies) / len(latencies)
    avg_ctx = sum(ctx_counts) / len(ctx_counts)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💬 Tổng lượt hỏi", len(chat_history))
    col2.metric("⏱ Latency TB", f"{avg_lat:.0f} ms")
    col3.metric("⏱ Latency Max", f"{max(latencies):.0f} ms")
    col4.metric("📦 Context TB", f"{avg_ctx:.1f} chunks")

    # Biểu đồ latency theo từng turn
    if latencies:
        st.markdown("**Latency theo từng lượt query:**")
        st.bar_chart(
            {f"Q{i+1}": lat for i, lat in enumerate(latencies)},
            height=200,
        )


def render_experiment_summary(summary: Optional[Dict[str, Any]] = None) -> None:
    """
    Render tóm tắt phiên thực nghiệm (từ ExperimentTracker.get_summary()).

    Args:
        summary: Dict từ ExperimentTracker.get_summary(). None = dùng data stub.
    """
    st.subheader("📋 Tóm tắt phiên thực nghiệm")

    if summary is None:
        # Sprint 1 stub: tổng hợp từ session_state
        indexed_docs = st.session_state.get("indexed_docs", [])
        chat_history = st.session_state.get("chat_history", [])
        indexing_results = st.session_state.get("indexing_results", [])
        cfg = st.session_state.get("config", {})

        summary = {
            "total_indexing_events": len(indexing_results),
            "total_query_events": len(chat_history),
            "total_docs_indexed": len(indexed_docs),
            "total_chunks": sum(
                r.num_chunks for _, r in indexing_results if r.success
            ),
            "avg_latency_ms": (
                sum(t.get("latency_ms", 0) for t in chat_history) / len(chat_history)
                if chat_history else 0
            ),
            "config_snapshot": cfg,
        }

    # Hiển thị
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Indexing events", summary.get("total_indexing_events", 0))
    col2.metric("💬 Query events", summary.get("total_query_events", 0))
    col3.metric("⏱ Avg latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")

    if summary.get("config_snapshot"):
        with st.expander("⚙️ Config snapshot"):
            st.json(summary["config_snapshot"])
