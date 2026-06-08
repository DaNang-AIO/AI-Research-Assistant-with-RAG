"""Metrics widget dùng chung cho trang Experiment Log (design.md §2.6, Yêu cầu 10.6).

Triển khai: S4-PE-05 — hiển thị thống kê tổng hợp (`get_summary()`), danh sách
sự kiện indexing/query theo thứ tự thời gian, và kết quả `compare_sessions()`
giữa hai phiên đã lưu.
"""

from typing import Any, Dict, List

import streamlit as st

from src.models import ExperimentLog


def render_summary(summary: Dict[str, Any]) -> None:
    """Hiển thị thống kê tổng hợp của phiên hiện tại (`ExperimentTracker.get_summary()`)."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số sự kiện", summary["total_events"])
    col2.metric("Lần indexing", summary["num_indexing"])
    col3.metric("Lần query", summary["num_query"])

    col4, col5 = st.columns(2)
    col4.metric("Latency indexing TB (ms)", f"{summary['avg_indexing_latency_ms']:.1f}")
    col5.metric("Latency query TB (ms)", f"{summary['avg_query_latency_ms']:.1f}")


def render_timeline(logs: List[ExperimentLog]) -> None:
    """Hiển thị danh sách sự kiện thực nghiệm theo thứ tự thời gian (Yêu cầu 10.6).

    Sự kiện mới nhất hiển thị trước để người dùng thấy ngay hoạt động gần đây.
    """
    if not logs:
        st.write(
            "Chưa có sự kiện thực nghiệm nào trong phiên này. Hãy thử upload "
            "tài liệu ở **Document Upload** hoặc đặt câu hỏi ở **Chat Interface** "
            "— mỗi thao tác sẽ tự động được ghi lại ở đây."
        )
        return

    for log in sorted(logs, key=lambda entry: entry.timestamp, reverse=True):
        timestamp = f"{log.timestamp:%H:%M:%S}"
        if log.event_type == "indexing":
            params, result = log.params, log.result
            st.markdown(
                f"`{timestamp}` 📥 **indexing** — tài liệu `{params.get('doc_id', '?')}` "
                f"→ {result.get('num_chunks', 0)} chunks "
                f"(chiến lược `{params.get('chunk_strategy', '?')}`, "
                f"chunk_size={params.get('chunk_size', '?')}) · "
                f"{result.get('latency_ms', 0.0):.1f} ms"
            )
        elif log.event_type == "query":
            params, result = log.params, log.result
            num_contexts = len(result.get("contexts", []))
            st.markdown(
                f"`{timestamp}` 💬 **query** — “{params.get('question', '?')}” "
                f"(top_k={params.get('top_k', '?')}) → {num_contexts} nguồn tham chiếu · "
                f"{result.get('latency_ms', 0.0):.1f} ms"
            )
            with st.expander("Xem câu trả lời"):
                st.write(result.get("answer", ""))
        else:
            st.markdown(f"`{timestamp}` — sự kiện `{log.event_type}` (id=`{log.experiment_id}`)")


def render_comparison(comparison: Dict[str, Any], session_a: str, session_b: str) -> None:
    """Hiển thị kết quả `compare_sessions()` — đối chiếu metrics giữa hai phiên đã lưu."""
    col_a, col_b, col_diff = st.columns(3)
    for col, name, key in (
        (col_a, session_a, session_a),
        (col_b, session_b, session_b),
        (col_diff, "Chênh lệch (B − A)", "diff"),
    ):
        metrics = comparison[key]
        with col:
            st.markdown(f"**{name}**")
            st.write(f"Tổng sự kiện: {metrics['total_events']}")
            st.write(f"Indexing: {metrics['num_indexing']} · Query: {metrics['num_query']}")
            st.write(f"Latency indexing TB: {metrics['avg_indexing_latency_ms']:.1f} ms")
            st.write(f"Latency query TB: {metrics['avg_query_latency_ms']:.1f} ms")
