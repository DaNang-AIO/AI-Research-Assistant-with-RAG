"""Trang 4 — Experiment Log (design.md §2.6, Yêu cầu 10.6).

Triển khai: S4-PE-05 — liệt kê sự kiện indexing/query theo thứ tự thời gian.
Sprint 1: đọc dữ liệu stub từ session_state (indexing_results, chat_history).
Sprint 4: tích hợp ExperimentTracker thật (S4-PE-01..S4-PE-05).
"""

import html
import datetime
import streamlit as st

st.set_page_config(
    page_title="Experiment Log — RAG Research",
    page_icon="📊",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .log-entry {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .log-icon {
        font-size: 20px;
        line-height: 1;
        flex-shrink: 0;
    }
    .log-body { flex: 1; }
    .log-title { color: #e2e8f0; font-weight: 600; margin-bottom: 2px; }
    .log-meta { color: #64748b; font-size: 12px; }
    .tag-index {
        background: #1e3a5f;
        color: #60a5fa;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .tag-query {
        background: #16543022;
        color: #4ade80;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 11px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _build_experiment_timeline() -> list:
    """
    Tổng hợp timeline thực nghiệm từ session_state.
    Sprint 4: thay bằng ExperimentTracker.get_summary() thật.
    """
    events = []

    # Sự kiện indexing từ trang Document Upload
    for file_name, result, timestamp in st.session_state.get("indexing_results", []):
        events.append({
            "type": "indexing",
            "icon": "📄",
            "title": f"Indexed: {file_name}",
            "details": (
                f"doc_id={result.doc_id} | "
                f"chunks={result.num_chunks} | "
                f"collection={result.collection_name} | "
                f"success={result.success}"
            ),
            "timestamp": timestamp,
            "success": result.success,
        })

    # Sự kiện query từ trang Chat Interface
    for turn in st.session_state.get("chat_history", []):
        events.append({
            "type": "query",
            "icon": "💬",
            "title": f'Query: "{turn["question"][:60]}{"…" if len(turn["question"]) > 60 else ""}"',
            "details": (
                f"contexts={len(turn.get('contexts', []))} | "
                f"latency={turn.get('latency_ms', 0):.0f} ms | "
                f"ts={turn.get('timestamp', '—')}"
            ),
            "timestamp": turn.get("timestamp", "—"),
            "success": True,
        })

    return events


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

    st.title("📊 Experiment Log")
    st.markdown(
        "Theo dõi **lịch sử thực nghiệm** của phiên làm việc hiện tại: "
        "các sự kiện indexing và querying theo thứ tự thời gian."
    )

    st.info(
        "⚠️ **Sprint 1 — Demo mode:** Dữ liệu đọc từ `st.session_state` "
        "(indexing_results + chat_history). "
        "`ExperimentTracker` thật sẽ được tích hợp từ Sprint 4 (S4-PE-04, S4-PE-05).",
        icon="🔧",
    )

    st.markdown("---")

    # ── Tóm tắt phiên ───────────────────────────────────────────────────────
    cfg = st.session_state["config"]
    indexing_results = st.session_state.get("indexing_results", [])
    chat_history = st.session_state.get("chat_history", [])
    indexed_docs = st.session_state.get("indexed_docs", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Tài liệu đã index", len(indexed_docs))
    col2.metric("💬 Lượt truy vấn", len(chat_history))
    col3.metric(
        "📦 Tổng số chunk",
        sum(r.num_chunks for _, r, _ts in indexing_results if r.success),
    )
    avg_latency = (
        sum(t.get("latency_ms", 0) for t in chat_history) / len(chat_history)
        if chat_history else 0
    )
    col4.metric("⏱ Latency TB", f"{avg_latency:.0f} ms")

    st.markdown("---")

    # ── Config hiện tại của phiên ────────────────────────────────────────────
    st.subheader("⚙️ Cấu hình Pipeline (phiên hiện tại)")
    cfg_cols = st.columns(5)
    cfg_cols[0].markdown(f"**LLM Model**\n\n`{cfg.get('ollama_model', '—')}`")
    cfg_cols[1].markdown(f"**Embedding**\n\n`{cfg.get('embedding_model', '—')}`")
    cfg_cols[2].markdown(f"**Chunk Size**\n\n`{cfg.get('chunk_size', '—')} ký tự`")
    cfg_cols[3].markdown(f"**Overlap**\n\n`{cfg.get('chunk_overlap', '—')} ký tự`")
    cfg_cols[4].markdown(f"**Top-K**\n\n`{cfg.get('top_k', '—')}`")

    st.markdown("---")

    # ── Timeline sự kiện ─────────────────────────────────────────────────────
    st.subheader("🕐 Timeline Thực Nghiệm")

    events = _build_experiment_timeline()

    if not events:
        st.markdown(
            """
            <div style="text-align:center; padding:40px; color:#475569">
                <div style="font-size:48px">📋</div>
                <p>Chưa có sự kiện nào được ghi lại.</p>
                <p style="font-size:13px">
                    Upload tài liệu trên trang <strong>Document Upload</strong> hoặc
                    đặt câu hỏi trên trang <strong>Chat Interface</strong> để bắt đầu.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Bộ lọc loại sự kiện
        filter_type = st.radio(
            "Lọc theo loại sự kiện:",
            options=["Tất cả", "Indexing", "Query"],
            horizontal=True,
            key="log_filter",
        )

        if filter_type == "Indexing":
            filtered_events = [e for e in events if e["type"] == "indexing"]
        elif filter_type == "Query":
            filtered_events = [e for e in events if e["type"] == "query"]
        else:
            filtered_events = events

        st.caption(f"Hiển thị {len(filtered_events)}/{len(events)} sự kiện")
        st.markdown("")

        def _ts_key(e: dict) -> datetime.datetime:
            try:
                return datetime.datetime.strptime(e["timestamp"], "%H:%M:%S")
            except (ValueError, KeyError):
                return datetime.datetime.min

        for i, event in enumerate(sorted(filtered_events, key=_ts_key, reverse=True)):
            tag_class = "tag-index" if event["type"] == "indexing" else "tag-query"
            tag_label = "INDEXING" if event["type"] == "indexing" else "QUERY"
            status_icon = "✅" if event["success"] else "❌"

            st.markdown(
                f"""
                <div class="log-entry">
                    <div class="log-icon">{event['icon']}</div>
                    <div class="log-body">
                        <div class="log-title">
                            {status_icon} {html.escape(event['title'])}
                            &nbsp;<span class="{tag_class}">{tag_label}</span>
                        </div>
                        <div class="log-meta">
                            ⏰ {event['timestamp']} &nbsp;|&nbsp; {html.escape(event['details'])}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Biểu đồ phân bố loại sự kiện ────────────────────────────────────────
    if events:
        st.markdown("---")
        st.subheader("📈 Thống kê phiên làm việc")

        idx_count = sum(1 for e in events if e["type"] == "indexing")
        qry_count = sum(1 for e in events if e["type"] == "query")

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown("**Phân bố sự kiện**")
            st.bar_chart(
                {"Indexing": idx_count, "Query": qry_count},
                height=200,
            )
        with stat_col2:
            st.markdown("**Latency các lượt query (ms)**")
            if chat_history:
                latency_data = {
                    f"Q{i+1}": t.get("latency_ms", 0)
                    for i, t in enumerate(chat_history)
                }
                st.bar_chart(latency_data, height=200)
            else:
                st.info("Chưa có dữ liệu latency.")

    # ── Placeholder: so sánh session (Sprint 4) ─────────────────────────────
    st.markdown("---")
    with st.expander("🔬 So sánh phiên thực nghiệm (Sprint 4)", expanded=False):
        st.info(
            "Tính năng **compare_sessions()** sẽ được triển khai ở Sprint 4 "
            "(S4-PE-03). Cho phép so sánh 2 phiên thực nghiệm để đánh giá "
            "ảnh hưởng của các tham số (chunk_size, top_k, model...) lên chất lượng.",
            icon="🔧",
        )
        c1, c2 = st.columns(2)
        c1.selectbox("Phiên A", options=["— chưa có —"], disabled=True, key="session_a")
        c2.selectbox("Phiên B", options=["— chưa có —"], disabled=True, key="session_b")
        st.button("📊 So sánh", disabled=True, key="btn_compare")


if __name__ == "__main__":
    main()
