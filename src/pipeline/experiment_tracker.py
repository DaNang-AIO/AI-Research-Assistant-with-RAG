"""ExperimentTracker (design.md §2.3).

Triển khai: S4-PE-01 (log_indexing, log_query, get_summary), S4-PE-02
(save_session/load_session — round-trip, Property 11), và S4-PE-03
(compare_sessions).
"""

import datetime
import json
import os
import uuid
from typing import Any, Dict, List

from src.models import ExperimentLog, ScoredChunk


class ExperimentTracker:
    """Ghi lại và lưu trữ các thực nghiệm để so sánh và phân tích.

    Dùng trong notebooks và Streamlit dashboard để theo dõi kết quả thực
    nghiệm (Yêu cầu 8) — mỗi sự kiện indexing/query trong phiên hiện tại
    được gom vào `_current_session`, có thể lưu ra JSON và tải lại nguyên
    vẹn (round-trip — Property 11).
    """

    def __init__(self, log_dir: str = "experiments/logs"):
        self.log_dir = log_dir
        self._current_session: List[ExperimentLog] = []

    def log_indexing(
        self,
        doc_id: str,
        chunk_strategy: str,
        chunk_size: int,
        num_chunks: int,
        latency_ms: float,
    ) -> None:
        """Ghi lại thông tin một lần indexing (Yêu cầu 8.1)."""
        self._current_session.append(
            ExperimentLog(
                experiment_id=str(uuid.uuid4()),
                event_type="indexing",
                params={
                    "doc_id": doc_id,
                    "chunk_strategy": chunk_strategy,
                    "chunk_size": chunk_size,
                },
                result={
                    "num_chunks": num_chunks,
                    "latency_ms": latency_ms,
                },
            )
        )

    def log_query(
        self,
        question: str,
        top_k: int,
        contexts: List[ScoredChunk],
        answer: str,
        latency_ms: float,
    ) -> None:
        """Ghi lại thông tin một lần query (Yêu cầu 8.2).

        `contexts` được chuyển thành dict JSON-serializable ngay khi ghi
        log để `save_session()`/`load_session()` khôi phục đúng cùng một
        cấu trúc dữ liệu (round-trip — Property 11).
        """
        self._current_session.append(
            ExperimentLog(
                experiment_id=str(uuid.uuid4()),
                event_type="query",
                params={
                    "question": question,
                    "top_k": top_k,
                },
                result={
                    "answer": answer,
                    "latency_ms": latency_ms,
                    "contexts": [self._scored_chunk_to_dict(sc) for sc in contexts],
                },
            )
        )

    def save_session(self, session_name: str) -> str:
        """Lưu session hiện tại ra file JSON, trả về đường dẫn file (Yêu cầu 8.3)."""
        os.makedirs(self.log_dir, exist_ok=True)
        file_path = os.path.join(self.log_dir, f"{session_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                [self._log_to_dict(log) for log in self._current_session],
                f,
                ensure_ascii=False,
                indent=2,
            )
        return file_path

    def load_session(self, session_name: str) -> List[ExperimentLog]:
        """Tải lại một session đã lưu — khôi phục đúng danh sách `ExperimentLog`
        ban đầu (round-trip, Yêu cầu 8.4, Property 11)."""
        file_path = os.path.join(self.log_dir, f"{session_name}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_logs = json.load(f)
        return [self._dict_to_log(raw) for raw in raw_logs]

    def compare_sessions(self, session_a: str, session_b: str) -> Dict[str, Any]:
        """So sánh hai session thực nghiệm, trả về dictionary chứa các metrics
        đối chiếu giữa hai phiên (Yêu cầu 8.5)."""
        summary_a = self._summarize(self.load_session(session_a))
        summary_b = self._summarize(self.load_session(session_b))
        return {
            session_a: summary_a,
            session_b: summary_b,
            "diff": {
                metric: summary_b[metric] - summary_a[metric]
                for metric in summary_a
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Tổng hợp thống kê của session hiện tại: tổng số sự kiện, tổng số
        lần indexing, tổng số lần query, latency trung bình (Yêu cầu 8.6)."""
        return self._summarize(self._current_session)

    @staticmethod
    def _summarize(logs: List[ExperimentLog]) -> Dict[str, Any]:
        indexing_logs = [log for log in logs if log.event_type == "indexing"]
        query_logs = [log for log in logs if log.event_type == "query"]

        def _avg_latency(entries: List[ExperimentLog]) -> float:
            if not entries:
                return 0.0
            latencies = [entry.result.get("latency_ms", 0.0) for entry in entries]
            return sum(latencies) / len(latencies)

        return {
            "total_events": len(logs),
            "num_indexing": len(indexing_logs),
            "num_query": len(query_logs),
            "avg_indexing_latency_ms": _avg_latency(indexing_logs),
            "avg_query_latency_ms": _avg_latency(query_logs),
        }

    @staticmethod
    def _scored_chunk_to_dict(scored_chunk: ScoredChunk) -> Dict[str, Any]:
        chunk = scored_chunk.chunk
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "score": scored_chunk.score,
            "rank": scored_chunk.rank,
        }

    @staticmethod
    def _log_to_dict(log: ExperimentLog) -> Dict[str, Any]:
        return {
            "experiment_id": log.experiment_id,
            "event_type": log.event_type,
            "params": log.params,
            "result": log.result,
            "timestamp": log.timestamp.isoformat(),
        }

    @staticmethod
    def _dict_to_log(raw: Dict[str, Any]) -> ExperimentLog:
        return ExperimentLog(
            experiment_id=raw["experiment_id"],
            event_type=raw["event_type"],
            params=raw["params"],
            result=raw["result"],
            timestamp=datetime.datetime.fromisoformat(raw["timestamp"]),
        )
