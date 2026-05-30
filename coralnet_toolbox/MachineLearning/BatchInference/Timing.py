"""Lightweight timing helpers for batch inference throughput diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class BatchTimingRecord:
    """Timing for one model call batch."""

    batch_size: int
    is_video: bool
    decode_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    gate_wait_seconds: float = 0.0

    @property
    def measured_seconds(self) -> float:
        return (
            self.decode_seconds
            + self.inference_seconds
            + self.postprocess_seconds
            + self.gate_wait_seconds
        )


@dataclass(slots=True)
class BatchInferenceTiming:
    """Aggregates per-batch timing records into a compact summary."""

    started_at: float = field(default_factory=time.perf_counter)
    records: list[BatchTimingRecord] = field(default_factory=list)

    def add_record(self, record: BatchTimingRecord) -> None:
        self.records.append(record)

    def summary(self) -> dict[str, float | int]:
        batches = len(self.records)
        items = sum(max(0, int(record.batch_size)) for record in self.records)
        video_items = sum(
            max(0, int(record.batch_size))
            for record in self.records
            if record.is_video
        )
        decode_seconds = sum(record.decode_seconds for record in self.records)
        inference_seconds = sum(record.inference_seconds for record in self.records)
        postprocess_seconds = sum(record.postprocess_seconds for record in self.records)
        gate_wait_seconds = sum(record.gate_wait_seconds for record in self.records)
        wall_seconds = max(0.0, time.perf_counter() - self.started_at)

        return {
            "batches": batches,
            "items": items,
            "video_items": video_items,
            "image_items": max(0, items - video_items),
            "avg_batch_size": (items / batches) if batches else 0.0,
            "decode_seconds": decode_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
            "gate_wait_seconds": gate_wait_seconds,
            "measured_seconds": sum(record.measured_seconds for record in self.records),
            "wall_seconds": wall_seconds,
            "items_per_second": (items / wall_seconds) if wall_seconds > 0 else 0.0,
            "model_items_per_second": (items / inference_seconds) if inference_seconds > 0 else 0.0,
        }

    @staticmethod
    def human_summary(summary: dict[str, float | int]) -> str:
        if not summary or not summary.get("items"):
            return "Batch inference timing: no processed items."
        return (
            "Batch inference timing: "
            f"{int(summary['items'])} items in {summary['wall_seconds']:.2f}s "
            f"({summary['items_per_second']:.2f} items/s), "
            f"decode={summary['decode_seconds']:.2f}s, "
            f"model={summary['inference_seconds']:.2f}s, "
            f"post={summary['postprocess_seconds']:.2f}s, "
            f"ui_gate={summary['gate_wait_seconds']:.2f}s, "
            f"avg_batch={summary['avg_batch_size']:.1f}."
        )