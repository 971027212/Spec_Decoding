from __future__ import annotations

import csv
import json
import math
import statistics
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(value: int | float) -> float:
    return float(value) / 1_000_000.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


class TimingRecorder:
    """Collects per-event timings for one generation run."""

    def __init__(
        self,
        run_id: str | None = None,
        mode: str | None = None,
        prompt_id: int | None = None,
        run_index: int | None = None,
        warmup: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.context: dict[str, Any] = {
            "run_id": run_id or str(uuid.uuid4()),
            "mode": mode,
            "prompt_id": prompt_id,
            "run_index": run_index,
            "warmup": warmup,
        }
        if extra:
            self.context.update({key: _json_safe(value) for key, value in extra.items()})
        self.events: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self._event_index = 0

    def record(self, phase: str, duration_ns: int | float, **metadata: Any) -> None:
        row = {
            **self.context,
            "event_index": self._event_index,
            "phase": phase,
            "duration_ns": int(duration_ns),
            "duration_ms": ns_to_ms(duration_ns),
        }
        row.update(
            {
                key: _json_safe(value)
                for key, value in metadata.items()
                if value is not None
            }
        )
        self.events.append(row)
        self._event_index += 1

    @contextmanager
    def time(self, phase: str, **metadata: Any):
        start = now_ns()
        try:
            yield
        finally:
            self.record(phase, now_ns() - start, **metadata)

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = _json_safe(value)

    def phase_totals_ns(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for event in self.events:
            phase = str(event["phase"])
            totals[phase] = totals.get(phase, 0) + int(event["duration_ns"])
        return totals

    def summary(self) -> dict[str, Any]:
        row = {**self.context, **self.metrics}
        for phase, duration_ns in sorted(self.phase_totals_ns().items()):
            row[f"{phase}_ns"] = duration_ns
            row[f"{phase}_ms"] = ns_to_ms(duration_ns)
        return row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    columns: list[str] = []
    for row in materialized:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(materialized)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_summaries(
    rows: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...] = ("mode",),
) -> list[dict[str, Any]]:
    materialized = [row for row in rows if not row.get("warmup", False)]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in materialized:
        key = tuple(row.get(group_key) for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        out["samples"] = len(group_rows)
        numeric_keys = sorted(
            {
                column
                for row in group_rows
                for column, value in row.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and column not in {"prompt_id", "run_index"}
            }
        )
        for column in numeric_keys:
            values = [float(row[column]) for row in group_rows if isinstance(row.get(column), (int, float))]
            if not values:
                continue
            out[f"{column}_mean"] = statistics.fmean(values)
            out[f"{column}_p50"] = percentile(values, 0.50)
            out[f"{column}_p95"] = percentile(values, 0.95)
            out[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(out)
    return aggregate_rows
