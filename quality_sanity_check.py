from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any, Iterable

from profiling import write_csv


def _is_warmup(row: dict[str, Any]) -> bool:
    return str(row.get("warmup", "")).lower() == "true"


def _int_value(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    return int(float(value))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


@dataclass(frozen=True)
class OutputSelector:
    placement: str
    network_profile: str | None = None

    @classmethod
    def parse(cls, value: str) -> "OutputSelector":
        if "/" in value:
            placement, network_profile = value.split("/", 1)
            return cls(placement=placement, network_profile=network_profile)
        return cls(placement=value)

    def label(self) -> str:
        return f"{self.placement}/{self.network_profile}" if self.network_profile else self.placement

    def matches(self, row: dict[str, Any]) -> bool:
        if row.get("placement") != self.placement:
            return False
        if self.network_profile is not None and row.get("network_profile") != self.network_profile:
            return False
        return True


def read_summary(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filter_output_rows(
    rows: Iterable[dict[str, str]],
    selector: OutputSelector,
    concurrency_level: int = 1,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if _is_warmup(row):
            continue
        if not selector.matches(row):
            continue
        if _int_value(row, "concurrency_level", 1) != concurrency_level:
            continue
        if "output_text" not in row or row["output_text"] == "":
            continue
        selected.append(row)
    return selected


def output_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        _int_value(row, "prompt_id"),
        _int_value(row, "run_index"),
        _int_value(row, "concurrent_worker"),
    )


def first_diff_index(left: str, right: str) -> int:
    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    if len(left) == len(right):
        return -1
    return min(len(left), len(right))


def compare_outputs(
    rows: Iterable[dict[str, str]],
    reference: OutputSelector,
    candidates: list[OutputSelector] | None = None,
    concurrency_level: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    materialized = list(rows)
    reference_rows = filter_output_rows(materialized, reference, concurrency_level=concurrency_level)
    if not reference_rows:
        raise ValueError(f"No reference output rows found for {reference.label()} at concurrency={concurrency_level}.")

    reference_by_key = {output_key(row): row for row in reference_rows}
    if candidates is None:
        seen: dict[tuple[str, str], OutputSelector] = {}
        for row in materialized:
            if _is_warmup(row):
                continue
            if _int_value(row, "concurrency_level", 1) != concurrency_level:
                continue
            placement = row.get("placement")
            network_profile = row.get("network_profile")
            if not placement or not network_profile:
                continue
            selector = OutputSelector(placement=placement, network_profile=network_profile)
            if selector == reference:
                continue
            seen[(placement, network_profile)] = selector
        candidates = list(seen.values())

    detail_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_rows = filter_output_rows(materialized, candidate, concurrency_level=concurrency_level)
        candidate_by_key = {output_key(row): row for row in candidate_rows}
        for key, reference_row in sorted(reference_by_key.items()):
            candidate_row = candidate_by_key.get(key)
            reference_text = reference_row["output_text"]
            candidate_text = candidate_row["output_text"] if candidate_row is not None else ""
            normalized_reference = _normalize_text(reference_text)
            normalized_candidate = _normalize_text(candidate_text)
            similarity = SequenceMatcher(None, normalized_reference, normalized_candidate).ratio() if candidate_text else 0.0
            detail_rows.append(
                {
                    "reference": reference.label(),
                    "candidate": candidate.label(),
                    "concurrency_level": concurrency_level,
                    "prompt_id": key[0],
                    "run_index": key[1],
                    "concurrent_worker": key[2],
                    "candidate_missing": candidate_row is None,
                    "exact_match": candidate_text == reference_text,
                    "normalized_exact_match": normalized_candidate == normalized_reference,
                    "similarity": similarity,
                    "first_diff_index": first_diff_index(reference_text, candidate_text),
                    "reference_chars": len(reference_text),
                    "candidate_chars": len(candidate_text),
                }
            )

    aggregate_rows = aggregate_quality(detail_rows)
    return detail_rows, aggregate_rows


def aggregate_quality(detail_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        groups.setdefault(str(row["candidate"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for candidate, rows in sorted(groups.items()):
        samples = len(rows)
        missing = sum(1 for row in rows if row["candidate_missing"])
        exact = sum(1 for row in rows if row["exact_match"])
        normalized_exact = sum(1 for row in rows if row["normalized_exact_match"])
        similarities = [float(row["similarity"]) for row in rows]
        aggregates.append(
            {
                "candidate": candidate,
                "samples": samples,
                "missing": missing,
                "exact_match_rate": exact / samples if samples else 0.0,
                "normalized_exact_match_rate": normalized_exact / samples if samples else 0.0,
                "mean_similarity": sum(similarities) / samples if samples else 0.0,
                "min_similarity": min(similarities) if similarities else 0.0,
            }
        )
    return aggregates


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare greedy outputs across target placements.")
    parser.add_argument("--run-summary", required=True, help="Path to target_placement_benchmark run_summary.csv.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference placement, optionally placement/network_profile.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Candidate placement, optionally placement/network_profile. Repeat to compare multiple candidates.",
    )
    parser.add_argument("--concurrency-level", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fail-under", type=float, default=None, help="Exit non-zero if any candidate mean similarity is below this value.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_summary(args.run_summary)
    reference = OutputSelector.parse(args.reference)
    candidates = [OutputSelector.parse(value) for value in args.candidate] if args.candidate else None
    detail_rows, aggregate_rows = compare_outputs(
        rows,
        reference=reference,
        candidates=candidates,
        concurrency_level=args.concurrency_level,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.run_summary).parent
    detail_path = output_dir / "quality_sanity_detail.csv"
    aggregate_path = output_dir / "quality_sanity_summary.csv"
    write_csv(detail_path, detail_rows)
    write_csv(aggregate_path, aggregate_rows)
    print(f"Wrote quality sanity detail: {detail_path}")
    print(f"Wrote quality sanity summary: {aggregate_path}")
    for row in aggregate_rows:
        print(
            f"{row['candidate']}: exact={row['exact_match_rate']:.3f} "
            f"normalized_exact={row['normalized_exact_match_rate']:.3f} "
            f"mean_similarity={row['mean_similarity']:.3f} "
            f"min_similarity={row['min_similarity']:.3f}"
        )

    if args.fail_under is not None:
        failed = [row for row in aggregate_rows if float(row["mean_similarity"]) < args.fail_under]
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
