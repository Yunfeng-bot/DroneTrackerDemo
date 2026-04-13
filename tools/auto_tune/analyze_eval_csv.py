#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    w = pos - lo
    return values[lo] * (1 - w) + values[hi] * w


def analyze(csv_path: Path, fps: float, conf_lock: float) -> dict[str, float | int]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    latencies: list[float] = []
    confidences: list[float] = []
    valid_box = 0
    tracking_like = 0

    first_lock_frame = None
    for i, r in enumerate(rows):
        latency = float(r.get("latency_ms", "nan"))
        conf = float(r.get("confidence_score", "nan"))
        x = int(r.get("predicted_x", "-1"))
        y = int(r.get("predicted_y", "-1"))
        w = int(r.get("predicted_w", "-1"))
        h = int(r.get("predicted_h", "-1"))

        if math.isfinite(latency):
            latencies.append(latency)
        if math.isfinite(conf):
            confidences.append(conf)

        has_box = x >= 0 and y >= 0 and w > 0 and h > 0
        if has_box:
            valid_box += 1
        if has_box and conf >= conf_lock:
            tracking_like += 1
            if first_lock_frame is None:
                first_lock_frame = i

    latencies_sorted = sorted(latencies)

    return {
        "frames": len(rows),
        "valid_box_frames": valid_box,
        "tracking_like_frames": tracking_like,
        "track_like_ratio": (tracking_like / len(rows)) if rows else 0.0,
        "first_lock_frame": (-1 if first_lock_frame is None else first_lock_frame),
        "first_lock_sec": (-1.0 if first_lock_frame is None else first_lock_frame / fps),
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else float("nan"),
        "p50_latency_ms": (median(latencies_sorted) if latencies_sorted else float("nan")),
        "p90_latency_ms": percentile(latencies_sorted, 0.90),
        "p95_latency_ms": percentile(latencies_sorted, 0.95),
        "avg_conf": (sum(confidences) / len(confidences)) if confidences else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze evaluation CSV exported by EvaluationActivity")
    parser.add_argument("csv", type=Path, help="Path to CSV file")
    parser.add_argument("--fps", type=float, default=15.0, help="Replay FPS used for frame->second conversion")
    parser.add_argument("--conf-lock", type=float, default=0.9, help="Confidence threshold treated as lock/tracking")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    result = analyze(args.csv, fps=args.fps, conf_lock=args.conf_lock)

    print(f"csv={args.csv}")
    for k in [
        "frames",
        "valid_box_frames",
        "tracking_like_frames",
        "track_like_ratio",
        "first_lock_frame",
        "first_lock_sec",
        "avg_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "avg_conf",
    ]:
        v = result[k]
        if isinstance(v, float):
            print(f"{k}={v:.4f}")
        else:
            print(f"{k}={v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
