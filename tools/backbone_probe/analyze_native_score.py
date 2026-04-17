#!/usr/bin/env python3
"""
Analyze native tracker score logs and estimate similarity discrimination ranges.

Expected log line format (from OpenCVTrackerAnalyzer):
  EVAL_NATIVE_SCORE src=native_img frame=120 action=accept conf=0.912 sim=0.887 meas=0.894 tracking=true
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCORE_PATTERN = re.compile(r"EVAL_NATIVE_SCORE\s+(.*)")
LOST_PATTERN = re.compile(r"EVAL_EVENT\s+type=LOST\b")
KV_PATTERN = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


@dataclass
class Sample:
    source: str
    frame: int
    action: str
    confidence: float
    similarity: float
    measurement: float
    tracking: bool
    file: str
    line_no: int


def parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    q = max(0.0, min(100.0, q))
    xs = sorted(values)
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    t = pos - lo
    return float(xs[lo] * (1.0 - t) + xs[hi] * t)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p10": float(percentile(values, 10)),
        "p50": float(percentile(values, 50)),
        "p90": float(percentile(values, 90)),
        "max": float(max(values)),
    }


def parse_line(line: str, file: str, line_no: int) -> Optional[Sample]:
    m = SCORE_PATTERN.search(line)
    if not m:
        return None
    payload = m.group(1)
    kv = {k: v for k, v in KV_PATTERN.findall(payload)}
    try:
        source = kv.get("src", "unknown")
        frame = int(kv.get("frame", "-1"))
        action = kv.get("action", "unknown")
        confidence = clamp01(float(kv.get("conf", "nan")))
        similarity = clamp01(float(kv.get("sim", "nan")))
        measurement = clamp01(float(kv.get("meas", "nan")))
        tracking = parse_bool(kv.get("tracking", "false"))
        if any(math.isnan(v) for v in (confidence, similarity, measurement)):
            return None
        return Sample(
            source=source,
            frame=frame,
            action=action,
            confidence=confidence,
            similarity=similarity,
            measurement=measurement,
            tracking=tracking,
            file=file,
            line_no=line_no,
        )
    except Exception:
        return None


def parse_logs(paths: Iterable[str]) -> Tuple[List[Sample], Dict[str, List[int]]]:
    samples: List[Sample] = []
    lost_markers: Dict[str, List[int]] = {}
    for path in paths:
        path_lost: List[int] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, 1):
                    sample = parse_line(line, path, idx)
                    if sample is not None:
                        samples.append(sample)
                    if LOST_PATTERN.search(line):
                        path_lost.append(idx)
        except OSError:
            continue
        if path_lost:
            lost_markers[path] = path_lost
    return samples, lost_markers


def best_threshold(scores: Sequence[float], labels: Sequence[bool]) -> Tuple[float, float, float]:
    """
    Return (threshold, tpr, fpr) maximizing Youden's J = TPR - FPR.
    Predict positive when score >= threshold.
    """
    assert len(scores) == len(labels)
    if not scores:
        return float("nan"), float("nan"), float("nan")
    unique = sorted(set(scores))
    candidates = [0.0] + unique + [1.0]
    pos_total = sum(1 for x in labels if x)
    neg_total = len(labels) - pos_total
    best_j = -1e9
    best = (float("nan"), float("nan"), float("nan"))
    for thr in candidates:
        tp = fp = 0
        for s, y in zip(scores, labels):
            pred = s >= thr
            if pred and y:
                tp += 1
            elif pred and not y:
                fp += 1
        tpr = tp / pos_total if pos_total > 0 else float("nan")
        fpr = fp / neg_total if neg_total > 0 else float("nan")
        if math.isnan(tpr) or math.isnan(fpr):
            continue
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best = (thr, tpr, fpr)
    return best


def infer_pseudo_negatives_from_lost(
    samples: Sequence[Sample],
    lost_markers: Dict[str, List[int]],
    max_gap_lines: int,
) -> List[Sample]:
    by_file: Dict[str, List[Sample]] = {}
    for s in samples:
        by_file.setdefault(s.file, []).append(s)
    for path in by_file:
        by_file[path].sort(key=lambda x: x.line_no)

    selected: List[Sample] = []
    seen: set[Tuple[str, int]] = set()
    for path, lost_lines in lost_markers.items():
        cand = by_file.get(path, [])
        if not cand:
            continue
        for lost_ln in lost_lines:
            best: Optional[Sample] = None
            best_gap = max_gap_lines + 1
            for s in cand:
                if s.line_no >= lost_ln:
                    break
                gap = lost_ln - s.line_no
                if gap <= max_gap_lines and gap < best_gap:
                    best = s
                    best_gap = gap
            if best is not None:
                key = (best.file, best.line_no)
                if key not in seen:
                    seen.add(key)
                    selected.append(best)
    return selected


def threshold_report(
    samples: Sequence[Sample],
    lost_markers: Dict[str, List[int]],
    infer_from_lost: bool = False,
    infer_max_gap_lines: int = 400,
) -> Dict[str, object]:
    accept_actions = {"accept"}
    reject_actions = {"drop_hard", "drop_soft", "drop_min_conf", "hold"}

    pseudo_negatives: List[Sample] = []
    if infer_from_lost:
        pseudo_negatives = infer_pseudo_negatives_from_lost(samples, lost_markers, infer_max_gap_lines)

    negatives = [s for s in samples if s.action in reject_actions]
    negative_keys = {(s.file, s.line_no) for s in negatives}
    if pseudo_negatives:
        for s in pseudo_negatives:
            key = (s.file, s.line_no)
            if key not in negative_keys:
                negatives.append(s)
                negative_keys.add(key)

    positives = [s for s in samples if s.action in accept_actions and (s.file, s.line_no) not in negative_keys]

    pos_sim = [s.similarity for s in positives]
    neg_sim = [s.similarity for s in negatives]
    pos_meas = [s.measurement for s in positives]
    neg_meas = [s.measurement for s in negatives]

    report: Dict[str, object] = {
        "positives": len(positives),
        "negatives": len(negatives),
        "lost_events": int(sum(len(v) for v in lost_markers.values())),
        "pseudo_negatives_from_lost": len(pseudo_negatives),
        "similarity": {
            "accept": summarize(pos_sim),
            "reject": summarize(neg_sim),
        },
        "measurement": {
            "accept": summarize(pos_meas),
            "reject": summarize(neg_meas),
        },
    }

    if positives and negatives:
        low = percentile(neg_sim, 95)
        high = percentile(pos_sim, 5)
        if high > low:
            suggested = (low + high) * 0.5
            strategy = "separation_band_midpoint"
        else:
            scores = [s.similarity for s in positives + negatives]
            labels = [True] * len(positives) + [False] * len(negatives)
            suggested, tpr, fpr = best_threshold(scores, labels)
            strategy = "youden_j"
            report["youden"] = {"threshold": suggested, "tpr": tpr, "fpr": fpr}
        report["suggested_similarity_threshold"] = {
            "value": clamp01(suggested),
            "strategy": strategy,
            "accept_p5": high,
            "reject_p95": low,
        }

    return report


def format_float(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def print_console_summary(samples: Sequence[Sample], report: Dict[str, object]) -> None:
    print(f"[native-score] samples={len(samples)}")
    if not samples:
        return

    by_action: Dict[str, int] = {}
    for s in samples:
        by_action[s.action] = by_action.get(s.action, 0) + 1
    for action in sorted(by_action):
        print(f"  action={action:<12} count={by_action[action]}")

    sim = report.get("similarity", {})
    if isinstance(sim, dict):
        a = sim.get("accept", {})
        r = sim.get("reject", {})
        if isinstance(a, dict) and isinstance(r, dict):
            print(
                "  sim.accept mean/p10/p50/p90="
                f"{format_float(a.get('mean', float('nan')))} / "
                f"{format_float(a.get('p10', float('nan')))} / "
                f"{format_float(a.get('p50', float('nan')))} / "
                f"{format_float(a.get('p90', float('nan')))}"
            )
            print(
                "  sim.reject mean/p10/p50/p90="
                f"{format_float(r.get('mean', float('nan')))} / "
                f"{format_float(r.get('p10', float('nan')))} / "
                f"{format_float(r.get('p50', float('nan')))} / "
                f"{format_float(r.get('p90', float('nan')))}"
            )

    suggested = report.get("suggested_similarity_threshold")
    if isinstance(suggested, dict):
        print(
            "  suggested_similarity_threshold="
            f"{format_float(float(suggested.get('value', float('nan'))))} "
            f"({suggested.get('strategy', 'unknown')})"
        )


def resolve_input_files(inputs: Sequence[str], recursive_glob: bool) -> List[str]:
    files: List[str] = []
    for item in inputs:
        if os.path.isfile(item):
            files.append(item)
            continue
        if os.path.isdir(item):
            pattern = "**/*.log" if recursive_glob else "*.log"
            files.extend(glob.glob(os.path.join(item, pattern), recursive=recursive_glob))
            continue
        matched = glob.glob(item, recursive=True)
        files.extend([p for p in matched if os.path.isfile(p)])
    dedup = sorted(set(os.path.abspath(p) for p in files))
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze EVAL_NATIVE_SCORE logs and estimate similarity thresholds.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["tools/auto_tune/out"],
        help="Input log files/directories/globs. Default: tools/auto_tune/out",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input includes directories, recursively include *.log",
    )
    parser.add_argument(
        "--out-dir",
        default="tools/backbone_probe/out",
        help="Output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix tag in output file names.",
    )
    parser.add_argument(
        "--infer-neg-from-lost",
        action="store_true",
        help="When explicit reject actions are absent, infer pseudo-negatives from native-score samples nearest before LOST events.",
    )
    parser.add_argument(
        "--infer-max-gap-lines",
        type=int,
        default=400,
        help="Max line distance between LOST and preceding EVAL_NATIVE_SCORE for pseudo-negative inference.",
    )
    args = parser.parse_args()

    files = resolve_input_files(args.input, args.recursive)
    if not files:
        print("[native-score] no log files found")
        return 2

    samples, lost_markers = parse_logs(files)
    report = threshold_report(
        samples,
        lost_markers,
        infer_from_lost=args.infer_neg_from_lost,
        infer_max_gap_lines=max(10, args.infer_max_gap_lines),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"native_score_report_{ts}{tag}.json")
    sample_path = os.path.join(args.out_dir, f"native_score_samples_{ts}{tag}.jsonl")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": files,
        "sample_count": len(samples),
        "report": report,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(sample_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "src": s.source,
                        "frame": s.frame,
                        "action": s.action,
                        "conf": s.confidence,
                        "sim": s.similarity,
                        "meas": s.measurement,
                        "tracking": s.tracking,
                        "file": s.file,
                        "line": s.line_no,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print_console_summary(samples, report)
    print(f"[native-score] report={os.path.abspath(json_path)}")
    print(f"[native-score] samples={os.path.abspath(sample_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
