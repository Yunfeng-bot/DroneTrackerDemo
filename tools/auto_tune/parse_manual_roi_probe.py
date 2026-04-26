#!/usr/bin/env python3
"""
Parse a logcat capture from a manual ROI probe session and emit:
  1. probe_<scenario>_summary.json
  2. probe_<scenario>_ncc.csv
  3. probe_<scenario>_timeline.csv

Usage:
  python tools/auto_tune/parse_manual_roi_probe.py <logcat_file> <scenario_name>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


LOG_PATTERNS = {
    "init_ok": re.compile(
        r"EVAL_EVENT type=MANUAL_ROI_INIT_OK.*?"
        r"patch_kp=(\d+).*?"
        r"patch_texture=([\d.]+).*?"
        r"box=(\d+),(\d+),(\d+)x(\d+)"
    ),
    "init_fail": re.compile(r"EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=([A-Za-z0-9_]+)"),
    "patch_stats": re.compile(
        r"EVAL_EVENT type=MANUAL_ROI_PATCH_STATS.*?"
        r"mean=([\d.]+) std=([\d.]+) min=([\d.]+) max=([\d.]+)"
    ),
    "lock": re.compile(
        r"DIAG_LOCK.*?backend=([A-Za-z0-9_]+).*?"
        r"reason=([A-Za-z0-9_]+).*?"
        r"box=(\d+),(\d+),(\d+)x(\d+)"
    ),
    "lost_enter": re.compile(r"action=lost_enter reason=([A-Za-z0-9_]+)"),
    "track_veto": re.compile(r"EVAL_EVENT type=MANUAL_ROI_TRACK_VETO.*?reason=([A-Za-z0-9_]+)"),
    "blocked": re.compile(r"EVAL_EVENT type=MANUAL_ROI state=relock_blocked trigger=(\S+)"),
    "unblocked": re.compile(
        r"EVAL_EVENT type=MANUAL_ROI state=relock_unblocked trigger=(\S+) ncc=([\d.]+)"
    ),
    "lock_gate": re.compile(
        r"DIAG_LOCK_GATE.*?stage=manual_relock_v2.*?"
        r"good=(\d+) inliers=(\d+) conf=([\d.]+) fallback=(\S+) blocked=(\w+).*?"
        r"pass=(\w+)"
        r"(?:.*?reason=([A-Za-z0-9_]+))?"
        r"(?:.*?ncc=([\d.]+))?"
    ),
}

TIMESTAMP_RE = re.compile(r"^(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


def bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def avg_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def parse_log(logfile: Path, scenario: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    summary: dict[str, Any] = {
        "scenario": scenario,
        "logfile": str(logfile),
        "init_ok": 0,
        "init_fail": {},
        "patch_stats_count": 0,
        "patch_mean_avg": None,
        "patch_std_avg": None,
        "patch_std_min": None,
        "patch_std_max": None,
        "patch_range_min": None,
        "patch_range_max": None,
        "init_ok_patch_kp_min": None,
        "init_ok_patch_kp_max": None,
        "init_ok_patch_texture_min": None,
        "init_ok_patch_texture_max": None,
        "lock": 0,
        "lost": {},
        "track_veto": {},
        "blocked": 0,
        "unblocked": 0,
        "lock_gate_pass": 0,
        "lock_gate_fail": {},
        "ncc_samples": [],
    }
    timeline: list[dict[str, str]] = []
    ncc_rows: list[dict[str, Any]] = []

    patch_means: list[float] = []
    patch_stds: list[float] = []
    patch_ranges: list[float] = []
    patch_kps: list[float] = []
    patch_textures: list[float] = []

    with logfile.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts_match = TIMESTAMP_RE.match(line)
            ts = ts_match.group(1) if ts_match else "?"

            if m := LOG_PATTERNS["patch_stats"].search(line):
                mean_val = float(m.group(1))
                std_val = float(m.group(2))
                min_val = float(m.group(3))
                max_val = float(m.group(4))
                range_val = max_val - min_val
                patch_means.append(mean_val)
                patch_stds.append(std_val)
                patch_ranges.append(range_val)
                summary["patch_stats_count"] += 1
                timeline.append(
                    {
                        "ts": ts,
                        "type": "PATCH_STATS",
                        "detail": f"mean={mean_val:.3f} std={std_val:.3f} min={min_val:.3f} max={max_val:.3f}",
                    }
                )
                continue

            if m := LOG_PATTERNS["init_ok"].search(line):
                kp = int(m.group(1))
                texture = float(m.group(2))
                patch_kps.append(float(kp))
                patch_textures.append(texture)
                summary["init_ok"] += 1
                timeline.append(
                    {
                        "ts": ts,
                        "type": "INIT_OK",
                        "detail": (
                            f"kp={kp} tex={texture:.3f} "
                            f"box={m.group(3)},{m.group(4)},{m.group(5)}x{m.group(6)}"
                        ),
                    }
                )
                continue

            if m := LOG_PATTERNS["init_fail"].search(line):
                reason = m.group(1)
                bump(summary["init_fail"], reason)
                timeline.append({"ts": ts, "type": "INIT_FAIL", "detail": f"reason={reason}"})
                continue

            if m := LOG_PATTERNS["lock"].search(line):
                summary["lock"] += 1
                timeline.append(
                    {
                        "ts": ts,
                        "type": "LOCK",
                        "detail": (
                            f"backend={m.group(1)} reason={m.group(2)} "
                            f"box={m.group(3)},{m.group(4)},{m.group(5)}x{m.group(6)}"
                        ),
                    }
                )
                continue

            if m := LOG_PATTERNS["lost_enter"].search(line):
                reason = m.group(1)
                bump(summary["lost"], reason)
                timeline.append({"ts": ts, "type": "LOST", "detail": f"reason={reason}"})
                continue

            if m := LOG_PATTERNS["track_veto"].search(line):
                reason = m.group(1)
                bump(summary["track_veto"], reason)
                timeline.append({"ts": ts, "type": "VETO", "detail": f"reason={reason}"})
                continue

            if m := LOG_PATTERNS["blocked"].search(line):
                summary["blocked"] += 1
                timeline.append({"ts": ts, "type": "BLOCKED", "detail": f"trigger={m.group(1)}"})
                continue

            if m := LOG_PATTERNS["unblocked"].search(line):
                ncc = float(m.group(2))
                summary["unblocked"] += 1
                summary["ncc_samples"].append({"ncc": ncc, "outcome": "unblock_pass"})
                ncc_rows.append(
                    {
                        "ts": ts,
                        "scenario": scenario,
                        "ncc": f"{ncc:.3f}",
                        "gate_pass": "true",
                        "context": "unblock",
                    }
                )
                timeline.append({"ts": ts, "type": "UNBLOCKED", "detail": f"ncc={ncc:.3f}"})
                continue

            if m := LOG_PATTERNS["lock_gate"].search(line):
                good = m.group(1)
                inliers = m.group(2)
                conf = m.group(3)
                gate_pass = m.group(6).lower()
                reason = m.group(7) or ""
                ncc_text = m.group(8)
                if gate_pass == "true":
                    summary["lock_gate_pass"] += 1
                else:
                    bump(summary["lock_gate_fail"], reason or "unknown")
                if ncc_text:
                    ncc = float(ncc_text)
                    summary["ncc_samples"].append({"ncc": ncc, "outcome": gate_pass})
                    ncc_rows.append(
                        {
                            "ts": ts,
                            "scenario": scenario,
                            "ncc": f"{ncc:.3f}",
                            "gate_pass": gate_pass,
                            "context": f"reason={reason or '-'}",
                        }
                    )
                timeline.append(
                    {
                        "ts": ts,
                        "type": "GATE",
                        "detail": (
                            f"pass={gate_pass} reason={reason or '-'} "
                            f"good={good} inliers={inliers} conf={conf} ncc={ncc_text or '-'}"
                        ),
                    }
                )

    summary["patch_mean_avg"] = avg_or_none(patch_means)
    summary["patch_std_avg"] = avg_or_none(patch_stds)
    summary["patch_std_min"] = min_or_none(patch_stds)
    summary["patch_std_max"] = max_or_none(patch_stds)
    summary["patch_range_min"] = min_or_none(patch_ranges)
    summary["patch_range_max"] = max_or_none(patch_ranges)
    summary["init_ok_patch_kp_min"] = min_or_none(patch_kps)
    summary["init_ok_patch_kp_max"] = max_or_none(patch_kps)
    summary["init_ok_patch_texture_min"] = min_or_none(patch_textures)
    summary["init_ok_patch_texture_max"] = max_or_none(patch_textures)
    return summary, ncc_rows, timeline


def write_outputs(logfile: Path, scenario: str, summary: dict[str, Any], ncc_rows: list[dict[str, Any]], timeline: list[dict[str, str]]) -> None:
    out_dir = logfile.parent
    base = f"probe_{scenario}"

    (out_dir / f"{base}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / f"{base}_ncc.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "scenario", "ncc", "gate_pass", "context"])
        writer.writeheader()
        writer.writerows(ncc_rows)

    with (out_dir / f"{base}_timeline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "type", "detail"])
        writer.writeheader()
        writer.writerows(timeline)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse manual ROI probe logcat into summary/NCC/timeline files")
    parser.add_argument("logcat_file", type=Path, help="Path to captured logcat file")
    parser.add_argument("scenario_name", help="Scenario label, e.g. S1 or S3_return")
    args = parser.parse_args()

    if not args.logcat_file.exists():
        raise SystemExit(f"logcat file not found: {args.logcat_file}")

    summary, ncc_rows, timeline = parse_log(args.logcat_file, args.scenario_name)
    write_outputs(args.logcat_file, args.scenario_name, summary, ncc_rows, timeline)
    print(
        f"Wrote probe_{args.scenario_name}_summary.json, "
        f"probe_{args.scenario_name}_ncc.csv, "
        f"probe_{args.scenario_name}_timeline.csv "
        f"({len(ncc_rows)} NCC samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
