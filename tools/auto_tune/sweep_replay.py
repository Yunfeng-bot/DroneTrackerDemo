#!/usr/bin/env python3
"""
Replay auto-tuning helper for DroneTrackerDemo.

This script runs replay sessions with different `eval_params`, captures Tracker logs,
extracts quality/perf metrics, computes a fitness score, and writes ranked outputs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PACKAGE_NAME = "com.example.dronetracker"
ACTIVITY_NAME = ".MainActivity"
TRACKER_TAG = "Tracker"


@dataclass
class Weights:
    first_lock_weight: float = 9.0
    lost_weight: float = 35.0
    streak_weight: float = 0.40
    track_ratio_weight: float = 65.0
    frame_ms_over_weight: float = 1.4
    lock_count_weight: float = 4.0
    lock_bonus: float = 35.0
    no_lock_penalty: float = 280.0
    no_metrics_penalty: float = 180.0
    early_lock_penalty: float = 160.0
    target_avg_frame_ms: float = 40.0


def parse_grid_item(text: str) -> Tuple[str, List[object]]:
    if "=" not in text:
        raise ValueError(f"invalid --grid item: {text}")
    key, raw_values = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty key in --grid item: {text}")
    values = []
    for token in raw_values.split("|"):
        t = token.strip()
        if not t:
            continue
        values.append(parse_scalar(t))
    if not values:
        raise ValueError(f"no values in --grid item: {text}")
    return key, values


def parse_scalar(text: str) -> object:
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)
    return str(value)


def make_default_grid(preset: str) -> Dict[str, List[object]]:
    if preset == "quick":
        return {
            "fallback_refine_ratio": [0.80, 0.86],
            "small_target_min_good": [5, 6],
            "small_target_min_inliers": [3, 4],
            "orb_ransac": [4.0, 6.0],
            "first_lock_min_iou": [0.58],
        }
    if preset == "overnight":
        return {
            "orb_ratio": [0.65, 0.68, 0.72, 0.75],
            "fallback_refine_ratio": [0.78, 0.82, 0.86],
            "fallback_refine_min_inliers": [4, 5],
            "small_target_min_good": [5, 6],
            "small_target_min_inliers": [3, 4],
            "orb_ransac": [4.0, 6.0],
            "first_lock_min_iou": [0.55, 0.62],
        }
    return {
        "fallback_refine_ratio": [0.78, 0.82, 0.86],
        "small_target_min_good": [5, 6],
        "small_target_min_inliers": [3, 4],
        "orb_ransac": [4.0, 6.0, 8.0],
        "first_lock_min_iou": [0.58, 0.65],
    }


def expand_grid(grid: Dict[str, List[object]]) -> List[Dict[str, object]]:
    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def parse_params_string(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not raw.strip():
        return out
    for entry in raw.replace(";", ",").split(","):
        token = entry.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        key = k.strip()
        value = v.strip()
        if key:
            out[key] = value
    return out


def make_eval_params(base_params: str, params: Dict[str, object]) -> str:
    merged = parse_params_string(base_params)
    for k in sorted(params.keys()):
        merged[k] = format_value(params[k])
    pairs = [f"{k}={merged[k]}" for k in sorted(merged.keys())]
    return ",".join(pairs)


def parse_kv(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in re.findall(r"([A-Za-z0-9_]+)=([^ \r\n]+)", line):
        out[k] = v
    return out


def to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def to_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def compute_score(
    first_lock_after_target_sec: float | None,
    early_lock: bool,
    locks: int,
    lost: int,
    max_track_streak: int,
    track_ratio: float | None,
    avg_frame_ms: float | None,
    frames: int,
    weights: Weights,
) -> float:
    score = 0.0
    if first_lock_after_target_sec is None or first_lock_after_target_sec < 0:
        score -= weights.no_lock_penalty
    else:
        score += weights.lock_bonus
        score -= first_lock_after_target_sec * weights.first_lock_weight

    if early_lock:
        score -= weights.early_lock_penalty

    score += locks * weights.lock_count_weight
    score -= lost * weights.lost_weight
    score += max_track_streak * weights.streak_weight

    if track_ratio is not None:
        score += track_ratio * weights.track_ratio_weight
    if avg_frame_ms is not None and avg_frame_ms > weights.target_avg_frame_ms:
        score -= (avg_frame_ms - weights.target_avg_frame_ms) * weights.frame_ms_over_weight
    if frames <= 0:
        score -= weights.no_metrics_penalty
    return score


class AdbRunner:
    def __init__(self, repo_root: Path, mode: str) -> None:
        self.repo_root = repo_root
        self.mode = mode
        self.wrapper = repo_root / "tools" / "adb_exec.ps1"
        self.args_file = repo_root / "tools" / "adb_args.json"

    def run(self, args: List[str], timeout_sec: int = 30, check: bool = True) -> str:
        if self.mode == "wrapper":
            self.args_file.write_text(
                json.dumps(args, ensure_ascii=False),
                encoding="utf-8",
            )
            cmd = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.wrapper),
            ]
        else:
            cmd = ["adb", *args]

        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        merged = (proc.stdout or "") + (proc.stderr or "")
        if check and proc.returncode != 0:
            raise RuntimeError(f"adb command failed: {' '.join(args)}\n{merged.strip()}")
        return merged


def parse_tracker_log(text: str) -> Dict[str, object]:
    summary_lines = []
    perf_lines = []
    lock_lines = []
    lost_lines = []
    refine_pass_count = 0

    for line in text.splitlines():
        if "EVAL_SUMMARY" in line:
            summary_lines.append(line)
        if "EVAL_PERF" in line:
            perf_lines.append(line)
        if "EVAL_EVENT type=LOCK" in line:
            lock_lines.append(line)
        if "EVAL_EVENT type=LOST" in line:
            lost_lines.append(line)
        if "EVAL_EVENT type=SEARCH_REFINE state=pass" in line:
            refine_pass_count += 1

    summary_kv = parse_kv(summary_lines[-1]) if summary_lines else {}
    perf_kv = parse_kv(perf_lines[-1]) if perf_lines else {}
    lock_kv = parse_kv(lock_lines[-1]) if lock_lines else {}

    locks = to_int(summary_kv.get("locks"))
    if locks is None:
        locks = to_int(perf_kv.get("locks"))
    if locks is None:
        locks = len(lock_lines)
    lost = to_int(summary_kv.get("lost"))
    if lost is None:
        lost = to_int(perf_kv.get("lost"))
    if lost is None:
        lost = len(lost_lines)

    first_lock_sec = to_float(summary_kv.get("firstLockSec"))
    if first_lock_sec is None:
        first_lock_sec = to_float(perf_kv.get("firstLockSec"))
    if first_lock_sec is None:
        first_lock_sec = to_float(lock_kv.get("firstLockSec"))

    track_ratio = to_float(summary_kv.get("trackRatio"))
    if track_ratio is None:
        track_ratio = to_float(perf_kv.get("trackRatio"))
    avg_frame_ms = to_float(summary_kv.get("avgFrameMs"))
    if avg_frame_ms is None:
        avg_frame_ms = to_float(perf_kv.get("avgFrameMs"))
    max_track_streak = to_int(summary_kv.get("maxTrackStreak")) or 0
    frames = to_int(summary_kv.get("frames"))
    if frames is None:
        frames = to_int(perf_kv.get("frames")) or 0

    return {
        "locks": locks,
        "lost": lost,
        "first_lock_sec": first_lock_sec,
        "track_ratio": track_ratio,
        "avg_frame_ms": avg_frame_ms,
        "max_track_streak": max_track_streak,
        "frames": frames,
        "refine_pass_count": refine_pass_count,
    }


def build_start_args(
    serial: str | None,
    video_path: str,
    target_path: str,
    target_paths: str,
    eval_params: str,
    tracker_mode: str,
) -> List[str]:
    args: List[str] = []
    if serial:
        args += ["-s", serial]
    args += [
        "shell",
        "am",
        "start",
        "-n",
        f"{PACKAGE_NAME}/{ACTIVITY_NAME}",
        "--ez",
        "eval_use_replay",
        "true",
        "--es",
        "tracker_mode",
        tracker_mode,
        "--es",
        "eval_video_path",
        video_path,
        "--es",
        "eval_target_path",
        target_path,
        "--es",
        "eval_params",
        eval_params,
    ]
    if target_paths.strip():
        # `adb shell am start ...` is executed via a shell on device; semicolons must be escaped
        # so multi-template lists are passed as one extra value instead of split commands.
        escaped_target_paths = target_paths.replace(";", r"\;")
        args += ["--es", "eval_target_paths", escaped_target_paths]
    return args


def build_simple_args(serial: str | None, *rest: str) -> List[str]:
    args: List[str] = []
    if serial:
        args += ["-s", serial]
    args += list(rest)
    return args


def ensure_device(adb: AdbRunner, serial: str | None) -> None:
    out = adb.run(build_simple_args(serial, "devices"), timeout_sec=20, check=True)
    if serial:
        if serial not in out:
            raise RuntimeError(f"device {serial} not present in adb devices output")
    else:
        device_lines = [ln for ln in out.splitlines() if "\tdevice" in ln]
        if not device_lines:
            raise RuntimeError("no adb device connected")


def compute_first_lock_after_target(first_lock_sec: float | None, target_appear_sec: float) -> float | None:
    if first_lock_sec is None or first_lock_sec < 0:
        return None
    target_ts = max(0.0, target_appear_sec)
    if first_lock_sec + 1e-6 < target_ts:
        return None
    return first_lock_sec - target_ts


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep replay params and rank with fitness score.")
    parser.add_argument("--serial", default="", help="adb device serial, optional if only one device is connected")
    parser.add_argument("--video-path", default="/sdcard/Download/Video_Search/scene.mp4")
    parser.add_argument("--target-path", default="/sdcard/Download/Video_Search/target.jpg")
    parser.add_argument(
        "--target-paths",
        default="",
        help="semicolon-separated template paths for multi-template replay, e.g. /sdcard/.../T1.jpg;/sdcard/.../T2.jpg",
    )
    parser.add_argument("--tracker-mode", default="enhanced", choices=["enhanced", "baseline"])
    parser.add_argument("--duration-sec", type=float, default=10.0, help="effective scoring window (seconds) after target appears")
    parser.add_argument("--target-appear-sec", type=float, default=0.0, help="target appears after this offset in replay (seconds)")
    parser.add_argument("--cooldown-sec", type=float, default=1.0, help="pause between runs")
    parser.add_argument("--preset", default="default", choices=["quick", "default", "overnight"])
    parser.add_argument("--grid", action="append", default=[], help="override/add grid key=val1|val2|...")
    parser.add_argument("--base-params", default="", help="extra fixed eval_params prefix")
    parser.add_argument("--max-runs", type=int, default=0, help="cap total runs after expansion (0 means no cap)")
    parser.add_argument("--shuffle", action="store_true", help="shuffle run order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-empty", type=int, default=1, help="retry count when a run captures empty metrics/log")
    parser.add_argument("--adb-mode", default="wrapper", choices=["wrapper", "direct"])
    parser.add_argument("--output-dir", default="tools/auto_tune/out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / args.output_dir
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_tag
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    grid = make_default_grid(args.preset)
    for item in args.grid:
        k, values = parse_grid_item(item)
        grid[k] = values

    combos = expand_grid(grid)
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(combos)
    if args.max_runs > 0:
        combos = combos[: args.max_runs]
    if not combos:
        print("No parameter combinations generated.")
        return 2

    weights = Weights()
    adb = AdbRunner(repo_root, args.adb_mode)
    serial = args.serial.strip() or None

    meta = {
        "run_tag": run_tag,
        "preset": args.preset,
        "count": len(combos),
        "duration_sec": args.duration_sec,
        "target_appear_sec": args.target_appear_sec,
        "cooldown_sec": args.cooldown_sec,
        "video_path": args.video_path,
        "target_path": args.target_path,
        "target_paths": args.target_paths,
        "tracker_mode": args.tracker_mode,
        "base_params": args.base_params,
        "adb_mode": args.adb_mode,
        "retry_empty": args.retry_empty,
        "serial": serial or "",
        "grid": grid,
        "weights": weights.__dict__,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sweep] output={run_dir}")
    total_wait_sec = max(0.1, args.duration_sec + max(0.0, args.target_appear_sec))
    print(
        f"[sweep] runs={len(combos)} preset={args.preset} "
        f"effective={args.duration_sec:.1f}s targetAppear={args.target_appear_sec:.1f}s total={total_wait_sec:.1f}s"
    )

    if args.dry_run:
        for i, p in enumerate(combos[:10], start=1):
            print(f"  dry#{i}: {make_eval_params(args.base_params, p)}")
        return 0

    try:
        ensure_device(adb, serial)
    except Exception as exc:  # noqa: BLE001
        if args.adb_mode == "wrapper" and "Cannot mkdir" in str(exc):
            print("[sweep] wrapper mode hit adb home permission issue, fallback to --adb-mode direct")
            adb = AdbRunner(repo_root, "direct")
            ensure_device(adb, serial)
        else:
            raise

    results: List[Dict[str, object]] = []
    for idx, params in enumerate(combos, start=1):
        eval_params = make_eval_params(args.base_params, params)
        print(f"[run {idx:03d}/{len(combos)}] params={eval_params}")

        start_args = build_start_args(
            serial=serial,
            video_path=args.video_path,
            target_path=args.target_path,
            target_paths=args.target_paths,
            eval_params=eval_params,
            tracker_mode=args.tracker_mode,
        )
        force_stop_args = build_simple_args(serial, "shell", "am", "force-stop", PACKAGE_NAME)
        clear_log_args = build_simple_args(serial, "logcat", "-c")
        dump_log_args = build_simple_args(serial, "logcat", "-d", "-s", f"{TRACKER_TAG}:V")
        pidof_args = build_simple_args(serial, "shell", "pidof", PACKAGE_NAME)

        error = ""
        log_text = ""
        parsed: Dict[str, object] = {}
        attempt = 0
        log_file = logs_dir / f"run_{idx:03d}.log"
        while True:
            attempt += 1
            error = ""
            log_text = ""
            app_pid = ""
            try:
                adb.run(clear_log_args, timeout_sec=20, check=True)
                adb.run(force_stop_args, timeout_sec=20, check=False)
                adb.run(start_args, timeout_sec=20, check=True)
                pid_out = adb.run(pidof_args, timeout_sec=10, check=False).strip()
                if pid_out:
                    app_pid = pid_out.split()[0].strip()
                time.sleep(total_wait_sec)
                adb.run(force_stop_args, timeout_sec=20, check=False)
                if app_pid:
                    dump_args = build_simple_args(serial, "logcat", "-d", f"--pid={app_pid}")
                    log_text = adb.run(dump_args, timeout_sec=30, check=True)
                else:
                    log_text = adb.run(dump_log_args, timeout_sec=30, check=True)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                try:
                    if app_pid:
                        dump_args = build_simple_args(serial, "logcat", "-d", f"--pid={app_pid}")
                        log_text = adb.run(dump_args, timeout_sec=30, check=False)
                    else:
                        log_text = adb.run(dump_log_args, timeout_sec=30, check=False)
                except Exception:  # noqa: BLE001
                    pass

            log_file = logs_dir / f"run_{idx:03d}_try{attempt}.log"
            log_file.write_text(log_text, encoding="utf-8", errors="ignore")
            parsed = parse_tracker_log(log_text)
            has_metrics = int(parsed.get("frames", 0) or 0) > 0 or parsed.get("avg_frame_ms") is not None
            if has_metrics or attempt > args.retry_empty:
                break
            print(f"  -> empty metrics/log on attempt {attempt}, retrying...")
            time.sleep(0.6)
        first_lock_after_target = compute_first_lock_after_target(
            first_lock_sec=parsed["first_lock_sec"],  # type: ignore[arg-type]
            target_appear_sec=args.target_appear_sec,
        )
        early_lock = False
        raw_first_lock = parsed["first_lock_sec"]  # type: ignore[assignment]
        if isinstance(raw_first_lock, (int, float)):
            early_lock = raw_first_lock >= 0 and raw_first_lock + 1e-6 < max(0.0, args.target_appear_sec)
        score = compute_score(
            first_lock_after_target_sec=first_lock_after_target,
            early_lock=early_lock,
            locks=int(parsed["locks"]),  # type: ignore[arg-type]
            lost=int(parsed["lost"]),  # type: ignore[arg-type]
            max_track_streak=int(parsed["max_track_streak"]),  # type: ignore[arg-type]
            track_ratio=parsed["track_ratio"],  # type: ignore[arg-type]
            avg_frame_ms=parsed["avg_frame_ms"],  # type: ignore[arg-type]
            frames=int(parsed["frames"]),  # type: ignore[arg-type]
            weights=weights,
        )
        if error:
            score -= 200.0

        record = {
            "run": idx,
            "attempts": attempt,
            "score": round(score, 4),
            "first_lock_sec": parsed["first_lock_sec"],
            "first_lock_after_target_sec": first_lock_after_target,
            "early_lock": int(early_lock),
            "locks": parsed["locks"],
            "lost": parsed["lost"],
            "max_track_streak": parsed["max_track_streak"],
            "track_ratio": parsed["track_ratio"],
            "avg_frame_ms": parsed["avg_frame_ms"],
            "frames": parsed["frames"],
            "refine_pass_count": parsed["refine_pass_count"],
            "params": eval_params,
            "error": error,
            "log_file": str(log_file.relative_to(repo_root)),
        }
        results.append(record)
        first_raw = "NA" if record["first_lock_sec"] is None else f"{record['first_lock_sec']}s"
        first_after = "NA" if record["first_lock_after_target_sec"] is None else f"{record['first_lock_after_target_sec']}s"
        print(
            f"  -> score={record['score']} lock={record['locks']} lost={record['lost']} "
            f"firstRaw={first_raw} firstAfter={first_after} early={record['early_lock']} "
            f"ratio={record['track_ratio']} avgMs={record['avg_frame_ms']}"
        )
        time.sleep(max(0.0, args.cooldown_sec))

    results_sorted = sorted(results, key=lambda x: float(x["score"]), reverse=True)

    csv_path = run_dir / "results.csv"
    fieldnames = [
        "run",
        "attempts",
        "score",
        "first_lock_sec",
        "first_lock_after_target_sec",
        "early_lock",
        "locks",
        "lost",
        "max_track_streak",
        "track_ratio",
        "avg_frame_ms",
        "frames",
        "refine_pass_count",
        "params",
        "error",
        "log_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    top_path = run_dir / "top10.json"
    top10 = results_sorted[:10]
    top_path.write_text(json.dumps(top10, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nTop candidates:")
    for i, row in enumerate(top10, start=1):
        top_first_raw = "NA" if row["first_lock_sec"] is None else f"{row['first_lock_sec']}s"
        top_first_after = (
            "NA" if row["first_lock_after_target_sec"] is None else f"{row['first_lock_after_target_sec']}s"
        )
        print(
            f"  {i:02d}. score={row['score']:<8} firstRaw={top_first_raw} firstAfter={top_first_after} "
            f"early={row['early_lock']} lock/lost={row['locks']}/{row['lost']} streak={row['max_track_streak']} "
            f"ratio={row['track_ratio']} avgMs={row['avg_frame_ms']}"
        )
        print(f"      {row['params']}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {top_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
