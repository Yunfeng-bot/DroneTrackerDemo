#!/usr/bin/env python3
"""
Replay auto-tuning helper for DroneTrackerDemo.

This script runs replay sessions with different `eval_params`, captures Tracker logs,
extracts quality/perf metrics, computes a fitness score, and writes ranked outputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import os
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
VIDEO_REPLAY_TAG = "VideoReplay"
MAIN_ACTIVITY_TAG = "MainActivity"
ANDROID_RUNTIME_TAG = "AndroidRuntime"
GPS_READY_ACTION = "com.example.dronetracker.GPS_READY"
LOGCAT_PRIMARY_TIMEOUT_SEC = 30
LOGCAT_FALLBACK_TIMEOUT_SEC = 25
LOGCAT_TIMEOUT_FALLBACK_TAIL_LINES = 6000


def build_logcat_tag_filters(profile: str) -> List[str]:
    tracker_level = "I" if profile == "l2" else "W"
    return [
        f"{TRACKER_TAG}:{tracker_level}",
        f"{VIDEO_REPLAY_TAG}:E",
        f"{MAIN_ACTIVITY_TAG}:E",
        f"{ANDROID_RUNTIME_TAG}:E",
    ]


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
    relock_latency_weight: float = 12.0
    relock_missing_penalty: float = 90.0
    wrong_lock_ratio_weight: float = 120.0


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


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError(f"invalid float list: {raw}")
    return values


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
    if preset == "fixed":
        return {}
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
    if preset == "phase2_demo":
        return {
            # TrackGuard cluster
            "track_guard_max_jump": [1.00, 1.10, 1.25],
            "track_guard_min_area_ratio": [0.40, 0.45],
            "track_guard_max_area_ratio": [2.00, 2.20],
            "track_guard_min_appearance": [-0.12, -0.08],
            # Temporal cluster
            "temporal_min_conf_small_refined": [0.24, 0.28],
            "temporal_min_conf_base": [0.32, 0.35],
            "temporal_live_conf_relax": [0.02, 0.03],
            # Kalman cluster
            "kalman_r_scale_low": [6.0, 8.0],
            "kalman_r_scale_occlusion": [10.0, 12.0],
            "kalman_prior_min_iou": [0.25, 0.30],
            "kalman_prior_stale_ms": [100, 140],
        }
    if preset == "phase2_overnight":
        return {
            # TrackGuard cluster
            "track_guard_max_jump": [0.95, 1.00, 1.10, 1.20],
            "track_guard_min_area_ratio": [0.35, 0.40, 0.45],
            "track_guard_max_area_ratio": [1.80, 2.00, 2.20, 2.50],
            "track_guard_drop_streak": [1, 2],
            "track_guard_min_appearance": [-0.16, -0.12, -0.08, -0.04],
            # Temporal cluster
            "temporal_min_conf_small_refined": [0.22, 0.24, 0.28, 0.32],
            "temporal_min_conf_base": [0.30, 0.34, 0.38],
            "temporal_live_conf_relax": [0.01, 0.02, 0.03, 0.04],
            "temporal_live_conf_floor": [0.08, 0.10, 0.12],
            # Kalman cluster
            "kalman_r_scale_high": [0.20, 0.25, 0.35],
            "kalman_r_scale_low": [5.0, 7.0, 9.0, 12.0],
            "kalman_r_scale_occlusion": [8.0, 12.0, 16.0],
            "kalman_prior_min_iou": [0.20, 0.25, 0.30, 0.35],
            "kalman_prior_stale_ms": [80, 120, 160, 220],
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


def int_or_default(raw: object, default: int = 0) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def parse_box(raw: str | None) -> Tuple[int, int, int, int] | None:
    if raw is None:
        return None
    m = re.fullmatch(r"(-?\d+),(-?\d+),(\d+)x(\d+)", raw.strip())
    if not m:
        return None
    try:
        x = int(m.group(1))
        y = int(m.group(2))
        w = int(m.group(3))
        h = int(m.group(4))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def parse_size(raw: str | None) -> Tuple[int, int] | None:
    if raw is None:
        return None
    m = re.fullmatch(r"(\d+)x(\d+)", raw.strip())
    if not m:
        return None
    try:
        w = int(m.group(1))
        h = int(m.group(2))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


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


def _metric_first_lock_after_target_sec(row: Dict[str, object]) -> float:
    raw = row.get("first_lock_after_target_sec")
    if isinstance(raw, (int, float)):
        return float(raw)
    # missing/invalid first-lock is always worst on this axis (minimize).
    return float("inf")


def _metric_track_ratio(row: Dict[str, object]) -> float:
    raw = row.get("track_ratio")
    if isinstance(raw, (int, float)):
        return float(raw)
    # missing metrics are always worst on this axis (maximize).
    return float("-inf")


def _metric_avg_frame_ms(row: Dict[str, object]) -> float:
    raw = row.get("avg_frame_ms")
    if isinstance(raw, (int, float)):
        return float(raw)
    # missing metrics are always worst on this axis (minimize).
    return float("inf")


def _metric_lost(row: Dict[str, object]) -> float:
    raw = row.get("lost")
    if isinstance(raw, (int, float)):
        return float(raw)
    return float("inf")


def _metric_relock_latency(row: Dict[str, object], key: str) -> float:
    raw = row.get(key)
    if isinstance(raw, (int, float)):
        return float(raw)
    return float("inf")


def _metric_wrong_lock_ratio(row: Dict[str, object]) -> float:
    raw = row.get("wrong_lock_ratio_in_windows")
    if isinstance(raw, (int, float)):
        return float(raw)
    return float("inf")


def pareto_dominates(a: Dict[str, object], b: Dict[str, object]) -> bool:
    """
    Multi-objective dominance:
    - minimize: first_lock_after_target_sec, avg_frame_ms, lost
    - maximize: track_ratio
    """
    a_first = _metric_first_lock_after_target_sec(a)
    b_first = _metric_first_lock_after_target_sec(b)
    a_ratio = _metric_track_ratio(a)
    b_ratio = _metric_track_ratio(b)
    a_ms = _metric_avg_frame_ms(a)
    b_ms = _metric_avg_frame_ms(b)
    a_lost = _metric_lost(a)
    b_lost = _metric_lost(b)
    a_relock20 = _metric_relock_latency(a, "relock_latency_20s")
    b_relock20 = _metric_relock_latency(b, "relock_latency_20s")
    a_relock24 = _metric_relock_latency(a, "relock_latency_24s")
    b_relock24 = _metric_relock_latency(b, "relock_latency_24s")
    a_wrong = _metric_wrong_lock_ratio(a)
    b_wrong = _metric_wrong_lock_ratio(b)

    not_worse = (
        a_first <= b_first
        and a_ratio >= b_ratio
        and a_ms <= b_ms
        and a_lost <= b_lost
        and a_relock20 <= b_relock20
        and a_relock24 <= b_relock24
        and a_wrong <= b_wrong
    )
    strictly_better = (
        a_first < b_first
        or a_ratio > b_ratio
        or a_ms < b_ms
        or a_lost < b_lost
        or a_relock20 < b_relock20
        or a_relock24 < b_relock24
        or a_wrong < b_wrong
    )
    return not_worse and strictly_better


def extract_pareto_front(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    front: List[Dict[str, object]] = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if pareto_dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(
        key=lambda r: (
            _metric_wrong_lock_ratio(r),
            _metric_relock_latency(r, "relock_latency_20s"),
            _metric_relock_latency(r, "relock_latency_24s"),
            -_metric_track_ratio(r),
            _metric_avg_frame_ms(r),
            _metric_lost(r),
            _metric_first_lock_after_target_sec(r),
            -float(r.get("score", 0.0)),
        )
    )
    return front


def load_p0_windows(repo_root: Path, raw_path: str) -> List[Dict[str, object]]:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        windows_raw = data
    else:
        windows_raw = data.get("windows") if isinstance(data, dict) else None
    if not isinstance(windows_raw, list):
        raise ValueError(f"invalid windows file (windows not list): {path}")

    windows: List[Dict[str, object]] = []
    for i, item in enumerate(windows_raw):
        if not isinstance(item, dict):
            continue
        start_sec = float(item.get("start_sec", 0.0))
        return_sec = float(item.get("return_sec", start_sec + 1.0))
        metric_key_sec_raw = item.get("metric_key_sec")
        metric_key_sec = (
            float(metric_key_sec_raw)
            if isinstance(metric_key_sec_raw, (int, float))
            else start_sec
        )
        eval_window_sec = float(item.get("eval_window_sec", 4.0))
        relock_max_sec = float(item.get("relock_max_sec", 8.0))
        min_track_ratio_raw = item.get("min_track_ratio")
        min_track_ratio = (
            float(min_track_ratio_raw)
            if isinstance(min_track_ratio_raw, (int, float))
            else None
        )
        x_range = item.get("target_center_x_ratio")
        if (
            isinstance(x_range, list)
            and len(x_range) == 2
            and all(isinstance(v, (int, float)) for v in x_range)
        ):
            low = float(x_range[0])
            high = float(x_range[1])
        else:
            # Compatibility for MVP-5 windows schema (state-only windows without GT x-range).
            low = 0.0
            high = 1.0
        if low > high:
            low, high = high, low
        end_sec_raw = item.get("end_sec")
        end_sec: float | None = None
        if isinstance(end_sec_raw, (int, float)) and end_sec_raw > start_sec:
            eval_window_sec = float(end_sec_raw) - start_sec
            end_sec = float(end_sec_raw)
        else:
            end_sec = start_sec + max(0.1, eval_window_sec)
        windows.append(
            {
                "name": str(item.get("name", item.get("label", f"w{i+1}"))),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "metric_key_sec": metric_key_sec,
                "return_sec": return_sec,
                "eval_window_sec": max(0.1, eval_window_sec),
                "relock_max_sec": max(0.1, relock_max_sec),
                "target_center_x_ratio": [max(0.0, low), min(1.0, high)],
                "min_track_ratio": min_track_ratio,
            }
        )
    return windows


def git_info(repo_root: Path) -> Dict[str, object]:
    def _run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()

    commit = _run("rev-parse", "HEAD")
    short = _run("rev-parse", "--short", "HEAD")
    status = _run("status", "--porcelain")
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "short_commit": short,
        "branch": branch,
        "dirty": bool(status),
    }


def device_package_info(adb: "AdbRunner", serial: str | None) -> Dict[str, object]:
    info: Dict[str, object] = {}
    pkg_dump = adb.run(
        build_simple_args(serial, "shell", "dumpsys", "package", PACKAGE_NAME),
        timeout_sec=30,
        check=False,
    )
    m_name = re.search(r"\bversionName=([^\s]+)", pkg_dump)
    m_code = re.search(r"\bversionCode=(\d+)", pkg_dump)
    info["package"] = PACKAGE_NAME
    info["version_name"] = m_name.group(1) if m_name else ""
    info["version_code"] = m_code.group(1) if m_code else ""

    model = adb.run(
        build_simple_args(serial, "shell", "getprop", "ro.product.model"),
        timeout_sec=10,
        check=False,
    ).strip()
    device = adb.run(
        build_simple_args(serial, "shell", "getprop", "ro.product.device"),
        timeout_sec=10,
        check=False,
    ).strip()
    info["device_model"] = model
    info["device_name"] = device
    return info


class AdbRunner:
    def __init__(self, repo_root: Path, mode: str) -> None:
        self.repo_root = repo_root
        self.mode = mode
        self.wrapper = repo_root / "tools" / "adb_exec.ps1"
        self.wrapper_args_file = repo_root / "tools" / "adb_args.json"
        self.android_home = repo_root / ".android_home"
        self.env = self._build_env()

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        android_home = str(self.android_home)
        os.makedirs(android_home, exist_ok=True)

        # Keep Android env deterministic to avoid AGP/adb conflicts:
        # - only ANDROID_USER_HOME + ADB_VENDOR_KEYS
        # - explicitly clear ANDROID_PREFS_ROOT / ANDROID_SDK_HOME
        env.pop("ANDROID_PREFS_ROOT", None)
        env.pop("ANDROID_SDK_HOME", None)
        env["ANDROID_USER_HOME"] = android_home
        env["ADB_VENDOR_KEYS"] = android_home
        env["HOME"] = str(self.repo_root)
        env["USERPROFILE"] = str(self.repo_root)
        drive, tail = os.path.splitdrive(str(self.repo_root))
        if drive:
            env["HOMEDRIVE"] = drive
            env["HOMEPATH"] = tail if tail else "\\"
        return env

    def _write_wrapper_args(self, args: List[str]) -> None:
        if self.mode != "wrapper":
            return
        payload = json.dumps(args, ensure_ascii=False)
        self.wrapper_args_file.write_text(payload, encoding="utf-8")

    def _build_cmd(self, args: List[str]) -> List[str]:
        if self.mode == "wrapper":
            return [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.wrapper),
            ]
        return ["adb", *args]

    @staticmethod
    def _detect_home_fallback(text: str) -> bool:
        low = text.lower()
        return (
            "codexsandboxoffline\\.android" in low
            or "codexsandboxoffline/.android" in low
            or "cannot mkdir" in low and ".android" in low
        )

    def run(self, args: List[str], timeout_sec: int = 30, check: bool = True) -> str:
        self._write_wrapper_args(args)
        cmd = self._build_cmd(args)

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            # Python 3.14 + msys can hang when subprocess.run() handles timeout.
            # Kill process tree first, then drain output to avoid deadlocks.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    cwd=str(self.repo_root),
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except Exception:  # noqa: BLE001
                stdout, stderr = "", ""
            merged_timeout = (stdout or "") + (stderr or "")
            raise RuntimeError(
                f"adb command timeout ({timeout_sec}s): {' '.join(args)}\n{merged_timeout.strip()}"
            )

        merged = (stdout or "") + (stderr or "")
        if self._detect_home_fallback(merged):
            raise RuntimeError(
                "adb home fallback detected (CodexSandboxOffline/.android). "
                "Use project-local env only: ANDROID_USER_HOME=<repo>/.android_home."
            )
        if check and proc.returncode != 0:
            raise RuntimeError(f"adb command failed: {' '.join(args)}\n{merged.strip()}")
        return merged

    def reset_server(self, serial: str | None) -> None:
        # Restart once to ensure adb daemon picks project-local env.
        self.run(build_simple_args(serial, "kill-server"), timeout_sec=20, check=False)
        self.run(build_simple_args(serial, "start-server"), timeout_sec=25, check=True)


def parse_tracker_log(text: str) -> Dict[str, object]:
    summary_lines = []
    perf_lines = []
    lock_lines = []
    lost_lines = []
    replay_lines = []
    replay_input_state = ""
    replay_first_frame_seen = False
    replay_fatal_seen = False
    replay_stop_seen = False
    replay_decoded_frames: int | None = None
    replay_null_frames: int | None = None
    replay_frame_w: int | None = None
    replay_frame_h: int | None = None
    refine_pass_count = 0
    lock_events: List[Dict[str, object]] = []
    lost_events: List[Dict[str, object]] = []
    perf_samples: List[Dict[str, object]] = []
    cand_dump_events: List[Dict[str, object]] = []
    descend_offset_events: List[Dict[str, object]] = []

    for line in text.splitlines():
        if "EVAL_SUMMARY" in line:
            summary_lines.append(line)
        if "EVAL_PERF" in line:
            perf_lines.append(line)
            kv = parse_kv(line)
            perf_samples.append(
                {
                    "replay_pts_sec": to_float(kv.get("replayPtsSec")),
                    "tracking": (kv.get("tracking", "").lower() == "true"),
                    "box": parse_box(kv.get("box")),
                    "backend_active": kv.get("backendActive", ""),
                    "line": line,
                }
            )
        if "EVAL_EVENT type=LOCK" in line:
            lock_lines.append(line)
            kv = parse_kv(line)
            lock_events.append(
                {
                    "replay_pts_sec": to_float(kv.get("replayPtsSec")),
                    "reason": kv.get("reason", ""),
                    "backend": kv.get("backend", ""),
                    "box": parse_box(kv.get("box")),
                    "line": line,
                }
            )
        if "EVAL_EVENT type=LOST" in line:
            lost_lines.append(line)
            kv = parse_kv(line)
            lost_events.append(
                {
                    "replay_pts_sec": to_float(kv.get("replayPtsSec")),
                    "reason": kv.get("reason", ""),
                    "backend": kv.get("backend", ""),
                    "line": line,
                }
            )
        if "EVAL_EVENT type=REPLAY " in line:
            replay_lines.append(line)
            kv = parse_kv(line)
            state = kv.get("state", "")
            if state == "first_frame":
                replay_first_frame_seen = True
                size = parse_size(kv.get("size"))
                if size is not None:
                    replay_frame_w, replay_frame_h = size
            elif state == "fatal":
                replay_fatal_seen = True
            elif state == "stop":
                replay_stop_seen = True
            if state in {"fatal", "stop"}:
                replay_decoded_frames = to_int(kv.get("decoded"))
                replay_null_frames = to_int(kv.get("nullCount"))
        if "EVAL_EVENT type=REPLAY_INPUT" in line:
            kv = parse_kv(line)
            replay_input_state = kv.get("state", "")
        if "EVAL_EVENT type=SEARCH_REFINE state=pass" in line:
            refine_pass_count += 1
        if "EVAL_EVENT type=CAND_DUMP" in line:
            kv = parse_kv(line)
            expected_x_raw = kv.get("expectedX", "")
            expected_x_min: float | None = None
            expected_x_max: float | None = None
            if "," in expected_x_raw:
                a, b = expected_x_raw.split(",", 1)
                expected_x_min = to_float(a)
                expected_x_max = to_float(b)
            cand_dump_events.append(
                {
                    "replay_pts_sec": to_float(kv.get("replayPtsSec")),
                    "session": to_int(kv.get("session")),
                    "src": kv.get("src", ""),
                    "rank": to_int(kv.get("rank")),
                    "cx": to_float(kv.get("cx")),
                    "cy": to_float(kv.get("cy")),
                    "w": to_int(kv.get("w")),
                    "h": to_int(kv.get("h")),
                    "appearance": to_float(kv.get("appearance")),
                    "spatial": to_float(kv.get("spatial")),
                    "d2": to_float(kv.get("d2")),
                    "fusion": to_float(kv.get("fusion")),
                    "conf": to_float(kv.get("conf")),
                    "good": to_int(kv.get("good")),
                    "inliers": to_int(kv.get("inliers")),
                    "homo": to_int(kv.get("homo")),
                    "tier": kv.get("tier", ""),
                    "expected_x_min": expected_x_min,
                    "expected_x_max": expected_x_max,
                    "line": line,
                }
            )
        if "EVAL_EVENT type=DESCEND_OFFSET" in line:
            kv = parse_kv(line)
            descend_offset_events.append(
                {
                    "x": to_float(kv.get("x")),
                    "y": to_float(kv.get("y")),
                    "conf": to_float(kv.get("conf")),
                    "state": kv.get("state", ""),
                    "t": to_float(kv.get("t")),
                    "session": to_int(kv.get("session")),
                    "line": line,
                }
            )

    summary_kv = parse_kv(summary_lines[-1]) if summary_lines else {}
    perf_kv = parse_kv(perf_lines[-1]) if perf_lines else {}
    lock_kv = parse_kv(lock_lines[-1]) if lock_lines else {}

    locks = to_int(summary_kv.get("locks"))
    if locks is None:
        locks = to_int(perf_kv.get("locks"))
    if locks is None:
        locks = len(lock_events)
    lost = to_int(summary_kv.get("lost"))
    if lost is None:
        lost = to_int(perf_kv.get("lost"))
    if lost is None:
        lost = len(lost_events)

    first_lock_sec = to_float(summary_kv.get("firstLockSec"))
    if first_lock_sec is None:
        first_lock_sec = to_float(perf_kv.get("firstLockSec"))
    if first_lock_sec is None:
        first_lock_sec = to_float(lock_kv.get("firstLockSec"))
    first_lock_replay_sec = to_float(summary_kv.get("firstLockReplaySec"))
    if first_lock_replay_sec is None:
        first_lock_replay_sec = to_float(perf_kv.get("firstLockReplaySec"))
    if first_lock_replay_sec is None:
        first_lock_replay_sec = to_float(lock_kv.get("firstLockReplaySec"))

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
    replay_pts_sec = to_float(summary_kv.get("replayPtsSec"))
    if replay_pts_sec is None:
        replay_pts_sec = to_float(perf_kv.get("replayPtsSec"))

    descend_offset_first_t: float | None = None
    descend_offset_last_state = ""
    descend_offset_oob_count = 0
    descend_offset_fail_count = 0
    for event in descend_offset_events:
        t = event.get("t")
        if isinstance(t, (int, float)) and math.isfinite(float(t)) and descend_offset_first_t is None:
            descend_offset_first_t = float(t)
        state = str(event.get("state", "")).upper()
        if state:
            descend_offset_last_state = state
            if state == "FAIL":
                descend_offset_fail_count += 1
        x = event.get("x")
        y = event.get("y")
        conf = event.get("conf")
        x_oob = isinstance(x, (int, float)) and math.isfinite(float(x)) and not (-1.0 <= float(x) <= 1.0)
        y_oob = isinstance(y, (int, float)) and math.isfinite(float(y)) and not (-1.0 <= float(y) <= 1.0)
        conf_oob = (
            isinstance(conf, (int, float))
            and math.isfinite(float(conf))
            and not (0.0 <= float(conf) <= 1.0)
        )
        if x_oob or y_oob or conf_oob:
            descend_offset_oob_count += 1

    return {
        "locks": locks,
        "lost": lost,
        "first_lock_sec": first_lock_sec,
        "first_lock_replay_sec": first_lock_replay_sec,
        "replay_pts_sec": replay_pts_sec,
        "track_ratio": track_ratio,
        "avg_frame_ms": avg_frame_ms,
        "max_track_streak": max_track_streak,
        "frames": frames,
        "refine_pass_count": refine_pass_count,
        "lock_events": lock_events,
        "lost_events": lost_events,
        "replay_input_state": replay_input_state,
        "replay_first_frame_seen": replay_first_frame_seen,
        "replay_fatal_seen": replay_fatal_seen,
        "replay_stop_seen": replay_stop_seen,
        "replay_decoded_frames": replay_decoded_frames,
        "replay_null_frames": replay_null_frames,
        "replay_event_count": len(replay_lines),
        "replay_frame_w": replay_frame_w,
        "replay_frame_h": replay_frame_h,
        "perf_samples": perf_samples,
        "cand_dump_events": cand_dump_events,
        "descend_offset_events": descend_offset_events,
        "descend_offset_first_t": descend_offset_first_t,
        "descend_offset_last_state": descend_offset_last_state,
        "descend_offset_oob_count": descend_offset_oob_count,
        "descend_offset_fail_count": descend_offset_fail_count,
    }


def _build_state_events(parsed: Dict[str, object]) -> List[Tuple[float, str, str]]:
    events: List[Tuple[float, str, str]] = []
    for item in parsed.get("lock_events", []):  # type: ignore[assignment]
        if not isinstance(item, dict):
            continue
        ts = item.get("replay_pts_sec")
        if isinstance(ts, (int, float)):
            events.append((float(ts), "lock", str(item.get("reason", ""))))
    for item in parsed.get("lost_events", []):  # type: ignore[assignment]
        if not isinstance(item, dict):
            continue
        ts = item.get("replay_pts_sec")
        if isinstance(ts, (int, float)):
            events.append((float(ts), "lost", str(item.get("reason", ""))))
    events.sort(key=lambda it: (it[0], 0 if it[1] == "lost" else 1))
    return events


def _is_locked_at(events: List[Tuple[float, str, str]], ts_sec: float) -> bool:
    locked = False
    for event_ts, state, _reason in events:
        if event_ts > ts_sec:
            break
        locked = state == "lock"
    return locked


def _box_center_x_ratio(box: Tuple[int, int, int, int], frame_w: int) -> float:
    x, _y, w, _h = box
    center_x = float(x) + float(w) * 0.5
    return center_x / float(max(1, frame_w))


def _is_correct_box_by_window(
    box: Tuple[int, int, int, int] | None,
    frame_w: int,
    x_ratio_range: Tuple[float, float],
) -> bool:
    if box is None:
        return False
    ratio = _box_center_x_ratio(box, frame_w)
    low, high = x_ratio_range
    return (ratio >= low) and (ratio <= high)


def _is_wrong_lock_lost_reason(reason: str) -> bool:
    low = reason.lower()
    if not low:
        return False
    keywords = (
        "track_guard",
        "verify_",
        "anchor",
        "appearance",
        "native_spatial_gate",
        "native_conf_",
    )
    return any(token in low for token in keywords)


def _compute_s1_with_windows_label(
    parsed: Dict[str, object],
    window_labels: List[Dict[str, object]],
) -> Dict[str, object]:
    frame_w = int(parsed.get("replay_frame_w") or 0)
    if frame_w <= 0:
        frame_w = 1280

    perf_samples_raw = parsed.get("perf_samples", [])
    perf_samples: List[Dict[str, object]] = []
    if isinstance(perf_samples_raw, list):
        for item in perf_samples_raw:
            if isinstance(item, dict):
                ts = item.get("replay_pts_sec")
                if isinstance(ts, (int, float)):
                    perf_samples.append(item)
    perf_samples.sort(key=lambda it: float(it.get("replay_pts_sec", 0.0)))

    relock_latency: Dict[str, float | None] = {}
    total_track_samples = 0
    total_wrong_samples = 0
    total_lock_events = 0
    total_wrong_lock_events = 0
    lock_events_raw = parsed.get("lock_events", [])
    lock_events: List[Dict[str, object]] = []
    if isinstance(lock_events_raw, list):
        for item in lock_events_raw:
            if isinstance(item, dict):
                ts = item.get("replay_pts_sec")
                if isinstance(ts, (int, float)):
                    lock_events.append(item)
    lock_events.sort(key=lambda it: float(it.get("replay_pts_sec", 0.0)))

    def _metric_name(raw_name: object, fallback: str) -> str:
        text = str(raw_name).strip().lower()
        if not text:
            text = fallback
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return text or fallback

    for window in window_labels:
        start_sec = float(window.get("start_sec", 0.0))
        end_sec_raw = window.get("end_sec")
        if isinstance(end_sec_raw, (int, float)) and float(end_sec_raw) > start_sec:
            end_sec = float(end_sec_raw)
        else:
            end_sec = start_sec + max(0.1, float(window.get("eval_window_sec", 4.0)))
        return_sec = float(window.get("return_sec", start_sec + 1.0))
        eval_window_sec = float(window.get("eval_window_sec", 4.0))
        relock_max_sec = float(window.get("relock_max_sec", 8.0))
        metric_key_sec_raw = window.get("metric_key_sec")
        if isinstance(metric_key_sec_raw, (int, float)):
            metric_key_sec = float(metric_key_sec_raw)
        else:
            metric_key_sec = start_sec
        range_raw = window.get("target_center_x_ratio", [0.0, 1.0])
        low = float(range_raw[0])  # type: ignore[index]
        high = float(range_raw[1])  # type: ignore[index]
        if low > high:
            low, high = high, low
        x_range = (max(0.0, low), min(1.0, high))
        metric_name = _metric_name(window.get("name"), f"w{int(round(start_sec))}")

        key = f"relock_latency_{int(round(metric_key_sec))}s"
        relock_value: float | None = None

        before_candidates = [
            s
            for s in perf_samples
            if float(s.get("replay_pts_sec", -1.0)) <= return_sec
        ]
        if before_candidates:
            last_before = before_candidates[-1]
            before_ts = float(last_before.get("replay_pts_sec", -1.0))
            before_tracking = bool(last_before.get("tracking", False))
            before_box = last_before.get("box")
            before_correct = (
                before_tracking
                and isinstance(before_box, tuple)
                and _is_correct_box_by_window(before_box, frame_w, x_range)
            )
            if before_correct and (return_sec - before_ts) <= 1.5:
                relock_value = 0.0

        if relock_value is None:
            deadline = return_sec + max(0.1, relock_max_sec)
            for lock in lock_events:
                ts = float(lock.get("replay_pts_sec", -1.0))
                if ts < return_sec:
                    continue
                if ts > deadline:
                    break
                box = lock.get("box")
                if not isinstance(box, tuple):
                    continue
                if _is_correct_box_by_window(box, frame_w, x_range):
                    relock_value = ts - return_sec
                    break

        if relock_value is None:
            deadline = return_sec + max(0.1, relock_max_sec)
            for sample in perf_samples:
                ts = float(sample.get("replay_pts_sec", -1.0))
                if ts < return_sec:
                    continue
                if ts > deadline:
                    break
                if not bool(sample.get("tracking", False)):
                    continue
                box = sample.get("box")
                if not isinstance(box, tuple):
                    continue
                if _is_correct_box_by_window(box, frame_w, x_range):
                    relock_value = ts - return_sec
                    break
        relock_latency[key] = relock_value

        eval_end = return_sec + max(0.1, eval_window_sec)

        # wrong-lock KPI is event-oriented: count locks that happen inside the
        # return window, then classify each lock box against label range.
        for lock in lock_events:
            ts = float(lock.get("replay_pts_sec", -1.0))
            if ts < return_sec:
                continue
            if ts > eval_end:
                break
            box = lock.get("box")
            if not isinstance(box, tuple):
                continue
            total_lock_events += 1
            if not _is_correct_box_by_window(box, frame_w, x_range):
                total_wrong_lock_events += 1

        # Keep sample-based counters as a fallback when logs miss LOCK events.
        for sample in perf_samples:
            ts = float(sample.get("replay_pts_sec", -1.0))
            if ts < return_sec:
                continue
            if ts > eval_end:
                break
            if not bool(sample.get("tracking", False)):
                continue
            box = sample.get("box")
            if not isinstance(box, tuple):
                continue
            total_track_samples += 1
            if not _is_correct_box_by_window(box, frame_w, x_range):
                total_wrong_samples += 1

        # Track-ratio KPI aligned with window labels (for MVP-5 steady window acceptance).
        samples_in_window = [
            s
            for s in perf_samples
            if start_sec <= float(s.get("replay_pts_sec", -1.0)) <= end_sec
        ]
        tracked_count = 0
        if samples_in_window:
            tracked_count = sum(1 for s in samples_in_window if bool(s.get("tracking", False)))
            window_track_ratio: float | None = tracked_count / len(samples_in_window)
        else:
            window_track_ratio = None
        out_key = f"{metric_name}_track_ratio"
        relock_latency[out_key] = window_track_ratio
        relock_latency[f"{metric_name}_track_ratio_samples"] = len(samples_in_window)
        relock_latency[f"{metric_name}_track_ratio_tracked"] = tracked_count

        min_track_ratio_raw = window.get("min_track_ratio")
        if isinstance(min_track_ratio_raw, (int, float)):
            min_track_ratio = float(min_track_ratio_raw)
            relock_latency[f"{metric_name}_min_track_ratio"] = min_track_ratio
            relock_latency[f"{metric_name}_track_ratio_pass"] = (
                window_track_ratio is not None and window_track_ratio >= min_track_ratio
            )
            # Primary acceptance alias for summary/readability.
            if "window_track_ratio" not in relock_latency:
                relock_latency["window_track_ratio"] = window_track_ratio
                relock_latency["window_track_ratio_min"] = min_track_ratio
                relock_latency["window_track_ratio_window_name"] = str(window.get("name", metric_name))
                relock_latency["window_track_ratio_samples"] = len(samples_in_window)
                relock_latency["window_track_ratio_tracked"] = tracked_count

    if total_lock_events > 0:
        wrong_lock_ratio = total_wrong_lock_events / total_lock_events
        window_lock_count = total_lock_events
        window_wrong_count = total_wrong_lock_events
        wrong_src = "window_label_lock_event"
    else:
        wrong_lock_ratio = (
            total_wrong_samples / total_track_samples if total_track_samples > 0 else None
        )
        window_lock_count = total_track_samples
        window_wrong_count = total_wrong_samples
        wrong_src = "window_label_sample_fallback"

    out: Dict[str, object] = dict(relock_latency)
    out["wrong_lock_ratio_in_windows"] = wrong_lock_ratio
    out["window_lock_count"] = window_lock_count
    out["window_wrong_lock_count"] = window_wrong_count
    out["wrong_lock_metric_source"] = wrong_src
    return out


def _find_window_for_ts(
    window_labels: List[Dict[str, object]],
    replay_pts_sec: float,
) -> Dict[str, object] | None:
    for window in window_labels:
        return_sec = float(window.get("return_sec", window.get("start_sec", 0.0)))
        eval_window_sec = float(window.get("eval_window_sec", 4.0))
        end_sec = return_sec + max(0.1, eval_window_sec)
        if return_sec <= replay_pts_sec <= end_sec:
            return window
    return None


def analyze_candidate_dumps(
    rows: List[Dict[str, object]],
    window_labels: List[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "groups_total": 0,
        "groups_in_windows": 0,
        "gt_missing_topk": 0,
        "gt_present_topk": 0,
        "rank0_wrong_when_gt_present": 0,
        "gt_rank_hist": {},
        "rank0_src_hist": {},
        "rank0_wrong_src_hist": {},
        "rank0_app_minus_gt_app_mean": None,
        "diagnosis": "insufficient_data",
    }
    if not rows:
        return summary

    def rank_or_default(item: Dict[str, object], default: int = 9999) -> int:
        return int_or_default(item.get("rank"), default)

    grouped: Dict[Tuple[int, int, float], List[Dict[str, object]]] = {}
    for row in rows:
        run = int(row.get("run", 0) or 0)
        attempt = int(row.get("attempt", 0) or 0)
        ts_raw = row.get("replay_pts_sec")
        if not isinstance(ts_raw, (int, float)):
            continue
        ts_key = round(float(ts_raw), 3)
        key = (run, attempt, ts_key)
        grouped.setdefault(key, []).append(row)

    gt_rank_hist: Dict[str, int] = {}
    rank0_src_hist: Dict[str, int] = {}
    rank0_wrong_src_hist: Dict[str, int] = {}
    app_deltas: List[float] = []
    groups_in_windows = 0
    gt_missing = 0
    gt_present = 0
    rank0_wrong = 0

    for (_run, _attempt, ts_key), group in sorted(grouped.items()):
        window = _find_window_for_ts(window_labels, float(ts_key))
        if window is None:
            continue
        groups_in_windows += 1
        range_raw = window.get("target_center_x_ratio", [0.0, 1.0])
        low = float(range_raw[0])  # type: ignore[index]
        high = float(range_raw[1])  # type: ignore[index]
        if low > high:
            low, high = high, low
        x_range = (max(0.0, low), min(1.0, high))

        candidates = sorted(
            group,
            key=lambda item: (
                rank_or_default(item),
                -float(item.get("fusion", -1.0) or -1.0),
            ),
        )
        frame_w = int(candidates[0].get("frame_w", 0) or 0)
        if frame_w <= 0:
            frame_w = 1280

        valid_candidates: List[Dict[str, object]] = []
        for cand in candidates:
            cx = cand.get("cx")
            rank = rank_or_default(cand)
            if not isinstance(cx, (int, float)) or not isinstance(rank, (int, float)):
                continue
            ratio = float(cx) / float(frame_w)
            is_gt = x_range[0] <= ratio <= x_range[1]
            cand["center_x_ratio"] = ratio
            cand["is_gt"] = is_gt
            cand["rank_norm"] = rank
            valid_candidates.append(cand)

        if not valid_candidates:
            continue

        rank0 = next((c for c in valid_candidates if int(c.get("rank_norm", 9999)) == 0), None)
        if rank0 is not None:
            src = str(rank0.get("src", ""))
            rank0_src_hist[src] = rank0_src_hist.get(src, 0) + 1

        gt_candidates = [c for c in valid_candidates if bool(c.get("is_gt", False))]
        if not gt_candidates:
            gt_missing += 1
            continue

        gt_present += 1
        best_gt = min(gt_candidates, key=lambda c: int(c.get("rank_norm", 9999)))
        best_gt_rank = int(best_gt.get("rank_norm", 9999))
        gt_rank_key = str(best_gt_rank if best_gt_rank < 4 else "3+")
        gt_rank_hist[gt_rank_key] = gt_rank_hist.get(gt_rank_key, 0) + 1

        if rank0 is not None:
            if not bool(rank0.get("is_gt", False)):
                rank0_wrong += 1
                src = str(rank0.get("src", ""))
                rank0_wrong_src_hist[src] = rank0_wrong_src_hist.get(src, 0) + 1
            rank0_app = rank0.get("appearance")
            gt_app = best_gt.get("appearance")
            if isinstance(rank0_app, (int, float)) and isinstance(gt_app, (int, float)):
                app_deltas.append(float(rank0_app) - float(gt_app))

    summary["groups_total"] = len(grouped)
    summary["groups_in_windows"] = groups_in_windows
    summary["gt_missing_topk"] = gt_missing
    summary["gt_present_topk"] = gt_present
    summary["rank0_wrong_when_gt_present"] = rank0_wrong
    summary["gt_rank_hist"] = gt_rank_hist
    summary["rank0_src_hist"] = rank0_src_hist
    summary["rank0_wrong_src_hist"] = rank0_wrong_src_hist
    if app_deltas:
        summary["rank0_app_minus_gt_app_mean"] = sum(app_deltas) / len(app_deltas)
    else:
        summary["rank0_app_minus_gt_app_mean"] = None

    diagnosis = "insufficient_data"
    if groups_in_windows > 0:
        missing_ratio = gt_missing / groups_in_windows
        wrong_ratio = (rank0_wrong / gt_present) if gt_present > 0 else 0.0
        if gt_present == 0 or missing_ratio >= 0.60:
            diagnosis = "recall_layer"
        elif wrong_ratio >= 0.50:
            diagnosis = "scoring_layer"
        else:
            diagnosis = "gate_or_temporal_layer"
    summary["diagnosis"] = diagnosis
    return summary


def _compute_s1_fallback(
    parsed: Dict[str, object],
    window_start_secs: List[float],
    return_offset_sec: float,
    window_sec: float,
    relock_max_sec: float,
    wrong_lock_horizon_sec: float,
) -> Dict[str, object]:
    events = _build_state_events(parsed)
    lock_events: List[Tuple[float, str]] = [
        (ts, reason) for ts, state, reason in events if state == "lock"
    ]
    lost_events: List[Tuple[float, str]] = [
        (ts, reason) for ts, state, reason in events if state == "lost"
    ]

    relock_latency: Dict[str, float | None] = {}
    total_window_locks = 0
    total_wrong_locks = 0

    for start_sec in window_start_secs:
        return_sec = start_sec + max(0.0, return_offset_sec)
        field = f"relock_latency_{int(round(start_sec))}s"

        if _is_locked_at(events, return_sec):
            relock_latency[field] = 0.0
        else:
            relock_value: float | None = None
            deadline = return_sec + max(0.1, relock_max_sec)
            for ts, _reason in lock_events:
                if ts < return_sec:
                    continue
                if ts > deadline:
                    break
                relock_value = ts - return_sec
                break
            relock_latency[field] = relock_value

        window_end = start_sec + max(0.1, window_sec)
        locks_in_window = [(ts, reason) for ts, reason in lock_events if start_sec <= ts <= window_end]
        total_window_locks += len(locks_in_window)

        for lock_ts, _lock_reason in locks_in_window:
            lost_deadline = lock_ts + max(0.1, wrong_lock_horizon_sec)
            for lost_ts, lost_reason in lost_events:
                if lost_ts < lock_ts:
                    continue
                if lost_ts > lost_deadline:
                    break
                if _is_wrong_lock_lost_reason(lost_reason):
                    total_wrong_locks += 1
                    break

    wrong_lock_ratio = (
        total_wrong_locks / total_window_locks if total_window_locks > 0 else None
    )
    out: Dict[str, object] = dict(relock_latency)
    out["wrong_lock_ratio_in_windows"] = wrong_lock_ratio
    out["window_lock_count"] = total_window_locks
    out["window_wrong_lock_count"] = total_wrong_locks
    out["wrong_lock_metric_source"] = "reason_heuristic_fallback"
    return out


def compute_s1_window_metrics(
    parsed: Dict[str, object],
    window_start_secs: List[float],
    return_offset_sec: float,
    window_sec: float,
    relock_max_sec: float,
    wrong_lock_horizon_sec: float,
    window_labels: List[Dict[str, object]] | None = None,
) -> Dict[str, object]:
    if window_labels:
        return _compute_s1_with_windows_label(parsed, window_labels)
    return _compute_s1_fallback(
        parsed=parsed,
        window_start_secs=window_start_secs,
        return_offset_sec=return_offset_sec,
        window_sec=window_sec,
        relock_max_sec=relock_max_sec,
        wrong_lock_horizon_sec=wrong_lock_horizon_sec,
    )


def has_eval_metrics_text(text: str) -> bool:
    if ("EVAL_PERF" in text) or ("EVAL_SUMMARY" in text):
        return True
    if ("EVAL_EVENT type=LOCK" in text) or ("EVAL_EVENT type=LOST" in text):
        return True
    if "EVAL_EVENT type=REPLAY_INPUT state=ready" in text:
        return True
    if "EVAL_EVENT type=REPLAY_INPUT state=video_missing" in text:
        return True
    if "EVAL_EVENT type=REPLAY_INPUT state=target_missing" in text:
        return True
    if "EVAL_EVENT type=REPLAY state=first_frame" in text:
        return True
    if "EVAL_EVENT type=REPLAY state=stop" in text:
        return True
    if "EVAL_EVENT type=REPLAY state=fatal" in text:
        return True
    low = text.lower()
    return ("video replay failed" in low) or ("exception" in low and "mainactivity" in low)


def build_start_args(
    serial: str | None,
    video_path: str,
    target_path: str,
    target_paths: str,
    eval_params: str,
    tracker_mode: str,
    replay_loop: bool,
    replay_start_sec: float,
    target_appear_sec: float,
    replay_catchup: bool,
    replay_fps: float,
) -> List[str]:
    args: List[str] = []
    if serial:
        args += ["-s", serial]
    args += [
        "shell",
        "am",
        "start",
        "-S",
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
        "--ef",
        "eval_replay_start_sec",
        format_value(max(0.0, replay_start_sec)),
        "--ef",
        "eval_target_appear_sec",
        format_value(max(0.0, target_appear_sec)),
        "--ez",
        "eval_replay_catchup",
        "true" if replay_catchup else "false",
        "--ef",
        "eval_replay_fps",
        format_value(max(0.0, replay_fps)),
        "--ez",
        "eval_loop",
        "true" if replay_loop else "false",
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


def build_gps_ready_broadcast_args(serial: str | None, ready: bool, reason: str) -> List[str]:
    args = build_simple_args(
        serial,
        "shell",
        "am",
        "broadcast",
        "-a",
        GPS_READY_ACTION,
        "--ez",
        "ready",
        "true" if ready else "false",
    )
    if reason.strip():
        args += ["--es", "reason", reason.strip()]
    return args


def build_logcat_dump_args(
    serial: str | None,
    tag_filters: List[str],
    tail_lines: int,
) -> List[str]:
    args = build_simple_args(
        serial,
        "logcat",
        "-d",
        "-b",
        "main",
    )
    if tail_lines > 0:
        args += ["-t", str(tail_lines)]
    args += ["-s", *tag_filters]
    return args


def check_device_file(adb: AdbRunner, serial: str | None, path: str) -> Tuple[bool, str]:
    args = build_simple_args(serial, "shell", "ls", "-la", path)
    out = adb.run(args, timeout_sec=20, check=False)
    low = out.lower()
    missing = ("no such file" in low) or ("not found" in low) or ("permission denied" in low)
    return (not missing), out.strip()


def build_replay_target_list(target_path: str, target_paths: str) -> List[str]:
    if target_paths.strip():
        return [p.strip() for p in target_paths.split(";") if p.strip()]
    return [target_path.strip()]


def preflight_replay_inputs(
    adb: AdbRunner,
    serial: str | None,
    video_path: str,
    target_path: str,
    target_paths: str,
) -> None:
    checks: List[Tuple[str, str]] = [("video", video_path)]
    for tp in build_replay_target_list(target_path, target_paths):
        checks.append(("target", tp))

    failures: List[str] = []
    for kind, path in checks:
        ok, detail = check_device_file(adb, serial, path)
        print(f"[preflight] {kind}: {path}")
        if detail:
            first_line = detail.splitlines()[0]
            print(f"[preflight]   {first_line}")
        if not ok:
            failures.append(f"{kind}:{path}")

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"replay input missing/unreadable on device: {joined}")


def ensure_device(adb: AdbRunner, serial: str | None) -> None:
    out = adb.run(build_simple_args(serial, "devices"), timeout_sec=20, check=True)
    if serial:
        if serial not in out:
            raise RuntimeError(f"device {serial} not present in adb devices output")
    else:
        device_lines = [ln for ln in out.splitlines() if "\tdevice" in ln]
        if not device_lines:
            raise RuntimeError("no adb device connected")


def wait_for_app_pid(
    adb: AdbRunner,
    serial: str | None,
    timeout_sec: float,
    poll_interval_sec: float = 0.3,
) -> str:
    pidof_args = build_simple_args(serial, "shell", "pidof", PACKAGE_NAME)
    deadline = time.time() + max(0.1, timeout_sec)
    while time.time() < deadline:
        out = adb.run(pidof_args, timeout_sec=10, check=False).strip()
        if out:
            return out.split()[0].strip()
        time.sleep(max(0.05, poll_interval_sec))
    return ""


def configure_logcat_buffer(adb: AdbRunner, serial: str | None, size: str) -> None:
    text = size.strip()
    if not text:
        return
    adb.run(build_simple_args(serial, "logcat", "-G", text), timeout_sec=20, check=False)


def write_manifest(
    run_dir: Path,
    repo_root: Path,
    adb: AdbRunner,
    serial: str | None,
    args: argparse.Namespace,
    run_tag: str,
) -> None:
    manifest = {
        "run_tag": run_tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(repo_root),
        "app": device_package_info(adb, serial),
        "device_serial": serial or "",
        "adb_mode": args.adb_mode,
        "log_profile": args.log_profile,
        "logcat_buffer_size": args.logcat_buffer_size,
        "logcat_tail_lines": args.logcat_tail_lines,
        "video_path": args.video_path,
        "replay_start_sec": args.replay_start_sec,
        "replay_catchup": args.replay_catchup,
        "replay_fps": args.replay_fps,
        "gps_ready_at_sec": args.gps_ready_at_sec,
        "gps_ready_reason": args.gps_ready_reason,
        "target_path": args.target_path,
        "target_paths": args.target_paths,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    parser.add_argument("--video-path", default="/sdcard/Download/Video_Search/scene_20260417.mp4")
    parser.add_argument("--target-path", default="/sdcard/Download/Video_Search/target0417_s640.jpg")
    parser.add_argument(
        "--target-paths",
        default="",
        help="semicolon-separated template paths for multi-template replay, e.g. /sdcard/.../T1.jpg;/sdcard/.../T2.jpg",
    )
    parser.add_argument("--tracker-mode", default="enhanced", choices=["enhanced", "baseline"])
    parser.add_argument(
        "--replay-loop",
        dest="replay_loop",
        action="store_true",
        help="request replay loop mode in MainActivity",
    )
    parser.add_argument(
        "--no-replay-loop",
        dest="replay_loop",
        action="store_false",
        help="disable replay loop mode in MainActivity (default)",
    )
    parser.set_defaults(replay_loop=False)
    parser.add_argument(
        "--replay-catchup",
        dest="replay_catchup",
        action="store_true",
        help="allow replay catchup (frame skip when processing lags)",
    )
    parser.add_argument(
        "--no-replay-catchup",
        dest="replay_catchup",
        action="store_false",
        help="disable replay catchup for strict 1x playback",
    )
    parser.set_defaults(replay_catchup=False)
    parser.add_argument("--duration-sec", type=float, default=10.0, help="effective scoring window (seconds) after target appears")
    parser.add_argument("--target-appear-sec", type=float, default=0.0, help="target appears after this offset in replay (seconds)")
    parser.add_argument(
        "--gps-ready-at-sec",
        type=float,
        default=-1.0,
        help="when >=0, send GPS_READY broadcast at this many seconds after each replay start",
    )
    parser.add_argument(
        "--gps-ready-reason",
        default="replay_script",
        help="reason string for GPS_READY broadcast",
    )
    parser.add_argument(
        "--replay-start-sec",
        type=float,
        default=0.0,
        help="start replay from this video offset (seconds); target_appear_sec remains relative to replay timeline",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help="override replay fps (>0 to enable); 0 means auto from metadata/frame count/default",
    )
    parser.add_argument(
        "--p0-window-label-json",
        default="tools/replay_sop/p0_windows.json",
        help="window label json for wrong_lock/relock scoring; empty to disable",
    )
    parser.add_argument(
        "--windows",
        default="",
        help="alias of --p0-window-label-json for checklist compatibility",
    )
    parser.add_argument(
        "--s1-window-start-secs",
        default="20,24",
        help="comma-separated replay window starts (seconds) for P0 relock evaluation",
    )
    parser.add_argument(
        "--s1-return-offset-sec",
        type=float,
        default=1.0,
        help="seconds after each window start when target is expected to return",
    )
    parser.add_argument(
        "--s1-window-sec",
        type=float,
        default=4.0,
        help="window length (seconds) for wrong-lock ratio counting",
    )
    parser.add_argument(
        "--s1-relock-max-sec",
        type=float,
        default=8.0,
        help="max relock wait (seconds) after target return before marking timeout",
    )
    parser.add_argument(
        "--s1-wrong-lock-horizon-sec",
        type=float,
        default=3.0,
        help="if a lock is followed by mismatch-lost within this horizon, count as wrong-lock",
    )
    parser.add_argument("--cooldown-sec", type=float, default=1.0, help="pause between runs")
    parser.add_argument(
        "--preset",
        default="default",
        choices=["fixed", "quick", "default", "overnight", "phase2_demo", "phase2_overnight"],
    )
    parser.add_argument("--grid", action="append", default=[], help="override/add grid key=val1|val2|...")
    parser.add_argument("--base-params", default="", help="extra fixed eval_params prefix")
    parser.add_argument("--max-runs", type=int, default=0, help="cap total runs after expansion (0 means no cap)")
    parser.add_argument("--shuffle", action="store_true", help="shuffle run order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-empty", type=int, default=1, help="retry count when a run captures empty metrics/log")
    parser.add_argument(
        "--retry-low-replay",
        type=int,
        default=1,
        help="retry count when replayPtsSec does not cover target window",
    )
    parser.add_argument(
        "--min-replay-coverage",
        type=float,
        default=0.90,
        help="minimum replay coverage ratio for target window [0,1]",
    )
    parser.add_argument(
        "--max-wait-sec",
        type=float,
        default=180.0,
        help="upper bound for adaptive wait when replay coverage is low",
    )
    parser.add_argument(
        "--logcat-buffer-size",
        default="16M",
        help="set device logcat buffer once before sweep, e.g. 8M/16M; empty to skip",
    )
    parser.add_argument(
        "--log-profile",
        default="l1",
        choices=["l1", "l2"],
        help="l1: Tracker:W summary logs; l2: Tracker:I evidence logs",
    )
    parser.add_argument(
        "--logcat-tail-lines",
        type=int,
        default=0,
        help="if >0, dump only last N logcat lines; for l2 recommended >=5000",
    )
    parser.add_argument(
        "--adb-mode",
        default="direct",
        choices=["wrapper", "direct"],
        help="direct: call adb binary with injected env (recommended); wrapper: invoke tools/adb_exec.ps1",
    )
    parser.add_argument(
        "--pid-wait-sec",
        type=float,
        default=8.0,
        help="wait time for app pid after cold start",
    )
    parser.add_argument(
        "--force-cold-start",
        dest="force_cold_start",
        action="store_true",
        help="always force-stop before launch (default)",
    )
    parser.add_argument(
        "--no-force-cold-start",
        dest="force_cold_start",
        action="store_false",
        help="skip force-stop before launch",
    )
    parser.set_defaults(force_cold_start=True)
    parser.add_argument(
        "--adb-reset-server",
        dest="adb_reset_server",
        action="store_true",
        help="restart adb server once before sweep to lock env (default)",
    )
    parser.add_argument(
        "--no-adb-reset-server",
        dest="adb_reset_server",
        action="store_false",
        help="do not restart adb server before sweep",
    )
    parser.set_defaults(adb_reset_server=True)
    parser.add_argument("--output-dir", default="tools/auto_tune/out")
    parser.add_argument(
        "--out",
        default="",
        help="exact output directory (alias for checklist); when set, no extra timestamp subdir is appended",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.windows.strip():
        args.p0_window_label_json = args.windows.strip()
    s1_window_starts = parse_float_list(args.s1_window_start_secs)
    if args.log_profile == "l2" and 0 < args.logcat_tail_lines < 5000:
        print("[sweep] l2 profile tail too small, bump to 5000 lines")
        args.logcat_tail_lines = 5000

    script_path = Path(__file__).resolve()
    # Support both original location (tools/auto_tune/) and root hotfix copy.
    if (script_path.parent / "tools" / "auto_tune").exists():
        repo_root = script_path.parent
    else:
        repo_root = script_path.parents[2]
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = repo_root / args.output_dir
    if args.out.strip():
        explicit = Path(args.out.strip())
        run_dir = explicit if explicit.is_absolute() else (repo_root / explicit)
    else:
        run_dir = output_root / run_tag
    logs_dir = run_dir / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback_root = repo_root / ".codex_tmp" / "auto_tune" / "out"
        run_dir = fallback_root / run_tag
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sweep] output dir not writable ({output_root}): {exc}. fallback={fallback_root}")

    window_labels: List[Dict[str, object]] = []
    if args.p0_window_label_json.strip():
        try:
            window_labels = load_p0_windows(repo_root, args.p0_window_label_json.strip())
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] failed to load p0 window labels: {exc}")
            window_labels = []

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
        "gps_ready_at_sec": args.gps_ready_at_sec,
        "gps_ready_reason": args.gps_ready_reason,
        "replay_start_sec": args.replay_start_sec,
        "replay_catchup": args.replay_catchup,
        "replay_fps": args.replay_fps,
        "s1_window_start_secs": s1_window_starts,
        "p0_window_label_json": args.p0_window_label_json,
        "p0_window_label_loaded_count": len(window_labels),
        "s1_return_offset_sec": args.s1_return_offset_sec,
        "s1_window_sec": args.s1_window_sec,
        "s1_relock_max_sec": args.s1_relock_max_sec,
        "s1_wrong_lock_horizon_sec": args.s1_wrong_lock_horizon_sec,
        "cooldown_sec": args.cooldown_sec,
        "video_path": args.video_path,
        "target_path": args.target_path,
        "target_paths": args.target_paths,
        "tracker_mode": args.tracker_mode,
        "replay_loop": args.replay_loop,
        "base_params": args.base_params,
        "adb_mode": args.adb_mode,
        "adb_reset_server": args.adb_reset_server,
        "force_cold_start": args.force_cold_start,
        "pid_wait_sec": args.pid_wait_sec,
        "retry_empty": args.retry_empty,
        "retry_low_replay": args.retry_low_replay,
        "min_replay_coverage": args.min_replay_coverage,
        "max_wait_sec": args.max_wait_sec,
        "logcat_buffer_size": args.logcat_buffer_size,
        "log_profile": args.log_profile,
        "logcat_tail_lines": args.logcat_tail_lines,
        "serial": serial or "",
        "grid": grid,
        "weights": weights.__dict__,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sweep] output={run_dir}")
    total_wait_sec = max(0.1, args.duration_sec + max(0.0, args.target_appear_sec))
    print(
        f"[sweep] runs={len(combos)} preset={args.preset} "
        f"effective={args.duration_sec:.1f}s targetAppear={args.target_appear_sec:.1f}s "
        f"replayStart={max(0.0, args.replay_start_sec):.1f}s catchup={int(args.replay_catchup)} "
        f"replayFps={args.replay_fps:.2f} gpsReadyAt={args.gps_ready_at_sec:.1f}s total={total_wait_sec:.1f}s"
    )
    print(
        f"[sweep] adbMode={args.adb_mode} resetServer={int(args.adb_reset_server)} "
        f"coldStart={int(args.force_cold_start)} pidWait={args.pid_wait_sec:.1f}s "
        f"loop={int(args.replay_loop)} minReplayCov={args.min_replay_coverage:.2f} retryLowReplay={args.retry_low_replay} "
        f"maxWait={args.max_wait_sec:.1f}s logcatBuf={args.logcat_buffer_size or 'skip'} "
        f"logProfile={args.log_profile} tail={args.logcat_tail_lines}"
    )
    print(
        f"[sweep] s1Windows={','.join([str(int(round(x))) for x in s1_window_starts])} "
        f"returnOffset={args.s1_return_offset_sec:.1f}s window={args.s1_window_sec:.1f}s "
        f"relockMax={args.s1_relock_max_sec:.1f}s wrongHorizon={args.s1_wrong_lock_horizon_sec:.1f}s "
        f"labelWindows={len(window_labels)}"
    )

    if args.dry_run:
        for i, p in enumerate(combos[:10], start=1):
            print(f"  dry#{i}: {make_eval_params(args.base_params, p)}")
        return 0

    try:
        if args.adb_reset_server:
            adb.reset_server(serial)
        ensure_device(adb, serial)
        preflight_replay_inputs(
            adb=adb,
            serial=serial,
            video_path=args.video_path,
            target_path=args.target_path,
            target_paths=args.target_paths,
        )
        configure_logcat_buffer(adb, serial, args.logcat_buffer_size)
        write_manifest(run_dir, repo_root, adb, serial, args, run_tag)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        wrapper_fallback_tokens = (
            "Cannot mkdir",
            "fallback detected",
            "adb command timeout",
        )
        if args.adb_mode == "wrapper" and any(token in err for token in wrapper_fallback_tokens):
            print(f"[sweep] wrapper bootstrap failed ({err}); fallback to --adb-mode direct")
            adb = AdbRunner(repo_root, "direct")
            if args.adb_reset_server:
                adb.reset_server(serial)
            ensure_device(adb, serial)
            preflight_replay_inputs(
                adb=adb,
                serial=serial,
                video_path=args.video_path,
                target_path=args.target_path,
                target_paths=args.target_paths,
            )
            configure_logcat_buffer(adb, serial, args.logcat_buffer_size)
            write_manifest(run_dir, repo_root, adb, serial, args, run_tag)
        else:
            raise

    logcat_tags = build_logcat_tag_filters(args.log_profile)
    results: List[Dict[str, object]] = []
    cand_dump_all: List[Dict[str, object]] = []
    descend_offset_all: List[Dict[str, object]] = []
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
            replay_loop=args.replay_loop,
            replay_start_sec=max(0.0, args.replay_start_sec),
            target_appear_sec=max(0.0, args.target_appear_sec),
            replay_catchup=args.replay_catchup,
            replay_fps=max(0.0, args.replay_fps),
        )
        force_stop_args = build_simple_args(serial, "shell", "am", "force-stop", PACKAGE_NAME)
        gps_ready_broadcast_args = build_gps_ready_broadcast_args(
            serial=serial,
            ready=True,
            reason=args.gps_ready_reason,
        )
        clear_log_args = build_simple_args(serial, "logcat", "-c")
        dump_log_args = build_logcat_dump_args(
            serial=serial,
            tag_filters=logcat_tags,
            tail_lines=max(0, int(args.logcat_tail_lines)),
        )
        dump_log_args_timeout_fallback = build_logcat_dump_args(
            serial=serial,
            tag_filters=logcat_tags,
            tail_lines=max(
                max(0, int(args.logcat_tail_lines)),
                LOGCAT_TIMEOUT_FALLBACK_TAIL_LINES,
            ),
        )

        error = ""
        log_text = ""
        parsed: Dict[str, object] = {}
        attempt = 0
        log_file = logs_dir / f"run_{idx:03d}.log"
        wait_sec = total_wait_sec
        low_replay_retry_left = max(0, int(args.retry_low_replay))
        required_replay_sec = max(0.0, args.target_appear_sec) + max(0.1, args.duration_sec)
        best_log_text = ""
        best_parsed: Dict[str, object] = {}
        best_log_file: Path | None = None
        best_error = ""
        best_replay_pts = -1.0
        best_frames = -1
        has_metrics = False
        while True:
            attempt += 1
            error = ""
            log_text = ""
            app_pid = ""
            try:
                adb.run(clear_log_args, timeout_sec=20, check=True)
                if args.force_cold_start:
                    adb.run(force_stop_args, timeout_sec=20, check=False)
                adb.run(start_args, timeout_sec=25, check=True)
                app_pid = wait_for_app_pid(adb, serial, timeout_sec=args.pid_wait_sec)
                if not app_pid:
                    note = f"pid missing after start (wait={args.pid_wait_sec}s)"
                    error = f"{error}; {note}" if error else note
                start_wait_ts = time.time()
                gps_ready_sent = args.gps_ready_at_sec < 0.0
                while True:
                    elapsed = time.time() - start_wait_ts
                    if (not gps_ready_sent) and elapsed >= args.gps_ready_at_sec:
                        try:
                            adb.run(gps_ready_broadcast_args, timeout_sec=20, check=True)
                            print(
                                f"  -> gps_ready broadcast sent at +{elapsed:.2f}s "
                                f"reason={args.gps_ready_reason}"
                            )
                        except Exception as gps_exc:  # noqa: BLE001
                            note = f"gps_ready broadcast failed: {gps_exc}"
                            error = f"{error}; {note}" if error else note
                        finally:
                            gps_ready_sent = True
                    remaining = wait_sec - elapsed
                    if remaining <= 0.0:
                        break
                    time.sleep(min(0.20, max(0.02, remaining)))
                try:
                    log_text = adb.run(dump_log_args, timeout_sec=LOGCAT_PRIMARY_TIMEOUT_SEC, check=True)
                    if not has_eval_metrics_text(log_text):
                        note = "logcat empty/weak"
                        error = f"{error}; {note}" if error else note
                except Exception as dump_exc:  # noqa: BLE001
                    note = f"logcat dump failed: {dump_exc}"
                    error = f"{error}; {note}" if error else note
                    dump_error_text = str(dump_exc).lower()
                    fallback_args = (
                        dump_log_args_timeout_fallback
                        if "timeout" in dump_error_text
                        else dump_log_args
                    )
                    log_text = adb.run(fallback_args, timeout_sec=LOGCAT_FALLBACK_TIMEOUT_SEC, check=False)
                if args.force_cold_start:
                    adb.run(force_stop_args, timeout_sec=20, check=False)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                try:
                    fallback_args = (
                        dump_log_args_timeout_fallback
                        if "timeout" in error.lower()
                        else dump_log_args
                    )
                    log_text = adb.run(fallback_args, timeout_sec=LOGCAT_FALLBACK_TIMEOUT_SEC, check=False)
                except Exception:  # noqa: BLE001
                    pass
                if (not log_text) and ("EVAL_" in error or "Tracker" in error):
                    log_text = error
                try:
                    if args.force_cold_start:
                        adb.run(force_stop_args, timeout_sec=20, check=False)
                except Exception:  # noqa: BLE001
                    pass

            log_file = logs_dir / f"run_{idx:03d}_try{attempt}.log"
            log_file.write_text(log_text, encoding="utf-8", errors="ignore")
            parsed = parse_tracker_log(log_text)
            has_metrics = int(parsed.get("frames", 0) or 0) > 0 or parsed.get("avg_frame_ms") is not None
            replay_pts = parsed.get("replay_pts_sec")
            if has_metrics:
                replay_pts_value = float(replay_pts) if isinstance(replay_pts, (int, float)) else -1.0
                frames_value = int(parsed.get("frames", 0) or 0)
                better = (
                    replay_pts_value > best_replay_pts
                    or (replay_pts_value == best_replay_pts and frames_value > best_frames)
                )
                if better:
                    best_log_text = log_text
                    best_parsed = dict(parsed)
                    best_log_file = log_file
                    best_error = error
                    best_replay_pts = replay_pts_value
                    best_frames = frames_value
            has_replay_window = (
                isinstance(replay_pts, (int, float))
                and replay_pts >= required_replay_sec * max(0.0, min(1.0, args.min_replay_coverage))
            )
            if has_metrics and (not has_replay_window) and low_replay_retry_left > 0:
                low_replay_retry_left -= 1
                prev_wait = wait_sec
                next_wait = max(wait_sec * 1.7, wait_sec + 10.0)
                if isinstance(replay_pts, (int, float)) and replay_pts > 0:
                    required_by_rate = wait_sec * (required_replay_sec / replay_pts) * 1.10
                    next_wait = max(next_wait, required_by_rate)
                wait_sec = min(max(5.0, next_wait), max(5.0, float(args.max_wait_sec)))
                print(
                    f"  -> replay coverage low: replayPts={replay_pts} need>="
                    f"{required_replay_sec * args.min_replay_coverage:.2f}s "
                    f"(wait {prev_wait:.1f}s -> {wait_sec:.1f}s), retrying..."
                )
                time.sleep(0.6)
                continue
            if has_metrics or attempt > args.retry_empty:
                break
            print(f"  -> empty metrics/log on attempt {attempt}, retrying...")
            time.sleep(0.6)

        if (not has_metrics) and best_parsed:
            parsed = dict(best_parsed)
            log_text = best_log_text
            if best_log_file is not None:
                log_file = best_log_file
            note = "used best previous attempt due latest empty metrics"
            if best_error:
                note += f" (bestError={best_error})"
            error = f"{error}; {note}" if error else note

        first_lock_metric = parsed.get("first_lock_replay_sec")
        first_lock_metric_source = "replay" if isinstance(first_lock_metric, (int, float)) and first_lock_metric >= 0 else "wall"
        if first_lock_metric_source == "wall":
            first_lock_metric = parsed.get("first_lock_sec")
        replay_pts_final = parsed.get("replay_pts_sec")
        replay_window_ok = (
            isinstance(replay_pts_final, (int, float))
            and replay_pts_final >= required_replay_sec * max(0.0, min(1.0, args.min_replay_coverage))
        )

        first_lock_after_target = compute_first_lock_after_target(
            first_lock_sec=first_lock_metric,  # type: ignore[arg-type]
            target_appear_sec=args.target_appear_sec,
        )
        s1_metrics = compute_s1_window_metrics(
            parsed=parsed,
            window_start_secs=s1_window_starts,
            return_offset_sec=args.s1_return_offset_sec,
            window_sec=args.s1_window_sec,
            relock_max_sec=args.s1_relock_max_sec,
            wrong_lock_horizon_sec=args.s1_wrong_lock_horizon_sec,
            window_labels=window_labels,
        )
        cand_dump_rows_run: List[Dict[str, object]] = []
        cand_events_raw = parsed.get("cand_dump_events", [])
        if isinstance(cand_events_raw, list):
            frame_w = int(parsed.get("replay_frame_w") or 1280)
            for item in cand_events_raw:
                if not isinstance(item, dict):
                    continue
                ts = item.get("replay_pts_sec")
                if not isinstance(ts, (int, float)):
                    continue
                row = dict(item)
                row["run"] = idx
                row["attempt"] = attempt
                row["frame_w"] = frame_w
                row["params"] = eval_params
                row["log_file"] = str(log_file.relative_to(repo_root))
                window = _find_window_for_ts(window_labels, float(ts))
                row["window_name"] = str(window.get("name", "")) if isinstance(window, dict) else ""
                cand_dump_rows_run.append(row)
            cand_dump_all.extend(cand_dump_rows_run)
        cand_dump_count = len(cand_dump_rows_run)
        descend_rows_run: List[Dict[str, object]] = []
        descend_events_raw = parsed.get("descend_offset_events", [])
        if isinstance(descend_events_raw, list):
            for item in descend_events_raw:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row["run"] = idx
                row["attempt"] = attempt
                row["params"] = eval_params
                row["log_file"] = str(log_file.relative_to(repo_root))
                descend_rows_run.append(row)
            descend_offset_all.extend(descend_rows_run)
        descend_offset_count = len(descend_rows_run)
        early_lock = False
        raw_first_lock = first_lock_metric  # type: ignore[assignment]
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
        for start in s1_window_starts:
            key = f"relock_latency_{int(round(start))}s"
            latency = s1_metrics.get(key)
            if isinstance(latency, (int, float)):
                score -= float(latency) * weights.relock_latency_weight
            else:
                score -= weights.relock_missing_penalty
        wrong_lock_ratio = s1_metrics.get("wrong_lock_ratio_in_windows")
        if isinstance(wrong_lock_ratio, (int, float)):
            score -= float(wrong_lock_ratio) * weights.wrong_lock_ratio_weight
        if error:
            error_l = str(error).lower()
            has_metrics_for_error = int(parsed.get("frames", 0) or 0) > 0 or parsed.get("avg_frame_ms") is not None
            timeout_with_metrics = "timeout" in error_l and has_metrics_for_error
            if not timeout_with_metrics:
                score -= 200.0
        if not replay_window_ok:
            score -= 80.0

        record = {
            "run": idx,
            "attempts": attempt,
            "score": round(score, 4),
            "first_lock_sec": parsed["first_lock_sec"],
            "first_lock_replay_sec": parsed.get("first_lock_replay_sec"),
            "first_lock_metric_source": first_lock_metric_source,
            "first_lock_after_target_sec": first_lock_after_target,
            "replay_pts_sec": parsed.get("replay_pts_sec"),
            "replay_window_ok": int(replay_window_ok),
            "required_replay_sec": required_replay_sec,
            "replay_start_sec": max(0.0, args.replay_start_sec),
            "replay_catchup": int(args.replay_catchup),
            "replay_fps": max(0.0, args.replay_fps),
            "gps_ready_at_sec": args.gps_ready_at_sec,
            "gps_ready_reason": args.gps_ready_reason,
            "early_lock": int(early_lock),
            "locks": parsed["locks"],
            "lost": parsed["lost"],
            "max_track_streak": parsed["max_track_streak"],
            "track_ratio": parsed["track_ratio"],
            "avg_frame_ms": parsed["avg_frame_ms"],
            "frames": parsed["frames"],
            "refine_pass_count": parsed["refine_pass_count"],
            "cand_dump_count": cand_dump_count,
            "descend_offset_count": descend_offset_count,
            "descend_offset_first_t": parsed.get("descend_offset_first_t"),
            "descend_offset_last_state": parsed.get("descend_offset_last_state"),
            "descend_offset_oob_count": parsed.get("descend_offset_oob_count"),
            "descend_offset_fail_count": parsed.get("descend_offset_fail_count"),
            "wrong_lock_ratio_in_windows": s1_metrics.get("wrong_lock_ratio_in_windows"),
            "wrong_lock_metric_source": s1_metrics.get("wrong_lock_metric_source"),
            "window_lock_count": s1_metrics.get("window_lock_count"),
            "window_wrong_lock_count": s1_metrics.get("window_wrong_lock_count"),
            "window_track_ratio": s1_metrics.get("window_track_ratio"),
            "window_track_ratio_min": s1_metrics.get("window_track_ratio_min"),
            "window_track_ratio_window_name": s1_metrics.get("window_track_ratio_window_name"),
            "window_track_ratio_samples": s1_metrics.get("window_track_ratio_samples"),
            "window_track_ratio_tracked": s1_metrics.get("window_track_ratio_tracked"),
            "steady_track_window_track_ratio": s1_metrics.get("steady_track_window_track_ratio"),
            "steady_track_window_min_track_ratio": s1_metrics.get("steady_track_window_min_track_ratio"),
            "steady_track_window_track_ratio_pass": s1_metrics.get("steady_track_window_track_ratio_pass"),
            "steady_track_window_track_ratio_samples": s1_metrics.get("steady_track_window_track_ratio_samples"),
            "steady_track_window_track_ratio_tracked": s1_metrics.get("steady_track_window_track_ratio_tracked"),
            "replay_input_state": parsed.get("replay_input_state"),
            "replay_first_frame_seen": int(bool(parsed.get("replay_first_frame_seen"))),
            "replay_stop_seen": int(bool(parsed.get("replay_stop_seen"))),
            "replay_fatal_seen": int(bool(parsed.get("replay_fatal_seen"))),
            "replay_decoded_frames": parsed.get("replay_decoded_frames"),
            "replay_null_frames": parsed.get("replay_null_frames"),
            "replay_event_count": parsed.get("replay_event_count"),
            "params": eval_params,
            "error": error,
            "log_file": str(log_file.relative_to(repo_root)),
        }
        for start in s1_window_starts:
            key = f"relock_latency_{int(round(start))}s"
            record[key] = s1_metrics.get(key)
        results.append(record)
        first_raw = "NA" if record["first_lock_sec"] is None else f"{record['first_lock_sec']}s"
        first_replay = "NA" if record["first_lock_replay_sec"] is None else f"{record['first_lock_replay_sec']}s"
        first_after = "NA" if record["first_lock_after_target_sec"] is None else f"{record['first_lock_after_target_sec']}s"
        relock_parts = []
        for start in s1_window_starts:
            key = f"relock_latency_{int(round(start))}s"
            raw = record.get(key)
            relock_parts.append(f"{int(round(start))}s={'NA' if raw is None else f'{raw}s'}")
        relock_text = ",".join(relock_parts)
        print(
            f"  -> score={record['score']} lock={record['locks']} lost={record['lost']} "
            f"firstRaw={first_raw} firstReplay={first_replay} src={record['first_lock_metric_source']} firstAfter={first_after} replayPts={record['replay_pts_sec']} early={record['early_lock']} "
            f"ratio={record['track_ratio']} avgMs={record['avg_frame_ms']} cov={record['replay_window_ok']} "
            f"relock[{relock_text}] wrongRatio={record.get('wrong_lock_ratio_in_windows')} "
            f"wrongSrc={record.get('wrong_lock_metric_source')} "
            f"replayState={record.get('replay_input_state')} firstFrame={record.get('replay_first_frame_seen')} "
            f"candDump={record.get('cand_dump_count')} "
            f"descend={record.get('descend_offset_count')} "
            f"descState={record.get('descend_offset_last_state')} "
            f"descOob={record.get('descend_offset_oob_count')}"
        )
        time.sleep(max(0.0, args.cooldown_sec))

    results_sorted = sorted(results, key=lambda x: float(x["score"]), reverse=True)

    csv_path = run_dir / "results.csv"
    fieldnames = [
        "run",
        "attempts",
        "score",
        "first_lock_sec",
        "first_lock_replay_sec",
        "first_lock_metric_source",
        "first_lock_after_target_sec",
        "replay_pts_sec",
        "replay_window_ok",
        "required_replay_sec",
        "replay_start_sec",
        "replay_catchup",
        "replay_fps",
        "gps_ready_at_sec",
        "gps_ready_reason",
        "early_lock",
        "locks",
        "lost",
        "max_track_streak",
        "track_ratio",
        "avg_frame_ms",
        "frames",
        "refine_pass_count",
        "cand_dump_count",
        "descend_offset_count",
        "descend_offset_first_t",
        "descend_offset_last_state",
        "descend_offset_oob_count",
        "descend_offset_fail_count",
        "wrong_lock_ratio_in_windows",
        "wrong_lock_metric_source",
        "window_lock_count",
        "window_wrong_lock_count",
        "window_track_ratio",
        "window_track_ratio_min",
        "window_track_ratio_window_name",
        "window_track_ratio_samples",
        "window_track_ratio_tracked",
        "steady_track_window_track_ratio",
        "steady_track_window_min_track_ratio",
        "steady_track_window_track_ratio_pass",
        "steady_track_window_track_ratio_samples",
        "steady_track_window_track_ratio_tracked",
        "replay_input_state",
        "replay_first_frame_seen",
        "replay_stop_seen",
        "replay_fatal_seen",
        "replay_decoded_frames",
        "replay_null_frames",
        "replay_event_count",
        "params",
        "error",
        "log_file",
    ]
    for start in s1_window_starts:
        key = f"relock_latency_{int(round(start))}s"
        if key not in fieldnames:
            fieldnames.insert(fieldnames.index("wrong_lock_ratio_in_windows"), key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)

    top_path = run_dir / "top10.json"
    top10 = results_sorted[:10]
    top_path.write_text(json.dumps(top10, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = run_dir / "summary.json"
    mvp5_summary_path = run_dir / "mvp5_summary.json"
    if results_sorted:
        best = results_sorted[0]
        summary = {
            "run_tag": run_tag,
            "best_run": best.get("run"),
            "best_attempts": best.get("attempts"),
            "score": best.get("score"),
            "video_path": args.video_path,
            "target_path": args.target_path,
            "target_paths": args.target_paths,
            "gps_ready_at_sec": args.gps_ready_at_sec,
            "gps_ready_reason": args.gps_ready_reason,
            "replay_fps": args.replay_fps,
            "replay_start_sec": args.replay_start_sec,
            "descend_offset_count": best.get("descend_offset_count"),
            "descend_offset_first_t": best.get("descend_offset_first_t"),
            "descend_offset_last_state": best.get("descend_offset_last_state"),
            "descend_offset_oob_count": best.get("descend_offset_oob_count"),
            "descend_offset_fail_count": best.get("descend_offset_fail_count"),
            "first_lock_replay_sec": best.get("first_lock_replay_sec"),
            "first_lock_after_target_sec": best.get("first_lock_after_target_sec"),
            "locks": best.get("locks"),
            "lost": best.get("lost"),
            "track_ratio": best.get("track_ratio"),
            "avg_frame_ms": best.get("avg_frame_ms"),
            "frames": best.get("frames"),
            "replay_window_ok": best.get("replay_window_ok"),
            "replay_pts_sec": best.get("replay_pts_sec"),
            "wrong_lock_ratio_in_windows": best.get("wrong_lock_ratio_in_windows"),
            "window_track_ratio": best.get("window_track_ratio"),
            "window_track_ratio_min": best.get("window_track_ratio_min"),
            "window_track_ratio_window_name": best.get("window_track_ratio_window_name"),
            "window_track_ratio_samples": best.get("window_track_ratio_samples"),
            "window_track_ratio_tracked": best.get("window_track_ratio_tracked"),
            "steady_track_window_track_ratio": best.get("steady_track_window_track_ratio"),
            "steady_track_window_min_track_ratio": best.get("steady_track_window_min_track_ratio"),
            "steady_track_window_track_ratio_pass": best.get("steady_track_window_track_ratio_pass"),
            "steady_track_window_track_ratio_samples": best.get("steady_track_window_track_ratio_samples"),
            "steady_track_window_track_ratio_tracked": best.get("steady_track_window_track_ratio_tracked"),
            "log_file": best.get("log_file"),
            "params": best.get("params"),
        }
    else:
        summary = {
            "run_tag": run_tag,
            "video_path": args.video_path,
            "target_path": args.target_path,
            "target_paths": args.target_paths,
            "gps_ready_at_sec": args.gps_ready_at_sec,
            "gps_ready_reason": args.gps_ready_reason,
            "replay_fps": args.replay_fps,
            "replay_start_sec": args.replay_start_sec,
            "descend_offset_count": 0,
            "descend_offset_first_t": None,
            "descend_offset_last_state": "",
            "descend_offset_oob_count": 0,
            "descend_offset_fail_count": 0,
        }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    mvp5_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pareto_front = extract_pareto_front(results_sorted)
    pareto_path = run_dir / "pareto.json"
    pareto_path.write_text(json.dumps(pareto_front, indent=2, ensure_ascii=False), encoding="utf-8")
    pareto_csv_path = run_dir / "pareto.csv"
    with pareto_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pareto_front)

    cand_dump_csv_path = run_dir / "candidate_dump.csv"
    cand_summary_path = run_dir / "candidate_dump_summary.json"
    if cand_dump_all:
        cand_fieldnames = [
            "run",
            "attempt",
            "window_name",
            "replay_pts_sec",
            "session",
            "src",
            "rank",
            "cx",
            "cy",
            "w",
            "h",
            "frame_w",
            "appearance",
            "spatial",
            "d2",
            "fusion",
            "conf",
            "good",
            "inliers",
            "homo",
            "tier",
            "expected_x_min",
            "expected_x_max",
            "params",
            "log_file",
            "line",
        ]
        cand_sorted = sorted(
            cand_dump_all,
            key=lambda r: (
                int(r.get("run", 0) or 0),
                int(r.get("attempt", 0) or 0),
                float(r.get("replay_pts_sec", -1.0) or -1.0),
                int_or_default(r.get("rank"), 9999),
            ),
        )
        with cand_dump_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cand_fieldnames)
            writer.writeheader()
            writer.writerows(cand_sorted)
    cand_summary = analyze_candidate_dumps(cand_dump_all, window_labels)
    cand_summary_path.write_text(json.dumps(cand_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    descend_offset_csv_path = run_dir / "descend_offset.csv"
    if descend_offset_all:
        descend_fieldnames = [
            "run",
            "attempt",
            "t",
            "state",
            "x",
            "y",
            "conf",
            "session",
            "params",
            "log_file",
            "line",
        ]
        descend_sorted = sorted(
            descend_offset_all,
            key=lambda r: (
                int_or_default(r.get("run"), 0),
                int_or_default(r.get("attempt"), 0),
                float(r.get("t", -1.0) or -1.0),
            ),
        )
        with descend_offset_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=descend_fieldnames)
            writer.writeheader()
            writer.writerows(descend_sorted)

    print("\nTop candidates:")
    for i, row in enumerate(top10, start=1):
        top_first_raw = "NA" if row["first_lock_sec"] is None else f"{row['first_lock_sec']}s"
        top_first_replay = "NA" if row.get("first_lock_replay_sec") is None else f"{row['first_lock_replay_sec']}s"
        top_first_after = (
            "NA" if row["first_lock_after_target_sec"] is None else f"{row['first_lock_after_target_sec']}s"
        )
        top_relock_parts = []
        for start in s1_window_starts:
            key = f"relock_latency_{int(round(start))}s"
            raw = row.get(key)
            top_relock_parts.append(f"{int(round(start))}s={'NA' if raw is None else f'{raw}s'}")
        top_relock_text = ",".join(top_relock_parts)
        print(
            f"  {i:02d}. score={row['score']:<8} firstRaw={top_first_raw} firstReplay={top_first_replay} src={row.get('first_lock_metric_source','wall')} firstAfter={top_first_after} "
            f"early={row['early_lock']} cov={row.get('replay_window_ok', 0)} lock/lost={row['locks']}/{row['lost']} streak={row['max_track_streak']} "
            f"ratio={row['track_ratio']} avgMs={row['avg_frame_ms']} relock[{top_relock_text}] "
            f"wrongRatio={row.get('wrong_lock_ratio_in_windows')} src={row.get('wrong_lock_metric_source')}"
        )
        print(f"      {row['params']}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {top_path}")
    print(f"Saved: {pareto_path}")
    print(f"Saved: {pareto_csv_path}")
    print(f"Saved: {cand_summary_path}")
    if descend_offset_all:
        print(f"Saved: {descend_offset_csv_path}")
        print(f"Descend rows: {len(descend_offset_all)}")
    else:
        print("Descend rows: 0")
    if cand_dump_all:
        print(f"Saved: {cand_dump_csv_path}")
        print(f"Cand dump rows: {len(cand_dump_all)}")
    else:
        print("Cand dump rows: 0")
    print(f"Saved: {run_dir / 'manifest.json'}")
    print(f"Pareto count: {len(pareto_front)}")
    if pareto_front:
        print("\nPareto front (top 10 by ratio->latency):")
        for i, row in enumerate(pareto_front[:10], start=1):
            p_first_after = (
                "NA" if row["first_lock_after_target_sec"] is None else f"{row['first_lock_after_target_sec']}s"
            )
            print(
                f"  P{i:02d}. ratio={row['track_ratio']} avgMs={row['avg_frame_ms']} "
                f"lost={row['lost']} firstAfter={p_first_after} score={row['score']}"
            )
            print(f"      {row['params']}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


