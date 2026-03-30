#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Dict, List, Optional


def run_capture(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def load_result(run_dir: pathlib.Path) -> Dict[str, object]:
    result_json = run_dir / "result.json"
    if not result_json.exists():
        return {}
    try:
        return json.loads(result_json.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_float(value: object, default: float = -1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def is_valid_result(result: Dict[str, object], require_lock: bool) -> bool:
    if not result:
        return False
    perf_count = parse_int(result.get("perf_count", 0))
    summary_count = parse_int(result.get("summary_count", 0))
    events = result.get("events", {}) if isinstance(result.get("events"), dict) else {}
    mode_events = parse_int(events.get("MODE", 0))
    locks = parse_int(events.get("LOCK", 0))
    has_eval_logs = perf_count > 0 or summary_count > 0 or mode_events > 0
    if not has_eval_logs:
        return False
    if require_lock and locks <= 0:
        return False
    return True


def collect_metric(result: Dict[str, object], key: str, default: float = -1.0) -> float:
    summary = result.get("latest_summary", {}) if isinstance(result.get("latest_summary"), dict) else {}
    perf = result.get("latest_perf", {}) if isinstance(result.get("latest_perf"), dict) else {}
    if key in summary:
        return parse_float(summary.get(key), default)
    if key in perf:
        return parse_float(perf.get(key), default)
    if key == "firstLockSec":
        return parse_float(result.get("first_lock_sec_from_event", default), default)
    return default


def collect_locks(result: Dict[str, object]) -> int:
    summary = result.get("latest_summary", {}) if isinstance(result.get("latest_summary"), dict) else {}
    perf = result.get("latest_perf", {}) if isinstance(result.get("latest_perf"), dict) else {}
    if "locks" in summary:
        return parse_int(summary.get("locks", 0))
    if "locks" in perf:
        return parse_int(perf.get("locks", 0))
    events = result.get("events", {}) if isinstance(result.get("events"), dict) else {}
    return parse_int(events.get("LOCK", 0))


def is_locked_run(result: Dict[str, object]) -> bool:
    if collect_locks(result) > 0:
        return True
    return collect_metric(result, "firstLockSec", -1.0) >= 0.0


def run_round(
    capture_script: pathlib.Path,
    serial: str,
    duration: int,
    mode: str,
    target_path: str,
    video_path: str,
    max_attempts: int,
    require_lock: bool,
    params: str,
) -> Optional[Dict[str, object]]:
    for attempt in range(1, max_attempts + 1):
        cmd = [
            sys.executable,
            str(capture_script),
            "--serial",
            serial,
            "--duration",
            str(duration),
            "--mode",
            mode,
            "--target-path",
            target_path,
            "--video-path",
            video_path,
            "--use-replay",
        ]
        if params:
            cmd.extend(["--params", params])
        proc = run_capture(cmd)
        if proc.returncode != 0:
            print(f"[WARN] mode={mode} attempt={attempt} capture failed rc={proc.returncode}")
            if proc.stderr.strip():
                print(proc.stderr.strip())
            continue

        run_dir = None
        for line in proc.stdout.splitlines():
            if line.startswith("[INFO] run_dir:"):
                run_dir = pathlib.Path(line.split(":", 1)[1].strip())
                break
        if run_dir is None:
            print(f"[WARN] mode={mode} attempt={attempt} missing run_dir in output")
            continue

        result = load_result(run_dir)
        if not is_valid_result(result, require_lock=require_lock):
            print(f"[WARN] mode={mode} attempt={attempt} invalid result, retrying...")
            continue

        result["_run_dir"] = str(run_dir)
        result["_attempt"] = attempt
        return result

    return None


def median(values: List[float], default: float = -1.0) -> float:
    valid = [v for v in values if v >= 0.0]
    if not valid:
        return default
    return float(statistics.median(valid))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-round A/B replay benchmark and summarize median metrics.")
    parser.add_argument("--serial", default="CRX0222215001153")
    parser.add_argument("--duration", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--modes", default="baseline,enhanced", help="comma-separated modes, e.g. enhanced")
    parser.add_argument("--target-path", default="/sdcard/Download/Video_Search/target.jpg")
    parser.add_argument("--video-path", default="/sdcard/Download/Video_Search/scene.mp4")
    parser.add_argument("--params", default="", help="runtime eval params passed to app")
    parser.add_argument("--require-lock", action="store_true", default=False)
    args = parser.parse_args()

    this_dir = pathlib.Path(__file__).resolve().parent
    capture_script = this_dir / "capture_tracker_metrics.py"
    runs_root = this_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = runs_root / f"{stamp}_ab_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    modes = [m for m in modes if m in ("baseline", "enhanced")]
    if not modes:
        print("[ERROR] no valid modes selected; use baseline and/or enhanced")
        return 2
    all_results: Dict[str, List[Dict[str, object]]] = {m: [] for m in modes}

    print(
        f"[INFO] benchmark start: rounds={args.rounds}, duration={args.duration}s, modes={','.join(modes)}, "
        f"max_attempts={args.max_attempts}, require_lock={args.require_lock}, params='{args.params}'"
    )

    for round_id in range(1, args.rounds + 1):
        print(f"[INFO] round {round_id}/{args.rounds}")
        for mode in modes:
            print(f"[INFO]   mode={mode} ...")
            result = run_round(
                capture_script=capture_script,
                serial=args.serial,
                duration=args.duration,
                mode=mode,
                target_path=args.target_path,
                video_path=args.video_path,
                max_attempts=args.max_attempts,
                require_lock=args.require_lock,
                params=args.params,
            )
            if result is None:
                print(f"[WARN]   mode={mode} round={round_id} failed after retries")
                continue
            all_results[mode].append(result)
            first_lock = collect_metric(result, "firstLockSec", -1.0)
            avg_ms = collect_metric(result, "avgFrameMs", -1.0)
            track_ratio = collect_metric(result, "trackRatio", 0.0)
            locks = collect_locks(result)
            print(
                f"[INFO]   mode={mode} ok run={result.get('_run_dir','')} "
                f"lock={locks} firstLockSec={first_lock:.3f} avgFrameMs={avg_ms:.2f} trackRatio={track_ratio:.3f}"
            )

    summary: Dict[str, Dict[str, object]] = {}
    for mode in modes:
        items = all_results.get(mode, [])
        first_locks = [collect_metric(r, "firstLockSec", -1.0) for r in items]
        avg_mss = [collect_metric(r, "avgFrameMs", -1.0) for r in items]
        track_ratios = [collect_metric(r, "trackRatio", 0.0) for r in items]
        locks = [1 if is_locked_run(r) else 0 for r in items]
        summary[mode] = {
            "count": len(items),
            "median_firstLockSec": median(first_locks, -1.0),
            "median_avgFrameMs": median(avg_mss, -1.0),
            "median_trackRatio": median(track_ratios, 0.0),
            "lock_rate": (sum(1 for x in locks if x > 0) / len(locks)) if locks else 0.0,
            "runs": [r.get("_run_dir", "") for r in items],
        }

    result_json = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rounds": args.rounds,
        "duration_sec": args.duration,
        "max_attempts": args.max_attempts,
        "require_lock": args.require_lock,
        "target_path": args.target_path,
        "video_path": args.video_path,
        "params": args.params,
        "summary": summary,
    }
    (bench_dir / "ab_result.json").write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("# A/B Replay Benchmark")
    md.append("")
    md.append(f"- rounds: `{args.rounds}`")
    md.append(f"- duration: `{args.duration}s`")
    md.append(f"- max-attempts: `{args.max_attempts}`")
    md.append(f"- require-lock: `{args.require_lock}`")
    md.append(f"- params: `{args.params}`")
    md.append(f"- target: `{args.target_path}`")
    md.append(f"- video: `{args.video_path}`")
    md.append("")
    md.append("| mode | valid_runs | lock_rate | median_firstLockSec | median_avgFrameMs | median_trackRatio |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for mode in modes:
        item = summary.get(mode, {})
        md.append(
            "| {mode} | {count} | {lock_rate:.2f} | {first:.3f} | {avg:.2f} | {ratio:.3f} |".format(
                mode=mode,
                count=parse_int(item.get("count", 0)),
                lock_rate=parse_float(item.get("lock_rate", 0.0), 0.0),
                first=parse_float(item.get("median_firstLockSec", -1.0), -1.0),
                avg=parse_float(item.get("median_avgFrameMs", -1.0), -1.0),
                ratio=parse_float(item.get("median_trackRatio", 0.0), 0.0),
            )
        )
    md.append("")
    md.append("## Run Directories")
    md.append("")
    for mode in modes:
        item = summary.get(mode, {})
        run_dirs = item.get("runs", []) if isinstance(item.get("runs"), list) else []
        md.append(f"- {mode}:")
        for run_dir in run_dirs:
            md.append(f"  - `{run_dir}`")
    (bench_dir / "ab_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[INFO] benchmark_dir: {bench_dir}")
    print(f"[INFO] report: {bench_dir / 'ab_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
