#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Dict, List

SUMMARY_RE = re.compile(r"EVAL_SUMMARY\s+(.*)$")
EVENT_RE = re.compile(r"EVAL_EVENT\s+type=([A-Z_]+)(.*)$")
PERF_RE = re.compile(r"EVAL_PERF\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


ADB_HOME = pathlib.Path(__file__).resolve().parent / ".adb-home"
ADB_USER_HOME = ADB_HOME / ".android"
ADB_USER_HOME.mkdir(parents=True, exist_ok=True)


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("ANDROID_SDK_HOME", str(ADB_HOME))
    env.setdefault("ANDROID_USER_HOME", str(ADB_USER_HOME))
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        capture_output=True,
        env=env,
    )


def parse_kv(blob: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in KV_RE.finditer(blob):
        out[m.group(1)] = m.group(2)
    return out


def parse_metrics(log_lines: List[str]) -> Dict[str, object]:
    summaries = []
    perfs = []
    events = {"LOCK": 0, "LOST": 0, "MODE": 0, "TEMPLATE_READY": 0}
    first_lock_sec = -1.0

    for line in log_lines:
        sm = SUMMARY_RE.search(line)
        if sm:
            summaries.append(parse_kv(sm.group(1)))

        pm = PERF_RE.search(line)
        if pm:
            perfs.append(parse_kv(pm.group(1)))

        em = EVENT_RE.search(line)
        if em:
            et = em.group(1)
            payload = parse_kv(em.group(2))
            events[et] = events.get(et, 0) + 1
            if et == "LOCK" and "firstLockSec" in payload and first_lock_sec < 0:
                try:
                    first_lock_sec = float(payload["firstLockSec"])
                except Exception:
                    pass

    latest_summary = summaries[-1] if summaries else {}
    latest_perf = perfs[-1] if perfs else {}
    return {
        "events": events,
        "summary_count": len(summaries),
        "perf_count": len(perfs),
        "latest_summary": latest_summary,
        "latest_perf": latest_perf,
        "first_lock_sec_from_event": first_lock_sec,
    }


def to_float(value: str, default: float = -1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def get_sdk_level(serial: str) -> int:
    out = run(["adb", "-s", serial, "shell", "getprop", "ro.build.version.sdk"], check=False).stdout.strip()
    try:
        return int(out)
    except Exception:
        return -1


def grant_permission(serial: str, package: str, permission: str) -> None:
    result = run(["adb", "-s", serial, "shell", "pm", "grant", package, permission], check=False)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        print(f"[WARN] grant failed: {permission}: {msg}")


def remote_exists(serial: str, path: str) -> bool:
    result = run(["adb", "-s", serial, "shell", "ls", path], check=False)
    return result.returncode == 0


def wait_process_gone(serial: str, package: str, timeout_sec: float = 4.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        pid = run(["adb", "-s", serial, "shell", "pidof", package], check=False).stdout.strip()
        if not pid:
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture DroneTrackerDemo eval metrics from logcat")
    parser.add_argument("--serial", default="CRX0222215001153")
    parser.add_argument("--duration", type=int, default=35)
    parser.add_argument("--mode", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--package", default="com.example.dronetracker")
    parser.add_argument("--activity", default=".MainActivity")
    parser.add_argument("--tag", default="Tracker")
    parser.add_argument("--target-path", default="/sdcard/Download/Video_Search/target.jpg")
    parser.add_argument("--video-path", default="/sdcard/Download/Video_Search/scene.mp4")
    parser.add_argument("--params", default="", help="runtime eval params, e.g. refine_interval=4,local_expand=2.0")
    parser.add_argument("--use-replay", dest="use_replay", action="store_true", default=True)
    parser.add_argument("--no-replay", dest="use_replay", action="store_false")
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    root = pathlib.Path(__file__).resolve().parent / "runs"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{stamp}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run(["adb", "-s", args.serial, "logcat", "-c"])
    run(["adb", "-s", args.serial, "shell", "am", "force-stop", args.package], check=False)
    if not wait_process_gone(args.serial, args.package, timeout_sec=4.0):
        run(["adb", "-s", args.serial, "shell", "am", "force-stop", args.package], check=False)
        wait_process_gone(args.serial, args.package, timeout_sec=2.0)

    sdk_level = get_sdk_level(args.serial)
    if args.use_replay:
        if sdk_level >= 33:
            grant_permission(args.serial, args.package, "android.permission.READ_MEDIA_IMAGES")
            grant_permission(args.serial, args.package, "android.permission.READ_MEDIA_VIDEO")
        else:
            grant_permission(args.serial, args.package, "android.permission.READ_EXTERNAL_STORAGE")

        if not remote_exists(args.serial, args.target_path):
            print(f"[WARN] target not found: {args.target_path}")
        if not remote_exists(args.serial, args.video_path):
            print(f"[WARN] video not found : {args.video_path}")
    else:
        grant_permission(args.serial, args.package, "android.permission.CAMERA")

    start_cmd = [
        "adb", "-s", args.serial, "shell", "am", "start",
        "-S",
        "-n", f"{args.package}/{args.activity}",
        "--es", "tracker_mode", args.mode,
    ]
    if args.use_replay:
        start_cmd.extend([
            "--ez", "eval_use_replay", "true",
            "--es", "eval_target_path", args.target_path,
            "--es", "eval_video_path", args.video_path,
        ])
    if args.params:
        start_cmd.extend(["--es", "eval_params", args.params])
    start_result = run(start_cmd)
    start_stdout = (start_result.stdout or "").strip()
    start_stderr = (start_result.stderr or "").strip()
    if "Error:" in start_stdout or "Exception" in start_stdout or start_stderr:
        print(f"[WARN] am start stdout: {start_stdout}")
        if start_stderr:
            print(f"[WARN] am start stderr: {start_stderr}")

    time.sleep(0.8)
    pid_check = run(["adb", "-s", args.serial, "shell", "pidof", args.package], check=False).stdout.strip()
    if not pid_check:
        print(f"[WARN] app is not running after start: {args.package}")

    print(f"[INFO] running mode={args.mode}, replay={args.use_replay}, duration={args.duration}s")
    time.sleep(max(args.duration, 1))

    # Try to close activity gracefully so onDestroy summary can be emitted.
    run(["adb", "-s", args.serial, "shell", "input", "keyevent", "4"], check=False)
    time.sleep(0.8)

    log_out = run(
        [
            "adb",
            "-s",
            args.serial,
            "logcat",
            "-d",
            "-s",
            f"{args.tag}:V",
            "MainActivity:V",
            "AndroidRuntime:E",
        ]
    ).stdout
    if not log_out.strip():
        full_out = run(["adb", "-s", args.serial, "logcat", "-d"], check=False).stdout
        filtered = []
        tag_token = f" {args.tag} :"
        for line in full_out.splitlines():
            if args.package in line or "MainActivity" in line or "AndroidRuntime" in line or tag_token in line:
                filtered.append(line)
        log_out = "\n".join(filtered)
    log_path = run_dir / "logcat_tracker.log"
    log_path.write_text(log_out, encoding="utf-8")

    lines = log_out.splitlines()
    metrics = parse_metrics(lines)
    summary = metrics.get("latest_summary", {})
    perf = metrics.get("latest_perf", {})

    result = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "serial": args.serial,
        "mode": args.mode,
        "use_replay": args.use_replay,
        "target_path": args.target_path if args.use_replay else "",
        "video_path": args.video_path if args.use_replay else "",
        "params": args.params,
        "duration_sec": args.duration,
        "run_dir": str(run_dir),
        "events": metrics.get("events", {}),
        "summary_count": metrics.get("summary_count", 0),
        "perf_count": metrics.get("perf_count", 0),
        "latest_summary": summary,
        "latest_perf": perf,
        "first_lock_sec_from_event": metrics.get("first_lock_sec_from_event", -1.0),
    }

    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    first_lock = to_float(
        summary.get(
            "firstLockSec",
            perf.get("firstLockSec", str(metrics.get("first_lock_sec_from_event", -1.0))),
        )
    )
    avg_ms = to_float(summary.get("avgFrameMs", perf.get("avgFrameMs", "-1")))
    track_ratio = to_float(summary.get("trackRatio", perf.get("trackRatio", "0")))
    locks = to_int(summary.get("locks", perf.get("locks", "0")))
    lost = to_int(summary.get("lost", perf.get("lost", str(metrics.get("events", {}).get("LOST", 0)))))

    md = []
    md.append("# DroneTracker Eval Report")
    md.append("")
    md.append(f"- mode: `{args.mode}`")
    md.append(f"- replay: `{args.use_replay}`")
    if args.params:
        md.append(f"- params: `{args.params}`")
    if args.use_replay:
        md.append(f"- target: `{args.target_path}`")
        md.append(f"- video: `{args.video_path}`")
    md.append(f"- duration: `{args.duration}s`")
    md.append(f"- log: `{log_path}`")
    md.append("")
    md.append("## Metrics")
    md.append("")
    md.append(f"- locks: `{locks}`")
    md.append(f"- lost: `{lost}`")
    md.append(f"- firstLockSec: `{first_lock}`")
    md.append(f"- avgFrameMs: `{avg_ms}`")
    md.append(f"- trackRatio: `{track_ratio}`")
    md.append(f"- summary_count: `{metrics.get('summary_count', 0)}`")
    md.append(f"- perf_count: `{metrics.get('perf_count', 0)}`")
    md.append("")
    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] report : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
