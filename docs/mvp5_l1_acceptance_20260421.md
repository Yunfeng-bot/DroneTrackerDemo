# MVP-5 L1 Acceptance Report (2026-04-21)

## Scope
- Variant: `l1`
- Video: `test/scene_mvp5_l1_20260421.mp4`
- Template: `test/target_rooftop_20260421.jpg`
- Command: `./tools/replay_sop/run_mvp5_variant.ps1 -Variant l1 -Date 20260421`

## Parameter Baseline (this round)
- `first_lock_min_iou=0.30`
- `center_roi_l3_timeout_ms=5000`
- `first_lock_gap_ms=2500`

## Tooling / Contract Changes
1. `OpenCVTrackerAnalyzer.kt`
- override clamp for `first_lock_min_iou`: `0.40 -> 0.30`
- override clamp for `first_lock_gap_ms`: `1200 -> 2500`

2. `tools/replay_sop/run_mvp5_variant.ps1`
- base params updated to include the three values above
- default `DurationSec` adjusted to `30.0` to align with MVP-5 steady window validation

3. `tools/auto_tune/sweep_replay.py`
- added window-level ratio metrics aligned with `mvp5_windows_l1.json` semantics:
  - `steady_track_window_track_ratio`
  - `steady_track_window_min_track_ratio`
  - `steady_track_window_track_ratio_pass`
  - `window_track_ratio_*` aliases
  - sample counters: `*_samples`, `*_tracked`
- metrics are now written into `summary.json` / `mvp5_summary.json`

## Evidence (latest run)
- Output dir: `tools/auto_tune/out/mvp5_l1_20260421_132319`
- Summary: `tools/auto_tune/out/mvp5_l1_20260421_132319/summary.json`

Key fields:
- `descend_offset_first_t=1.188`
- `descend_offset_last_state=TRACKING`
- `descend_offset_oob_count=0`
- `descend_offset_fail_count=0`
- `locks=1`
- `lost=0`
- `wrong_lock_ratio_in_windows=0.0`
- `track_ratio=0.567` (global/session-level)
- `steady_track_window_track_ratio=1.0`
- `steady_track_window_min_track_ratio=0.9`
- `steady_track_window_track_ratio_pass=true`
- `steady_track_window_track_ratio_samples=3`
- `replay_window_ok=0`
- `replay_pts_sec=17.226` (below 30s run coverage gate)

## Assessment
- L1 core state-chain is working (`last_state=TRACKING`, `fail_count=0`, `oob_count=0`).
- Window-based acceptance ratio passes by current computed semantics (`steady_track_window_track_ratio_pass=true`).
- Replay coverage is still unstable (`replay_window_ok=0`, only 17.226s effective replay points), so this run is acceptable for state-chain evidence but not yet a full-coverage benchmark run.

## Next Step
- Keep current params.
- Prioritize replay coverage stability investigation before using `track_ratio` as final KPI for long-window acceptance.
