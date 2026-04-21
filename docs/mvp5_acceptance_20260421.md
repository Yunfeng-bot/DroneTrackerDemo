# MVP-5 Acceptance Report (2026-04-21)

## Scope
- Variant set: `l1`, `l2`, `l3`, `fail`
- Runtime profile: replay fps 15, duration 32s, GPS ready @2.0s
- Current baseline params include:
  - `center_roi_l3_timeout_ms=10000`
  - `first_lock_min_iou=0.30`
  - `first_lock_gap_ms=2500`
- Additional fix for off-center hold:
  - `track_guard_anchor_enabled=false` (applied to `l2`/`l3` in replay script)

## Results (Final)
| Variant | Summary Path | last_state | fail_count | locks | lost | first_lock_replay_sec | wrong_lock_ratio | steady_pass | replay_stop_missing |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| l1 | `tools/auto_tune/out/mvp5_l1_20260421_163553/summary.json` | TRACKING | 0 | 1 | 0 | 7.722 | 0.0 | true | false |
| l2 | `tools/auto_tune/out/mvp5_l2_20260421_163352/summary.json` | TRACKING | 0 | 1 | 0 | 3.102 | 0.0 | true | false |
| l3 | `tools/auto_tune/out/mvp5_l3_20260421_163445/summary.json` | TRACKING | 0 | 1 | 0 | 3.102 | 0.0 | true | false |
| fail | `tools/auto_tune/out/mvp5_fail_20260421_163645/summary.json` | FAIL | 1 | 0 | 0 | -1.0 | n/a | n/a | false |

## FAIL Window Check
- Expected window: `expected_fail_at_sec=14.0 ± 1.0` (`[13.0, 15.0]`)
- Actual log:
  - `EVAL_EVENT type=DESCEND_OFFSET ... state=FAIL t=13.728`
  - Source: `tools/auto_tune/out/mvp5_fail_20260421_163645/logs/run_001_try1.log`
- Verdict: PASS

## Root-Cause Evidence (L2 hold issue)
- Diagnostic run with subtype telemetry:
  - `tools/auto_tune/out/mvp5_l2_20260421_163238/logs/run_001_try1.log`
- Key lines:
  - `EVAL_EVENT type=TRACK_GUARD_REJECT reason=appearance ... smallTarget=true`
  - `EVAL_EVENT type=TRACK_GUARD_REJECT reason=appearance_hard ... smallTarget=true`
- Interpretation:
  - Loss trigger is appearance/anchor path in small-target condition; not geometry jump/area.

## Conclusion
- MVP-5 four-variant acceptance is GREEN under current baseline + L2/L3 anchor-guard relaxation.
