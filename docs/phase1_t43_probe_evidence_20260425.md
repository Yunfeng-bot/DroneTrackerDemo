# Phase 1 T4.3 Manual ROI Probe Evidence (2026-04-25)

HEAD: `737f8cd`

## 场景证据表

| 场景 | init_ok | init_fail | lock | lost reasons | veto reasons | blocked | unblocked | gate_fail reasons | NCC pass min | NCC fail max |
|---|---:|---|---:|---|---|---:|---:|---|---:|---:|
| S1 静/静 | 1 | template_rebuild_failed x6, fallback_forbidden x6 | 1 | - | - | 0 | 0 | - | - | - |
| S2 静/慢移 | 1 | template_rebuild_failed x1, fallback_forbidden x1 | 1 | manual_track_veto x1 | area_grow x1 | 1 | 0 | - | - | - |
| S3 挥开-回归 | 1 | - | 3 | manual_track_veto x3 | area_shrink x3 | 1 | 0 | ban_refined_area_small x14, low_good x6 | - | - |
| S4 相似异物 | 1 | - | 5 | manual_track_veto x4 | area_shrink x3, edge_hug x1 | 1 | 0 | low_good x27, ban_refined_area_small x65, low_inliers x26, fingerprint_mismatch x10, low_conf x3 | - | 0.666 |
| S5 瞬时遮挡 | 2 | template_rebuild_failed x1, fallback_forbidden x1 | 5 | manual_track_veto x2 | edge_hug x1, area_shrink x1 | 2 | 0 | ban_refined_area_small x41, low_good x75, fingerprint_mismatch x3 | - | 0.221 |

## NCC 分布（合并所有场景）

| 区间 | gate_pass=true 数量 | gate_pass=false 数量 | 备注 |
|---|---:|---:|---|
| >= 0.85 | 0 | 0 | 本轮未采到 true-pass NCC |
| 0.70~0.85 | 0 | 0 | 当前阈值边界附近无样本 |
| 0.55~0.70 | 0 | 9 | 全部为 fail，集中在相似异物/误候选 |
| < 0.55 | 0 | 4 | 明显异物或极弱候选 |

## Patch 健康分布

| 场景 | patch_stats_count | patch_mean_avg | patch_std[min,max] | patch_range[min,max] | init_ok patch_kp[min,max] | init_ok texture[min,max] |
|---|---:|---:|---|---|---|---|
| S1 静/静 | 7 | 110.756 | 23.522~27.437 | 163.000~197.000 | 151~151 | 2018.399~2018.399 |
| S2 静/慢移 | 2 | 116.138 | 26.681~29.366 | 202.000~204.000 | 67~67 | 1255.177~1255.177 |
| S3 挥开-回归 | 1 | 126.201 | 29.935~29.935 | 220.000~220.000 | 54~54 | 1559.516~1559.516 |
| S4 相似异物 | 1 | 109.305 | 18.036~18.036 | 111.000~111.000 | 578~578 | 7500.187~7500.187 |
| S5 瞬时遮挡 | 3 | 125.008 | 25.068~27.866 | 204.000~220.000 | 95~117 | 1123.985~1489.124 |

## 观察

- S1 有效首锁成功，但在成功前出现多次 `template_rebuild_failed -> fallback_forbidden`。
- S2 慢移场景未能全程稳定，后段触发 `area_grow` veto。
- S3 挥开后扫回原目标，没有进入 NCC 判定；主要卡在一层 gate：`ban_refined_area_small` 与 `low_good`。
- S4 相似异物场景采到 10 个 `fingerprint_mismatch`，最高 `NCC=0.666`，说明 NCC 二层验证能挡住 B。
- S5 短遮挡不够平滑，实际出现 2 次 `manual_track_veto`，说明当前 TRACKING veto 对遮挡也较敏感。
- blocked 态下 `orb_temporal_confirm` 仍会漏进自动锁定路径。实测出现次数：
  - S3: 2 次
  - S4: 4 次
  - S5: 3 次

## 阈值判定

- 真匹配最低分（pass NCC 的最小值）：`N/A（本轮未采到 pass 样本）`
- 假匹配最高分（fail NCC 的最大值）：`0.666`
- 结论：`保持 0.70，暂不调整`

理由：

1. 本轮 13 个 NCC 样本全部为 fail，最高仅 `0.666`，没有穿透当前 `0.70` 阈值。
2. 这说明 `0.70` 至少没有把已观测到的假匹配放进来。
3. 但本轮没有采到 `pass=true` 的 NCC 样本，因此不能靠这批 probe 去下调或上调阈值。
4. 当前 T4.3 暴露的主瓶颈不是 NCC 阈值，而是：
   - `manual_track_veto` 之后，一层 gate 候选不足，回不到 NCC 身份验证；
   - blocked 态下 `orb_temporal_confirm` 自动路径仍会漏进。

## 工程结论

T4.3 证据已经足够支持两件事：

1. `MANUAL_ROI_RELOCK_MIN_FINGERPRINT_NCC = 0.70` 当前没有被假匹配数据证伪，Phase 1 维持不变。
2. 下一修复重点不应放在改 NCC 阈值，而应放在 blocked-path 行为收敛：
   - blocked 态禁止 `orb_temporal_confirm` 自动接管；
   - 提升原目标回归时的一层候选质量，避免一直卡在 `ban_refined_area_small` / `low_good`。

## 产物

- `tools/auto_tune/out/probe_S1_20260425.log`
- `tools/auto_tune/out/probe_S2_20260425.log`
- `tools/auto_tune/out/probe_S3_20260425.log`
- `tools/auto_tune/out/probe_S4_20260425.log`
- `tools/auto_tune/out/probe_S5_20260425.log`
- `tools/auto_tune/out/probe_S1_summary.json`
- `tools/auto_tune/out/probe_S2_summary.json`
- `tools/auto_tune/out/probe_S3_summary.json`
- `tools/auto_tune/out/probe_S4_summary.json`
- `tools/auto_tune/out/probe_S5_summary.json`
- `tools/auto_tune/out/probe_S1_ncc.csv`
- `tools/auto_tune/out/probe_S2_ncc.csv`
- `tools/auto_tune/out/probe_S3_ncc.csv`
- `tools/auto_tune/out/probe_S4_ncc.csv`
- `tools/auto_tune/out/probe_S5_ncc.csv`
- `tools/auto_tune/out/probe_S1_timeline.csv`
- `tools/auto_tune/out/probe_S2_timeline.csv`
- `tools/auto_tune/out/probe_S3_timeline.csv`
- `tools/auto_tune/out/probe_S4_timeline.csv`
- `tools/auto_tune/out/probe_S5_timeline.csv`
