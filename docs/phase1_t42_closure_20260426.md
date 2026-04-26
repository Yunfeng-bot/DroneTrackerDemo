# Phase 1 T4.2 闭环报告 · 2026-04-26

## 1. 业务结论

MVP-5 收官时暴露的真机可用性硬伤（"距离>50cm 锁不到 / 锁了易丢 / 丢了找不回"）通过 Phase 1 屏幕圈选机制完成首轮闭环。当前手动圈选路径在真机上可达成：

- **首锁**：圈选后 ≤ 1s 内 `MANUAL_ROI_INIT_OK + DIAG_LOCK reason=manual`
- **持续追踪**：首锁后稳定追踪 ≥ 5s（V3 实测 10.7s，无干扰场景下更长）
- **正确丢锁**：目标真出画时通过 NCC panic / sustained drift 路径正确触发 LOST
- **自动重锁**：原目标回归画面时通过 NCC fingerprint 验证（≥ 0.70）自动 relock_unblocked
- **拒绝错锁**：相似但不同物体的候选被 fingerprint_mismatch 持续拒绝，不会误锁

## 2. 病灶清单（病因 → 修复批次对照）

| # | 病灶 | 修复批次 | 关键 commit |
|---|---|---|---|
| 1 | 手动圈选 patch 内容废（`kp=0 texture=0`） | Batch 3 加 patch_stats + 转储 + 低对比度早失败 | `fe8d596` |
| 2 | 弱 ORB 候选错锁（good=8 conf=0.75 通过） | Batch 5 硬阈值（good≥25 / inliers≥12 / conf≥0.40 / anchor_dist / silence） | `8777a67` |
| 3 | TRACKING 阶段 native 持续 accept 渐进漂移 | Batch 6 五维几何 veto（anchor_drift / area / frame_jump / edge） | `84b49fa` |
| 4 | "硬阻断" 误用空集换安全（不代用户猜被错误推广） | Batch 7v2 改为模板指纹 NCC 二层验证（不阻断，识别） | `737f8cd` |
| 5 | `orb_temporal_confirm` 路径完全 bypass manual gate | Batch 8 `passesManualRoiRelockGate` 覆盖 + blocked 态 Layer 1 让位给 NCC | `ec2947c` |
| 6 | 七处 `initializeTracker` 入口三处无 manual 守门 | Batch 9 把 NCC 守门集中到 `initializeTracker` 入口 + TRACKING 周期 NCC | `3823651` |
| 7 | `matchTemplate` 等尺寸误用，对像素级对齐误差极敏感 | Batch 10 改为 search-window matchTemplate（patch 跟随当前 box，±20% 取 max） | (含在 Batch 10) |
| 8 | 单帧 NCC 阈值 veto 把 NCNN init 头帧抖动误判漂移 | Batch 11 grace 窗 + 滑窗 4 样本中位数 + panic 双阈值 | `09d2c6a` |

## 3. 验收证据（V3 真机 · 2026-04-26 19:24）

**事件链**：
```
19:24:03.413  LOCK reason=manual                              ← 首锁
19:24:14.081  LOST reason=manual_track_veto                   ← 移出后 panic
19:24:18.853  MANUAL_ROI state=relock_unblocked ncc=0.889     ← 回归识别
19:24:18.859  LOCK reason=manual_roi_direct                   ← 二次锁定
19:24:37.011  日志结束，无 LOST                               ← 持续 18.152s
```

**NCC 分阶段统计**：
- `grace`  期：count=4, avg=0.922, median=0.924, range [0.849, 0.993]
- `warmup` 期：count=6, avg=0.882, median=0.880, range [0.846, 0.935]
- `stable` 期：未稀疏触发输出（每 60 帧一次），但 18s 内未中断证明已稳

**硬指标对照**：
- ✓ 首锁后持续追踪 ≥ 5s（实测 10.668s）
- ✓ 挥开后能 fingerprint_pass（实测 ncc=0.889）
- ✓ 回归后再持续追踪 ≥ 5s（实测 18.152s）

**回归确认（V5 · 同日）**：
- l1：first_lock=7.722s, L3, ratio=0.400（落已知 slow-path bucket，非 Batch 9~11 引入）
- l2：first_lock=3.036s, TRACKING, ratio=0.883（确定性绿）
- l3：first_lock=5.412s, TRACKING, ratio=0.817（确定性绿）

## 4. 关键阈值与默认值（数据驱动）

| 参数 | 当前值 | 数据依据 |
|---|---|---|
| `MANUAL_ROI_FINGERPRINT_SEARCH_MARGIN` | 0.20 | 容忍 NCNN 亚像素 + 中等像素级框抖动 |
| `MANUAL_ROI_RELOCK_MIN_FINGERPRINT_NCC` | 0.70 | V3 实测 fingerprint_pass = 0.889；mismatch 候选最高 0.666；间隙清晰 |
| `MANUAL_ROI_TRACK_NCC_GRACE_FRAMES` | 30 (~1s) | NCNN init 后稳定时间观察 |
| `MANUAL_ROI_TRACK_NCC_HISTORY_SIZE` | 4 | 4 × 15 帧间隔 = 60 帧滚动覆盖 |
| `MANUAL_ROI_TRACK_NCC_DRIFT_MEDIAN` | 0.40 | 真同目标 stable 期 median 0.85+，0.40 留 0.45 安全余量 |
| `MANUAL_ROI_TRACK_NCC_PANIC_SINGLE` | 0.20 | 异物 NCC 通常 < 0.30，0.20 catch 灾难无误伤 |
| `MANUAL_ROI_RELOCK_MIN_GOOD` (Layer 1, 非 blocked) | 25 | Batch 5 数据 |
| `MANUAL_ROI_RELOCK_BLOCKED_MIN_GOOD` (Layer 1, blocked) | 12 | Batch 8 让 NCC 做权威，Layer 1 只过滤垃圾 |

## 5. 已识别边界（不在 T4.2 范围）

参见 [phase1_manual_roi_selection_plan.md §6](phase1_manual_roi_selection_plan.md)：

- Scale 大变化 / 视角剧变 → Phase 2 议题
- `verify_realign` in-vivo 验证缺失 → unit test 补强
- L1 外场稳定性 33% slow path → L1 验收前独立债

## 6. 工程方法论沉淀

本轮 T4.2 经历 11 个 batch 的迭代，核心反思已沉淀为持久记忆，避免后续重蹈：

- **不要用"目标可能变了"代替"识别不可靠"**（Batch 7 v1→v2 转折）—— 不要用空集换安全，要做识别工程
- **不要靠"来回真机验证"补齐代码逻辑**（Batch 9 之前）—— 多路径状态机修复必须先系统读完架构，一次出方案
- **matchTemplate 在等尺寸 Mat 上是误用**（Batch 10 修复）—— 等尺寸 NCC 退化为零偏移点积
- **连续噪声信号不能用单帧阈值 veto**（Batch 11 修复）—— 必须配合 grace / 滑窗 / 连续失败计数

## 7. 下一步（Phase 1.x → Phase 2）

T4.2 闭环后，Phase 1 剩余任务：

- [可选] T4.3 完整 5 场景 probe（V3 已顺手覆盖核心场景，剩余 S1/S2/S5 是 nice-to-have）
- [推荐] verify_realign unit test 补强（已识别边界）
- [Phase 2 启动] VLM 自动圈选取代手动圈选（详见 `docs/REQUIREMENTS.md §1.6.2`）
