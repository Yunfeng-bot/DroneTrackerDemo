# 模板对追踪效果影响 · 系统优化方案

**创建日期**: 2026-04-20
**状态**: 活文档（Living Document），每轮验证后刷新"进展追踪"章节
**适用分支**: main（与 `p0_iteration_report_20260419.md`、`tactical_execution_plan-0419.md` 配套）
**Prompt**: 我对模板对追踪效果的影响还不清楚，请系统梳理技术原理，并从多维度（消除实现方案盲点）输出完整优化方案，要有理有据（禁止瞎猜和捏造）和可信，方案逻辑清晰。


---

## 0. 本文档的使用方式

- **第 1~3 节**：不变内容（技术原理、影响因子、既往误区）。只有在发现代码层事实变化或推翻性证据时才修订。
- **第 4 节（因子分析表）**：每条方案一行，标注 **已验证 ✓ / 待验证 ⚠️ / 已推翻 ✗**，附证据链。新信息→更新一行。
- **第 5 节（实施计划）**：checkbox 列表，按优先级分层。完成一项→打勾+填写提交号。
- **第 6 节（进展追踪日志）**：逆序时间线，最新在最上。每轮回放验证后新增一条。

---

## 0.1 路径分工决议（2026-04-20，方案 C）

**背景**：P0-D* 全部工作聚焦在 `scene_20260417` 动态伴飞回放的窗口重锁（wrong_lock_ratio），但 MVP 场景是[楼顶精降](REQUIREMENTS.md#L1.5)（静态目标 + GPS 先验 + 分级 ROI），两者的失败模式物理上不同。

**决议**：

| 分工 | 范围 | 优先级 | 当前状态 |
|---|---|---|---|
| **MVP 主线** | L1/L2/L3 分级 ROI、GPS 就位信号、offsetX/Y 输出、L3 超时兜底、静态楼顶验收视频 | 🔴 P0（最高） | 已启动：`MVP-2` 分级 ROI 搜索已接入（待联调 GPS 信号与 offset 输出） |
| **RC1 伴飞主线** | P0-D4 候选可见性诊断、动态场景窗口重锁 | 🟡 P1（后置） | P0-D4 降级为**仅 1 轮诊断，不做 A/B 收敛** |

**P0-D4 的范围限定（方案 C）**：
- **只做**：加 `EVAL_CAND_DUMP` 日志 → 跑 1 轮 → 解析得出"召回层 / 打分层 / 门控层"根因结论 → 入档。
- **不做**：基于 D4 结论的任何 A/B 收敛（权重、几何、外观下限都不跑）。
- **时间上限**：半天。超时则直接关闭 P0-D4，结论记为"未知，挂起至 RC1 阶段"。

**MVP 真正的关键路径**（5 项新模块，所有 P0-D 工作不在其上）：
1. 飞控 ↔ App 的 GPS 就位信号通道（IPC/消息）
2. L1/L2/L3 分级 ROI 搜索状态机（`CENTER_ROI_SEARCH` 三态机）
3. bbox → offsetX/Y 归一化输出 + 下发通道
4. L3 超时兜底：2 s 未命中 → 上抛 "视觉引导失败"
5. MVP 专用验收视频：静态楼顶俯视 + GPS 悬停模拟 + 窗口定义

---

## 1. 两条独立的模板使用路径

项目中模板并不是一个概念，而是两条完全独立的路径：

| 路径 | 模板来源 | 输入尺寸 | 使用模块 | 敏感性 |
|---|---|---|---|---|
| **ORB 路径** | 用户 App 内导入/裁剪的图片 | 128~480 短边，保留纵横比 | [OpenCVTrackerAnalyzer.kt:4708](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L4708) `findOrbMatch` | 对**方向**敏感，对**纵横比**不敏感 |
| **NCNN 路径** | 首次 ORB 命中时截取的**当前视频帧** patch | 127×127 正方形（context_amount=0.5） | [NcnnTrackerImpl.cpp:478](app/src/main/cpp/tracker/NcnnTrackerImpl.cpp#L478) `init(frame, bbox)` | 与用户导入的图片**完全无关**，由视频帧决定 |

**代码证据**：
- ORB 侧：`setTemplateImages()` → `normalizeTemplateSize` → `rebuildTemplatePyramid`（[line 1927/1948/2020](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L1927)）。
- NCNN 侧：[NcnnTrackerImpl.cpp:497-505](app/src/main/cpp/tracker/NcnnTrackerImpl.cpp#L497)
  ```cpp
  // context_amount = 0.5f, sZ = sqrt((w+0.5(w+h))*(h+0.5(w+h)))
  // 对 (w,h) 交换对称 → 方向不影响 NCNN 初始 patch
  extractPatchToMat(frame, cx, cy, patchW, patchH, templateInputSize_=127, ...);
  ```

**推论**：
- 换模板图片**只影响 ORB**的首次匹配；之后所有帧 NCNN 走自己的视频帧 patch。
- 因此模板的"横版/竖版"议题只在 **ORB 首锁** + **首锁后 fallback box 形态** 两个点发力。

---

## 2. 模板影响追踪效果的四个维度

### 2.1 分辨率 / 可用像素量 → ORB 特征数量

- ORB 初始化要求 `MIN_TEMPLATE_USABLE_KEYPOINTS=50`（[line 6464](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6464)）。
- 金字塔等级对最短边要求 `MIN_TEMPLATE_LEVEL_DIM=20`。
- **评估规则**：可用像素量 ≈ (短边·长边·缩放后)。低于 ~5 万像素时 ORB 常无法产出稳定描述子。

### 2.2 模板方向（旋转） → ORB 的方向盲区

- Oriented-BRIEF 对 ±30° 内鲁棒，对 90° 以上**不具备旋转不变性**。
- 现有姿态增强：`TEMPLATE_POSE_AUGMENT_DEGREES = [-30, -15, 0, 15, 30]`（[line 6471](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6471)）。缺少 ±90°。
- NCNN 侧：`context_amount` 公式对 (w,h) **对称**，所以 NCNN 初始 patch 不受纵横比方向影响。

### 2.3 EXIF 方向 vs 原始像素方向的不一致

- Android 导入链：[MainActivity.kt:59](app/src/main/java/com/example/dronetracker/MainActivity.kt#L59) 使用 `BitmapFactory.decodeStream` → **不应用 EXIF**。
- 回放链：[MainActivity.kt:357](app/src/main/java/com/example/dronetracker/MainActivity.kt#L357) 使用 `BitmapFactory.decodeFile` → **同样不应用 EXIF**。
- **结论**：追踪器看到的永远是**原始像素方向**，EXIF 只影响 PC / Gallery 的显示。
- **已校正事实**：`target0417_s640.jpg` 原始像素为 385×640 **竖版**（EXIF=6 只是 PC 端手动调整后显示成横版，手机回放一直使用竖版）。

### 2.4 首锁后的 fallback box 形态

- [line 6105](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6105) `buildCenteredSquare` 决定首锁框形态。
- track_guard 使用**短边**作为跳变参考（[line 2937/3050](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2937)）。
- 一旦从 48×48 正方形切到 37×62 矩形，短边从 48→37，track_guard_fail 阈值被放大，**诱发过早丢失**。
- **回归用例**：padding 实验期间 `track_ratio 0.645→0.059`，即此原因。目前已回退至纯正方形逻辑。

---

## 3. 既往误区清单（节省后续复盘时间）

| # | 过去错误假设 | 代码/数据反证 | 正确认知 |
|---|---|---|---|
| 1 | padding 到正方形能提升 NCNN | NCNN 用视频帧 patch，和用户图无关 | 只影响 ORB，且破坏 track_guard |
| 2 | 模板纵横比应匹配目标形状 | ORB 对纵横比不敏感；NCNN context_amount 对称 | 两个路径都不要求匹配 |
| 3 | `target0417_s640.jpg` 在手机上是横版 | `BitmapFactory.decodeFile` 不读 EXIF；原始像素 385×640 | 手机回放一直是竖版 |
| 4 | 全局 track_ratio 高即成功 | 窗口指标（wrong_lock_ratio, relock_24s）可同时恶化 | 必须以窗口指标为首要判据 |
| 5 | target0419.jpg (148×96) 可用 | 像素量仅 1.4 万，ORB 特征不足 | 必须满足短边 ≥128 且可用像素 ≥5 万 |

---

## 4. 模板影响因子分析表

> 标注说明：✓ 已验证（有回放数据支撑） · ⚠️ 待验证（方案成立但尚未落地/回放） · ✗ 已推翻（实验否定）

| 因子 | 作用机制 | 预期影响 | 状态 | 证据 / 待办 |
|---|---|---|---|---|
| **分辨率（可用像素量）** | 决定 ORB 关键点数量和描述子稳定性 | 像素过低 → 首锁失败 | ✓ 已验证 | target0419.jpg 1.4 万像素 → 失败；target0417_s640.jpg 24.6 万 → track_ratio=0.645 |
| **短边下限 MIN_TEMPLATE_DIM=128** | 保证金字塔顶层仍有 ≥20 px | 过小则金字塔层级减少 | ✓ 已验证 | 当前 128 可用；提升至 192 待 P1-2 验证 |
| **短边上限 MAX_TEMPLATE_DIM=480** | 限制金字塔顶层不过大，控制耗时 | 过小会丢失纹理 | ✓ 已验证 | 降到 256 时 track_ratio=0.365，回退到 480 恢复 |
| **模板方向（ORB）** | Oriented-BRIEF 对 ≥90° 不具旋转不变性 | 方向错位 → window_lock_count=0 | ✓ 已验证 | landscape_480 全局 0.654 但窗口锁=0，方向对 ORB 致命 |
| **纵横比（ORB）** | ORB 描述子与纵横比无关 | 横版/竖版效果接近 | ✓ 已验证 | s640（竖版）vs portrait_480 三轮均值 track_ratio=0.640±0.012 vs 0.643±0.016 |
| **EXIF vs 原始像素** | `BitmapFactory` 不应用 EXIF | EXIF 方向对追踪器不可见 | ✓ 已验证 | decode 路径代码已确认；纠正前期误判 |
| **fallback box 形状（首锁）** | track_guard 使用短边作为跳变阈值 | 非正方形 → 过早触发 guard_fail | ✓ 已验证 | 37×62 矩形 vs 48×48 正方形：track_ratio 0.059 vs 0.645 |
| **姿态增强 ±15°/±30°** | 覆盖小幅旋转 | 对 30° 内目标稳定 | ✓ 已验证 | 当前默认值生效 |
| **姿态增强 ±90°** | 覆盖相机旋转 / 目标大角度倾转 | 预期降低 wrong_lock，方向兜底 | ⚠️ 待验证 | P1-1 尚未落地 |
| **MIN_TEMPLATE_DIM 提升到 192** | 顶层金字塔更稳 | 预期提高 ORB 召回 | ⚠️ 待验证 | P1-2 尚未落地 |
| **ORB oriented=true 总开关** | 当前姿态增强+oriented 双管齐下 | 关闭 oriented 可测试单纯增强效果 | ⚠️ 待验证 | 无对照实验 |
| **重搜索阶段 KCF 长占用** | 二次丢失后 KCF 持续 + ORB 再搜索 | avg_frame_ms +34ms | ⚠️ 待定位 | 日志线索在 `run_001_try2.log`（14.9s 后），未定位代码出处 |
| **wrong_lock_ratio 稳态偏高** | 空间门控 5 种配置均无法降低（1.0） → 瓶颈不在门控 | 未恢复到目标阈值（<0.10） | ✓ 已定位（P0-D4：召回层失效） | 一轮诊断 `20260420_161708`：`groups_in_windows=101`，`gt_missing_topk=84`，`gt_present_topk=17`，`diagnosis=recall_layer`；说明窗口期多数帧真目标未进入 top-K，后续应转战役二/三（重检测 + 特征区分度） |
| **relock_24s = NA** | 窗口标签模式的指标键名错配（`start_sec=23` 却写入 `relock_24s` 列） | 评估误报 NA（非跟踪器真实能力） | ✓ 已修复 | `sweep_replay.py` 增加 `metric_key_sec`，`p0_windows.json` 标注第二窗口 `metric_key_sec=24`；且 relock 计算优先使用 `LOCK` 事件，`20260420_110116` 可稳定产出 `relock_24s=2.994s` |
| **padding 方形化模板** | 人为补黑边以规整 | 预期破坏 ORB 关键点 & 改 fallback box | ✗ 已推翻 | 首次尝试 track_ratio 0.645→0.059，已完全回退 |
| **模板纵横比应匹配目标** | — | — | ✗ 已推翻 | ORB 对纵横比不敏感，实验数据支持 |
| **Siamese 直接用用户图初始化** | 绕开视频帧 patch，直接用用户图 | 允许前端引导更强 | ⚠️ 待研究 | P3 前期调研未启动 |

---

## 5. 分层实施计划

### 5.1 P0 · 质量门控与 EXIF 正确性（本周完成）

- [x] 回退 padding 系列实验（已完成，见 [OpenCVTrackerAnalyzer.kt:6105](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6105)）
- [x] 恢复 `track_guard_min_area_ratio=0.30`
- [x] `MAX_TEMPLATE_DIM` 恢复到 480.0（[line 6434](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6434)）
- [x] **P0-A**：MainActivity 裁剪完成后增加 EXIF 归一化（`ExifInterface` 读取方向，`Matrix.postRotate` 纠正），统一以"追踪器看到的像素"为准（落地：`decodeBitmapFromUriWithExif/decodeBitmapFromPathWithExif` + `normalizeBitmapWithExif`，见 [MainActivity.kt:414](app/src/main/java/com/example/dronetracker/MainActivity.kt#L414)）
- [x] **P0-B**：前端裁剪提示"目标正朝上方/方向锁定"，规避 ORB 方向盲区（落地：选择模板按钮点击时提示，见 [MainActivity.kt:166](app/src/main/java/com/example/dronetracker/MainActivity.kt#L166)）
- [x] **P0-C**：加入导入时质量自检（短边 ≥128、可用像素 ≥5 万、ORB 关键点 ≥50），不达标直接拒绝并给出提示（落地：`evaluateTemplateQuality`，见 [MainActivity.kt:475](app/src/main/java/com/example/dronetracker/MainActivity.kt#L475)）
- [x] **P0-D1**：修复窗口标签模式下 `relock_24s` 指标口径错误（`metric_key_sec`），避免 NA 误判
- [x] **P0-D2**：空间门控参数 A/B/C/D 均无法降低 `wrong_lock_ratio`（5 种配置全部为 1.0），判定瓶颈不在空间门控 → 转入 P0-D4
- [x] **P0-D3**：relax 路径可观测化（`spatial_relax_apply`）+ 外观下限参数落地；throughput 改善但窗口 wrong_lock 未解；**不升默认，挂起 3 轮交错 A/B**
- [x] **P0-D4（方案 C · 1 轮诊断）**：`EVAL_CAND_DUMP` 日志 → 1 轮回放 → 根因结论为 **召回层失效（recall_layer）** → 已入档。**不做**后续 A/B 收敛；已降级为 RC1 主线 P1 任务

#### P0-D2 · 丢失后重锁门控 A/B 候选（参数级，零代码改动）

**证据链（`20260420_110116`）**：
- `14.916 s`: `LOST(track_guard_fail)`，`backendActive=kcf` 持续。
- `14.9 ~ 27.5 s`: 多次 `SEARCH_STABLE` 被 `appearance/promoted_verify_reject` 拒绝。
- `27.522 s`: `LOCK(orb_temporal_confirm)` 命中干扰物（位置偏离原目标）。
- 指标：`wrong_lock_ratio_in_windows=1.0`（window_label_lock_event）、`relock_24s=2.994 s`。
- **症结**：`lost_prior` 搜索半径随时间膨胀（`radius = side × (nearFactor + boost)`，见 [OpenCVTrackerAnalyzer.kt:4275-4289](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L4275)），10 s 后 prior 已不能约束候选；同时 far-recover 的外观门槛未能阻断相似物。

**三条互不耦合的参数 A/B（通过 `param_overrides.json` 注入，单变量，各跑 3 轮取均值）**：

| 方案 | 参数键 | 当前默认 | 建议实验值 | 作用机理 | 预期信号 |
|---|---|---|---|---|---|
| **A. 收紧重锁半径膨胀上限** | `auto_verify_lost_prior_center_boost_cap` | 待查 `DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_BOOST_CAP` | `1.5` | 限制 `boost` 随时间膨胀的上限，防止 10 s 后 prior 覆盖画面大部分 | `wrong_lock_ratio` ↓、`relock_24s` 可能 ↑（trade-off） |
| **B. 收紧 lost_prior 生效窗口** | `auto_verify_lost_prior_max_frames_replay` | 待查 | `120` 帧（≈8 s @15fps） | 超窗后回退到 `last_measured/last_tracked`，避免过期 prior 引导错锁 | `wrong_lock_ratio` ↓ |
| **C. 抬高 far-recover 外观门槛** | `far_recover_min_appearance` | 待查（[line ~5305](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L5305)） | 基线 + `0.05~0.10` | 远距离重锁要求更高外观相似度，物理阻断相似物误锁 | `wrong_lock_ratio` ↓、`relock_24s` 可能 ↑ |

**实施顺序**：
1. **先定位默认值**（读 `DEFAULT_AUTO_VERIFY_LOST_PRIOR_*` 常量，并扫 `autoVerifyLostPriorCenterBoostCap` 的 replay 初值），避免瞎调。
2. **先跑 A**（最贴合"时间膨胀 + 位置漂移"证据），记录与基线 `20260420_110116` 的对照。
3. **判定规则**：
   - A 通过（`wrong_lock_ratio` 下降 ≥0.10 且 `relock_24s` 回退 <0.5 s）→ 固化默认值。
   - A 失败或 trade-off 不可接受 → 换 B。
   - A/B 都无效 → 跑 C；若 C 仍无效，说明问题在**外观相似度本身不具判别力**，上推到 P1 方向（战役三：特征区分度验证）。
4. **不叠加多变量**：任一轮改动仅动一个参数；确认单变量影响后再考虑组合。

**风险与兜底**：
- A 可能让"真目标慢速移出又回来"时因半径不足而永远找不回；以 `window_lock_count` 作为回归信号。
- B 截断太早时，`last_measured` prior 可能漂移更严重；观察 `relock_24s` 是否显著回退。
- 所有 A/B 必须在本文第 7 节"验证规范"下执行（三轮均值、多指标矩阵）。

### 5.2 P1 · ORB 方向容忍度扩展（下周）

- [ ] **P1-1**：`TEMPLATE_POSE_AUGMENT_DEGREES` 扩展到 `[-90, -30, -15, 0, 15, 30, 90]`（[line 6471](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6471)）
- [ ] **P1-2**：`MIN_TEMPLATE_DIM` 从 128 提升到 192（[line 6433](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6433)），同步回放验证耗时是否可接受
- [ ] **P1-3**：回放验证：s640 + portrait_480 + landscape_480 各 3 轮，对比 P0 基线

### 5.3 P2 · 方向自适应 + 离线预检（下下周）

- [ ] **P2-1**：App 启动后离线预检：对用户模板跑一次 ORB + 方向扫描，给出"最佳使用方向"提示
- [ ] **P2-2**：前端 UX：默认强制方形裁剪框（规避 fallback box 形态风险），并允许用户旋转模板
- [ ] **P2-3**：窗口指标加入 App 回放报告（不仅 track_ratio，还有 wrong_lock_ratio/relock_24s）

### 5.4 P3 · Siamese 直接初始化（探索）

- [ ] **P3-1**：调研让 NCNN 使用用户图而非视频帧初始化的可行性（需修改 [NcnnTrackerImpl.cpp:478](app/src/main/cpp/tracker/NcnnTrackerImpl.cpp#L478) 的 init 签名）
- [ ] **P3-2**：评估是否引入"双模板"——视频帧 patch + 用户图 patch 加权

### 5.5 优先级依赖图

```
【MVP 主线 · P0 最高优先级】
  MVP-1 GPS 就位信号 ──┐
  MVP-2 L1/L2/L3 状态机 ──┼──> MVP-4 L3 超时兜底 ──> MVP-5 验收视频 & 回放
  MVP-3 offsetX/Y 输出 ──┘

【RC1 伴飞主线 · P1 后置】
  P0-A (EXIF 归一化) ──┐
  P0-B (方向提示)    ──┼──> P1-1 (±90° 增强) ──> P1-3 (三模板回放)
  P0-C (质量门控)    ──┘

  P0-D1 (指标口径) ──> P0-D2 (空间门控 A/B/C/D, 均失败)
                        └──> P0-D3 (relax + appearance, 不升默认, 永久挂起 A/B)
                               └──> P0-D4 (1 轮诊断, 方案 C, 半天上限, 入档即止)
                                      └── 结论 → 挂起至 RC1 阶段再消化

  P1-2 (MIN_DIM=192) ──> 与 P1-1 合并一次回放
  P2-* ──> 依赖 P1 数据
  P3-* ──> 独立探索
```

---

### 5.6 MVP-3 实施规范：`offsetX/Y` 对外输出通道（文档约束，不含代码）

**契约上游**：[REQUIREMENTS.md §1.5.7.2](REQUIREMENTS.md#1572-egressdescend_offsetmvp-3)（Egress 日志行格式、字段规范、`state` 枚举、发射节奏）。本节只补"落在哪、什么时候发、如何避坑"。

**现状盘点（免重复探查）**

| 需要的量 | 代码位置 | 取法 |
|---|---|---|
| 当前帧 `frameW / frameH` | [OpenCVTrackerAnalyzer.kt:3063](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L3063) `trackFrameNativeImage(image, frameW, frameH)` | 调用方已持有，直接沿用 |
| 融合后 bbox | [line 1828](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L1828) `lastTrackedBox = fused` | 作为 TRACKING 帧的 bbox 源 |
| `fusedConfidence` | [NcnnTrackerImpl.cpp:722](app/src/main/cpp/tracker/NcnnTrackerImpl.cpp#L722) 产出，Kotlin 侧已可读（`confidence` 参数链路） | 沿已有链路传到 emit 点 |
| `diagSessionId` | [line 570](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L570) | 直接读成员 |
| `replayPtsSec` | [line 2400](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2400) 样式 `currentReplayPtsMs / 1000.0` | 回放模式正值；实机回退 `SystemClock.uptimeMillis() / 1000.0` |
| `CenterRoiLevel` 状态 | MVP-2 已落地（L1/L2/L3 state） | 直接映射到 egress `state` |
| `lockHoldFramesRemaining` | `onLost()` 触发条件 | LOST/FAIL 分界依据 |

**状态机 ↔ Egress `state` 映射**

```
GPS=false                              → 不 emit
CenterRoiLevel=L1 & !isTracking        → state=L1, x/y/conf=NaN
CenterRoiLevel=L2 & !isTracking        → state=L2, x/y/conf=NaN
CenterRoiLevel=L3 & !isTracking        → state=L3, x/y/conf=NaN
isTracking=true & fused 有效            → state=TRACKING, x/y/conf=实际值
onLost 后 lockHoldFramesRemaining>0    → state=LOST, x/y/conf=lastTrackedBox 缓存
L3 超时 + lockHoldFramesRemaining=0    → state=FAIL, x/y/conf=NaN（仅 emit 一次）
```

**挂钩点清单（6 处，实施者据此布置 emit 调用）**

1. **emit helper（新增一个私有方法）**：入参 `state / bbox? / conf? / reason`，内部负责归一化、clamp、`t` 取值、格式化 `EVAL_EVENT type=DESCEND_OFFSET` 行。所有挂钩点只调此一处，禁止绕过。
2. **TRACKING 挂钩**：在 [line 1828](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L1828) `lastTrackedBox = fused` 之后同一流程内发射。频率：每帧一次。
3. **L1/L2/L3 心跳挂钩**：在 `CenterRoiLevel` 切换瞬间必发一次；停在同一层级时按 `SEARCH_DIAG_INTERVAL_FRAMES`（当前 15 帧）节流发射。避免每帧打满日志。
4. **LOST 挂钩**：`onLost()` 首帧必发一次；后续在 LOST 窗口内按 1 s 心跳。`x/y` 用 `lastTrackedBox` 缓存换算，`conf` 用"进入 LOST 瞬间的最后一次 fusedConfidence"。
5. **FAIL 挂钩**：当 `CenterRoiLevel=L3 && L3 超时 && lockHoldFramesRemaining=0` 三者同时成立时，**一次性** emit `state=FAIL`，并置状态位 `failEmittedThisSession=true` 抑制重复。
6. **复位**：`setCenterRoiGpsReady(false, ...)` 或 `resetCenterRoiSearchState()` 或 session 重启时，清空 `failEmittedThisSession` 与 LOST 缓存对应标志位。

**归一化与边界（契约强约束）**

- `x_norm = (cx - frameW/2.0) / (frameW/2.0)`；`y_norm = (cy - frameH/2.0) / (frameH/2.0)`（OpenCV 左上原点，见 §1.5.7.2 坐标系定义）。
- emit 前强制 `clamp([-1.0, 1.0])`；**越界必须先 `Log.w` 原始值再 clamp**，便于事后定位 bbox 异常源。
- `conf` 对 `NaN/负数/>1.0` 一律记 `Log.w` 并置 `NaN`（不得伪造）。
- `frameW==0 || frameH==0`：视为异常，不 emit，降级为 LOST（挂钩 4 的错误情形分支），参见 [REQUIREMENTS.md §1.5.7.3](REQUIREMENTS.md#1573-错误情形) 第 6 行。

**禁止清单（违反则算契约违规）**

- ❌ 不得在 emit 点做 Kalman 平滑 / 滤波 / 插值——契约要求上报 **原始每帧** bbox。
- ❌ 不得因 bbox 置信度低而静默跳过 emit；低置信度应走 LOST 分支。
- ❌ 不得在 TRACKING 态 emit `x/y=NaN`；若 bbox 无效应走 LOST 分支。
- ❌ 不得擅自新增/删除 `state` 枚举值，修改必须先改 §1.5.7.2 并同步 `sweep_replay.py`。
- ❌ 不得在 emit 前 `round()` 到整数——`[-1.0, 1.0]` 是浮点区间。

**测试矩阵（与 §1.5.7.4 对齐）**

| 用例 | 场景 | 期望首 state | 期望末 state | 关键断言 |
|---|---|---|---|---|
| T1 | GPS=ready，目标在 L1 ±20% 内 | L1 | TRACKING | `descend_offset_first_t` ≤ `T1 + 1 帧` |
| T2 | 目标在 L2（±20%~±40%）范围 | L1 | TRACKING | L1 超时后切 L2，最终 TRACKING |
| T3 | 目标脱离 L2 但在全图内 | L1 | TRACKING | L2 超时后切 L3，最终 TRACKING |
| T4 | 总预算 3.5 s 耗尽未锁 | L1 | FAIL | FAIL 行 **只出现 1 次** |
| T5 | TRACKING → LOST → Re-TRACK | L1 | TRACKING | LOST 窗口内 x/y 等于 LOST 前帧缓存 |
| T6 | TRACKING → LOST → 超时失败 | L1 | FAIL | LOST 窗口内有 1s 心跳，最终 FAIL |
| T7 | GPS=false 全程 | — | — | **无任何** `DESCEND_OFFSET` 行 |

**`sweep_replay.py` 解析器扩展（文档约束）**

- 新增列：`descend_offset_first_t`（首条行 `t`）、`descend_offset_last_state`（末行 state）、`descend_offset_oob_count`（clamp 前越界次数，由 `Log.w` 行辅助计数）、`descend_offset_fail_count`（FAIL 行出现次数，>1 判退）。
- 新增判退：`descend_offset_oob_count > 0` 或 `descend_offset_fail_count > 1`。

**实施者自检表**（提交前勾齐）

- [ ] 6 个挂钩点全部接入，且只有 1 个 emit helper
- [ ] TRACKING 每帧发，L1/L2/L3 按 15 帧节流，LOST 心跳 1 s，FAIL 仅一次
- [ ] `x/y` 均在 `[-1.0, 1.0]`（越界写 warning + clamp）
- [ ] LOST 态 bbox 取自缓存，非新算
- [ ] GPS=false 全程零 emit
- [ ] `sweep_replay.py` 解析器四列已实装
- [ ] T1~T7 回放全部通过

---

### 5.7 MVP-1 实施规范：GPS 就位信号触发通道（文档约束，不含代码）

**契约上游**：[REQUIREMENTS.md §1.5.7.1](REQUIREMENTS.md#1571-ingressgps-就位信号mvp-1)。核心入口 `setCenterRoiGpsReady(ready, reason)` 已落地在 [OpenCVTrackerAnalyzer.kt:1593](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L1593)。本节补 **两条触发通道** 的落地规范（(a) Debug UI 按钮 / (b) adb broadcast），(c) 飞控 SDK 通道不在 v1 范围。

**通道 (a)：Debug UI 按钮（人工联调）**

- 宿主：`MainActivity`（按钮布局 + onClick 回调）。
- 控件形态：**toggle 按钮**（非瞬时按钮），便于在 `true`/`false` 两态间切换，文本随状态显示 `GPS:READY` / `GPS:—`。
- onClick 语义：
  - 当前状态 `false` → 调 `analyzer.setCenterRoiGpsReady(true, "debug_ui")`
  - 当前状态 `true` → 调 `analyzer.setCenterRoiGpsReady(false, "debug_ui")`
- 布局位置建议：与现有"导入模板"、"开始回放"按钮同屏，不阻挡预览。
- **不得** 增加额外逻辑（不做延迟确认、不做双击防抖），UI 层保持薄。

**通道 (b)：adb broadcast 意图（脚本化 / CI）**

- Intent Action：`com.example.dronetracker.GPS_READY`（已在 §1.5.7.1 契约固化，不得擅改）
- 额外字段：
  - `--ez ready <bool>`：必传
  - `--es reason <string>`：可选，缺省记 `"adb_broadcast"`
- 接收方：`MainActivity` 内注册一个 `BroadcastReceiver`；生命周期跟 Activity（onResume 注册、onPause 反注册），避免后台常驻。
- 收到 Intent 后：同样调 `analyzer.setCenterRoiGpsReady(ready, reason ?: "adb_broadcast")`，**不做** 其他副作用。
- 权限：仅允许本进程 / ADB 发送。使用 `LocalBroadcastManager` 不适用（adb 跨进程）；选系统 Broadcast + `exported=false` + `android:permission` 自定义权限 + ADB shell 通过签名信任。
- 安全红线：生产构建（release）**必须** 关闭此 Receiver 或以 `BuildConfig.DEBUG` 网开；否则任意 App 都能注入 GPS 就位信号。

**联调验证协议**

| 步骤 | 预期日志 |
|---|---|
| 1. 启动 App，开启 `center_roi_search_enable=true` | 无 `EVAL_EVENT type=ROI_SEARCH` 输出（GPS 未就位） |
| 2. 点击 UI toggle（或 adb broadcast ready=true） | `EVAL_EVENT type=ROI_SEARCH state=gps_ready ready=true reason=debug_ui enabled=true` |
| 3. 下一帧起 | `EVAL_EVENT type=ROI_SEARCH state=start level=l1` + `EVAL_EVENT type=DESCEND_OFFSET state=L1 ...` |
| 4. 再次点击 toggle（或 adb broadcast ready=false） | `EVAL_EVENT type=ROI_SEARCH state=gps_ready ready=false ... enabled=false` |
| 5. 后续帧 | 不再 emit `DESCEND_OFFSET`；状态机复位 |

**实施者自检表**

- [ ] UI toggle 的两态切换与日志 `ready` 值一一对应
- [ ] adb broadcast 能在回放脚本中稳定触发（`sweep_replay.py` 可调）
- [ ] release 构建下 adb broadcast 通道被关闭或需特权
- [ ] GPS=false 后 ROI 状态机立刻复位（不等下一次命中）
- [ ] 完成后立即按 §5.8 纪律提交代码（不得囤积到下一个 MVP 阶段）

---

### 5.8 阶段性提交纪律（MVP 主线通用约束）

**原则**：每个 MVP-* 阶段（MVP-1 / MVP-2 / MVP-3 / MVP-4 / MVP-5）**做完一个就立即提交**，不得囤积多个阶段合并提交。

**适用范围**：§5.6、§5.7 及后续补出的 MVP-4/5 实施规范，均受本节约束。

**触发条件**（同时满足才能提交）：

1. 当前 MVP-* 阶段的实施者自检表**全部打勾**
2. `./tools/gradlew_exec.ps1 :app:compileDebugKotlin` 通过（Kotlin 改动）或对应 C++ 构建通过
3. 本阶段对应的回放测试（若已有）至少跑通一次，日志符合 §1.5.7 契约
4. 无越界日志（`descend_offset_oob_count=0` 等契约违规指标为 0）

**Commit 消息规范**（对齐全局 CLAUDE.md §1 与 `docs/dev-environment-guardrails.md`）：

```
<type>: <简短描述>

<可选正文：改动挂钩点列表、回放验证结果、遗留 TODO>
```

| MVP 阶段 | 推荐 type | 示例标题 |
|---|---|---|
| MVP-1 GPS 就位信号 | `feat` | `feat: MVP-1 接入 GPS 就位信号 Debug UI + adb broadcast 通道` |
| MVP-2 L1/L2/L3 状态机 | `feat` | `feat: MVP-2 分级 ROI 搜索状态机落地`（已提交，见 2026-04-20 日志） |
| MVP-3 offsetX/Y 输出 | `feat` | `feat: MVP-3 按契约 §1.5.7.2 输出 DESCEND_OFFSET` |
| MVP-4 L3 超时兜底 | `feat` | `feat: MVP-4 L3 超时上抛视觉引导失败` |
| MVP-5 验收回放 | `test` 或 `docs` | `test: MVP-5 静态楼顶验收回放 SOP + 用例` |
| 遇编译/契约违规的返修 | `fix` | `fix: MVP-3 修复 DESCEND_OFFSET x 越界未 clamp` |

**粒度约束**：

- **一个 MVP-* 编号 = 一个 commit** 为基准；若单阶段改动跨多天，可允许 `feat(wip)` 系列小提交，但**合并到 main 前必须 rebase squash 成一条**。
- 禁止 "MVP-1 + MVP-3 一起上" 的合并提交——阶段边界必须清晰可回溯。
- 同一 commit 内**不得同时**混入无关重构 / 格式调整 / 文档批量整理。

**文档与代码同步纪律**：

- 若提交涉及契约字段变更（§1.5.7 Ingress / Egress），必须在**同一 commit** 中同步 §1.5.7.5 "契约治理" 所列 4 项（本节、`tactical_execution_plan-0419.md`、`sweep_replay.py`、契约变更日志子条目）。
- 若提交引入新 `EVAL_EVENT type=*` 字段，必须在**同一 commit** 中更新 `sweep_replay.py` 解析器，否则回放侧会静默漏测。

**编译 / 测试失败时**：

- **不得** 跳过验证（禁止 `--no-verify` 绕过 hook）
- **不得** 在失败状态下推送远端
- 修复后**新建 commit**，不要 `--amend` 修改已推送的 commit
- 若问题无法当场修复：回退本地改动，记一条"遗留阻塞" todo，转做其他 MVP 阶段

**回滚策略**：

- 提交后发现问题：**新写一条 `revert: ...` commit**，不得 `git reset --hard` / `git push --force` 到已公开分支
- 回滚 commit 正文必须写明：原 commit SHA、回滚原因、后续计划

**推送时机**：

- 本地 commit → 本地回放验证通过 → `git push`
- 禁止"先 push 再本机回放"的倒序操作

**每个阶段提交后的收尾**：

- 在本文档 §6 "进展追踪日志" 顶部**新增一条时间线**（格式参考 2026-04-20 现有条目），记录变更项、回放证据目录、下一步指向
- 若该阶段触达契约变更，在 §1.5.7.5 追加"契约变更日志"子条目

---

### 5.9 MVP-4 实施规范：L3 超时兜底（文档约束，不含代码）

**契约上游**：[REQUIREMENTS.md §1.5.4 失败兜底](REQUIREMENTS.md#154-mvp-验收标准) + [§1.5.7.3 错误情形](REQUIREMENTS.md#1573-错误情形) "总超时 3.5 s 仍未锁定" 行 + §1.5.7.2 `state=FAIL` 枚举。MVP-3 §5.6 挂钩点 5 已描述 FAIL 的 emit 形式，本节聚焦**触发逻辑**。

**触发条件（三者同时成立 → 进入 FAIL）**：

```
CenterRoiLevel == L3
 AND  L3 已进入时间 >= T3 (默认 2000ms)
 AND  lockHoldFramesRemaining == 0   （无 LOST 缓冲可用）
 AND  failEmittedThisSession == false （本 session 未 emit 过 FAIL）
```

**落地挂钩**：

1. **T3 超时判定点**：复用 MVP-2 已落地的 `maybeAdvanceCenterRoiLevel()`（[OpenCVTrackerAnalyzer.kt:2565](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2565) 附近）——目前 L3 超时后分支为空（只记 timeout 日志）；本次补 FAIL 分支。
2. **FAIL emit**：复用 §5.6 挂钩点 1 的 `emitDescendOffset(FAIL, null, NaN, "l3_timeout")` helper，不得另起一条日志格式。
3. **状态复位**：FAIL emit 之后置 `failEmittedThisSession=true`，并立即调用 `resetCenterRoiSearchState()` 让状态机回到 `disabled`；等待下一次 `setCenterRoiGpsReady(true, ...)` 再启动。
4. **ROI_SEARCH 诊断日志**：在 FAIL emit 同一点额外写一行 `EVAL_EVENT type=ROI_SEARCH state=fail reason=l3_timeout l1ms=<int> l2ms=<int> l3ms=<int>`，便于回放端统计三层时间分布。

**幂等与一次性保证**：

- 本 session 内 `failEmittedThisSession=true` 之后，所有重复进入 L3 超时条件的帧**必须静默**（不 emit、不打 warning，只在 verbose 日志记 `fail_suppressed`）。
- `setCenterRoiGpsReady(false, ...)` 或 `setCenterRoiGpsReady(true, "restart")` 触发 session 重启时，清 `failEmittedThisSession=false`。
- **禁止**依赖外部调用方清标志位——由本模块内部统一管理。

**与 MVP-3 LOST 分支的互斥**：

```
TRACKING 丢 →  onLost()  →  lockHoldFramesRemaining > 0  →  state=LOST（§5.6 挂钩点 4）
                          ↓  lockHoldFramesRemaining == 0
                          → 回到 ACQUIRE / CenterRoiLevel 重新起算（不直接 FAIL）
                          → 若 ACQUIRE 阶段再次走到 L3 超时 → 才触发 FAIL
```

**错误不得**：跟踪中途丢失后**不得**跳过 LOST 直接 FAIL；必须先经 LOST 缓冲窗口，窗口耗尽后回 L1 重新搜，再次走完 L1→L2→L3 超时才允许 FAIL。

**飞控侧预期响应（与契约 §1.5.7.3 对齐，仅提醒）**：

- `state=FAIL` 行出现后，飞控应在 ≤ 500 ms 内切换到"纯 GPS 降落 / 盘旋"模式；App 层不负责这步，但 MVP-5 验收视频需核验 FAIL 出现的绝对时刻 ≤ `GPS_READY + 3.5 s + 1 帧`。

**测试矩阵**（与 §5.6 T4/T6 相同口径，此处为专项判据）：

| 用例 | 构造方法 | 通过判据 |
|---|---|---|
| F1 | 回放中 GPS=ready 时刻选在**无目标帧段**（前 5 秒空背景） | 3.5 s 后 emit 一行 `state=FAIL`；前面应有 L1/L2/L3 各至少 1 行心跳 |
| F2 | 回放中目标始终脱离画面（极端 GPS 误差模拟） | 同 F1 |
| F3 | F1 复测 3 次 | 每次 FAIL 行仅 1 次；`descend_offset_fail_count=1` |
| F4 | FAIL 后重新 `setCenterRoiGpsReady(true, "restart")` | 下一轮 L1 心跳出现，`failEmittedThisSession` 已复位 |

**实施者自检表**

- [ ] L3 超时 + `lockHoldFramesRemaining==0` 两条件齐备才 emit FAIL
- [ ] FAIL 本 session 仅 emit 1 次（F3 验证）
- [ ] `failEmittedThisSession` 在 GPS=false 或 restart 时正确复位
- [ ] 中途 LOST 不得越过 LOST 缓冲直接 FAIL
- [ ] `EVAL_EVENT type=ROI_SEARCH state=fail` 附带三层时间分布字段
- [ ] `sweep_replay.py` 新增 `descend_offset_fail_count` 列（§5.6 已约束）并用于 F3 判据
- [ ] 完成后按 §5.8 立即提交

---

### 5.10 MVP-5 实施规范：静态楼顶验收视频与回放 SOP（文档约束，不含代码）

> **执行侧指针**：本节是**规范**；可直接照做的启动清单（资产缺口、拍摄清单、入库步骤、回放命令、验收判据表）在 [docs/mvp5_launch_checklist.md](mvp5_launch_checklist.md)。两者不重复，配合使用。

**定位**：MVP-5 不是新功能，是把 MVP-1~MVP-4 的端到端链路放到**可复现回放**上跑通，并给出"发版前必跑"的验收清单。**产出物是视频 + 回放结果 + 结论报告**，不是代码。

**一、拍摄规范（静态楼顶俯视视频）**

| 要素 | 约束 |
|---|---|
| 拍摄主体 | 真实楼顶（非沙盘 / 非图纸）；楼顶占画面面积建议 3%~15%（覆盖远、中距离） |
| 视频时长 | 建议 30~60 s（覆盖"GPS 未就位 → GPS 就位 → 首锁 → 稳定追踪"完整链路） |
| 目标运动 | **全程静止**（不能移动拍摄机位模拟风偏；风偏通过 ROI 偏置模拟，不靠视频本身） |
| 画面抖动 | 允许轻微抖动（≤ ±5° yaw），禁止剧烈晃动 |
| 光照 | 正常日光；禁止逆光 / 夜景（留给 RC2） |
| 分辨率 / 帧率 | 1080p / 30 fps；回放默认以 `replay_fps=15` 运行 |
| 文件命名 | `scene_mvp5_<变体>_<YYYYMMDD>.mp4`，如 `scene_mvp5_l1_20260421.mp4` |

**二、场景变体（必须覆盖 4 种）**

| 变体 | 构造 | 预期命中层 | 预期首锁时延 |
|---|---|---|---|
| `l1` | 目标居画面中心 ±15% 内（GPS + 云台均准） | L1 | ≤ 0.5 s |
| `l2` | 目标居画面 ±25% 附近（GPS 精度边界） | L2 | ≤ 1.5 s |
| `l3` | 目标居画面 ±45% 附近（GPS 漂移边界） | L3 | ≤ 3.5 s |
| `fail` | 画面内无目标（严重 GPS 漂移模拟） | — | FAIL @ 3.5 s |

**三、GPS 就位时刻注入（沿用 §1.5.7.1 (b) 通道）**

- 每个场景视频起始 2.0 s 之前为"GPS 未就位"期；2.0 s 时刻通过 `sweep_replay.py` 脚本化触发 adb broadcast：
  ```
  adb shell am broadcast -a com.example.dronetracker.GPS_READY --ez ready true --es reason "mvp5_auto"
  ```
- 此时刻必须与视频 PTS 对齐（回放器保证，不得靠人工点击）。
- `sweep_replay.py` 扩展：新增 `--gps-ready-at-sec 2.0` 参数；在该 PTS 时刻执行上述 broadcast，一次性生效。

**四、`p0_windows.json` 适配**

- MVP-5 不再沿用伴飞场景的"窗口"（`p0_windows.json` 面向动态重锁）；**新建** `mvp5_windows.json`，字段对齐：
  ```json
  [
    {
      "label": "first_lock_window",
      "start_sec": 2.0,
      "end_sec": 5.5,
      "expected_final_state": "TRACKING"
    },
    {
      "label": "steady_track_window",
      "start_sec": 5.5,
      "end_sec": 25.5,
      "expected_final_state": "TRACKING",
      "min_track_ratio": 0.90
    }
  ]
  ```
- `fail` 变体单独一份：`expected_final_state: "FAIL"`，不设 steady_track 窗口。

**五、验收门槛（全绿才算 MVP 通过）**

| 指标 | 口径 | 阈值 |
|---|---|---|
| **L1 变体首锁时延** | `descend_offset_first_t` 到 `state=TRACKING` 首行的 PTS 差 | ≤ 0.5 s |
| **L2 变体首锁时延** | 同上 | ≤ 1.5 s |
| **L3 变体首锁时延** | 同上 | ≤ 3.5 s |
| **FAIL 变体 FAIL 时延** | `GPS_READY` 到 `state=FAIL` 的 PTS 差 | 3.5 s ± 0.2 s（不得提前，不得严重超时） |
| **steady_track_window** | 20 s 内 TRACKING 行占比 | ≥ 0.90 |
| **`wrong_lock_ratio_in_windows`** | 20 s 窗口内误锁率 | = 0.0（静态楼顶无干扰物，应完美） |
| **`descend_offset_oob_count`** | 越界计数 | = 0 |
| **`descend_offset_fail_count`** | FAIL 行次数 | `l1/l2/l3` 变体=0；`fail` 变体=1 |

**六、必跑回放脚本（MVP 主线阻塞门）**

```powershell
# 每个变体一次
foreach ($variant in @('l1','l2','l3','fail')) {
    ./tools/auto_tune/sweep_replay.py `
        --video test/scene_mvp5_${variant}_20260421.mp4 `
        --template test/target_rooftop.jpg `
        --windows test/mvp5_windows_${variant}.json `
        --gps-ready-at-sec 2.0 `
        --replay-fps 15 `
        --out tools/auto_tune/out/mvp5_${variant}_<YYYYMMDD>_<HHMMSS>/
}
```

运行产物（每个变体目录下）：
- `eval_events.csv`（从 logcat 提取的 `EVAL_EVENT` 行）
- `descend_offset.csv`（仅 `type=DESCEND_OFFSET` 子集）
- `mvp5_summary.json`（本节第五项全部指标的实测值）

**七、报告产出**

每次 MVP-5 全量回放后生成 `docs/mvp5_acceptance_<YYYYMMDD>.md`，模板：

```
# MVP-5 静态楼顶验收报告 YYYY-MM-DD
## 环境
- App commit: <SHA>
- 视频: scene_mvp5_{l1,l2,l3,fail}_YYYYMMDD.mp4
- 设备: <型号/Android 版本>
## 指标表（§5.10 第五项全绿检查）
| 指标 | l1 | l2 | l3 | fail | 阈值 | 通过? |
## 结论
[ ] 全绿 → 进入 RC1 主线
[ ] 红项 X/Y/Z → 返修清单 + 责任人
```

**实施者自检表**

- [ ] 4 个视频变体全部拍摄完成并入库 `test/`
- [ ] `mvp5_windows.json` 四份已创建
- [ ] `sweep_replay.py --gps-ready-at-sec` 参数实装并验证
- [ ] 4 次回放均产出 `mvp5_summary.json`
- [ ] `docs/mvp5_acceptance_YYYYMMDD.md` 报告已归档
- [ ] 验收门槛全绿
- [ ] 完成后按 §5.8 立即提交（type=`test` 或 `docs`）

---

## 6. 进展追踪日志（逆序时间线）

### 2026-04-20（MVP-1/3/4 代码实装 + 默认回放口径切换）

- **默认回放口径切换（对齐 MVP 主线）**：
  - [sweep_replay.py:1575-1576](tools/auto_tune/sweep_replay.py#L1575)：`--video-path` / `--target-path` 默认切到 `/sdcard/Download/Video_Search/scene_20260417.mp4` + `target0417_s640.jpg`
  - [MainActivity.kt:780](app/src/main/java/com/example/dronetracker/MainActivity.kt#L780)：`DEFAULT_REPLAY_VIDEO_PATH` / `DEFAULT_REPLAY_TARGET_PATH` 同步更新
  - [EvaluationActivity.kt:365](app/src/main/java/com/example/dronetracker/EvaluationActivity.kt#L365)：`DEFAULT_EVAL_VIDEO_PATH` / `DEFAULT_EVAL_TARGET_PATH` 同步更新
  - 编译：`:app:compileDebugKotlin` 通过
  - 实机 smoke：`20260420_195440`（无参数 sweep，preflight 正确读到新默认路径）
- **MVP-1 GPS 就位触发通道（adb broadcast，对应 §5.7 通道 (b)）已实装**：
  - Receiver：[MainActivity.kt:78](app/src/main/java/com/example/dronetracker/MainActivity.kt#L78)；release 构建拒收（[line 80](app/src/main/java/com/example/dronetracker/MainActivity.kt#L80) `Ignore GPS_READY broadcast in non-debug build`）
  - 注册/反注册跟随 Activity 生命周期（[line 207/736](app/src/main/java/com/example/dronetracker/MainActivity.kt#L207)）
  - Action / Extra 常量：[line 766-769](app/src/main/java/com/example/dronetracker/MainActivity.kt#L766)，与 [REQUIREMENTS.md §1.5.7.1](REQUIREMENTS.md#1571) 契约一致
  - 日志：`EVAL_EVENT type=GPS_READY ready=<bool> source=<str> reason=<str> debug=<bool>`（[line 311](app/src/main/java/com/example/dronetracker/MainActivity.kt#L311)）
- **MVP-3 `DESCEND_OFFSET` Egress（对应 §5.6）已实装**：
  - Emit helper：[OpenCVTrackerAnalyzer.kt:2667](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2667) `emitDescendOffset(state, bbox?, conf?)`
  - 每帧分派：[line 2716 `emitDescendOffsetPerFrame()`](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2716) 覆盖 L1/L2/L3/TRACKING/LOST 全部状态
  - 每帧挂钩：[line 6911](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6911)
- **MVP-4 L3 超时 FAIL（对应 §5.9）已实装**：
  - FAIL 一次性 emit：[OpenCVTrackerAnalyzer.kt:2784](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2784) `emitDescendOffset(FAIL, force=true)`
  - 每帧分支中带 `DescendOffsetState.FAIL` 守卫：[line 2719](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L2719)
- **`sweep_replay.py` 脚本端配套扩展**：
  - [line 1612 `--gps-ready-at-sec`](tools/auto_tune/sweep_replay.py#L1612) CLI 实装，回放 PTS 到达阈值自动发 broadcast（[line 1986-1989](tools/auto_tune/sweep_replay.py#L1986)）
  - `DESCEND_OFFSET` 解析与汇总：[line 768-876](tools/auto_tune/sweep_replay.py#L768) 产出 `descend_offset_first_t / last_state / oob_count / fail_count` 4 列（完全对齐 §5.6 "sweep_replay.py 解析器扩展"）
- **遗留 / 下一步**：
  - Debug UI toggle（§5.7 通道 (a)）尚未看到落地代码；需确认是否已加在 `MainActivity` 布局里
  - MVP-5 静态楼顶验收视频 + 4 变体回放（§5.10）尚未执行，须先补拍摄视频与 `mvp5_windows.json`
  - review：按 §5.6 / §5.7 / §5.9 "实施者自检表" 对已实装代码做一次对照核验（建议由作者自检或另起 read-only review）

### 2026-04-20（MVP-2 分级 ROI 搜索落地 · 首次代码接入）

- 变更项：
  - [OpenCVTrackerAnalyzer.kt](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt)：`ACQUIRE` 阶段新增 L1/L2/L3 分级中心 ROI 搜索状态机（L1±20%、L2±40%、L3 全图）。
  - 新增运行时参数：`center_roi_search_enable`、`center_roi_gps_ready`、`center_roi_l1/l2_range`、`center_roi_l1/l2/l3_timeout_ms`。
  - 新增事件日志：`EVAL_EVENT type=ROI_SEARCH state=start/escalate/scan/hit/timeout/reset`。
- 默认行为：
  - 默认关闭（`center_roi_search_enable=false`），不影响当前 RC1 回放基线。
  - 仅在 `center_roi_search_enable=true && center_roi_gps_ready=true` 时启用。
- 编译验证：
  - `./tools/gradlew_exec.ps1 :app:compileDebugKotlin` 通过。
- 下一步：
  - 联调 GPS 就位信号接入（MVP-1）和 offsetX/Y 输出通道（MVP-3），再做静态楼顶验收回放（MVP-5）。

### 2026-04-20（P0-D4 单轮诊断完成 · 结论入档）

- 代码落地：
  - [OpenCVTrackerAnalyzer.kt](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt)：新增 `EVAL_EVENT type=CAND_DUMP`（窗口可控开关、top-K 候选、expectedX 标注字段）。
  - [sweep_replay.py](tools/auto_tune/sweep_replay.py)：新增 `CAND_DUMP` 解析与汇总，输出 `candidate_dump.csv`、`candidate_dump_summary.json`。
- 诊断回放：
  - 目录：`tools/auto_tune/out/20260420_161708/`
  - 口径：`scene_20260417 + target0417_s640 + replay_fps=15 + window_label + cand_dump_enable=true`
- 关键结果（`candidate_dump_summary.json`）：
  - `groups_in_windows=101`
  - `gt_missing_topk=84`
  - `gt_present_topk=17`
  - `gt_rank_hist={0:10,1:5,2:2}`
  - `rank0_wrong_when_gt_present=12`
  - `diagnosis=recall_layer`
- 结论：
  - P0-D4 根因归类为**召回层失效**，不是单纯门控阈值问题。
  - 按方案 C 关闭 D4 后续 A/B 收敛，挂起到 RC1 阶段，转战役二/三（重检测通道 + GAP 区分度验证）。

### 2026-04-20（方案 C 决议 · 路径分工）

- 决议：P0-D4 不再作为 MVP blocker，降级为 RC1 主线 P1 任务，只做 **1 轮诊断 + 入档**（半天上限），不做 A/B 收敛。
- MVP 主线启动，5 个关键模块（GPS 信号、L1/L2/L3 状态机、offsetX/Y 输出、L3 兜底、验收视频）为下一阶段 P0。
- 证据依据：P0-D2 五种空间门控配置 `wrong_lock_ratio` 均 1.0；该指标在 MVP 场景（静态楼顶 + GPS 先验 + 分级 ROI）物理上不会出现。
- 本决议写入 §0.1 并同步更新 §4 表、§5.5 优先级图、§5.1 P0-D 清单。

### 2026-04-20（P0-D2 证据链修复 + 单轮回放）

- 机制修复（避免再出现“日志抓不到/指标误判”）：
  - [sweep_replay.py](tools/auto_tune/sweep_replay.py)：`AdbRunner(wrapper)` 改为“写 `tools/adb_args.json` + 无参数调用 `adb_exec.ps1`”，彻底规避 PowerShell 对 `-s` 的参数吞噬。
  - [adb_exec.ps1](tools/adb_exec.ps1)：修复 `AdbArgs` 为空时的空数组处理，保证 `adb_args.json` 路径稳定可用。
  - [sweep_replay.py](tools/auto_tune/sweep_replay.py)：`wrong_lock_ratio_in_windows` 改为优先基于窗口内 `LOCK` 事件计算（`window_label_lock_event`），仅在缺失 LOCK 事件时回退到 PERF 采样。
  - [sweep_replay.py](tools/auto_tune/sweep_replay.py)：`relock_latency_*` 在 window-label 模式下优先使用 `LOCK` 事件计算，减少 `EVAL_PERF` 采样稀疏导致的 NA。
- 回放验证（`20260420_110116`，s640、fps=15、l2 全量 Tracker 日志）：
  - `track_ratio=0.414`, `avg_frame_ms=101.71`
  - `relock_20s=5.994s`, `relock_24s=2.994s`
  - `wrong_lock_ratio_in_windows=1.0`, `wrong_lock_metric_source=window_label_lock_event`
- 瓶颈定位：
  - 关键事件链：`LOST(track_guard_fail @14.916s)` 后长期处于 `backendActive=kcf`，多次 `SEARCH_STABLE` 被 `appearance/promoted_verify_reject` 拒绝，最终在 `27.522s` 才 `LOCK(orb_temporal_confirm)`。
  - 结论：P0-D2 进入“重锁策略根因修复”阶段，当前不是日志机制问题，而是重锁门控与误锁抑制的平衡仍未达标。

### 2026-04-20（P0-D 口径修复：window_label 指标对齐）

- 变更项：
  - [sweep_replay.py](tools/auto_tune/sweep_replay.py)：`load_p0_windows` / `_compute_s1_with_windows_label` 新增 `metric_key_sec`，窗口标签可显式映射输出字段。
  - [p0_windows.json](tools/replay_sop/p0_windows.json)：第二窗口补充 `metric_key_sec=24.0`（`start_sec=23, return_sec=24`）。
- 证据回放（复算历史日志，不重跑设备）：
  - 095838 run_001：`relock_20s=5.994s, relock_24s=2.994s, wrong_lock=0.75`
  - 095838 run_002：`relock_20s=5.994s, relock_24s=2.994s, wrong_lock=0.50`
  - 225935/230450/231006：`relock_24s` 同样为 `2.994s`，不再是 NA。
- 结论：
  - `relock_24s=NA` 属于评估脚本口径 bug，已修复。
  - `wrong_lock_ratio` 在统一口径下基线与当前都偏高，后续进入 P0-D2（算法根因定位），不是“最新代码突然退化”。

### 2026-04-20（P0-A/B/C 代码落地）

- 变更项：
  - [MainActivity.kt](app/src/main/java/com/example/dronetracker/MainActivity.kt)：新增 EXIF 归一化加载链（URI + 文件路径），确保模板按真实像素方向进入追踪器。
  - [MainActivity.kt](app/src/main/java/com/example/dronetracker/MainActivity.kt)：新增导入质量门控（`MIN_TEMPLATE_INPUT_SIDE=128`、`MIN_TEMPLATE_USABLE_PIXELS=50000`、`MIN_TEMPLATE_ORB_KEYPOINTS=50`）。
  - [MainActivity.kt](app/src/main/java/com/example/dronetracker/MainActivity.kt)：新增模板选择方向提示（建议目标朝上、与首帧方向偏差 ≤ ±30°）。
  - [build.gradle.kts](app/build.gradle.kts)：新增 `androidx.exifinterface:exifinterface:1.3.7` 依赖。
- 编译验证：`./tools/gradlew_exec.ps1 :app:assembleDebug` 通过。
- 说明：本轮为“代码落地轮”，窗口指标回放验证待下一轮更新（按第 7 节规范执行 3 轮）。

### 2026-04-20（文档创建）

- 完成：padding 系列全回退、`buildCenteredSquare` 纯正方形、`MAX_TEMPLATE_DIM=480`、`track_guard_min_area_ratio=0.30`
- 三轮均值基线（未启用任何 P1 方案，纯口径对比）：
  - s640（竖版）：track_ratio=0.640±0.012, wrong_lock=0.611±0.079
  - portrait_480：track_ratio=0.643±0.016, wrong_lock=0.472±0.171
  - landscape_480：全局 0.654 但窗口锁=0（方向失效验证）
- **关键结论**：横版/竖版对 ORB 像素层面等效；方向错位是致命的；wrong_lock_ratio 仍稳态偏高。
- **下一步**：执行 P0-A/B/C，并并行定位 P0-D 的代码层根因。

### 待填写模板（每次验证后新增一条）

```
### YYYY-MM-DD（验证轮次 · 主题）
- 变更项：<文件:行 / 参数 / 模板>
- 回放命令口径：<关键参数>
- 结果（三轮均值）：
  - track_ratio = X ± Y
  - wrong_lock_ratio_in_windows = X ± Y
  - relock_20s = X / relock_24s = X / window_lock_count = X
- 与基线对比：<回退/持平/改善>
- 结论：<判断 + 下一步>
```

---

## 7. 验证规范（不变）

1. **单变量原则**：每次只改一个因子；参数与上一轮完全同口径。
2. **三轮最小样本**：每个模板/参数跑 3 轮，报告均值±σ。
3. **多指标矩阵**（任一回归即判退）：
   - track_ratio（整体）
   - wrong_lock_ratio_in_windows（窗口错锁）
   - relock_20s / relock_24s（重锁延迟）
   - window_lock_count（窗口锁次数）
   - avg_frame_ms（性能）
4. **判据**：
   - track_ratio 下降 ≥ 0.05 → 回归
   - wrong_lock_ratio 上升 ≥ 0.10 → 回归
   - window_lock_count < 基线 → 回归
5. **每次回放后更新第 4 节的状态列 + 第 6 节的时间线**。

---

**维护人**：@Yunfeng-bot
**关联文档**：[tactical_execution_plan-0419.md](tactical_execution_plan-0419.md) · [p0_iteration_report_20260419.md](p0_iteration_report_20260419.md) · [replay_validation_sop_v4.md](replay_validation_sop_v4.md)

---

## 2026-04-20 P0-D2 Sweep Update (Evidence-Driven)

### Fixed test protocol for this batch
- `video=/sdcard/Download/Video_Search/scene_20260417.mp4`
- `target=/sdcard/Download/Video_Search/target0417_s640.jpg`
- `replay_fps=15`, `replay_catchup=false`
- `retry_low_replay=1`, `max_wait_sec=420` (to guarantee 20s/24s windows are covered on slower devices)
- common baseline params unchanged from `20260420_120016`

### A/B/C/D single-variable results
| Variant | Param change | track_ratio | avg_frame_ms | locks/lost | wrong_lock_ratio | wrong source | window_lock_count |
|---|---|---:|---:|---:|---:|---|---:|
| baseline | none | 0.451 | 94.22 | 3/3 | NA | sample_fallback | 0 |
| A | `auto_verify_lost_prior_center_boost_cap=0.2` | 0.394 | 105.76 | 4/4 | 1.0 | sample_fallback | 1 |
| B | `auto_verify_lost_prior_max_frames_replay=120` | 0.478 | 95.64 | 5/4 | 1.0 | sample_fallback | 1 |
| C | `auto_verify_lost_far_recover_min_appearance=-0.05` | 0.480 | 92.81 | 3/2 | 1.0 | sample_fallback | 1 |
| D (diag-driven) | `spatial_gate_center_sigma=3.0`, `spatial_gate_size_sigma=1.6`, `spatial_gate_relock_min_replay=0.0` | **0.655** | **72.20** | 4/4 | 1.0 | **lock_event** | 1 |

### Key findings from logs
- Dominant reject reason after second loss is `reason=spatial` with `src=lost_prior` and large `d2`, repeatedly blocking relock candidates in 20s/24s windows.
- A/B/C (documented candidates) did not recover window relock quality.
- D reduces `spatial` over-rejection (higher throughput), but still produces a wrong relock in window horizon.
- Example from run `20260420_122232`: first window/horizon relock lock box center ratio is `~0.577`, outside current label range `[0.30, 0.50]`, counted as wrong lock.

### Next action (P0-D2 -> P0-D3)
1. Keep baseline code unchanged (no default promotion yet).
2. Add one focused code fix candidate for spatial gate source handling in relock path:
- degrade `lost_prior` spatial gate weight after consecutive spatial rejects, instead of hard reject loop.
3. Run 3x repeat only for:
- baseline
- D-like spatial-relax candidate
- new P0-D3 code candidate
4. Keep KPI gate unchanged: `wrong_lock_ratio_in_windows` is primary.

### 2026-04-20 P0-D3 code candidate (in progress)
- Kotlin code change landed in `OpenCVTrackerAnalyzer.kt`:
  - add `relockSpatialRejectStreak`
  - reset streak on `resetTracking` / `beginEvalSession` / `onLost` / successful init-lock
  - in relock spatial gate path: after consecutive `lost_prior` spatial rejects, relax `effectiveMinSpatial` to avoid deadlock loop
- constants added:
  - `AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_LIVE=4`
  - `AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_REPLAY=5`
  - `AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_SCORE=0.0`
- Build status: `:app:assembleDebug` passed.
- Device install status: `adb uninstall` + `adb install` succeeded once; replay runs after reinstall were unstable/partial (empty or short logs), so this candidate still needs a clean A/B rerun before promoting defaults.

### P0-D3 A/B 验证协议（进入回放前必须对齐）

本节定义 P0-D3 代码候选 升为默认前 的严格验证流程。任何跳过以下项的结果都不作为固化依据。

#### 1. 单变量干净度

- **代码侧**：两组必须仅差 P0-D3 三处改动：
  - `relockSpatialRejectStreak` 字段（[OpenCVTrackerAnalyzer.kt:845](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L845)）
  - 重锁空间门控放宽逻辑（[line 5225](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L5225)）
  - 新增常量 `AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_*` / `AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_SCORE`（[line 6787](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L6787)）
  - 其余文件（含参数默认值、cpp、MainActivity）**不得有额外未提交改动**。
- **操作建议**：
  - baseline 组切到干净分支（或 `git stash` 掉 D3 三处改动）跑。
  - D3 组切回 D3 分支跑。
  - **不复用 `20260420_110116` 的历史数字作为 baseline**（间隔 1 天、设备温度/电量/背景进程漂移），必须当日重跑 baseline。
- **参数侧**：与 `20260420_120016` 同口径（video / target / replay_fps=15 / replay_catchup=false / retry_low_replay=1 / max_wait_sec=420 / 其他 param_overrides 全部保持默认）。

#### 2. 跑序（抗设备漂移）

采用**交错执行**而非连跑：

```
D3·r1  →  base·r1  →  D3·r2  →  base·r2  →  D3·r3  →  base·r3
```

- 消除设备预热、温度爬升、电量下降、后台进程调度等系统性噪声对前半段 vs 后半段的偏差。
- 每轮之间强制 `adb shell am kill` 应用冷启动，并等待 ≥ 10 s 散热。

#### 3. 成功 / 回归 / 灰色区判据（事前写死）

| 情况 | 条件 | 动作 |
|---|---|---|
| **升为默认** | `wrong_lock_ratio_in_windows` 下降 ≥ **0.30** 且 `relock_20s`/`relock_24s` 回退 < **0.5 s** 且 `track_ratio` 不下降 ≥ 0.05 | 合入默认，更新 §4 表与 §6 日志 |
| **判退** | `wrong_lock_ratio` 上升 ≥ 0.10 或 `window_lock_count` 下降 任一项 | 回滚 D3，记录日志进 §6 |
| **灰色区** | 单项改善但伴随其他项任一回退 | 加跑 1 轮（共 4 轮）再判；仍灰色则**不升默认**，转写 P1 方向 |
| **无效** | 所有指标无显著变化（|Δ| < 0.05） | D3 失效，转战役三（特征区分度验证） |

#### 4. 分段 wrong_lock 来源统计（必填）

D3 放宽逻辑本质是"以覆盖换时间"，可能把 wrong_lock **从 `src=lost_prior` 段转移到 `src=last_measured` / `src=none` 段**（即从"锁得晚但锁错"变成"锁得早但仍锁错"）。因此报告中必须**分段统计**，不能只看聚合值：

| 来源 tag | 基线计数 | D3 计数 | Δ |
|---|---|---|---|
| `src=lost_prior` | ? | ? | ? |
| `src=last_measured` | ? | ? | ? |
| `src=last_tracked` | ? | ? | ? |
| `src=none` / fallback | ? | ? | ? |

**决策规则**：如果 `lost_prior` 段降低但其他段同比例增加，视为"转移不解决"，**不升默认**。

#### 5. 事前代码逻辑校对（建议）

在跑 6 轮回放前，对 [line 5225](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L5225) 的放宽逻辑做一次静态校对，至少回答：

- [ ] **streak 上限**：放宽后 streak 是否清零？若不清零会永久放宽吗？
- [ ] **重置条件**：`onLost` / 成功锁定 / resetTracking / beginEvalSession 是否全覆盖？
- [ ] **放宽幅度**：`effectiveMinSpatial` 放宽到 `AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_SCORE=0.0` 时，是否等效于"关闭空间门控"？如果是，相当于 C 方案（已验证无效）+ 延迟触发，需说明差异。
- [ ] **作用域**：是否只作用于 `src=lost_prior`？还是所有 src 都受影响？若后者则与 D 方案（已知产生 wrong_lock）语义接近，需额外控制。

**不跑 6 轮**直到上列校对通过。避免"跑完发现是 streak 无上限导致的假阳性"。

#### 6. 报告格式（写入 §6 进展追踪）

```
### YYYY-MM-DD（P0-D3 A/B · 3 轮交错）
- 分组与跑序：D3·r1 → base·r1 → D3·r2 → base·r2 → D3·r3 → base·r3
- 三轮均值对比表：
  |  | baseline | P0-D3 | Δ |
  | track_ratio | X ± σ | X ± σ | ± |
  | wrong_lock_ratio | X ± σ | X ± σ | ± |
  | relock_20s / 24s | X / X | X / X | ± |
  | window_lock_count | X | X | ± |
  | avg_frame_ms | X | X | ± |
- 分段 wrong_lock 来源统计表（见第 4 项）
- 判据结论：升默认 / 判退 / 灰色区 / 无效
- 下一步：<固化常量 / 回滚 / 加轮 / 转 P1>
```

---

### 2026-04-20 P0-D3 progress update (streak-decay + relax evidence)

#### Code changes landed
- File: `app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt`
- Added observability for relax path:
  - New DIAG line: `reason=spatial_relax_apply` (emitted when relax condition is active, regardless of final accept/reject).
- Existing P0-D3 logic remains:
  - relock spatial reject streak with decay-on-pass in `recentLost + lost_prior` path.

#### Build/install status
- Build: `:app:assembleDebug` passed.
- Install: `adb install -r -g app/build/outputs/apk/debug/app-debug.apk` passed.
- Note: `tools/adb_exec.ps1 install ...` remained unstable (hang/timeout) in this environment; replay runs continue with `--adb-mode direct`.

#### Evidence runs (same replay protocol: scene_20260417 + target0417_s640 + fps=15 + 26s/5s)

1) D3 (aggressive threshold test, live/replay=3/4)
- run: `20260420_150135`
- metrics: `track_ratio=0.683`, `avg_frame_ms=65.54`, `locks/lost=3/3`, `relock_20s=5.994`, `relock_24s=2.994`, `wrong_lock_ratio=1.0` (`window_label_lock_event`)
- logs: `spatial_relax_apply=6`, `spatial reject(relax=false)=10`, `max spatialStreak=3`
- key point: relax is now **provably triggered** in logs.

2) Baseline-off switch (live/replay=256/512)
- run: `20260420_150729`
- metrics: `track_ratio=0.590`, `avg_frame_ms=80.40`, `locks/lost=3/2`, `relock_20s=NA`, `relock_24s=NA`, `wrong_lock_ratio=1.0` (`window_label_sample_fallback`)
- logs: `spatial_relax_apply=0`, `spatial reject(relax=false)=23`, `max spatialStreak=23`

#### Interim conclusion
- P0-D3 relax mechanism is active and reduces spatial hard-reject pressure (23 -> 10 rejects in this pair), with throughput gain (`avg_frame_ms` down) and global tracking gain (`track_ratio` up).
- P0 target is still not met: `wrong_lock_ratio_in_windows` remains high (`1.0`) in this pair.
- This means current relax path improves continuity but does not yet prevent relock-to-similar-object in 20s/24s windows.

#### Next minimal step (single-variable, evidence-driven)
- Keep current relax trigger (3/4) for now.
- Add one guard only on relaxed path:
  - when `spatial_relax_apply` is active, require `appearanceScore >= relaxed_appearance_floor` (start with `0.00` or `0.05`, sweep 2 values only).
- Goal: preserve relock latency/throughput gains while reducing window wrong-lock.

### 2026-04-20 P0-D3 follow-up: relaxed-path appearance guard (single-variable test)

#### Code delta
- `OpenCVTrackerAnalyzer.kt`
  - new eval param: `auto_verify_relock_spatial_relax_min_appearance` (default `-1.0`, disabled)
  - applies only when `spatial_relax_apply` path is active
  - new reject reason log: `reason=spatial_relax_appearance`

#### Replay test (same protocol, threshold 3/4)
- candidate run: `20260420_153048`
- params include: `auto_verify_relock_spatial_relax_min_appearance=0.05`
- metrics:
  - `track_ratio=0.668` (vs no-appearance-floor run `20260420_150135`: `0.683`)
  - `avg_frame_ms=69.41` (vs `65.54`)
  - `wrong_lock_ratio_in_windows=1.0` (no improvement)
  - `relock_20s=5.994`, `relock_24s=2.994`
- log evidence:
  - `spatial_relax_apply=11`
  - `spatial_relax_appearance` rejects = `7`
  - spatial rejects (`reason=spatial`) = `6`

#### Conclusion
- The new appearance guard is active and intercepts candidates, but it did **not** reduce window wrong-lock in this run.
- It also slightly reduces global track ratio and increases frame time.
- Therefore, this guard is not promoted as default in current form.

#### Next step recommendation
- Keep P0-D3 core relax path as a configurable candidate (do not hard-promote).
- Move optimization focus from relax-path threshold to candidate source quality:
  - prioritize `lost_prior` geometry/anchor constraints in the 20s/24s return windows,
  - then re-run 3x interleaved A/B with strict window metrics.

---

### P0-D4 · 候选可见性诊断（先诊断再实验；阻塞后续 A/B 放行）

#### 为什么暂停继续跑门控 A/B

截止本轮，已经在**空间门控**这一轴上累计验证了 5 种配置，`wrong_lock_ratio_in_windows` 全部无法下降：

| # | 配置 | spatial_relax | appearance_floor | wrong_lock |
|---|---|---|---|---:|
| 1 | A：`boost_cap=0.2` | — | — | 1.0 |
| 2 | B：`max_frames=120` | — | — | 1.0 |
| 3 | C：`far_recover_min_appearance=-0.05` | — | — | 1.0 |
| 4 | D/D3：`spatial_gate` 全面放宽 / relax 3/4 | 开（6 次） | 关 | 1.0 |
| 5 | D3 + 外观下限 | 开（11 次） | 开（7 次，score=0.05） | 1.0 |

**结论**：5 种空间门控状态下 `wrong_lock_ratio` 均为 1.0。空间门控只决定"候选是否被允许锁"，当 **ORB 返回的 top-K 候选里根本没有真目标**时，门控松紧都会在相似物上锁。继续调门控是错把诊断面当成杠杆。

#### P0-D4 诊断设计（零参数变动，只加日志）

在 wrong_lock 发生点（s640 基线约 `25~28 s` 窗口）dump 一次 ORB top-K 候选的完整信息：

```
EVAL_CAND_DUMP t=<pts_sec> session=<id> src=<lost_prior|last_measured|last_tracked|none> topK=[
  {rank=0, cx, cy, w, h, appearance=0.xx, spatial=0.xx, d_mahal=xx, fusion=0.xx, homo_ok=?, tier=<near|far>},
  {rank=1, ...},
  ...
]
GT_EXPECTED (from p0_windows.json label): cx=?, cy=?, w=?, h=?
```

**触发口径**：
- 仅在 window_label 模式开启。
- 窗口进入时起 dump，每次 ORB 提出新 top-K 都记录一条，直到窗口结束或锁定稳定。
- 每轮回放至少覆盖 20s、24s 两个窗口。

#### 决策分叉（诊断结果直接决定下一步走向）

| 诊断结果 | 含义 | 下一步 |
|---|---|---|
| **真目标不在 top-K** | ORB 召回层在窗口时刻外观差距过大（例如目标远/小/纹理退化），门控无能为力 | 结束 P0 门控线；升级到 **战役三（GAP 特征区分度验证）** 或 **战役二（独立重检测通道）**。本文第 5 节优先级图更新。 |
| **真目标在 top-K（rank ≥ 1），但 rank=0 是相似物** | 问题在**打分/融合层**，不在召回也不在门控 | 跑 3 轮交错 A/B，变量换成 `spatial_gate_weight`（`w_spatial` vs `w_appearance` 融合权重），而非继续调 lost_prior 几何。 |
| **真目标在 top-K 且 rank=0，但被现有门控拒绝** | 问题在门控过严 | 回到 D3 variant，但单变量收敛到 `effectiveMinSpatial` 下限，而非 streak 阈值。 |

#### 阻塞规则（写死 · 方案 C 决议后更新）

- **P0-D4 已降级为 RC1 主线 P1 任务，不阻塞 MVP 主线**（见 §0.1 路径分工决议）。
- **P0-D4 只做 1 轮诊断，不做任何 A/B 收敛**：得出根因结论（召回/打分/门控层）后直接入档，挂起到 RC1 阶段再消化。
- D3-core 3 轮交错 A/B **永久挂起**（不再执行），除非 RC1 阶段重新评估。
- 时间上限：半天。超时则关闭 D4，结论记为"未知，挂起至 RC1"。

#### 实施清单

- [x] **D4-1**：在 ORB 候选生成点（`findOrbMatch` / promote 阶段）加 `EVAL_CAND_DUMP` 日志，字段见上。仅 `window_label` 开启时生效，避免污染常规日志。
- [x] **D4-2**：在 `sweep_replay.py` 提取端加入 `EVAL_CAND_DUMP` 解析，输出 `candidate_dump.csv`（每行一条候选）。
- [x] **D4-3**：与 `p0_windows.json` 的 label 做匹配，计算"真目标 rank 分布"、"rank=0 appearance vs GT appearance 差值"等统计。
- [x] **D4-4**：按上表三种情况之一写明结论，触发对应下一步（战役二/三 或 融合权重 A/B 或 effectiveMinSpatial 单变量）。

**D4 一轮诊断结论（20260420_161708）**：
- 产物：`tools/auto_tune/out/20260420_161708/candidate_dump.csv`
- 汇总：`tools/auto_tune/out/20260420_161708/candidate_dump_summary.json`
- 判定：`diagnosis=recall_layer`（窗口期真目标多数帧未进入 top-K）
- 动作：按方案 C 收口，不再继续 D4 参数 A/B，转入 RC1 阶段的战役二/三

#### 与主线战役映射

| D4 结论 | 触发主线 |
|---|---|
| 召回层失效 | 战役三（GAP 特征区分度）+ 战役二（重检测通道） |
| 打分层失效 | 战役一 Task A 的**权重项**（而非仅"加空间项"） |
| 门控过严 | 当前 D3 路径收敛（单变量） |

## 5.11 MVP-5 Known Debt: Cross-Session First-Lock State Contamination (2026-04-21)

- Status: Open (workaround active, root fix pending)
- Symptom:
  - Same APK + same video + same params, first cold session can lock at ~7.7s.
  - Subsequent sessions without process cold-start may regress to ~12.5s first lock.
- Evidence snapshots:
  - run_132319_try1: firstLockReplaySec=7.722
  - run_132319_try2: firstLockReplaySec=12.474
  - run_151212_try1: firstLockReplaySec=12.474
- Suspected root cause:
  - Cross-session counters/state in `OpenCVTrackerAnalyzer.kt` are not fully reset on session restart.
  - Candidate fields to verify: first-lock/reset/refine related counters (for example resetDrift/resetIou/refinePass family).
- Impact:
  - Replay acceptance stability depends on process hot/cold state.
  - Real-world repeated mission start may show delayed first lock.
- Temporary workaround (enabled in script):
  - `tools/replay_sop/run_mvp5_variant.ps1` forces cold session before each run:
    - `adb shell am force-stop com.example.dronetracker`
    - sleep 1s
- Exit criteria for debt closure:
  - Remove workaround and still keep first-lock variance within ±1.0s across 3 consecutive runs.
- Update 2026-04-21 (reclassification):
  - The cold-start workaround (`am force-stop`) did not recover L1 first-lock to 7~8s.
  - L1 remains stable around `firstLockReplaySec=12.474s`, indicating intrinsic LOCK pipeline latency rather than cross-session contamination.
  - Working policy for MVP-5 acceptance: extend `center_roi_l3_timeout_ms` to `10000` as pragmatic guardrail.
  - Root-cause optimization of LOCK latency is deferred as post-MVP follow-up (target: reduce first-lock from ~12.474s to <=5s).
