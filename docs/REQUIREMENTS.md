# 无人机伴飞单目标追踪系统项目规范（Project Spec�?
> 最后更新：2026-04-20 · 新增 MVP 首发场景（楼顶精准降落）

## 1. 核心定调：从像素博弈走向物理认知
本系统的核心追求不再是”通过无限逼近的参数调优来适配视频”�?**认知铁律（The Paradigm Axiom�?*：追踪的第一性原理，是在时间的切片中，依�?*表观流形特征（长什么样�?*�?*空间动力学（它能在哪�?*维持目标身份的连续性。决不能用纯视觉的防抖参数来掩盖 6 自由度飞行器带来的空间物理运动�?
## 1.5 MVP 首发场景：GPS 引导 + 视觉精降（楼顶送快递）

**产品定位**：取消任意场景的通用伴飞目标，先聚焦一�?*闭环可交�?*的落地场景—�?无人机把快递送到指定楼顶"。此场景�?**GPS 粗定�?+ 视觉精定�?* 串联，是整个追踪栈最轻量的一条路径，也是团队验证 ORB + NanoTrack 闭环的最小可用切片�?
### 1.5.1 场景描述

- **输入**�?  1. 目标楼宇�?*俯视楼顶照片**（用户在 App 内导入，无需 ArUco / 二维�?/ 人工地标）�?  2. 楼顶中心点的 GPS 坐标（经纬度）�?- **输出**：降落点相对当前画面中心�?`offsetX / offsetY`（归一化到 [-1, 1]）→ 供飞控做横向修正 + 下降指令�?- **不含**：多目标、远距离伴飞、动态目标追逃、遮挡恢复（这些留给 RC1/RC2）�?
### 1.5.2 执行流（分级 ROI 搜索闭环�?
**设计原则**：不能假�?GPS 就位后目标必然在画面中心（GPS 精度、风偏、云台初始指向误差都会让目标偏离中心）。采�?*由小到大的三�?ROI 搜索**，优先在小区域快速匹配，失败后再逐级扩大，兼顾速度与兜底�?
> **风险警示**：GPS 精度 ±5 m + 云台初始指向误差 ±5° + 悬停时风偏可叠加出目标脱�?L1（�?0%）甚�?L2（�?0%）的情形，因�?L3 全图兜底不可省。实机调优时应记�?`ROI_SEARCH` 的实际命中层级分布，�?L2/L3 命中率长�?> 30% 需重新校验 GPS 精度与云台初始指向�?
| 步骤 | 触发 | 动作 | 成功 �?| 失败 �?|
|---|---|---|---|---|
| 1 | 任务开�?| GPS 导航到目标楼宇上空，悬停 | GPS 就位信号 | �?|
| 2 | GPS 就位信号 | 进入 `ROI_SEARCH_L1` 状�?| state = L1 | �?|
| 3 | **L1（中�?±20%�?* | 在画面中�?±20% ROI（占全图 ~16%）跑 ORB 模板匹配 | 命中 �?step 6 | 超时 T1 �?step 4 |
| 4 | **L2（中�?±40%�?* | 扩大到画面中�?±40% ROI（占全图 ~64%）继续搜 | 命中 �?step 6 | 超时 T2 �?step 5 |
| 5 | **L3（全图兜底）** | 全图 ORB 搜索 | 命中 �?step 6 | 超时 T3 �?失败兜底 |
| 6 | ORB 命中 | NanoTrack 用该 bbox + 当前视频�?patch 初始化，进入 `TRACKING` | 每帧 bbox | �?|
| 7 | 每帧更新 | 计算 bbox 中心与画面中心的偏移，归一化输�?`offsetX/Y`；结合高度信号下发横向修�?+ 下降 | offsetX/Y + descend flag | 触发 onLost()（lockHoldFramesRemaining 耗尽后连续失败达阈值）�?回到 step 3 |

**超时与预算参�?*（仅 MVP 初值，后续以回放实测调整）�?- `T1 = 0.5 s`（L1 搜索窗口），`T2 = 1.0 s`（L2），`T3 = 2.0 s`（L3 全图兜底�?- 总时间预�?�?3.5 s；超出则上抛"视觉引导失败"�?
**为什么分级而不是一上来就全�?*�?- ORB 耗时与搜索面积近似线性；L1 ROI 只有全图�?~1/6，单次搜索耗时�?~80 ms 降到 ~13 ms�?- 画面中心区域相似物体数量远少于全图，L1 命中�?`wrong_lock` 概率显著低于全图�?- L3 只作为最后兜底，避免 GPS 漂移或风偏导致目标偏�?±40% 时系统直接失败�?
### 1.5.3 为什么这条路径是 MVP

- **依赖收敛**：只用现役模块（ORB + NanoTrack + Kalman），**不需�?*重检测通道、ReID、检测器升级（战役一/�?四的成果是锦上添花，不是 MVP 前置）�?- **分级 ROI 天然�?wrong_lock**：GPS 把候选范围先压到画面中心 ±20%（再逐级扩大），从物理上排除"锁到画面另一端相似楼�?的失败模式，等效�?**最小版本的马氏空间门控**（战役一 Task A �?MVP 下退化为硬裁�?+ 分级扩张）�?- **模板生成路径清晰**：用户导入的俯视�?�?�?[template_optimization_plan.md](template_optimization_plan.md) �?P0 规范（EXIF 归一�?+ 质量自检 + 方向提示），最小化模板质量风险�?- **评估指标直观**：`offsetX/Y` 的稳态误�?+ 降落点偏移实测米数，不依�?LaSOT AUC 等抽象指标�?
### 1.5.4 MVP 验收标准

- **首锁**�?  - 理想场景（目标在 L1 ±20% 内）�?*�?0.5 s**
  - 一般场景（目标�?L2 ±40% 内）�?*�?1.5 s**
  - 最坏场景（命中 L3 全图）：**�?3.5 s**，仍优于 s640 基线�?5.2 s�?- **跟踪稳定**：降落前 20 s �?`window_lock_count �?3`、`wrong_lock_ratio_in_windows = 0`（楼顶静止无干扰物，应接近完美）�?- **横向误差**：最终降落点与楼顶中心地面投影偏�?**�?1.0 m**（给飞控 + GPS 漂移一定容忍，后续迭代再收紧）�?- **失败兜底**：L3 全图搜索超时 T3=2.0 s 后（即总预�?3.5 s 耗尽，见 §1.5.2 表格 step 5）→ 上抛 "视觉引导失败" 信号，飞控回退到纯 GPS 降落或盘旋重试�?
### 1.5.5 与主线战役的映射关系

| MVP 需�?| 映射到主�?| MVP 下简化为 |
|---|---|---|
| 分级 ROI 搜索（L1/L2/L3�?| 战役一 Task A（马氏空间门控） | 硬裁�?+ 逐级扩张（GPS 先验�?|
| 模板质量 | template_optimization_plan P0 | 复用 |
| 单楼顶静止目�?| �?| 不涉及战役二、三 |
| NanoTrack 初始�?| 现役 | 不改 |
| ROI 超时兜底 | 战役二（重检测通道�?| 降级�?L3 全图 ORB + 失败上抛 |

### 1.5.6 非目标（明确不做�?
- 不识别楼顶数�?logo（无 OCR、无二维码、无 ArUco）�?- 不处理楼顶被�?阴影遮挡的情况（遮挡场景进入 RC1）�?- 不支持动态目标（运动楼顶/车辆楼顶不在 MVP 范围）�?- 不做多楼顶判别（GPS 已选定唯一目标，ROI 内只有一个候选区域）�?
### 1.5.7 数据契约（Ingress / Egress · v1�?
> **版本**：v1（MVP 首发）�?适用分支：main · 契约冻结前任何字段变动必须同步本�?> **范围**：仅定义 App �?外部调用方（飞控 / 调试工具 / 回放脚本）之间的接口�?*�?*耦合任何具体飞控 SDK 实现
> **设计原则**：调试可脱机（adb / UI 按钮可触发）· 回放可解析（`sweep_replay.py` 可提取）· 实机可扩展（预留 UDP / JNI 通道，但不在 v1 实现�?
#### 1.5.7.1 Ingress：GPS 就位信号（MVP-1�?
**接口定义**（Kotlin 层，已落地）�?
```kotlin
OpenCVTrackerAnalyzer.setCenterRoiGpsReady(ready: Boolean, reason: String = "external")
```

位置：[OpenCVTrackerAnalyzer.kt:1593](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L1593)

**触发通道（v1 只实�?a+b，c 预留�?*�?
| 通道 | 用�?| 调用方式 | 何时使用 |
|---|---|---|---|
| **(a) Debug UI 按钮** | 人工联调 / 设备上手动测�?| MainActivity 增加 "GPS Ready" toggle 按钮，点击时调用 `analyzer.setCenterRoiGpsReady(true, "debug_ui")` | 开发期手动验证 |
| **(b) adb broadcast 意图** | 脚本化回�?/ 自动化测�?| `adb shell am broadcast -a com.example.dronetracker.GPS_READY --ez ready true --es reason "replay_script"` | `sweep_replay.py` 注入、CI |
| **(c) 飞控 SDK 回调（预留）** | 真机集成 | 未来�?MAVLink listener / 厂商 SDK 回调调用 | RC1 阶段落地，MVP 不做 |

**字段语义**�?- `ready: Boolean` �?`true`=已到达目标上空悬停；`false`=离开目标区域，立即回�?`CENTER_ROI_SEARCH=disabled` 状�?- `reason: String` �?仅用于日志归因（`debug_ui` / `adb_broadcast` / `fc_sdk` / `replay_script` 等），不影响行为

**状态切换语�?*�?- 调用 `setCenterRoiGpsReady(true, ...)` 后，分级 ROI 状态机在下一帧从 L1 起算（参�?§1.5.2 step 3~5�?- 调用 `setCenterRoiGpsReady(false, ...)` 后，状态机复位到初始态，`CENTER_ROI_SEARCH_ENABLED=false`

**可观测�?*：每次切换必�?emit 一�?`EVAL_EVENT type=ROI_SEARCH state=gps_ready ready=<bool> reason=<str> enabled=<bool>`（已落地，见 [line 1597-1600](app/src/main/java/com/example/dronetracker/OpenCVTrackerAnalyzer.kt#L1597)）�?
#### 1.5.7.2 Egress：降落偏移量（MVP-3�?
**v1 主通道：结构化日志**（零外部依赖，`sweep_replay.py` 可直接解析）�?
**日志行格�?*�?
```
EVAL_EVENT type=DESCEND_OFFSET x=<float> y=<float> conf=<float> state=<enum> t=<float> session=<id>
```

**字段规范**�?
| 字段 | 类型 | 范围 | 定义 |
|---|---|---|---|
| `x` | float | `[-1.0, 1.0]` | bbox 中心相对画面中心的归一�?X 偏移；正值表示目标在画面**�?*，需向右修正 |
| `y` | float | `[-1.0, 1.0]` | bbox 中心相对画面中心的归一�?Y 偏移；正值表示目标在画面**�?*，需向下修正 |
| `conf` | float | `[0.0, 1.0]` | 当前帧的 `fusedConfidence`（NCNN 融合分数，参�?[NcnnTrackerImpl.cpp:722](app/src/main/cpp/tracker/NcnnTrackerImpl.cpp#L722)�?|
| `state` | enum | 见下 | 当前 MVP 状态机阶段 |
| `t` | float | `�?0.0` | 视频 PTS 秒数（回放）�?System uptime 秒数（实机） |
| `session` | string | �?| `diagSessionId`，便于跨行关联同一�?|

**`state` 枚举**�?
| �?| 含义 | `x/y/conf` 是否有效 |
|---|---|---|
| `L1` | �?±20% ROI 内搜索（未锁定） | x/y/conf �?`NaN`；仅表示"正在搜索" |
| `L2` | �?±40% ROI 内搜索（未锁定） | 同上 |
| `L3` | 全图搜索（未锁定�?| 同上 |
| `TRACKING` | 已锁定目标，NanoTrack 持续输出 | **x/y/conf 有效**，飞控据此执行横向修�?+ 下降 |
| `LOST` | 跟踪丢失，等待重锁（�?`lockHoldFramesRemaining` 窗口内） | `x/y/conf` 为上一帧缓存值（允许短暂继续下降�?|
| `FAIL` | 总预�?3.5 s 耗尽仍未锁定 | x/y/conf �?`NaN`；飞控必须回退到纯 GPS 或盘�?|

**发射节奏**�?- `TRACKING` 状态：每帧必发（与视频帧率一致，15 fps 时约 67 ms/行）
- `L1/L2/L3` 状态：�?`SEARCH_DIAG_INTERVAL_FRAMES`（当�?15 帧）发一次（防止日志淹没�?- `LOST/FAIL`：状态切换瞬�?+ �?1 s 心跳

**坐标系定�?*（一次性对齐，避免实现歧义）：
- **画面原点**：左上角 `(0, 0)`；X 向右递增，Y 向下递增（OpenCV / Android Surface 标准�?- **归一化基�?*：`x_norm = (bbox_center_x - frame_w/2) / (frame_w/2)`；`y_norm` 同理
- **飞控语义**（建议）：`x > 0` �?无人机需�?*�?*平移；`y > 0` �?需�?*�?*平移（接近地面）
- 若飞控坐标系与此不同（例�?NED、机体系），**由飞控侧做一次线性变�?*，不�?App 层做适配

**v2 预留通道（不�?MVP 落地�?*�?- **UDP 广播**：`127.0.0.1:<port>`，同一 JSON schema 序列化，供同机飞控进程订�?- **JNI 回调**：`nativeDescendOffset(x, y, conf, state, t)`，供 C++ HAL 层直接拉�?- **决定�?*：v2 通道只在"实机联调出现日志解析延迟 > 100 ms"时启用；MVP 阶段不预先实现�?
#### 1.5.7.3 错误情形（必须由 App 主动上报，不得静默吞�?
| 情形 | Egress 行为 | state | x/y/conf | 飞控预期响应 |
|---|---|---|---|---|
| GPS 未就位（`ready=false`�?| �?emit `DESCEND_OFFSET` | �?| �?| 飞控继续 GPS 导航，不期待视觉输出 |
| L1/L2/L3 搜索�?| emit `state=L1/L2/L3`，x/y/conf=NaN | L1/L2/L3 | NaN | 飞控**悬停**，不下降 |
| 首锁成功 | emit `state=TRACKING` | TRACKING | 有效�?| 飞控开始横向修�?+ 下降 |
| 跟踪中短暂丢�?| emit `state=LOST`，x/y/conf=上一帧缓�?| LOST | 缓存�?| 飞控**减�?*但可继续下降（≤ 1 s 缓冲�?|
| 总超�?3.5 s 仍未锁定 | emit `state=FAIL` 一�?| FAIL | NaN | 飞控**上抛告警**，回退�?GPS 降落或盘旋重�?|
| ORB 候选异常（`bbox` 越界 / 尺寸非法�?| �?emit，降级为 LOST | LOST | 缓存�?| �?LOST |

**禁止**：在任何错误情形下静默跳�?emit，或发送越界值（�?`x=10.5`）——飞控侧不做边界检查，错误数据会直接转化为错误姿态�?
#### 1.5.7.3a MANUAL_ROI �¼���Լ��Phase 1 ���䣩

�ֶ�Ȧѡ��·�������¹۲��ֶΣ�д�� `EVAL_EVENT type=MANUAL_ROI` / `MANUAL_ROI_INIT_OK` / `MANUAL_ROI_INIT_FAIL`��

- `state=active|clear`
- `reason=<string>`
- `patch_kp=<int>`
- `patch_texture=<float>`
- `bbox_clamped=<true|false>`
- `init_path=live|fallback_disk`

Լ����

- `init_path=fallback_disk` �� Phase 1 ��Ϊʧ��̬��������� `reason=fallback_forbidden`�����þ�Ĭ���˲�����������
- `bbox_clamped=true` ��ʾȦѡԽ�类�ü������ɹ۲⡣
#### 1.5.7.4 验证协议（契约端到端�?
**回放测试脚本（伪流程�?*�?
```
1. sweep_replay.py 启动回放 scene_20260417.mp4
2. 视频播放�?T_SIMULATE_GPS_READY（例�?2.0 s）时，broadcast:
   adb shell am broadcast -a com.example.dronetracker.GPS_READY --ez ready true --es reason "replay_sim"
3. 等待 6 s 完成全部 L1/L2/L3 �?TRACKING 流程
4. �?logcat 提取所�?EVAL_EVENT type=DESCEND_OFFSET �?5. 校验项：
   - 首行是否�?state=L1 / L2 / L3 之一
   - 最终应�?state=TRACKING �?state=FAIL
   - 无任�?x/y 越出 [-1, 1]
   - state=TRACKING �?conf �?0.3（可配置阈值）
```

**`sweep_replay.py` 解析器扩�?*�?- 新增�?`descend_offset_first_t` / `descend_offset_last_state` / `descend_offset_oob_count`（越界计数）
- 新增判据：`descend_offset_oob_count > 0` �?直接判退（契约违反）

#### 1.5.7.5 契约治理

- 契约字段变动（新�?删除/语义改变）必须：
  1. 更新本节（�?.5.7�?  2. 同步更新 [tactical_execution_plan-0419.md](tactical_execution_plan-0419.md)（若已引用）
  3. `sweep_replay.py` 解析器同步更�?  4. �?`docs/` 新增一�?契约变更日志"（可在本节末追加子条目）
- 不允许在代码里静默新�?`EVAL_EVENT type=DESCEND_OFFSET` 字段而不更新本节�?
## 1.6 外场可用性目标拆�?(Phase 1~3 路线�?

�?MVP-5 (基于先验图片的楼顶降�? 完成后，为应对真实外场环境，系统衍生出以下阶段性可用性目标：

### 1.6.1 阶段一：消灭域偏差（Phase 1 屏幕即时圈选）
- **状态（2026-04-26）**：T4.2 已闭环。手动圈选 + NCC 模板指纹两层守门 + TRACKING 时间稳定化均通过 V3 真机验收。详见 [phase1_t42_closure_20260426.md](phase1_t42_closure_20260426.md) 与 [phase1_manual_roi_selection_plan.md](phase1_manual_roi_selection_plan.md) §5~6。下一步进 Phase 2 VLM 自动圈选（§1.6.2）。
- **核心问题**：预先准备的模板图由于光照、季节、设备内参差异，易导致首锁失败�?- **可验证子目标**�?  - `交互响应率`：在屏幕圈选后，`TRACKING` 启动延迟需 �?500 ms�?  - `时间稳定性`�?0s 内的 `steady_track_window` (TRACKING 状态占�? �?95%�?  - `极短距离无崩锁`：圈选后，目标占据屏幕超�?40% 时防爆护栏正常拦截，不发生致命丢失�?
### 1.6.2 阶段二：释放操作员（Phase 2 VLM 语义框选）
- **核心问题**：人工圈选在强光室外操作困难，且偏离全自动无人降落的终极商业愿景�?- **可验证子目标**�?  - `语义锚定准确率`：给定包�?3 个相似楼顶的场景，VLM 模型根据提示词返�?BBox 并命中目标中心（误差 �?10 像素）�?  - `端到端首锁耗时`：从发起视觉请求到云/端模型返回并完成 `TRACKING` 初始化，总延迟控制在合理区间（如 �?3.0 s）�?
### 1.6.3 阶段三：突破距离极值（Phase 3 小目标特征补强）
- **核心问题**：当目标在高空呈现的像素过少时，无论圈选多准，纯视觉跟踪依然会因特征匮乏而崩锁�?- **可验证子目标**�?  - `极限分辨率追溯`：系统能�?20x20 pixel 的极小目标下维持 `TRACKING` 状态不跳变�?  - `抗缩放突变`：无人机急速拉高时，NanoTrack 能通过尺度自适应/时序特征记忆维持稳定，不断锁�?
## 2. 现役底盘准则（RC1 稳态基线）
- Android CameraX + JNI/C++ HAL 底座�?- C++ 层追踪引擎：NanoTrack（SiamRPN 家族轻量版），通过 NCNN 推理，双网络流水线（Backbone + RPN Head）�?- Kotlin 层决策引擎：`OpenCVTrackerAnalyzer.kt`（当�?**6107 �?*），包含 ORB 特征验证、Kalman 预测、多层级置信度门控�?- **废弃通牒**：全面停止对 Kotlin �?`if-else` 防抖逻辑（如 `TrackGuard` 的面�?长宽比生硬检测）继续增加补丁。当前已膨胀�?**40+ 可调参数**（`FirstLockConfig` 14项、`FirstLockAdaptiveConfig` 20+项、`TemporalGateConfig`、`TrackGuardConfig` 等），参数间高度耦合，形成复杂性陷阱�?
## 3. 当前三大瓶颈（代码级诊断�?
基于 2026-04-19 代码评审，确认以下瓶颈均有代码层面的根因�?
### 瓶颈一：首次锁定慢 / 锁不�?- **根因**：模板照�?�?实景存在域差距（Domain Gap）；NanoTrack 的互相关机制是模板匹配器而非身份分类器，对视�?光照变化敏感�?- **代码证据**：`NcnnTrackerImpl.cpp:752` �?`cosineSimilarity(templateFeature_, feature)` 只衡量”像不像模板”，不具备跨域泛化能力�?
### 瓶颈二：锁到旁边相似物体
- **根因**：Score 融合（`NcnnTrackerImpl.cpp:722`：`fusedConfidence = calibrated * 0.90 + pScore * 0.10`）是**纯外观分数，没有空间约束**。画面另一端的相似物体与紧邻预测位置的真正目标获得几乎相同的分数�?- **代码证据**：整�?`app/src/main` 目录中没有任何马氏距离实现，尽管方案文档早已提出�?
### 瓶颈三：移出后再次进入锁不回
- **根因**：缺乏独立的重检测（Re-detection）通道。`onLost()` 触发后回退�?`ACQUIRE` 阶段依赖 ORB 全图搜索，�?ORB 在远距离/小目标场景下特征点严重不足�?- **加剧因素**：Hanning 窗（`NcnnTrackerImpl.cpp:664`，`kWindowInfluence = 0.462`）近一半得分来自空间位置先验，边缘�?Hanning 值趋近于 0，直接扼杀从画面边缘重入的可能性�?
## 4. SOTA 演进的四大战役（执行通牒与落地序位）
基于前沿技术调研（SAMURAI、DaSiamRPN、MixFormerV2-S、CVPR 2025 Anti-UAV Workshop）与严苛的排雷审查，团队**必须严格遵守以下优先�?*�?
### 战役一（P0�?-2天）：马氏距离空间门�?+ 模板更新门控
- **Task A �?空间门控**：在 Kotlin 层候选框匹配时，利用已有�?`BoxKalmanPredictor` 预测位置，计算候选框与预测位置的马氏距离，作为空间惩罚项融入最终得分：`FinalScore = w_app * Sim + w_spatial * exp(-0.5 * d_mahal²)`。物理上不可能在一帧内跳到画面另一端的候选框，其空间项直接归零。只�?Kotlin，不�?C++�?- **Task B �?模板更新门控**：修�?`NcnnTrackerImpl.cpp:903` �?Embedding 模式下无条件更新模板的缺陷。当�?`beta �?0.20` 的在线更新没有任何门控，连续误锁到干扰物后模板会”中毒”偏向干扰物形成正反馈。增加三重门控：(1) 马氏距离 < 阈值；(2) cosine similarity > 高阈值；(3) 目标不在画面边缘�?- **验收**：`sweep_replay.py` 回放中”锁到旁边相似物体”的比率下降 �?80%�?
### 战役二（P1�?-3天）：重检测通道（Re-detection Pipeline�?- **Task**：当 `onLost()` 触发后，不再仅依�?ORB 全图搜索。新增一条检�?ReID 二级流水线：(1) 用现�?NCNN backbone 对降采样全图做滑窗推理，产生 top-K 候选区域；(2) 对每个候选区域提�?GAP embedding，与**冻结的初始模�?embedding**（非在线更新版本）做 cosine similarity�?3) 选最高且过阈值的候选重新初始化跟踪�?- **关键设计**：重检测使用冻结初始模板，避免被在线更新中毒的当前模板污染。参�?SAMURAI 的三重记忆门控机制（mask similarity + appearance + motion score）�?- **验收**：`sweep_replay.py` 中”移出重入恢复率（Re-entry）”提升至 �?70%�?
### 战役三（P1，脱机验证）：GAP 特征区分度验证，提防”度量塌缩�?- **Task**：在 C++ 拦截 Siam 网络�?Backbone 输出�?GAP 向量化，通过 Python（`tools/backbone_probe/`）进行脱机混淆矩阵验证�?- **排雷与红�?*：SiamRPN 基于局部互相关训练，特征图极易发生”度量空间塌缩”。验收必须使用困难负样本集（Hard Negatives）。若 `Sim(真目�? = 0.95` �?`Sim(假白�? = 0.93`，直接判定废�?GAP 路线�?- **兜底方案**：如果区分度失败，立即转接轻�?ReID 模型（裁剪版 MobileNetV3�?MB 级别）。参�?DaSiamRPN �?distractor-aware 训练思路：不是在推理时区分，而是在特征空间就把相似干扰物推远�?
### 战役四（P2�?-7天）：Backbone 升级评估
- **短期**：尝�?DaSiamRPN 预训练权重替换当�?NanoTrack backbone，无需改推理代码结构�?- **中期**：评�?MixFormerV2-S �?NCNN 移植可行性。MixFormerV2-S �?CPU 上实时运行，LaSOT AUC 66.1%（CompressTracker-4 保留 96% 性能�?.17× 加速），精度远�?SiamRPN 家族�?- **长期**：如需 ReID 能力，挂�?2MB 级别 MobileNetV3-ReID head�?
## 5. 回放与验证底�?(SOP)
- 继续依赖 `sweep_replay.py`�?- 评估指标体系新增�?  - **首锁时间（First Lock Time�?*：从模板设置到稳定锁定的耗时
  - **相似物干扰漂移抵抗率（Anti-Drift�?*：存在相似干扰物场景下的跟踪准确�?  - **移出重入恢复率（Re-entry Rate�?*：目标移出画面后重新出现时的成功重锁比率
  - **模板中毒检�?*：连续跟踪中 cosine similarity 与初始模板的偏移量监�?- **Kotlin 层参数简化目�?*：将 40+ 参数逐步收敛�?2-3 个语义明确的决策函数，利�?`sweep_replay.py` 评测数据做参数敏感性分析，移除不敏感参数�?
## 6. 降级雷区：IMU 前融合（Ego-Motion�?面对 Android 传感器（ASensorManager）灾难级的时间戳碎片化现象，严禁在未彻底解决（CameraX PTS �?Gyroscope 物理钟对齐、差值插值、以及防抖乱序降频回退安全域）的前提下搞盲目坐标相减。当前全压光流补偿做平替兜底，维持保守策略�?
## 7. 前沿技术参考索�?| 技�?| 核心贡献 | 与本项目关联 | 来源 |
|------|---------|-------------|------|
| SAMURAI (2024.11) | SAM2 + Kalman 运动感知记忆选择，零样本跟踪 | 三重门控 memory bank 实现范式 | arXiv:2411.11922 |
| DaSiamRPN (ECCV 2018) | 训练阶段引入难负样本提升判别�?| 解决”锁到相似物体”的根源方法 | github.com/foolwood/DaSiamRPN |
| MixFormerV2-S (2023) | CPU 实时 Transformer 跟踪，LaSOT 66.1% AUC | Backbone 升级候�?| NeurIPS 2023 |
| Detector-Augmented SAMURAI (2026.01) | 检测器辅助长时无人机跟�?| 重检测通道设计参�?| arXiv:2601.04798 |
| CVPR 2025 Anti-UAV Workshop | 无人机反跟踪前沿对抗技�?| 鲁棒性测试场景设�?| openaccess.thecvf.com |


