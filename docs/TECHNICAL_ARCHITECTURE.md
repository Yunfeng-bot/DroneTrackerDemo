# 无人机单目标追踪技术方案（2026-04-14）

## 1. 目标与范围
本方案定义当前 Android 端单目标追踪系统从 MVP 走向工业化的技术底座，覆盖：
- ORB SEARCH -> KCF TRACK 主链路的生产化约束
- JNI/C++ Native Bridge 与后续 NCNN/RKNN 迁移接口
- 离线回放评测闭环（EvaluationActivity + CSV + 自动分析）

不包含：飞控 PID 参数本身、云台控制器固件逻辑。

## 2. 当前生产链路（已落地）

### 2.1 搜索与追踪双状态机
- SEARCH：ORB 多尺度模板匹配 + RANSAC + 时序确认
- TRACK：KCF 高频跟踪
- 自愈：KCF 失锁/漂移后回退 SEARCH

核心原则：
- 宁可漏检，不可误锁
- 首锁必须经过时序稳定性确认
- tracker 对象失效后必须重建，不复用污染状态

### 2.2 ORB 预算熔断（长尾治理）
新增预算参数：
- `orb_feature_cap`（别名：`orb_max_feature_cap` / `orb_budget`）

策略：
- 仅 SEARCH 主分支强制执行 cap，限制每帧进入匹配层的特征规模上限
- TRACK_VERIFY/模板构建分支不强制 cap，避免误伤稳定跟踪

默认值：
- `DEFAULT_ORB_FEATURE_HARD_CAP = 700`
- 可在高纹理场景试验下压到 500（需回放验证后再启用）

### 2.3 Native 架构底座（已接入）
- Kotlin/Java: `nativebridge/NativeTrackerBridge.kt`
- JNI: `cpp/jni/NativeTrackerBridge.cpp`
- HAL:
  - `ITracker` 抽象接口
  - `NcnnTrackerImpl`（已接入 `ncnn::Net` 与模型加载，保留 no-runtime fallback）
  - `RknnTrackerImpl`（当前 stub）
  - `NanoTrackerEngine` 单例管理

目标：上层调用稳定、下层后端可热切换，不破坏业务接口。

### 2.4 离线评测基础设施（已接入）
- Activity：`EvaluationActivity`
- 输出：每帧 CSV
  - `frame_id, latency_ms, predicted_x, predicted_y, predicted_w, predicted_h, confidence_score`
- 分析脚本：`tools/auto_tune/analyze_eval_csv.py`

### 2.5 构建环境基线（JNI/NCNN 强制）
- Android CLI 构建统一使用 Android Studio JBR（Java 21）
- 禁止系统 JDK 25 直接参与 Gradle/Kotlin 脚本编译
- 标准入口：`tools/gradlew_jbr.ps1`
- 目的：避免 Kotlin/Gradle 与 JNI/NDK 混合编译时出现版本解析异常、构建挂起与非确定性崩溃

## 3. 关键参数与推荐基线（BASELINE_v1）
当前推荐基线（回放验证表现最优）：
- `search_max_long_edge=480`
- `search_short_edge=480`
- `orb_features=700`
- `orb_feature_cap=700`（默认）

说明：
- 420 档实测退化（召回与稳定性下降）
- cap=500 会明显伤首锁与跟踪占比，仅可作为场景化兜底策略

## 4. 已验证结论（回放）
代表性结果：
- 历史较优：`first_lock_sec=0.40`, `avg_latency=22.94ms`, `track_like_ratio=0.7869`
- 过激 cap（500）会导致首锁变慢和跟踪占比下降

结论：
- 当前阶段优先稳态：采用 480 + 700 cap 基线
- 继续通过离线回放做小步参数搜索

补充（2026-04-13）：
- 当前 NCNN 路径已打通 JNI/Zero-Copy/状态机，但追踪核仍为占位实现
- 因此与成熟 KCF 直接做性能对标时，出现 `track_ratio` 与时延不占优属于预期
- 本阶段回放汇总：`tools/auto_tune/out/p0_10run_summary_20260413.csv`
  - `first_lock_after_target_sec` median `2.267s`
  - `track_ratio` median `0.345`
  - `avgFrameMs` median `81.1ms`

## 5. 下一阶段架构演进

### P0（当前迭代）
- 保持 ORB/KCF 主链路
- 继续压缩 P95 尾延迟（预算熔断 + 限频）
- 固化自动评测评分口径

### P1（下一阶段）
- 将 NCNN 后端从 stub 变为真实 Siam 推理路径（必须加载模型权重）
- 在同一评测基线上 A/B 对比 OpenCV 与 NCNN

### P2（量产预研）
- RKNN 后端接入
- 保持同一 HAL 接口，不改上层业务状态机

## 6. 深度孪生神经网络（Siamese）大方案

### 6.1 目标
在保持当前 ORB/KCF 方案可用的前提下，引入深度孪生追踪器，解决以下痛点：
- 远距离/小目标召回不足
- 背景高纹理干扰下误锁与漂移
- 参数调优成本高、场景迁移弱

### 6.2 推荐模型路线
优先级顺序：
1. `NanoTrack / LightTrack`（移动端友好）
2. `SiamRPN++`（精度更高，算力需求更高）
3. `MixFormer/SparseTrack`（仅在 NPU 充足时评估）

### 6.3 标准推理形态（Template/Search 双分支）
- Template 分支（单次或低频）：
  - 以初始化框裁剪 `127x127`（可配置）
  - 提取 `Z feature` 并缓存
- Search 分支（逐帧高频）：
  - 以上一帧目标中心裁剪 `255x255`（可配置）
  - 提取 `X feature`
  - `Z/X` 相关计算输出分类分数图 + 回归框

实现约束（新增）：
- 禁止继续用占位相关性核替代真模型做“性能优劣结论”
- `NcnnTrackerImpl` 必须显式持有 `ncnn::Net`，并加载固定版本 `.param/.bin`
- 每次模型切换都必须登记模型 hash 与导出脚本版本

### 6.4 工程接入约束
- 必须继续使用现有 `ITracker` HAL 接口
- 允许在运行时切换后端：`OpenCV` / `NCNN` / `RKNN`
- 新后端必须复用当前评测链路（EvaluationActivity + CSV）

### 6.5 模型转换与部署规范
- ONNX -> NCNN：固定导出脚本、固定输入 shape、固定后处理
- ONNX -> RKNN：固定量化集（至少 200~500 帧代表场景）
- 每个模型版本必须记录：
  - 模型 hash
  - 量化配置
  - 导出脚本版本
  - 评测结果摘要

### 6.6 升级门槛（必须满足）
深度模型替代 ORB/KCF 前，必须同时满足：
- 在同一回放集上 `first_lock_sec` 不劣化
- `track_like_ratio` 提升或持平
- `p95_latency_ms` 不高于现网基线 + 10ms
- 连续 30 分钟运行无崩溃/无内存持续增长

推荐增强（优先级顺序）：
1. 真模型接入后在响应图后处理中加入 Hanning/Cosine Window
2. 默认开启 FP16（`use_fp16_arithmetic` / `use_fp16_storage`）
3. 在模型稳定后再评估 INT8 量化（避免过早量化干扰归因）

### 6.7 回退策略
- 发布阶段保留 `OpenCV` 兜底开关
- 深度后端触发异常时自动切回 ORB/KCF
- 严禁“单后端不可回退”上线

