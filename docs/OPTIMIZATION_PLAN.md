# 追踪优化计划（BASELINE_v1）

## 1. 版本基线
- Baseline 名称：`BASELINE_v1_480`
- 固定参数：
  - `search_max_long_edge=480`
  - `search_short_edge=480`
  - `orb_features=700`
  - `orb_feature_cap=700`

## 2. 迭代目标

### 2.1 性能目标
- `avg_latency_ms < 35`
- `p95_latency_ms < 90`（阶段目标）
- 长期目标：`p95 < 70`

### 2.2 识别目标
- `first_lock_sec <= 1.0`（目标出现后）
- `track_like_ratio >= 0.70`

## 3. 执行策略（小步快跑）

### Step A：单轴扫参（优先）
只允许一次改一个轴，避免归因混乱：
1. `orb_ratio`（例如 0.65/0.68/0.72/0.75）
2. `fallback_refine_ratio`（例如 0.78/0.82/0.86）
3. `orb_feature_cap`（700 -> 650 -> 600，500 仅对照）

### Step B：尾延迟治理
- 保持 cap 机制
- 引入 SEARCH 限频（预算超限帧隔帧搜索）
- 严禁直接放宽到失控特征数量

### Step C：稳态回归
每次参数变更后必须执行：
- 同视频回放验证
- 同脚本分析
- 与上一个基线做 A/B 数据对照

## 4. 通过门槛（Gate）
任何参数集要升级为新基线，必须同时满足：
- 首锁不劣化（first_lock_sec 不高于基线 + 0.2s）
- track_like_ratio 不低于基线 - 0.05
- p95 有改善，或在同级别前提下 avg 明显改善

## 5. 产物要求
每轮迭代固定产物：
1. 一份 CSV（原始）
2. 一份分析输出（脚本结果）
3. 一段结论（采用/拒绝 + 原因）

## 6. 角色无关交接规则（Claude/Codex 通用）
- 任何 AI 工具接手前，先读取：
  - `docs/TECHNICAL_ARCHITECTURE.md`
  - `docs/REQUIREMENTS.md`
  - 本文档
- 禁止跳过基线与 Gate，禁止凭肉眼直接改默认参数上线。

## 7. 深度孪生迁移计划（P1/P2）

### 7.1 里程碑
- M1：NCNN 真模型后端跑通（`ncnn::Net` 加载 `.param/.bin`，非 stub）
- M2：与 ORB/KCF A/B 对比通过 Gate
- M3：RKNN 量化模型上板跑通
- M4：默认后端切换为深度孪生，ORB/KCF 作为 fallback

### 7.2 每个里程碑的交付物
1. 可复现命令（build/install/run）
2. CSV 原始结果
3. `analyze_eval_csv.py` 输出
4. 采用/拒绝结论（含风险）

### 7.3 深度模型专属 Gate
- `first_lock_sec <= BASELINE_v1 + 0.2s`
- `track_like_ratio >= BASELINE_v1`
- `p95_latency_ms <= BASELINE_v1 + 10ms`
- 连续 3 轮回放无 crash

### 7.4 禁止事项
- 未经 A/B 验证直接替换默认后端
- 只用肉眼效果判定“变好”
- 缺少模型版本与量化信息登记

## 8. 最新执行快照（2026-04-13）

### 8.1 P0 已完成项
1. Native 追踪链路加入置信熔断（soft/hard/min）与可配置参数入口
2. 首锁保护：新增 `native_fuse_warmup_frames`（默认 12 帧）
3. UI 提示修正：启动时明确显示 `SEARCH=OpenCV, TRACK=<backend>`
4. 构建链路固化：新增 `tools/gradlew_jbr.ps1`，强制 Java 21（Android Studio JBR）

### 8.2 10 次回放结果（同视频、目标出现在 6s）
- 汇总文件：`tools/auto_tune/out/p0_10run_summary_20260413.csv`
- `first_lock_after_target_sec`：median `2.267s`，p95 `2.703s`
- `track_ratio`：median `0.345`，p95 `0.382`
- `avgFrameMs`：median `81.1ms`，p95 `85.37ms`

### 8.3 当前结论（必须同步给后续 AI）
1. 当前 `NcnnTrackerImpl` 仍属工程占位实现（stub 相关性内核），不是正式 Siam/NanoTrack 推理
2. 因此现阶段 NCNN 指标劣于成熟 OpenCV/KCF，不代表“深度路线失败”
3. P0 参数微调收益已接近上限，继续微调无法实现质变

### 8.4 下一阶段主战役（P1）
1. 接入真模型：`ncnn::Net` 加载真实 `NanoTrack/Siam` 权重（`.param/.bin`）
2. 后处理稳态增强：在响应图引入 Hanning/Cosine Window 抑制边缘跳变
3. 性能压缩：默认启用 FP16（后续评估 INT8 量化）并复用现有回放 Gate 做 A/B
