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
