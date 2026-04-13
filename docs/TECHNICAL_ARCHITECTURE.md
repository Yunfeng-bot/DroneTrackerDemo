# 无人机单目标追踪技术方案（2026-04-13）

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
  - `NcnnTrackerImpl`（当前 stub）
  - `RknnTrackerImpl`（当前 stub）
  - `NanoTrackerEngine` 单例管理

目标：上层调用稳定、下层后端可热切换，不破坏业务接口。

### 2.4 离线评测基础设施（已接入）
- Activity：`EvaluationActivity`
- 输出：每帧 CSV
  - `frame_id, latency_ms, predicted_x, predicted_y, predicted_w, predicted_h, confidence_score`
- 分析脚本：`tools/auto_tune/analyze_eval_csv.py`

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

## 5. 下一阶段架构演进

### P0（当前迭代）
- 保持 ORB/KCF 主链路
- 继续压缩 P95 尾延迟（预算熔断 + 限频）
- 固化自动评测评分口径

### P1（下一阶段）
- 将 NCNN 后端从 stub 变为可运行推理路径
- 在同一评测基线上 A/B 对比 OpenCV 与 NCNN

### P2（量产预研）
- RKNN 后端接入
- 保持同一 HAL 接口，不改上层业务状态机
