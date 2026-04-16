# 无人机伴飞单目标追踪系统项目规范（Project Spec）

## 1. 项目目标
- 构建可量化、可复现、可迁移的单目标追踪系统
- 追求“稳定可控”的工程表现，而非单次演示效果

## 2. 当前规范化技术栈
- Android + CameraX + OpenCV（ORB SEARCH + KCF TRACK）
- JNI/C++ HAL 预留（NCNN/RKNN）
- 离线回放评测（EvaluationActivity + CSV）

## 3. 回放自动化验证 SOP（强制执行）

### 3.1 目的
- 在同输入条件下做可复现 A/B 对比
- 避免现场环境噪声导致误判

### 3.2 准备
1. 设备连接：`adb devices`
2. 安装最新包：`adb -s <serial> install -r app-debug.apk`
3. 确认测试素材：
   - 视频：`/sdcard/Download/Video_Search/scene.mp4`
   - 模板：`/sdcard/Download/Video_Search/target.jpg`
4. 构建环境（强制）：
   - 命令行必须使用 Android Studio 自带 JBR（Java 21）
   - 禁止使用系统 JDK 25 直接构建
   - 推荐命令：`powershell -ExecutionPolicy Bypass -File tools/gradlew_jbr.ps1 :app:assembleDebug`

### 3.3 执行
使用 ADB 显式启动 EvaluationActivity：
- Activity：`com.example.dronetracker/.EvaluationActivity`
- 建议参数：
  - `eval_loop=false`
  - `eval_video_path=<video>`
  - `eval_target_path=<template>`
  - `eval_csv_path=<固定输出路径>`
  - `eval_params=<本轮参数串>`

运行 20~30 秒后强制停止：
- `adb shell am force-stop com.example.dronetracker`

### 3.4 拉取与分析
1. 拉取 CSV 到本地：
   - `adb pull <eval_csv_path> tools/auto_tune/eval_csv/`
2. 执行分析脚本：
   - `python tools/auto_tune/analyze_eval_csv.py <csv> --fps 15 --conf-lock 0.9`

### 3.5 指标解释
- `first_lock_sec`：首次进入高置信跟踪的时间
- `avg_latency_ms`：平均单帧耗时
- `p90/p95_latency_ms`：尾延迟（重点关注）
- `track_like_ratio`：稳定跟踪占比

### 3.6 判定规则
- 不满足 Gate 条件不得升级为新基线
- 任何“看起来更快”但 `track_like_ratio` 明显下降的方案一律拒绝

## 4. AI 工具切换交接清单（Claude/Codex 通用）
每次切换代理前必须记录：
1. 当前基线参数
2. 最近 3 次 CSV 与分析结果
3. 本轮改动的 commit id
4. 采用/拒绝结论与原因

## 5. 约束
- 所有优化结论以 CSV 与脚本输出为准
- 禁止仅凭 logcat 或肉眼判断作为最终结论
- 禁止未经回放验证直接修改默认参数并发布

## 6. 深度孪生后端切换 SOP（Claude/Codex 通用）

### 6.1 切换前检查
1. 确认当前基线与 Gate（来自 `docs/OPTIMIZATION_PLAN.md`）
2. 确认模型元信息齐全（hash/量化配置/导出脚本版本）
3. 确认 fallback 开关仍可切回 ORB/KCF

### 6.2 切换执行步骤
1. 编译并安装最新 APK
2. 以相同视频和模板分别运行：
   - `OpenCV` 后端
   - `NCNN/RKNN` 后端
3. 产出两份 CSV
4. 使用同一分析脚本对比

### 6.3 结论模板（强制）
- 结论：采用 / 暂不采用
- 原因：首锁、稳定性、p95 三项数据
- 风险：异常场景与回退策略
- 下一步：参数/模型/量化要改什么

### 6.4 文档回写要求
每次切换尝试后必须回写：
- `docs/TECHNICAL_ARCHITECTURE.md`（架构状态）
- `docs/OPTIMIZATION_PLAN.md`（基线与 Gate）
- commit id 与评测 CSV 路径

## 7. 当前状态声明（2026-04-14）

### 7.1 必须认知
1. 当前 Native NCNN 路径已完成工程地基（JNI/Zero-Copy/状态机/日志）
2. `NcnnTrackerImpl` 已接入 `ncnn::Net` 与 `.param/.bin` 加载，默认开启 FP16；当 ABI 或模型缺失时会回退到 `ncnn-stub`
3. 已完成 NCNN Dual 真模型链路的同源 A/B 复测（2026-04-14 晚），当前结果可进入 P1 Gate 审核；默认后端切换仍需补充多场景回归。

### 7.2 最新 P0 回放结果（10 次）
- 结果文件：`tools/auto_tune/out/p0_10run_summary_20260413.csv`
- `first_lock_after_target_sec` median `2.267s`，p95 `2.703s`
- `track_ratio` median `0.345`，p95 `0.382`
- `avgFrameMs` median `81.1ms`，p95 `85.37ms`
- 该结果仍是当前已确认的公开基线，P1 需在同口径下给出可复现实验结论

### 7.3 下一步强制优先级
1. P1-1：固定模型版本并完成 OpenCV vs NCNN 同源 A/B（CSV + 脚本分析 + Gate 判定）
2. P1-2：在 NCNN 路径启用 Hanning/Cosine Window 抑制跳变，并验证收益是否满足 Gate
3. P1-3：维持 FP16 默认配置，模型稳定后再进入 INT8 量化评估



### 7.4 最新执行记录（2026-04-14）
- OpenCV CSV：`tools/auto_tune/eval_csv/eval_opencv_20260414_132205.csv`
- NCNN CSV：`tools/auto_tune/eval_csv/eval_ncnn_20260414_132319.csv`
- 结果：两组指标接近，但设备缺少 `nanotrack.param/bin`，本轮不纳入 P1 Gate 判定。
- 详情见：`docs/P1_AB_RUN_20260414.md`


### 7.5 追加执行记录（2026-04-14 晚）
- OpenCV CSV（午间对照）：`tools/auto_tune/eval_csv/eval_opencv_20260414_132205.csv`
- NCNN 早期 CSV（模型缺失）：`tools/auto_tune/eval_csv/eval_ncnn_20260414_132319.csv`
- NCNN Dual 最新 CSV：`tools/auto_tune/eval_csv/eval_ncnn_dual_full_20260414.csv`
- 晚间复测结果（`--conf-lock 0.9`）：`frames=301`, `track_like_ratio=0.8206`, `first_lock_sec=0.4667`, `avg_latency_ms=27.4019`, `p95_latency_ms=132.5030`
- 结论：NCNN Dual 真模型链路已跑通且达到 P1 Gate 审核口径；默认后端切换前仍需补充多场景回归。
- 详情见：`docs/P1_AB_RUN_20260414.md`
