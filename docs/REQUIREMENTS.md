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
