# P1 A/B 回放验证清单（OpenCV vs NCNN）

## 1. 目的
- 在同一视频、同一模板、同一时长下对比 OpenCV 与 NCNN。
- 以 CSV 与 analyze_eval_csv.py 输出作为唯一结论依据。

## 2. 构建与安装
1. 使用 JBR 21 构建：
   `powershell -ExecutionPolicy Bypass -File tools/gradlew_jbr.ps1 :app:assembleDebug`
2. 安装 APK：
   `adb -s <serial> install -r app/build/outputs/apk/debug/app-debug.apk`

## 3. 固定输入素材
- 视频：`/sdcard/Download/Video_Search/scene.mp4`
- 模板：`/sdcard/Download/Video_Search/target.jpg`

## 4. OpenCV 对照组
1. 启动：
   `adb shell am start -n com.example.dronetracker/.EvaluationActivity --ez eval_loop false --es eval_video_path /sdcard/Download/Video_Search/scene.mp4 --es eval_target_path /sdcard/Download/Video_Search/target.jpg --es eval_csv_path /sdcard/Download/Video_Search/eval_opencv.csv --es eval_params "backend=opencv"`
2. 运行 20~30 秒后停止：
   `adb shell am force-stop com.example.dronetracker`

## 5. NCNN 试验组
1. 启动：
   `adb shell am start -n com.example.dronetracker/.EvaluationActivity --ez eval_loop false --es eval_video_path /sdcard/Download/Video_Search/scene.mp4 --es eval_target_path /sdcard/Download/Video_Search/target.jpg --es eval_csv_path /sdcard/Download/Video_Search/eval_ncnn.csv --es eval_params "backend=ncnn,native_model_param=/sdcard/Download/Video_Search/nanotrack.param,native_model_bin=/sdcard/Download/Video_Search/nanotrack.bin"`
2. 运行 20~30 秒后停止：
   `adb shell am force-stop com.example.dronetracker`

## 6. 拉取与分析
1. 拉取：
   `adb pull /sdcard/Download/Video_Search/eval_opencv.csv tools/auto_tune/eval_csv/`
   `adb pull /sdcard/Download/Video_Search/eval_ncnn.csv tools/auto_tune/eval_csv/`
2. 分析：
   `python tools/auto_tune/analyze_eval_csv.py tools/auto_tune/eval_csv/eval_opencv.csv --fps 15 --conf-lock 0.9`
   `python tools/auto_tune/analyze_eval_csv.py tools/auto_tune/eval_csv/eval_ncnn.csv --fps 15 --conf-lock 0.9`

## 7. Gate 判定
- `first_lock_sec <= BASELINE_v1 + 0.2s`
- `track_like_ratio >= BASELINE_v1`
- `p95_latency_ms <= BASELINE_v1 + 10ms`

## 8. 回写要求
- 在 `docs/OPTIMIZATION_PLAN.md` 回写采用/拒绝结论。
- 在 `docs/TECHNICAL_ARCHITECTURE.md` 回写当前后端状态。
- 记录本轮 commit id 与 CSV 路径。
