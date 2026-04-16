# 追踪优化复盘（2026-04-13）

## 1. 本轮目标
1. 落地 Native 路径的 P0 能力：置信熔断、自愈、首锁保护、可观测性
2. 固化构建环境：彻底规避 JDK 25 与 JNI/Gradle 混编冲突
3. 用离线回放给出可重复的结论，不凭肉眼判断

## 2. 已完成改动

### 2.1 追踪逻辑
- 新增 Native 置信门控参数：`native_min_conf` / `native_fuse_soft_conf` / `native_fuse_hard_conf` / `native_fuse_soft_streak`
- 新增首锁暖启动窗口：`native_fuse_warmup_frames`（默认 12）
- 暖启动策略：短窗口内抑制 DROP，但允许框正常更新，避免“锁上即熔断”与“框冻结”

### 2.2 观测与可视化
- `EVAL_PERF` 增加 native 置信分布与 fuse 计数
- 主界面启动提示改为动态后端说明：`SEARCH=OpenCV, TRACK=<backend>`

### 2.3 构建环境
- 新增脚本：`tools/gradlew_jbr.ps1`
- 强制命令行构建使用 Android Studio JBR（Java 21）

## 3. 关键事实（残酷但必要）
1. 当前 `NcnnTrackerImpl` 仍是工程占位追踪核（stub），不是正式 Siam/NanoTrack 神经网络推理。
2. 现阶段将其与成熟 OpenCV/KCF 做稳定性与时延对标，出现“NCNN 不如 KCF”是预期现象。
3. 目前所有阈值微调只能优化边缘行为，无法替代“真模型接入”带来的质变。

## 4. 回放验证结果（10 次）
- 汇总文件：`tools/auto_tune/out/p0_10run_summary_20260413.csv`
- 测试窗口：`duration=40s`，`target_appear=6s`

指标：
- `first_lock_after_target_sec`：median `2.267s`，p95 `2.703s`
- `track_ratio`：median `0.345`，p95 `0.382`
- `avgFrameMs`：median `81.1ms`，p95 `85.37ms`

结论：
- P0 机制已经生效（可配置、可观测、可回放复现）
- 但当前指标仍未达到生产门槛，瓶颈不在参数微调，而在追踪核能力

## 5. 风险清单
1. 若继续把时间投入在 stub 参数炼丹，收益极低且容易误导方向
2. 若不统一 JBR 21，构建会反复出现 `java 25.0.1` 兼容错误
3. 若不保留统一回放口径，A/B 结论会被环境噪声污染

## 6. 下一步（P1 战役）
1. 真模型接入（最高优先级）：
   - 在 `NcnnTrackerImpl` 中引入 `ncnn::Net`
   - 固定加载 `.param/.bin`，登记模型 hash
2. 稳态增强：
   - 响应图后处理加入 Hanning/Cosine Window，抑制边缘高分误跳
3. 性能压缩：
   - 开启 FP16（`use_fp16_arithmetic` / `use_fp16_storage`）
   - 模型稳定后再进入 INT8 量化

## 7. 交接说明（给后续 AI/同事）
1. 先读：`docs/TECHNICAL_ARCHITECTURE.md`、`docs/OPTIMIZATION_PLAN.md`、本文件
2. 先跑：`powershell -ExecutionPolicy Bypass -File tools/gradlew_jbr.ps1 :app:assembleDebug`
3. 再测：`tools/auto_tune/sweep_replay.py`，并把 CSV 结论回写文档


## 8. 状态更新（2026-04-14）
1. NcnnTrackerImpl 已接入 ncnn::Net 与 .param/.bin 加载逻辑，具备 no-runtime fallback。
2. 复盘文档中的 stub-only 结论已过时，当前阶段应以 Gate 口径做 OpenCV vs NCNN 同源 A/B。
3. 下一步重点从是否接入真模型切换为后处理稳定性（Hanning/Cosine）与可复现验证结果回写。

