# 开发环境防踩坑（机制化）

## 根因
1. `ANDROID_*` 变量在同一会话被混用（`ANDROID_PREFS_ROOT` / `ANDROID_SDK_HOME` / `ANDROID_USER_HOME`），AGP 会拒绝或行为不稳定。  
2. 部分流程绕过了仓库包装器，直接调用 `adb` / `gradlew`，导致回退到不可写用户目录（如 `C:\Users\CodexSandboxOffline\.android`）。  
3. 自动化依赖 `tools/adb_args.json` 作为中间文件，权限抖动时会阻塞整条链路。  
4. 自动调参输出目录权限异常会直接中断。

## 已做的机制修复
1. `tools/dev-env.ps1`：统一会话环境，只保留 `ANDROID_USER_HOME`，并清除冲突变量。  
2. `tools/fix-dev-permissions.ps1`：持久化逻辑改为只写 `ANDROID_USER_HOME`/`ADB_VENDOR_KEYS`，并删除用户级冲突变量。  
3. `tools/gradlew_exec.ps1`（新增）：Gradle 单入口，自动清理冲突 `ANDROID_*` 后再执行。  
4. `tools/gradlew_jbr.ps1`：改为转调 `gradlew_exec.ps1 -UseAndroidStudioJbr`。  
5. `tools/adb_exec.ps1`：增强环境隔离（清 `ANDROID_PREFS_ROOT`，补 `HOMEDRIVE/HOMEPATH`，支持直接传 adb 参数）。  
6. `tools/auto_tune/sweep_replay.py`：wrapper 模式改为直接传参，不再依赖 `adb_args.json`；输出目录不可写时自动回退到 `.codex_tmp/auto_tune/out`。

## 固化使用规约
1. 新开终端先执行：`.\tools\dev-env.ps1`（首次可加 `-PersistUser`）。  
2. Gradle 一律走：`.\tools\gradlew_exec.ps1 ...`（或 `.\tools\gradlew_jbr.ps1 ...`）。  
3. ADB 一律走：`.\tools\adb_exec.ps1 ...`。  
4. 回放/调参使用 `sweep_replay.py` 时保持 `--adb-mode wrapper`（默认即 wrapper）。  
5. 遇到 `.git\objects` 写入异常，再执行一次：`.\tools\fix-dev-permissions.ps1 -AggressiveGitAcl`（管理员 PowerShell）。

## 快速自检
1. `.\tools\dev-env.ps1`  
2. `.\tools\adb_exec.ps1 devices`  
3. `.\tools\gradlew_exec.ps1 -UseAndroidStudioJbr :app:compileDebugKotlin --no-daemon`
