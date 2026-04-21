param(
    [ValidateSet("l1", "l2", "l3", "fail")]
    [string]$Variant = "l1",
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [double]$GpsReadyAtSec = 2.0,
    [int]$ReplayFps = 15,
    [string]$TrackerMode = "enhanced",
    [double]$DurationSec = 32.0,
    [double]$TargetAppearSec = 2.0,
    [switch]$StrictFirstLockCenter
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$videoLocal = Join-Path $repoRoot ("test/scene_mvp5_{0}_{1}.mp4" -f $Variant, $Date)
$targetLocal = Join-Path $repoRoot ("test/target_rooftop_{0}.jpg" -f $Date)
$windowsLocal = Join-Path $repoRoot ("tools/replay_sop/mvp5_windows_{0}.json" -f $Variant)

if (-not (Test-Path -LiteralPath $videoLocal)) {
    throw "Missing local video: $videoLocal"
}
if (-not (Test-Path -LiteralPath $targetLocal)) {
    throw "Missing local target: $targetLocal"
}
if (-not (Test-Path -LiteralPath $windowsLocal)) {
    throw "Missing windows json: $windowsLocal"
}

$videoRemote = "/sdcard/Download/Video_Search/scene_mvp5_{0}_{1}.mp4" -f $Variant, $Date
$targetRemote = "/sdcard/Download/Video_Search/target_rooftop_{0}.jpg" -f $Date

Write-Host "[mvp5] force-stop app for cold session"
& "$repoRoot/tools/adb_exec.ps1" shell am force-stop com.example.dronetracker
Start-Sleep -Seconds 1

Write-Host "[mvp5] push video => $videoRemote"
& "$repoRoot/tools/adb_exec.ps1" push "$videoLocal" "/sdcard/Download/Video_Search/"

Write-Host "[mvp5] push target => $targetRemote"
& "$repoRoot/tools/adb_exec.ps1" push "$targetLocal" "/sdcard/Download/Video_Search/"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "tools/auto_tune/out/mvp5_{0}_{1}" -f $Variant, $stamp

$baseParams = "center_roi_search_enable=true,center_roi_gps_ready=false"
$baseParams += ",fallback_refine_ratio=0.8,small_target_min_good=5,small_target_min_inliers=3,orb_ransac=4,first_lock_min_iou=0.30,center_roi_l3_timeout_ms=5000"
if ($Variant -eq "l1" -or $Variant -eq "l2" -or $Variant -eq "l3") {
    # Acquire boost profile for positive variants:
    # - Lower soft-match floor and enable earlier relax under long miss streaks
    # - Slightly loosen Lowe ratio to recover sparse rooftop features
    # - Extend first-lock candidate gap to survive sparse replay hits
    $baseParams += ",orb_ratio=0.82,orb_soft_min_matches=3,soft_relax_miss=4,soft_relax_max_ratio=0.82,first_lock_gap_ms=2500"
    # Optional strict center guard (can reduce wrong-lock, but may increase miss-lock).
    if ($StrictFirstLockCenter) {
        $baseParams += ",auto_verify_first_lock_center_guard_replay=true,auto_verify_first_lock_center_factor_replay=0.22"
        $baseParams += ",auto_verify_first_lock_min_inliers_replay=5"
    }
}

Write-Host "[mvp5] run sweep => $outDir"
python -u tools/auto_tune/sweep_replay.py `
    --video-path "$videoRemote" `
    --target-path "$targetRemote" `
    --windows "tools/replay_sop/mvp5_windows_${Variant}.json" `
    --gps-ready-at-sec $GpsReadyAtSec `
    --gps-ready-reason "mvp5_auto" `
    --replay-fps $ReplayFps `
    --replay-catchup `
    --tracker-mode $TrackerMode `
    --max-runs 1 `
    --preset fixed `
    --duration-sec $DurationSec `
    --target-appear-sec $TargetAppearSec `
    --retry-empty 0 `
    --retry-low-replay 2 `
    --cooldown-sec 0 `
    --base-params $baseParams `
    --out $outDir

$summaryPath = Join-Path $repoRoot ($outDir + "/summary.json")
if (-not (Test-Path -LiteralPath $summaryPath)) {
    throw "summary.json not found: $summaryPath"
}

$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
Write-Host "[mvp5] summary: $summaryPath"
Write-Host ("  descend_offset_first_t={0}" -f $summary.descend_offset_first_t)
Write-Host ("  descend_offset_last_state={0}" -f $summary.descend_offset_last_state)
Write-Host ("  descend_offset_oob_count={0}" -f $summary.descend_offset_oob_count)
Write-Host ("  descend_offset_fail_count={0}" -f $summary.descend_offset_fail_count)
