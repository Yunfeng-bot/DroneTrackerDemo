param(
    [switch]$PersistUser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$androidHome = Join-Path $projectRoot ".android_home"
if (-not (Test-Path -LiteralPath $androidHome)) {
    New-Item -ItemType Directory -Path $androidHome -Force | Out-Null
}

# AGP is sensitive to conflicting ANDROID_* definitions.
$null = Remove-Item Env:ANDROID_PREFS_ROOT -ErrorAction SilentlyContinue
$null = Remove-Item Env:ANDROID_SDK_HOME -ErrorAction SilentlyContinue
$env:ANDROID_USER_HOME = $androidHome
$env:ADB_VENDOR_KEYS = $androidHome

if ($PersistUser) {
    [Environment]::SetEnvironmentVariable("ANDROID_PREFS_ROOT", $null, "User")
    [Environment]::SetEnvironmentVariable("ANDROID_SDK_HOME", $null, "User")
    [Environment]::SetEnvironmentVariable("ANDROID_USER_HOME", $androidHome, "User")
    [Environment]::SetEnvironmentVariable("ADB_VENDOR_KEYS", $androidHome, "User")
}

Write-Host "ANDROID_PREFS_ROOT=<unset>"
Write-Host "ANDROID_SDK_HOME=<unset>"
Write-Host "ANDROID_USER_HOME=$($env:ANDROID_USER_HOME)"
Write-Host "ADB_VENDOR_KEYS=$($env:ADB_VENDOR_KEYS)"
if ($PersistUser) {
    Write-Host "Persistent user env updated. Reopen terminal for new sessions."
}
