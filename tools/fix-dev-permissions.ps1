param(
    [switch]$PersistEnv,
    [switch]$AggressiveGitAcl
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$androidHome = Join-Path $projectRoot ".android_home"
$gitObjects = Join-Path $projectRoot ".git\objects"
$userName = $env:USERNAME

if (-not (Test-Path $androidHome)) {
    New-Item -ItemType Directory -Path $androidHome -Force | Out-Null
}

# Use a single Android home root to avoid AGP env conflicts.
$null = Remove-Item Env:ANDROID_PREFS_ROOT -ErrorAction SilentlyContinue
$null = Remove-Item Env:ANDROID_SDK_HOME -ErrorAction SilentlyContinue
$env:ANDROID_USER_HOME = $androidHome
$env:ADB_VENDOR_KEYS = $androidHome
$env:HOME = $projectRoot
$env:USERPROFILE = $projectRoot
$env:HOMEDRIVE = [System.IO.Path]::GetPathRoot($projectRoot).TrimEnd('\')
$env:HOMEPATH = $projectRoot.Substring($env:HOMEDRIVE.Length)

if ($PersistEnv) {
    [Environment]::SetEnvironmentVariable("ANDROID_PREFS_ROOT", $null, "User")
    [Environment]::SetEnvironmentVariable("ANDROID_SDK_HOME", $null, "User")
    [Environment]::SetEnvironmentVariable("ANDROID_USER_HOME", $androidHome, "User")
    [Environment]::SetEnvironmentVariable("ADB_VENDOR_KEYS", $androidHome, "User")
}

if (Test-Path $gitObjects) {
    # PowerShell requires quoting ACE strings containing parentheses.
    if ($AggressiveGitAcl) {
        & takeown /f $gitObjects /r /d y | Out-Null
    }

    $icaclsOutput = & icacls $gitObjects /grant "${userName}:(OI)(CI)F" /T /C 2>&1
    $outputText = ($icaclsOutput | Out-String)
    if ($LASTEXITCODE -ne 0 -or $outputText -match "拒绝访问|Access is denied|失败|Successfully processed 0 files") {
        Write-Error "Git objects ACL repair failed. Run this script in Administrator PowerShell with -AggressiveGitAcl."
        Write-Host $outputText
        exit 1
    }
}

Write-Host "Done."
Write-Host "ANDROID_PREFS_ROOT=<unset>"
Write-Host "ANDROID_SDK_HOME=<unset>"
Write-Host "ANDROID_USER_HOME=$($env:ANDROID_USER_HOME)"
Write-Host "ADB_VENDOR_KEYS=$($env:ADB_VENDOR_KEYS)"
Write-Host "HOME=$($env:HOME)"
Write-Host "Git objects ACL repaired for user: $userName"
if ($PersistEnv) {
    Write-Host "Persistent user env updated. Reopen terminal for new sessions."
}
