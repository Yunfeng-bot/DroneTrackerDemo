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

# Make adb use a writable location in the repo instead of %USERPROFILE%\.android.
$env:ANDROID_SDK_HOME = $androidHome
$env:ANDROID_USER_HOME = $androidHome
$env:ADB_VENDOR_KEYS = $androidHome

if ($PersistEnv) {
    setx ANDROID_SDK_HOME $androidHome | Out-Null
    setx ANDROID_USER_HOME $androidHome | Out-Null
    setx ADB_VENDOR_KEYS $androidHome | Out-Null
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
Write-Host "ANDROID_SDK_HOME=$($env:ANDROID_SDK_HOME)"
Write-Host "ANDROID_USER_HOME=$($env:ANDROID_USER_HOME)"
Write-Host "ADB_VENDOR_KEYS=$($env:ADB_VENDOR_KEYS)"
Write-Host "Git objects ACL repaired for user: $userName"
if ($PersistEnv) {
    Write-Host "Persistent env vars written with setx. Reopen terminal for new sessions."
}
