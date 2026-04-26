param(
    [switch]$RestartServer,
    [switch]$ShowEnv,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdbArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$argsFile = Join-Path $scriptDir "adb_args.json"
$adbUserHome = Join-Path $projectRoot ".android_home"

if (-not (Test-Path -LiteralPath $adbUserHome)) {
    New-Item -ItemType Directory -Force -Path $adbUserHome | Out-Null
}

# Keep adb home deterministic and AGP-safe:
# - only ANDROID_USER_HOME is defined
# - ANDROID_PREFS_ROOT / ANDROID_SDK_HOME are unset
# - HOME / USERPROFILE point to project root for stable subprocess behavior
$null = Remove-Item Env:ANDROID_PREFS_ROOT -ErrorAction SilentlyContinue
$null = Remove-Item Env:ANDROID_SDK_HOME -ErrorAction SilentlyContinue
$env:ANDROID_USER_HOME = $adbUserHome
$env:ADB_VENDOR_KEYS = $adbUserHome
$env:HOME = $projectRoot
$env:USERPROFILE = $projectRoot
$env:HOMEDRIVE = [System.IO.Path]::GetPathRoot($projectRoot).TrimEnd('\')
$env:HOMEPATH = $projectRoot.Substring($env:HOMEDRIVE.Length)

# PowerShell may bind an empty trailing argument as a one-item array when
# using ValueFromRemainingArguments; normalize it so "no adb args" falls back
# to adb_args.json as intended.
$AdbArgs = @($AdbArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

if (@($AdbArgs).Count -eq 0) {
    if (-not (Test-Path -LiteralPath $argsFile)) {
        Write-Error "adb args file not found: $argsFile"
        exit 2
    }

    try {
        $raw = Get-Content -LiteralPath $argsFile -Raw -Encoding UTF8
        $parsed = ConvertFrom-Json -InputObject $raw
    } catch {
        Write-Error "failed to parse adb args json: $($_.Exception.Message)"
        exit 3
    }

    $AdbArgs = @()
    if ($parsed -is [System.Array]) {
        foreach ($item in $parsed) {
            if ($null -ne $item) {
                $AdbArgs += [string]$item
            }
        }
    } elseif ($null -ne $parsed) {
        $AdbArgs += [string]$parsed
    }
}

if (@($AdbArgs).Count -eq 0) {
    Write-Error "adb args are empty (neither cli args nor $argsFile)"
    exit 4
}

if ($ShowEnv) {
    Write-Host "ANDROID_PREFS_ROOT=<unset>"
    Write-Host "ANDROID_SDK_HOME=<unset>"
    Write-Host "ANDROID_USER_HOME=$($env:ANDROID_USER_HOME)"
    Write-Host "ADB_VENDOR_KEYS=$($env:ADB_VENDOR_KEYS)"
    Write-Host "HOME=$($env:HOME)"
}

if ($RestartServer) {
    & adb kill-server | Out-Null
    & adb start-server | Out-Null
}

& adb @AdbArgs
exit $LASTEXITCODE
