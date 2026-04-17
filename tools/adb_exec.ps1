param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdbArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$argsFile = Join-Path $scriptDir "adb_args.json"
$adbHome = Join-Path $scriptDir ".adb-home"
$adbUserHome = Join-Path $adbHome ".android"

if (-not (Test-Path -LiteralPath $adbUserHome)) {
    New-Item -ItemType Directory -Force -Path $adbUserHome | Out-Null
}

# Force adb to use workspace-local home to avoid sandbox permission jitter on
# C:\Users\CodexSandboxOffline\.android.
$null = Remove-Item Env:ANDROID_PREFS_ROOT -ErrorAction SilentlyContinue
$env:ANDROID_SDK_HOME = $adbHome
$env:ANDROID_USER_HOME = $adbUserHome
$env:ADB_VENDOR_KEYS = $adbUserHome
$env:HOME = $adbHome
$env:USERPROFILE = $adbHome
$env:HOMEDRIVE = [System.IO.Path]::GetPathRoot($adbHome).TrimEnd('\')
$env:HOMEPATH = $adbHome.Substring($env:HOMEDRIVE.Length)

if ($AdbArgs.Count -eq 0) {
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

if ($AdbArgs.Count -eq 0) {
    Write-Error "adb args are empty (neither cli args nor $argsFile)"
    exit 4
}

& adb @AdbArgs
exit $LASTEXITCODE
