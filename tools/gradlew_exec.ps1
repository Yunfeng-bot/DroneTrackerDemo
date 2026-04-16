param(
    [switch]$UseAndroidStudioJbr,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GradleArgs = @(":app:assembleDebug")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$androidHome = Join-Path $projectRoot ".android_home"
if (-not (Test-Path -LiteralPath $androidHome)) {
    New-Item -ItemType Directory -Path $androidHome -Force | Out-Null
}

# Keep AGP environment deterministic: use only ANDROID_USER_HOME.
$null = Remove-Item Env:ANDROID_PREFS_ROOT -ErrorAction SilentlyContinue
$null = Remove-Item Env:ANDROID_SDK_HOME -ErrorAction SilentlyContinue
$env:ANDROID_USER_HOME = $androidHome
$env:ADB_VENDOR_KEYS = $androidHome

if ($UseAndroidStudioJbr) {
    $jbr = "C:\Program Files\Android\Android Studio\jbr"
    if (-not (Test-Path -LiteralPath $jbr)) {
        throw "Android Studio JBR not found: $jbr"
    }
    $env:JAVA_HOME = $jbr
    $env:Path = "$($env:JAVA_HOME)\bin;$($env:Path)"
}

Write-Host "[gradlew_exec] ANDROID_USER_HOME=$env:ANDROID_USER_HOME"
if ($UseAndroidStudioJbr) {
    Write-Host "[gradlew_exec] JAVA_HOME=$env:JAVA_HOME"
}
Write-Host "[gradlew_exec] gradle args=$($GradleArgs -join ' ')"

Push-Location $projectRoot
try {
    & ".\gradlew.bat" @GradleArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
