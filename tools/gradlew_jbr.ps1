param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GradleArgs = @(":app:assembleDebug")
)

$ErrorActionPreference = "Stop"

$jbr = "C:\Program Files\Android\Android Studio\jbr"
if (-not (Test-Path -LiteralPath $jbr)) {
    throw "Android Studio JBR not found: $jbr"
}

$env:JAVA_HOME = $jbr
$env:Path = "$($env:JAVA_HOME)\bin;$($env:Path)"

Write-Host "[gradlew_jbr] JAVA_HOME=$env:JAVA_HOME"
Write-Host "[gradlew_jbr] gradle args=$($GradleArgs -join ' ')"

& ".\gradlew.bat" @GradleArgs
exit $LASTEXITCODE
