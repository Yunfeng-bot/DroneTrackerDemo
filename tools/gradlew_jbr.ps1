param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GradleArgs = @(":app:assembleDebug")
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$execScript = Join-Path $scriptDir "gradlew_exec.ps1"

& $execScript -UseAndroidStudioJbr @GradleArgs
exit $LASTEXITCODE
