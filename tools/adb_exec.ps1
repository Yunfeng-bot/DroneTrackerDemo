param()

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$argsFile = Join-Path $scriptDir "adb_args.json"

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

$adbArgs = @()
if ($parsed -is [System.Array]) {
    foreach ($item in $parsed) {
        if ($null -ne $item) {
            $adbArgs += [string]$item
        }
    }
} elseif ($null -ne $parsed) {
    $adbArgs += [string]$parsed
}

if ($adbArgs.Count -eq 0) {
    Write-Error "adb args are empty in $argsFile"
    exit 4
}

& adb @adbArgs
exit $LASTEXITCODE
