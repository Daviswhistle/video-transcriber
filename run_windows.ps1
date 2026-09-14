param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$InputFile,
    [switch]$Diarize,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

# Load KEY=VALUE pairs from .env into this process.
$EnvFile = Join-Path $ScriptDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $name = $parts[0].Trim()
            $value = $parts[1].Trim().Trim('"').Trim("'")
            if ($name) {
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "가상환경 생성..."
    python -m venv (Join-Path $ScriptDir ".venv")
    & $Python -m pip install -U pip
}

# Keep the venv synchronized with the repository after updates.
& $Python -m pip install -q -r (Join-Path $ScriptDir "requirements.txt")

$argsList = @((Join-Path $ScriptDir "transcribe_video.py"), $InputFile)

if ($Diarize) {
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        throw "-Diarize에는 ffmpeg가 필요합니다. 먼저 'winget install Gyan.FFmpeg'를 실행하세요."
    }
    & $Python -m pip install -r (Join-Path $ScriptDir "requirements-diarize.txt")
    $argsList += "--diarize"
}

if ($ExtraArgs) {
    $argsList += $ExtraArgs
}

& $Python @argsList
exit $LASTEXITCODE
