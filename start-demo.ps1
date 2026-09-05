param(
    [switch]$ResetEnvironment
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 --version 2>$null
        if ($LASTEXITCODE -eq 0) { return @("py", "-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python --version 2>$null
        if ($LASTEXITCODE -eq 0) { return @("python") }
    }
    throw "Python 3 is required. Install it with: winget install Python.Python.3.12"
}

$python = Get-PythonCommand
$pythonArguments = @($python | Select-Object -Skip 1)
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$needsEnvironment = $ResetEnvironment -or -not (Test-Path $venvPython)
if (-not $needsEnvironment) {
    & $venvPython -c "import fastapi, uvicorn" 2>$null
    $needsEnvironment = $LASTEXITCODE -ne 0
}
if ($needsEnvironment) {
    # A venv stores an absolute path to its original Python executable.  Safely
    # recreate this project-local venv when that interpreter is no longer present.
    if (Test-Path (Join-Path $projectRoot ".venv")) {
        Remove-Item -LiteralPath (Join-Path $projectRoot ".venv") -Recurse -Force
    }
    & $python[0] @pythonArguments -m venv .venv
    & $venvPython -m pip install -r backend/requirements.txt
}

& $venvPython seed.py

$nodeRoot = "C:\Program Files\nodejs"
if (-not (Test-Path (Join-Path $nodeRoot "npm.cmd"))) {
    throw "Node.js LTS is required. Install it with: winget install OpenJS.NodeJS.LTS"
}
& (Join-Path $nodeRoot "npm.cmd") install

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$venvPython' -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$nodeRoot\npm.cmd' run dev --workspace web"

Write-Host "ThreatNet is starting. Open http://127.0.0.1:3000 after both terminal windows show ready."
