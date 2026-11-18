param(
    [int]$StartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root ".dev-run"
$apiUrl = "http://127.0.0.1:8000/api/health"
$webUrl = "http://127.0.0.1:5173/"

function Test-Service {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Wait-Service {
    param(
        [string]$Url,
        [System.Diagnostics.Process]$Process,
        [string]$Name
    )

    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Service -Url $Url) {
            return
        }

        if ($Process.HasExited) {
            throw "$Name exited during startup. Check its log in $runDir."
        }

        Start-Sleep -Milliseconds 500
    }

    throw "$Name did not respond at $Url within $StartupTimeoutSeconds seconds. Check its log in $runDir."
}

if ($StartupTimeoutSeconds -lt 1) {
    throw "StartupTimeoutSeconds must be at least 1."
}

$python = Get-Command python -ErrorAction SilentlyContinue
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python was not found on PATH."
}
if (-not $npm) {
    throw "npm.cmd was not found on PATH."
}
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    throw "Frontend dependencies are missing. Run 'npm.cmd ci' from the frontend folder first."
}

New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$apiProcess = $null
$webProcess = $null

if (Test-Service -Url $apiUrl) {
    Write-Host "API already responds at http://127.0.0.1:8000; reusing it."
} else {
    $apiArgs = @(
        "-m", "uvicorn", "alien_garden.api:app",
        "--app-dir", "src",
        "--host", "127.0.0.1",
        "--port", "8000"
    )
    $apiProcess = Start-Process `
        -FilePath $python.Source `
        -ArgumentList $apiArgs `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $runDir "api.log") `
        -RedirectStandardError (Join-Path $runDir "api-error.log") `
        -PassThru
    Set-Content -Path (Join-Path $runDir "api.pid") -Value $apiProcess.Id
}

try {
    if ($apiProcess) {
        Wait-Service -Url $apiUrl -Process $apiProcess -Name "FastAPI"
    }

    if (Test-Service -Url $webUrl) {
        Write-Host "Frontend already responds at $webUrl; reusing it."
    } else {
        $webProcess = Start-Process `
            -FilePath $env:ComSpec `
            -ArgumentList @("/c", "npm.cmd run dev -- --host 127.0.0.1") `
            -WorkingDirectory (Join-Path $root "frontend") `
            -RedirectStandardOutput (Join-Path $runDir "frontend.log") `
            -RedirectStandardError (Join-Path $runDir "frontend-error.log") `
            -PassThru
        Set-Content -Path (Join-Path $runDir "frontend.pid") -Value $webProcess.Id
        Wait-Service -Url $webUrl -Process $webProcess -Name "Vite"
    }
} catch {
    if ($webProcess -and -not $webProcess.HasExited) {
        Stop-Process -Id $webProcess.Id -ErrorAction SilentlyContinue
    }
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -ErrorAction SilentlyContinue
    }
    throw
}

Write-Host ""
Write-Host "Alien Garden services are ready:"
Write-Host "  Frontend: http://127.0.0.1:5173/"
Write-Host "  API docs: http://127.0.0.1:8000/docs"
Write-Host "  Logs:     $runDir"
