# ============================================
# Road Crack Detection System - Service Script
# ============================================
# Usage:
#   .\start-all-services.ps1                    # Start all services
#   .\start-all-services.ps1 -Services gateway,auth  # Start selected backend services
#   .\start-all-services.ps1 -Frontend          # Start frontend only
#   .\start-all-services.ps1 -Backend           # Start all backend services
#   .\start-all-services.ps1 -Infra             # Show infra instructions
#   .\start-all-services.ps1 -Stop              # Stop all services
# ============================================

param(
    [string[]]$Services = @(),
    [switch]$Frontend,
    [switch]$Backend,
    [switch]$Infra,
    [switch]$All,
    [switch]$Stop,
    [switch]$Build,
    [switch]$Help
)

function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

$RootPath = Split-Path -Parent $PSScriptRoot
$BackendPath = $PSScriptRoot
$FrontendPath = Join-Path $RootPath "cloud-frontend"
$PythonPath = Join-Path $RootPath "python-inference"

$AllBackendServices = @{
    "gateway"   = @{ Port = 8080; Name = "cloud-gateway"; Desc = "Gateway" }
    "auth"      = @{ Port = 8081; Name = "cloud-auth"; Desc = "Auth" }
    "dataset"   = @{ Port = 8082; Name = "cloud-dataset"; Desc = "Dataset" }
    "inference" = @{ Port = 8083; Name = "cloud-inference"; Desc = "Inference" }
    "visual"    = @{ Port = 8084; Name = "cloud-visual"; Desc = "Visual" }
    "report"    = @{ Port = 8085; Name = "cloud-report"; Desc = "Report" }
}

function Show-Help {
    Write-Host ""
    Write-Host "Road Crack Detection System - Service Script" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\start-all-services.ps1 [options]"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Services <list>  Backend services to start (comma-separated)"
    Write-Host "                   Available: gateway, auth, dataset, inference, visual, report"
    Write-Host "  -Frontend          Start frontend only"
    Write-Host "  -Backend           Start all backend services"
    Write-Host "  -Infra             Show infra instructions"
    Write-Host "  -All               Start all services (default)"
    Write-Host "  -Build             Build before start"
    Write-Host "  -Stop              Stop all services"
    Write-Host "  -Help              Show this help"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start-all-services.ps1                         # Start all services"
    Write-Host "  .\start-all-services.ps1 -Services gateway,auth  # Start gateway and auth"
    Write-Host "  .\start-all-services.ps1 -Backend -Build         # Build and start backend"
    Write-Host "  .\start-all-services.ps1 -Frontend               # Start frontend"
    Write-Host "  .\start-all-services.ps1 -Infra                  # Show infra help"
    Write-Host ""
}

function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Stop-ProcessOnPort {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Warn "Stopping process on port ${Port}: $($process.ProcessName) (PID: $($process.Id))"
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Build-Backend {
    Write-Info "Building backend..."
    Push-Location $BackendPath
    try {
        & mvn clean package -DskipTests -q
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Backend build succeeded"
        } else {
            Write-Err "Backend build failed"
            exit 1
        }
    } finally {
        Pop-Location
    }
}

function Start-BackendService {
    param([string]$ServiceKey)

    $service = $AllBackendServices[$ServiceKey]
    if (-not $service) {
        Write-Err "Unknown service: $ServiceKey"
        return
    }

    $servicePath = Join-Path $BackendPath $service.Name
    $jarFile = Get-ChildItem -Path "$servicePath\target\*.jar" -Exclude "*-sources.jar", "*.original" -ErrorAction SilentlyContinue | Select-Object -First 1

    if (-not $jarFile) {
        Write-Err "$($service.Desc) JAR not found. Build first: mvn clean package -DskipTests"
        return
    }

    $port = $service.Port

    if (Test-PortInUse -Port $port) {
        Write-Warn "Port $port is in use. Stopping existing process..."
        Stop-ProcessOnPort -Port $port
        Start-Sleep -Seconds 1
    }

    Write-Info "Starting $($service.Desc) (port: $port)..."

    $logFile = Join-Path $servicePath "target\$ServiceKey.log"
    Start-Process -FilePath "java" -ArgumentList "-jar", $jarFile.FullName -WindowStyle Hidden -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"

    $maxWait = 30
    $waited = 0
    while (-not (Test-PortInUse -Port $port) -and $waited -lt $maxWait) {
        Start-Sleep -Seconds 1
        $waited++
    }

    if (Test-PortInUse -Port $port) {
        Write-Success "$($service.Desc) started (port: $port)"
    } else {
        Write-Err "$($service.Desc) startup timed out. Check logs: $logFile"
    }
}

function Start-Frontend {
    if (-not (Test-Path $FrontendPath)) {
        Write-Err "Frontend directory not found: $FrontendPath"
        return
    }

    Write-Info "Starting frontend..."
    Push-Location $FrontendPath
    try {
        if (-not (Test-Path "node_modules")) {
            Write-Info "Installing frontend dependencies..."
            & npm install
        }

        Start-Process -FilePath "cmd" -ArgumentList "/c", "npm run dev" -WorkingDirectory $FrontendPath
        Write-Success "Frontend starting... (port: 3000)"
        Write-Info "URL: http://localhost:3000"
    } finally {
        Pop-Location
    }
}

function Start-PythonInference {
    if (-not (Test-Path $PythonPath)) {
        Write-Warn "Python API directory not found: $PythonPath"
        return
    }

    Write-Info "Starting Python API..."
    Push-Location $PythonPath
    try {
        Start-Process -FilePath "cmd" -ArgumentList "/c", "python api_server.py" -WorkingDirectory $PythonPath
        Write-Success "Python API starting... (port: 5000)"
    } finally {
        Pop-Location
    }
}

function Stop-AllServices {
    Write-Info "Stopping all services..."

    foreach ($key in $AllBackendServices.Keys) {
        $service = $AllBackendServices[$key]
        if (Test-PortInUse -Port $service.Port) {
            Write-Info "Stopping $($service.Desc)..."
            Stop-ProcessOnPort -Port $service.Port
        }
    }

    if (Test-PortInUse -Port 3000) {
        Write-Info "Stopping frontend..."
        Stop-ProcessOnPort -Port 3000
    }

    if (Test-PortInUse -Port 5000) {
        Write-Info "Stopping Python API..."
        Stop-ProcessOnPort -Port 5000
    }

    Write-Success "All services stopped"
}

function Show-ServiceStatus {
    Write-Host ""
    Write-Host "Service status:" -ForegroundColor Cyan
    Write-Host "==========" -ForegroundColor Cyan

    foreach ($key in $AllBackendServices.Keys | Sort-Object) {
        $service = $AllBackendServices[$key]
        $status = if (Test-PortInUse -Port $service.Port) { "Running" } else { "Stopped" }
        $color = if ($status -eq "Running") { "Green" } else { "Red" }
        Write-Host "  $($service.Desc.PadRight(12)) (Port $($service.Port)): " -NoNewline
        Write-Host $status -ForegroundColor $color
    }

    $frontendStatus = if (Test-PortInUse -Port 3000) { "Running" } else { "Stopped" }
    $color = if ($frontendStatus -eq "Running") { "Green" } else { "Red" }
    Write-Host "  Frontend     (Port 3000): " -NoNewline
    Write-Host $frontendStatus -ForegroundColor $color

    $pythonStatus = if (Test-PortInUse -Port 5000) { "Running" } else { "Stopped" }
    $color = if ($pythonStatus -eq "Running") { "Green" } else { "Red" }
    Write-Host "  Python API   (Port 5000): " -NoNewline
    Write-Host $pythonStatus -ForegroundColor $color

    Write-Host ""
}

if ($Help) {
    Show-Help
    exit 0
}

if ($Stop) {
    Stop-AllServices
    Show-ServiceStatus
    exit 0
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Road Crack Detection System - Service Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($Build) {
    Build-Backend
}

$startBackend = $false
$startFrontend = $false
$servicesToStart = @()

if ($Services.Count -gt 0) {
    $servicesToStart = $Services -split ','
    $startBackend = $true
} elseif ($Backend) {
    $servicesToStart = $AllBackendServices.Keys
    $startBackend = $true
} elseif ($Frontend) {
    $startFrontend = $true
} elseif ($Infra) {
    Write-Warn "Infra services must be started manually."
    Write-Host ""
    Write-Host "Please start:" -ForegroundColor Yellow
    Write-Host "  - MySQL:    port 3307 (db: crack_detection)"
    Write-Host "  - Redis:    port 6379"
    Write-Host "  - MinIO:    port 9000/9001"
    Write-Host "  - RabbitMQ: port 5672/15672"
    Write-Host ""
    Write-Host "See INSTALL-SERVICES.md for details." -ForegroundColor Cyan
    exit 0
} else {
    $servicesToStart = $AllBackendServices.Keys
    $startBackend = $true
    $startFrontend = $true
}

if ($startBackend -and $servicesToStart.Count -gt 0) {
    Write-Info "Starting backend services: $($servicesToStart -join ', ')"
    Write-Host ""

    $orderedServices = $servicesToStart | Where-Object { $_ -ne "gateway" }
    if ($servicesToStart -contains "gateway") {
        $orderedServices += "gateway"
    }

    foreach ($svc in $orderedServices) {
        Start-BackendService -ServiceKey $svc.Trim()
    }
}

if ($startFrontend) {
    Write-Host ""
    Start-Frontend
}

Write-Host ""
Show-ServiceStatus

Write-Host "Tip: Use .\start-all-services.ps1 -Stop to stop all services" -ForegroundColor Gray
Write-Host ""