# ============================================
# Road Crack Detection System - Startup Script
# ============================================
# Usage:
#   .\start.ps1              # Start all services
#   .\start.ps1 -Dev         # Dev mode (minimal services)
#   .\start.ps1 -Stop        # Stop all services
#   .\start.ps1 -Status      # Show status
# ============================================

param(
    [switch]$Dev,         # Dev mode: gateway, auth, frontend
    [switch]$Stop,        # Stop all
    [switch]$Status,      # Show status
    [switch]$Build,       # Build backend
    [switch]$Help
)

$RootPath = $PSScriptRoot
$BackendScript = Join-Path $RootPath "cloud-backend\start-all-services.ps1"
$InfraScript = Join-Path $RootPath "start-infrastructure.ps1"

function Write-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Road Crack Detection System" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Write-Banner
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\start.ps1              # Start all services"
    Write-Host "  .\start.ps1 -Dev         # Dev mode (gateway + auth + frontend)"
    Write-Host "  .\start.ps1 -Build       # Build then start"
    Write-Host "  .\start.ps1 -Stop        # Stop all services"
    Write-Host "  .\start.ps1 -Status      # Show service status"
    Write-Host ""
    Write-Host "Services:" -ForegroundColor Yellow
    Write-Host "  Infra: MySQL, Redis, MinIO, RabbitMQ"
    Write-Host "  Backend: gateway, auth, dataset, inference, visual, report"
    Write-Host "  Frontend: Vue3 + Vite (port 3000)"
    Write-Host "  Python API: FastAPI (port 5000)"
    Write-Host ""
}

function Show-AllStatus {
    Write-Banner

    if (Test-Path $InfraScript) {
        & $InfraScript -Status
    }

    if (Test-Path $BackendScript) {
        Write-Host "Backend status:" -ForegroundColor Cyan
        Write-Host "==================" -ForegroundColor Cyan

        $services = @(
            @{Name="Gateway"; Port=8080},
            @{Name="Auth"; Port=8081},
            @{Name="Dataset"; Port=8082},
            @{Name="Inference"; Port=8083},
            @{Name="Visual"; Port=8084},
            @{Name="Report"; Port=8085}
        )

        foreach ($svc in $services) {
            $conn = Get-NetTCPConnection -LocalPort $svc.Port -ErrorAction SilentlyContinue
            $status = if ($conn) { "Running" } else { "Stopped" }
            $color = if ($conn) { "Green" } else { "Red" }
            Write-Host "  $($svc.Name.PadRight(12)) (Port $($svc.Port)): " -NoNewline
            Write-Host $status -ForegroundColor $color
        }

        $frontConn = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
        $frontStatus = if ($frontConn) { "Running" } else { "Stopped" }
        $color = if ($frontConn) { "Green" } else { "Red" }
        Write-Host "  Frontend     (Port 3000): " -NoNewline
        Write-Host $frontStatus -ForegroundColor $color

        $pyConn = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
        $pyStatus = if ($pyConn) { "Running" } else { "Stopped" }
        $color = if ($pyConn) { "Green" } else { "Red" }
        Write-Host "  Python API   (Port 5000): " -NoNewline
        Write-Host $pyStatus -ForegroundColor $color
    }

    Write-Host ""
}

if ($Help) {
    Show-Help
    exit 0
}

if ($Status) {
    Show-AllStatus
    exit 0
}

if ($Stop) {
    Write-Banner
    Write-Host "Stopping all services..." -ForegroundColor Yellow

    if (Test-Path $BackendScript) {
        & $BackendScript -Stop
    }

    if (Test-Path $InfraScript) {
        & $InfraScript -Stop
    }

    Write-Host ""
    Write-Host "All services stopped" -ForegroundColor Green
    exit 0
}

Write-Banner

Write-Host "[1/3] Checking infrastructure services..." -ForegroundColor Yellow
$mysqlOk = Get-NetTCPConnection -LocalPort 3307 -ErrorAction SilentlyContinue
$redisOk = Get-NetTCPConnection -LocalPort 6379 -ErrorAction SilentlyContinue

if (-not $mysqlOk -or -not $redisOk) {
    Write-Host ""
    Write-Host "  WARN: Infrastructure services are not fully started" -ForegroundColor Yellow
    Write-Host "  Please start MySQL (port 3307) and Redis (port 6379)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Options:" -ForegroundColor Cyan
    Write-Host "    1. Run .\start-infrastructure.ps1"
    Write-Host "    2. Start MySQL and Redis manually"
    Write-Host "    3. See INSTALL-SERVICES.md for details"
    Write-Host ""

    $continue = Read-Host "Continue starting backend services? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        exit 1
    }
}

Write-Host "  OK: Infrastructure check complete" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] Starting backend services..." -ForegroundColor Yellow

$backendArgs = @{}
if ($Build) {
    $backendArgs.Build = $true
}

if ($Dev) {
    $backendArgs.Services = "gateway,auth,dataset"
} else {
    $backendArgs.Backend = $true
}

if (Test-Path $BackendScript) {
    & $BackendScript @backendArgs
} else {
    Write-Host "  WARN: Backend script not found" -ForegroundColor Yellow
}

Write-Host ""

Write-Host "[3/3] Starting frontend..." -ForegroundColor Yellow

$frontendPath = Join-Path $RootPath "cloud-frontend"
if (Test-Path $frontendPath) {
    Push-Location $frontendPath
    try {
        if (-not (Test-Path "node_modules")) {
            Write-Host "  Installing frontend dependencies..." -ForegroundColor Gray
            & npm install
        }
        Start-Process -FilePath "cmd" -ArgumentList "/c", "npm run dev" -WorkingDirectory $frontendPath
        Write-Host "  Frontend starting..." -ForegroundColor Green
    } finally {
        Pop-Location
    }
} else {
    Write-Host "  WARN: Frontend directory not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Startup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  API Gateway: http://localhost:8080" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8081/doc.html" -ForegroundColor Gray
Write-Host ""
Write-Host "  Use .\start.ps1 -Status to view service status" -ForegroundColor Gray
Write-Host "  Use .\start.ps1 -Stop to stop all services" -ForegroundColor Gray
Write-Host ""