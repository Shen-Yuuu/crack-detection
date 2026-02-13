# ============================================
# 基础设施服务启动脚本 (不使用 Docker)
# ============================================
# 用法:
#   .\start-infrastructure.ps1           # 启动所有基础设施
#   .\start-infrastructure.ps1 -Stop     # 停止所有基础设施
#   .\start-infrastructure.ps1 -Status   # 查看状态
# ============================================

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Help
)

# ============================================
# 配置 - 请根据实际安装路径修改
# ============================================
$Config = @{
    # Redis 配置
    Redis = @{
        Path = "C:\Redis\redis-server.exe"        # Redis 可执行文件路径
        Port = 6379
        Enabled = $true
    }
    # MinIO 配置
    MinIO = @{
        Path = "C:\minio\minio.exe"               # MinIO 可执行文件路径
        DataDir = "C:\minio-data"                  # 数据目录
        Port = 9000
        ConsolePort = 9001
        User = "minioadmin"
        Password = "minioadmin123"
        Enabled = $true
    }
    # MySQL 配置 (如果作为 Windows 服务安装)
    MySQL = @{
        ServiceName = "MySQL80"                    # Windows 服务名称
        Port = 3307
        Enabled = $true
    }
    # RabbitMQ 配置 (如果作为 Windows 服务安装)
    RabbitMQ = @{
        ServiceName = "RabbitMQ"                   # Windows 服务名称
        Port = 5672
        ManagementPort = 15672
        Enabled = $true
    }
}

# 颜色输出
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# 检查端口
function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

# 显示帮助
function Show-Help {
    Write-Host ""
    Write-Host "基础设施服务启动脚本" -ForegroundColor Cyan
    Write-Host "========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "用法:" -ForegroundColor Yellow
    Write-Host "  .\start-infrastructure.ps1           # 启动所有服务"
    Write-Host "  .\start-infrastructure.ps1 -Stop     # 停止所有服务"
    Write-Host "  .\start-infrastructure.ps1 -Status   # 查看服务状态"
    Write-Host ""
    Write-Host "配置:" -ForegroundColor Yellow
    Write-Host "  请编辑脚本开头的 `$Config 变量，设置各服务的安装路径"
    Write-Host ""
}

# 显示状态
function Show-Status {
    Write-Host ""
    Write-Host "基础设施服务状态:" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    
    # MySQL
    $mysqlStatus = if (Test-PortInUse -Port $Config.MySQL.Port) { "运行中" } else { "已停止" }
    $color = if ($mysqlStatus -eq "运行中") { "Green" } else { "Red" }
    Write-Host "  MySQL     (端口 $($Config.MySQL.Port)):  " -NoNewline
    Write-Host $mysqlStatus -ForegroundColor $color
    
    # Redis
    $redisStatus = if (Test-PortInUse -Port $Config.Redis.Port) { "运行中" } else { "已停止" }
    $color = if ($redisStatus -eq "运行中") { "Green" } else { "Red" }
    Write-Host "  Redis     (端口 $($Config.Redis.Port)):  " -NoNewline
    Write-Host $redisStatus -ForegroundColor $color
    
    # MinIO
    $minioStatus = if (Test-PortInUse -Port $Config.MinIO.Port) { "运行中" } else { "已停止" }
    $color = if ($minioStatus -eq "运行中") { "Green" } else { "Red" }
    Write-Host "  MinIO     (端口 $($Config.MinIO.Port)):  " -NoNewline
    Write-Host $minioStatus -ForegroundColor $color
    
    # RabbitMQ
    $rmqStatus = if (Test-PortInUse -Port $Config.RabbitMQ.Port) { "运行中" } else { "已停止" }
    $color = if ($rmqStatus -eq "运行中") { "Green" } else { "Red" }
    Write-Host "  RabbitMQ  (端口 $($Config.RabbitMQ.Port)):  " -NoNewline
    Write-Host $rmqStatus -ForegroundColor $color
    
    Write-Host ""
}

# 启动 Redis
function Start-Redis {
    if (-not $Config.Redis.Enabled) { return }
    
    if (Test-PortInUse -Port $Config.Redis.Port) {
        Write-Success "Redis 已在运行"
        return
    }
    
    if (Test-Path $Config.Redis.Path) {
        Write-Info "启动 Redis..."
        Start-Process -FilePath $Config.Redis.Path -WindowStyle Hidden
        Start-Sleep -Seconds 2
        if (Test-PortInUse -Port $Config.Redis.Port) {
            Write-Success "Redis 启动成功"
        } else {
            Write-Err "Redis 启动失败"
        }
    } else {
        Write-Warn "Redis 未安装或路径不正确: $($Config.Redis.Path)"
        Write-Host "  下载地址: https://github.com/tporadowski/redis/releases" -ForegroundColor Gray
    }
}

# 启动 MinIO
function Start-MinIO {
    if (-not $Config.MinIO.Enabled) { return }
    
    if (Test-PortInUse -Port $Config.MinIO.Port) {
        Write-Success "MinIO 已在运行"
        return
    }
    
    if (Test-Path $Config.MinIO.Path) {
        Write-Info "启动 MinIO..."
        
        # 创建数据目录
        if (-not (Test-Path $Config.MinIO.DataDir)) {
            New-Item -ItemType Directory -Path $Config.MinIO.DataDir -Force | Out-Null
        }
        
        # 设置环境变量并启动
        $env:MINIO_ROOT_USER = $Config.MinIO.User
        $env:MINIO_ROOT_PASSWORD = $Config.MinIO.Password
        
        Start-Process -FilePath $Config.MinIO.Path -ArgumentList "server", $Config.MinIO.DataDir, "--console-address", ":$($Config.MinIO.ConsolePort)" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        
        if (Test-PortInUse -Port $Config.MinIO.Port) {
            Write-Success "MinIO 启动成功"
            Write-Host "  控制台: http://localhost:$($Config.MinIO.ConsolePort)" -ForegroundColor Gray
        } else {
            Write-Err "MinIO 启动失败"
        }
    } else {
        Write-Warn "MinIO 未安装或路径不正确: $($Config.MinIO.Path)"
        Write-Host "  下载地址: https://min.io/download#/windows" -ForegroundColor Gray
    }
}

# 启动 MySQL (Windows 服务)
function Start-MySQL {
    if (-not $Config.MySQL.Enabled) { return }
    
    if (Test-PortInUse -Port $Config.MySQL.Port) {
        Write-Success "MySQL 已在运行"
        return
    }
    
    $service = Get-Service -Name $Config.MySQL.ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        Write-Info "启动 MySQL 服务..."
        Start-Service -Name $Config.MySQL.ServiceName -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
        if (Test-PortInUse -Port $Config.MySQL.Port) {
            Write-Success "MySQL 启动成功"
        } else {
            Write-Err "MySQL 启动失败"
        }
    } else {
        Write-Warn "MySQL Windows 服务未找到: $($Config.MySQL.ServiceName)"
        Write-Host "  请确保 MySQL 已正确安装，或手动启动 MySQL" -ForegroundColor Gray
    }
}

# 启动 RabbitMQ (Windows 服务)
function Start-RabbitMQ {
    if (-not $Config.RabbitMQ.Enabled) { return }
    
    if (Test-PortInUse -Port $Config.RabbitMQ.Port) {
        Write-Success "RabbitMQ 已在运行"
        return
    }
    
    $service = Get-Service -Name $Config.RabbitMQ.ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        Write-Info "启动 RabbitMQ 服务..."
        Start-Service -Name $Config.RabbitMQ.ServiceName -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
        if (Test-PortInUse -Port $Config.RabbitMQ.Port) {
            Write-Success "RabbitMQ 启动成功"
            Write-Host "  管理界面: http://localhost:$($Config.RabbitMQ.ManagementPort)" -ForegroundColor Gray
        } else {
            Write-Err "RabbitMQ 启动失败"
        }
    } else {
        Write-Warn "RabbitMQ Windows 服务未找到: $($Config.RabbitMQ.ServiceName)"
        Write-Host "  请确保 RabbitMQ 已正确安装" -ForegroundColor Gray
    }
}

# 停止服务
function Stop-AllInfra {
    Write-Info "停止基础设施服务..."
    
    # 停止 Redis
    Get-Process redis-server -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # 停止 MinIO
    Get-Process minio -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # 停止 MySQL 服务
    $mysqlService = Get-Service -Name $Config.MySQL.ServiceName -ErrorAction SilentlyContinue
    if ($mysqlService -and $mysqlService.Status -eq 'Running') {
        Stop-Service -Name $Config.MySQL.ServiceName -Force -ErrorAction SilentlyContinue
    }
    
    # 停止 RabbitMQ 服务
    $rmqService = Get-Service -Name $Config.RabbitMQ.ServiceName -ErrorAction SilentlyContinue
    if ($rmqService -and $rmqService.Status -eq 'Running') {
        Stop-Service -Name $Config.RabbitMQ.ServiceName -Force -ErrorAction SilentlyContinue
    }
    
    Write-Success "基础设施服务已停止"
}

# ============================================
# 主逻辑
# ============================================

if ($Help) {
    Show-Help
    exit 0
}

if ($Status) {
    Show-Status
    exit 0
}

if ($Stop) {
    Stop-AllInfra
    Show-Status
    exit 0
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  基础设施服务启动脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 启动服务
Start-MySQL
Start-Redis
Start-MinIO
Start-RabbitMQ

Write-Host ""
Show-Status

Write-Host "提示: 首次运行后请创建 MinIO bucket 'crack-detection'" -ForegroundColor Yellow
Write-Host "      访问 http://localhost:9001 进行配置" -ForegroundColor Yellow
Write-Host ""
