# 不使用 Docker 部署基础设施服务

由于无法使用 Docker Desktop，可以通过以下方式在 Windows 上安装和运行所需的基础设施服务。

## 方式一：使用 Windows 版本的服务（推荐）

### 1. MySQL 8.0

**下载安装:**
1. 访问 https://dev.mysql.com/downloads/mysql/
2. 下载 MySQL Installer for Windows
3. 安装时选择 "Server only" 或 "Custom"

**配置:**
```sql
-- 创建数据库和用户
CREATE DATABASE crack_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'crack_user'@'localhost' IDENTIFIED BY 'crack_password123';
GRANT ALL PRIVILEGES ON crack_detection.* TO 'crack_user'@'localhost';
FLUSH PRIVILEGES;
```

**修改配置文件 my.ini（如需要修改端口为 3307）:**
```ini
[mysqld]
port=3307
```

**启动服务:**
```powershell
# 作为 Windows 服务启动
net start mysql80

# 或手动启动
mysqld --console
```

---

### 2. Redis

**方式 A: 使用 Windows 版 Redis（推荐）**

1. 下载: https://github.com/tporadowski/redis/releases
2. 解压到任意目录，如 `C:\Redis`
3. 启动:
```powershell
cd C:\Redis
.\redis-server.exe
```

**方式 B: 使用 Memurai (Redis 兼容)**
- 下载: https://www.memurai.com/
- 安装后自动作为 Windows 服务运行

---

### 3. MinIO

**下载安装:**
1. 访问 https://min.io/download#/windows
2. 下载 minio.exe

**启动:**
```powershell
# 设置数据目录
$env:MINIO_ROOT_USER = "minioadmin"
$env:MINIO_ROOT_PASSWORD = "minioadmin123"

# 启动 MinIO
.\minio.exe server C:\minio-data --console-address ":9001"
```

**创建 Bucket:**
1. 访问 http://localhost:9001
2. 使用 minioadmin / minioadmin123 登录
3. 创建名为 `crack-detection` 的 bucket
4. 设置 bucket 为 public（可选）

---

### 4. RabbitMQ

**下载安装:**
1. 先安装 Erlang: https://www.erlang.org/downloads
2. 下载 RabbitMQ: https://www.rabbitmq.com/install-windows.html
3. 安装完成后启用管理插件:
```powershell
cd "C:\Program Files\RabbitMQ Server\rabbitmq_server-xxx\sbin"
.\rabbitmq-plugins.bat enable rabbitmq_management
```

**创建用户:**
```powershell
.\rabbitmqctl.bat add_user crack_user crack_password123
.\rabbitmqctl.bat set_permissions -p / crack_user ".*" ".*" ".*"
.\rabbitmqctl.bat set_user_tags crack_user administrator
```

**访问管理界面:** http://localhost:15672

---

## 方式二：使用便携式启动脚本

我们提供了一个 PowerShell 脚本来管理这些服务：

### 创建基础设施启动脚本

将以下服务的可执行文件放在统一目录，例如 `C:\DevServices\`:
```
C:\DevServices\
├── mysql\
├── redis\
│   └── redis-server.exe
├── minio\
│   └── minio.exe
└── rabbitmq\
```

### 使用启动脚本

```powershell
# 启动所有基础设施
.\start-infrastructure.ps1

# 停止所有基础设施
.\start-infrastructure.ps1 -Stop
```

---

## 方式三：使用 WSL2 (Windows Subsystem for Linux)

如果你安装了 WSL2，可以在 Linux 环境中运行这些服务：

```bash
# 在 WSL2 Ubuntu 中
sudo apt update

# 安装 MySQL
sudo apt install mysql-server
sudo systemctl start mysql

# 安装 Redis
sudo apt install redis-server
sudo systemctl start redis

# MinIO
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server ~/minio-data

# RabbitMQ
sudo apt install rabbitmq-server
sudo systemctl start rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
```

---

## 服务端口配置

确保项目配置文件中的端口与实际服务端口一致：

| 服务 | 默认端口 | 配置位置 |
|------|---------|---------|
| MySQL | 3307 | `cloud-backend/cloud-auth/src/main/resources/application.yml` |
| Redis | 6379 | 同上 |
| MinIO | 9000/9001 | 同上 |
| RabbitMQ | 5672/15672 | 同上 |

如果使用标准端口 (如 MySQL 3306)，需要修改配置文件中的连接地址。

---

## 快速检查服务状态

```powershell
# 检查端口占用
netstat -an | findstr "3306 3307 6379 9000 5672"

# 或使用 PowerShell
Test-NetConnection -ComputerName localhost -Port 3307
Test-NetConnection -ComputerName localhost -Port 6379
Test-NetConnection -ComputerName localhost -Port 9000
Test-NetConnection -ComputerName localhost -Port 5672
```

---

## 数据库初始化

首次启动后，需要初始化数据库表结构：

```powershell
# 导入初始化 SQL
mysql -h localhost -P 3307 -u crack_user -pcrack_password123 crack_detection < docker\init-db\01-init.sql
```

或在 MySQL 客户端中执行 `docker\init-db\01-init.sql` 文件的内容。
