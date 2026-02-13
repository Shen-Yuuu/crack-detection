# 道路裂纹检测系统 - 后端服务

## 项目结构

```
cloud-backend/
├── cloud-common/      # 公共模块 - 工具类、实体、配置
├── cloud-auth/        # 认证服务 - 用户登录、注册、JWT
├── cloud-gateway/     # API网关 - 路由、鉴权、限流
├── cloud-dataset/     # 数据集服务 - 数据集、图像管理
├── cloud-inference/   # 推理服务 - 调用Python AI服务
├── cloud-visual/      # 可视化服务 - 叠加图、热力图
├── cloud-report/      # 报告服务 - PDF/Excel生成
├── docker/            # Docker配置文件
│   ├── docker-compose.yml
│   └── init-db/
│       └── 01-init.sql
└── pom.xml            # 父POM
```

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Spring Boot | 3.2.0 | 核心框架 |
| Spring Cloud | 2023.0.0 | 微服务基础设施 |
| Spring Cloud Alibaba | 2023.0.0.0-RC1 | Nacos服务发现 |
| MyBatis Plus | 3.5.5 | ORM框架 |
| MySQL | 8.0+ | 主数据库 |
| Redis | 7+ | 缓存 |
| MinIO | Latest | 对象存储 |
| RabbitMQ | 3+ | 消息队列 |

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| cloud-gateway | 8080 | API网关入口 |
| cloud-auth | 8081 | 认证服务 |
| cloud-dataset | 8082 | 数据集服务 |
| cloud-inference | 8083 | 推理服务 |
| cloud-visual | 8084 | 可视化服务 |
| cloud-report | 8085 | 报告服务 |
| Python API | 8090 | Python推理API |

## 快速开始

### 1. 启动基础设施

```bash
cd docker
docker-compose up -d
```

这将启动：
- MySQL (3306)
- Redis (6379)
- MinIO (9000/9001)
- RabbitMQ (5672/15672)
- Nacos (8848)

### 2. 等待服务就绪

```bash
# 检查MySQL
docker exec crack-mysql mysqladmin ping -h localhost -u root -proot123456

# 检查MinIO
curl http://localhost:9000/minio/health/live

# 检查Nacos
curl http://localhost:8848/nacos/v1/console/health/liveness
```

### 3. 启动Python推理服务

```bash
cd python-inference

# 安装依赖
pip install -r requirements_api.txt

# 启动服务
python api_server.py
```

### 4. 编译Java项目

```bash
cd cloud-backend

# 编译整个项目
mvn clean install -DskipTests
```

### 5. 启动微服务

按以下顺序启动各服务：

```bash
# 1. 启动认证服务
cd cloud-auth
mvn spring-boot:run

# 2. 启动网关服务
cd cloud-gateway
mvn spring-boot:run

# 3. 启动其他服务（可并行）
cd cloud-dataset && mvn spring-boot:run
cd cloud-inference && mvn spring-boot:run
cd cloud-visual && mvn spring-boot:run
cd cloud-report && mvn spring-boot:run
```

或者使用打包后的jar：

```bash
java -jar cloud-auth/target/cloud-auth-1.0.0.jar
java -jar cloud-gateway/target/cloud-gateway-1.0.0.jar
# ...
```

## API文档

启动服务后访问 Swagger UI：

- 认证服务: http://localhost:8081/doc.html
- 数据集服务: http://localhost:8082/doc.html
- 推理服务: http://localhost:8083/doc.html
- 可视化服务: http://localhost:8084/doc.html
- 报告服务: http://localhost:8085/doc.html
- Python API: http://localhost:8090/docs

## 主要API

### 认证API

```
POST /api/v1/auth/register    # 用户注册
POST /api/v1/auth/login       # 用户登录
POST /api/v1/auth/logout      # 用户登出
POST /api/v1/auth/refresh     # 刷新Token
GET  /api/v1/auth/user/info   # 获取用户信息
```

### 数据集API

```
POST   /api/v1/dataset                    # 创建数据集
GET    /api/v1/dataset                    # 数据集列表
GET    /api/v1/dataset/{id}               # 数据集详情
DELETE /api/v1/dataset/{id}               # 删除数据集
POST   /api/v1/dataset/{id}/images        # 上传图像
POST   /api/v1/dataset/{id}/images/batch  # 批量上传
GET    /api/v1/dataset/{id}/images        # 图像列表
```

### 推理API

```
POST /api/v1/inference/detect     # 单张检测
POST /api/v1/inference/batch      # 批量检测
GET  /api/v1/inference/result/{jobId}  # 获取结果
POST /api/v1/inference/jobs/{jobId}/retry  # 重试任务
```

### 可视化API

```
POST /api/v1/visual/overlay/{resultId}    # 生成叠加图
POST /api/v1/visual/heatmap/{resultId}    # 生成热力图
GET  /api/v1/visual/statistics/{resultId} # 获取统计
```

### 报告API

```
POST   /api/v1/report/generate        # 生成报告
GET    /api/v1/report                 # 报告列表
GET    /api/v1/report/{id}            # 报告详情
DELETE /api/v1/report/{id}            # 删除报告
GET    /api/v1/report/{id}/download   # 下载报告
```

## 配置说明

### 数据库配置

修改各服务的 `application.yml`：

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/crack_detection?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai&allowPublicKeyRetrieval=true
    username: crack_user
    password: crack_password123
```

### MinIO配置

```yaml
minio:
  endpoint: http://localhost:9000
  access-key: minioadmin
  secret-key: minioadmin123
  bucket-name: crack-detection
```

### JWT配置

```yaml
jwt:
  secret: your-secret-key-at-least-256-bits
```

## 开发指南

### 添加新接口

1. 在对应服务的 `controller` 包添加 Controller
2. 在 `service` 包定义接口和实现
3. 在 `mapper` 包添加 MyBatis Mapper
4. 在 `dto` 包定义请求/响应对象

### 跨服务调用

使用 OpenFeign：

```java
@FeignClient(name = "cloud-auth", url = "http://localhost:8081")
public interface AuthClient {
    @GetMapping("/api/v1/auth/user/{id}")
    Result<UserInfo> getUserInfo(@PathVariable Long id);
}
```

## 故障排查

### 常见问题

1. **数据库连接失败**
   - 检查 MySQL 是否启动
   - 验证连接配置是否正确

2. **MinIO 上传失败**
   - 确保 bucket 已创建
   - 检查访问凭据

3. **推理服务无响应**
   - 确保 Python API 已启动
   - 检查模型文件是否存在

## License

MIT License
