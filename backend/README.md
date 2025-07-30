# Suna 后端

## 快速设置

配置后端最简单的方法是使用项目根目录的设置向导：

```bash
cd .. # 如果你在 backend 目录中，导航到项目根目录
python setup.py
```

这将自动配置所有必要的环境变量和服务。

## 运行后端

在 backend 目录中，运行以下命令来停止并启动后端：

```bash
docker compose down && docker compose up --build
```

## 运行单个服务

你可以从 docker-compose 文件中运行单个服务。这在开发过程中特别有用：

### 仅运行 Redis 和 RabbitMQ

```bash
docker compose up redis rabbitmq
```

### 仅运行 API 和 Worker

```bash
docker compose up api worker
```

## 开发设置

对于本地开发，你可能只需要运行 Redis 和 RabbitMQ，同时在本地开发 API。这在以下情况下很有用：

- 你正在对 API 代码进行更改并希望直接测试它们
- 你希望避免每次更改都重建 API 容器
- 你直接在机器上运行 API 服务

要仅运行 Redis 和 RabbitMQ 进行开发：

```bash
docker compose up redis rabbitmq
```

然后你可以使用以下命令在本地运行 API 服务：

```sh
# 在一个终端中
cd backend
uv run api.py

# 在另一个终端中
cd backend
uv run dramatiq --processes 4 --threads 4 run_agent_background
```

### 环境配置

设置向导会自动创建一个包含所有必要配置的 `.env` 文件。如果你需要手动配置或了解设置：

#### 必要的环境变量

```sh
# 环境模式
ENV_MODE=local

# 数据库（Supabase）
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# 基础设施
REDIS_HOST=redis  # 本地运行 API 时使用 'localhost'
REDIS_PORT=6379
RABBITMQ_HOST=rabbitmq  # 本地运行 API 时使用 'localhost'
RABBITMQ_PORT=5672

# LLM 提供商（至少需要提供一个）
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
OPENROUTER_API_KEY=your-openrouter-key
MODEL_TO_USE=anthropic/claude-sonnet-4-20250514

# 搜索和网络抓取
TAVILY_API_KEY=your-tavily-key
FIRECRAWL_API_KEY=your-firecrawl-key
FIRECRAWL_URL=https://api.firecrawl.dev

# 代理执行
DAYTONA_API_KEY=your-daytona-key
DAYTONA_SERVER_URL=https://app.daytona.io/api
DAYTONA_TARGET=us

# 后台作业处理（必需）
QSTASH_URL=https://qstash.upstash.io
QSTASH_TOKEN=your-qstash-token
QSTASH_CURRENT_SIGNING_KEY=your-current-signing-key
QSTASH_NEXT_SIGNING_KEY=your-next-signing-key
WEBHOOK_BASE_URL=https://yourdomain.com

# MCP 配置
MCP_CREDENTIAL_ENCRYPTION_KEY=your-generated-encryption-key

# 可选 API
RAPID_API_KEY=your-rapidapi-key
SMITHERY_API_KEY=your-smithery-api-key

NEXT_PUBLIC_URL=http://localhost:3000
```

单独运行服务时，请确保：

1. 检查你的 `.env` 文件并根据需要调整任何环境变量
2. 确保 Redis 连接设置与你的本地设置匹配（默认：`localhost:6379`）
3. 确保 RabbitMQ 连接设置与你的本地设置匹配（默认：`localhost:5672`）
4. 根据需要更新任何特定于服务的环境变量

### 重要：Redis 主机配置

在 Docker 中运行 Redis 时本地运行 API，你需要在 `.env` 文件中设置正确的 Redis 主机：

- 对于 Docker 到 Docker 通信（在 Docker 中运行两个服务时）：使用 `REDIS_HOST=redis`
- 对于本地到 Docker 通信（本地运行 API 时）：使用 `REDIS_HOST=localhost`

### 重要：RabbitMQ 主机配置

在 Docker 中运行 RabbitMQ 时本地运行 API，你需要在 `.env` 文件中设置正确的 RabbitMQ 主机：

- 对于 Docker 到 Docker 通信（在 Docker 中运行两个服务时）：使用 `RABBITMQ_HOST=rabbitmq`
- 对于本地到 Docker 通信（本地运行 API 时）：使用 `RABBITMQ_HOST=localhost`

本地开发的 `.env` 配置示例：

```sh
REDIS_HOST=localhost #（而不是 'redis'）
REDIS_PORT=6379
REDIS_PASSWORD=

RABBITMQ_HOST=localhost #（而不是 'rabbitmq'）
RABBITMQ_PORT=5672
```

---

## 功能开关

后端包含一个基于 Redis 的功能开关系统，允许你在不部署代码的情况下控制功能可用性。

### 设置

功能开关系统使用现有的 Redis 服务，当 Redis 运行时自动可用。

### CLI 管理

使用 CLI 工具管理功能开关：

```bash
cd backend/flags
python setup.py <command> [arguments]
```

#### 可用命令

**启用功能开关：**

```bash
python setup.py enable test_flag "测试描述"
```

**禁用功能开关：**

```bash
python setup.py disable test_flag
```

**列出所有功能开关：**

```bash
python setup.py list
```

### API 端点

功能开关可通过 REST API 访问：

**获取所有功能开关：**

```bash
GET /feature-flags
```

**获取特定功能开关：**

```bash
GET /feature-flags/{flag_name}
```

响应示例：

```json
{
  "test_flag": {
    "enabled": true,
    "description": "测试标志",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

### 后端集成

在你的 Python 代码中使用功能开关：

```python
from flags.flags import is_enabled

# 检查功能是否启用
if await is_enabled('test_flag'):
    # 功能特定逻辑
    pass

# 使用回退值
enabled = await is_enabled('new_feature', default=False)
```

### 当前功能开关

系统目前支持以下功能开关：

- **`custom_agents`**：控制自定义代理创建和管理
- **`agent_marketplace`**：控制代理市场功能

### 错误处理

功能开关系统包含强大的错误处理：

- 如果 Redis 不可用，标志默认为 `False`
- API 端点在 Redis 错误时返回空对象
- CLI 操作显示清晰的错误消息

### 缓存

- 后端操作是直接 Redis 调用（无缓存）
- 前端包括 5 分钟缓存以提高性能
- 在前端使用 `clearCache()` 强制刷新

---

## 生产设置

对于生产部署，使用以下命令设置资源限制

```sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
