# Suna 自托管指南

本指南提供了设置和托管您自己的 Suna 实例的详细说明，Suna 是一个开源的通用 AI 代理。

## 目录

- [概述](#概述)
- [前提条件](#前提条件)
- [安装步骤](#安装步骤)
- [手动配置](#手动配置)
- [安装后步骤](#安装后步骤)
- [故障排除](#故障排除)

## 概述

Suna 由四个主要组件组成：

1. **后端 API** - Python/FastAPI 服务，用于 REST 端点、线程管理和 LLM 集成
2. **后端 Worker** - Python/Dramatiq worker 服务，用于处理代理任务
3. **前端** - Next.js/React 应用程序，提供用户界面
4. **代理 Docker** - 每个代理的隔离执行环境
5. **Supabase 数据库** - 处理数据持久化和身份验证

## 前提条件

在开始安装过程之前，您需要设置以下内容：

### 1. Supabase 项目

1. 在 [Supabase](https://supabase.com/) 创建账户
2. 创建一个新项目
3. 记下以下信息（在项目设置 → API 中找到）：
   - 项目 URL（例如，`https://abcdefg.supabase.co`）
   - API 密钥（匿名密钥和服务角色密钥）

### 2. API 密钥

获取以下 API 密钥：

#### 必需

- **LLM 提供商**（至少一个）：

  - [Anthropic](https://console.anthropic.com/) - 推荐用于最佳性能
  - [OpenAI](https://platform.openai.com/)
  - [Groq](https://console.groq.com/)
  - [OpenRouter](https://openrouter.ai/)
  - [AWS Bedrock](https://aws.amazon.com/bedrock/)

- **搜索和网络抓取**：

  - [Tavily](https://tavily.com/) - 用于增强搜索功能
  - [Firecrawl](https://firecrawl.dev/) - 用于网络抓取功能

- **代理执行**：
  - [Daytona](https://app.daytona.io/) - 用于安全代理执行

- **后台作业处理**：
  - [QStash](https://console.upstash.com/qstash) - 用于工作流、自动化任务和 webhook 处理

#### 可选

- **RapidAPI** - 用于访问额外的 API 服务（启用 LinkedIn 抓取和其他工具）
- **Smithery** - 用于自定义代理和工作流（[获取 API 密钥](https://smithery.ai/)）

### 3. 必需软件

确保您的系统上安装了以下工具：

- **[Docker](https://docs.docker.com/get-docker/)**
- **[Supabase CLI](https://supabase.com/docs/guides/local-development/cli/getting-started)**
- **[Git](https://git-scm.com/downloads)**
- **[Python 3.11](https://www.python.org/downloads/)**

对于手动设置，您还需要：

- **[uv](https://docs.astral.sh/uv/)**
- **[Node.js & npm](https://nodejs.org/en/download/)**

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/kortix-ai/suna.git
cd suna
```

### 2. 运行设置向导

设置向导将指导您完成安装过程：

```bash
python setup.py
```

向导将：

- 检查是否安装了所有必需的工具
- 收集您的 API 密钥和配置信息
- 设置 Supabase 数据库
- 配置环境文件
- 安装依赖项
- 使用您首选的方法启动 Suna

设置向导有 14 个步骤，包括进度保存，因此如果中断，您可以恢复。

### 3. Supabase 配置

在设置期间，您需要：

1. 登录到 Supabase CLI
2. 将本地项目链接到您的 Supabase 项目
3. 推送数据库迁移
4. 在 Supabase 中手动公开 'basejump' 模式：
   - 转到您的 Supabase 项目
   - 导航到项目设置 → API
   - 将 'basejump' 添加到公开模式部分

### 4. Daytona 配置

作为设置的一部分，您需要：

1. 创建 Daytona 账户
2. 生成 API 密钥
3. 创建快照：
   - 名称：`kortix/suna:0.1.3`
   - 镜像名称：`kortix/suna:0.1.3`
   - 入口点：`/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf`

### 5. QStash 配置

QStash 是后台作业处理、工作流和 webhook 处理所必需的：

1. 在 [Upstash 控制台](https://console.upstash.com/qstash) 创建账户
2. 获取您的 QStash 令牌和签名密钥
3. 配置公开可访问的 webhook 基本 URL 用于工作流回调

## 手动配置

如果您希望手动配置安装，或者需要在安装后修改配置，以下是您需要了解的内容：

### 后端配置 (.env)

后端配置存储在 `backend/.env` 中

示例配置：

```sh
# 环境模式
ENV_MODE=local

# 数据库
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# REDIS
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_SSL=false

# RABBITMQ
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672

# LLM 提供商
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
OPENROUTER_API_KEY=your-openrouter-key
MODEL_TO_USE=anthropic/claude-sonnet-4-20250514

# 网络搜索
TAVILY_API_KEY=your-tavily-key

# 网络抓取
FIRECRAWL_API_KEY=your-firecrawl-key
FIRECRAWL_URL=https://api.firecrawl.dev

# 沙箱容器提供商
DAYTONA_API_KEY=dtn_0881fbd32371e9d5d6200b3c9f8eb344b9d06341401f91d0bfde1e249bd892cd
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
SMITHERY_API_KEY=your-smithery-key

NEXT_PUBLIC_URL=http://localhost:3000
```

### 前端配置 (.env.local)

前端配置存储在 `frontend/.env.local` 中，包括：

- Supabase 连接详情
- 后端 API URL

示例配置：

```sh
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000/api
NEXT_PUBLIC_URL=http://localhost:3000
NEXT_PUBLIC_ENV_MODE=LOCAL
```

## 安装后步骤

完成安装后，您需要：

1. **创建账户** - 使用 Supabase 身份验证创建您的第一个账户
2. **验证安装** - 检查所有组件是否正确运行

## 启动选项

Suna 可以通过两种方式启动：

### 1. 使用 Docker Compose（推荐）

此方法在 Docker 容器中启动所有必需的服务：

```bash
docker compose up -d # 稍后使用 `docker compose down` 停止
# 或
python start.py # 稍后使用相同的命令停止
```

### 2. 手动启动

此方法需要您分别启动每个组件：

1. 启动 Redis 和 RabbitMQ（后端必需）：

```bash
docker compose up redis rabbitmq -d
# 或
python start.py # 稍后使用相同的命令停止
```

2. 启动前端（在一个终端中）：

```bash
cd frontend
npm run dev
```

3. 启动后端（在另一个终端中）：

```bash
cd backend
uv run api.py
```

4. 启动工作进程（在另一个终端中）：

```bash
cd backend
uv run dramatiq run_agent_background
```

## 故障排除

### 常见问题

1. **Docker 服务未启动**

   - 检查 Docker 日志：`docker compose logs`
   - 确保 Docker 正确运行
   - 验证端口可用性（前端 3000，后端 8000）

2. **数据库连接问题**

   - 验证 Supabase 配置
   - 检查 'basejump' 模式是否在 Supabase 中公开

3. **LLM API 密钥问题**

   - 验证 API 密钥是否正确输入
   - 检查 API 使用限制或限制

4. **Daytona 连接问题**

   - 验证 Daytona API 密钥
   - 检查容器镜像是否正确配置

5. **QStash/Webhook 问题**

   - 验证 QStash 令牌和签名密钥
   - 确保 webhook 基本 URL 公开可访问
   - 检查 QStash 控制台中的传递状态

6. **设置向导问题**

   - 删除 `.setup_progress` 文件以重置设置向导
   - 检查是否安装了所有必需的工具并可访问

### 日志

查看日志并诊断问题：

```bash
# Docker Compose 日志
docker compose logs -f

# 前端日志（手动设置）
cd frontend
npm run dev -- --turbopack

# 后端日志（手动设置）
cd backend
uv run api.py

# 工作进程日志（手动设置）
cd backend
uv run dramatiq run_agent_background
```

### 恢复设置

如果设置向导中断，您可以通过运行以下命令从上次中断的地方继续：

```bash
python setup.py
```

向导将检测您的进度并从最后完成的步骤继续。

---

如需进一步帮助，请加入 [Suna Discord 社区](https://discord.gg/Py6pCBUUPw) 或查看 [GitHub 仓库](https://github.com/kortix-ai/suna) 获取更新和问题。
