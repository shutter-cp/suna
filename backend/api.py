# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入FastAPI相关模块
from fastapi import FastAPI, Request, HTTPException, Response, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# 导入项目内部模块
from services import redis
import sentry
from contextlib import asynccontextmanager
from agentpress.thread_manager import ThreadManager
from services.supabase import DBConnection

# 导入标准库模块
from datetime import datetime, timezone
from utils.config import config, EnvMode
import asyncio
from utils.logger import logger, structlog
import time
from collections import OrderedDict
from typing import Dict, Any

# 导入Pydantic和UUID
from pydantic import BaseModel
import uuid

# 导入各个子模块的API
from agent import api as agent_api
from agent import workflows as workflows_api
from sandbox import api as sandbox_api
from services import billing as billing_api
from flags import api as feature_flags_api
from services import transcription as transcription_api
import sys
from services import email_api
from triggers import api as triggers_api

# Windows平台特殊设置
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 初始化数据库连接
db = DBConnection()
# 设置实例ID
instance_id = "single"

# 速率限制器状态
ip_tracker = OrderedDict()
# 最大并发IP数
MAX_CONCURRENT_IPS = 25

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting up FastAPI application with instance ID: {instance_id} in {config.ENV_MODE.value} mode")
    try:
        # 初始化数据库
        await db.initialize()
        
        # 初始化agent API
        agent_api.initialize(
            db,
            instance_id
        )
        
        # 初始化workflow API
        workflows_api.initialize(db, instance_id)
        
        # 初始化sandbox API
        sandbox_api.initialize(db)
        
        # 初始化Redis连接
        from services import redis
        try:
            await redis.initialize_async()
            logger.info("Redis connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            # 即使Redis初始化失败也继续运行
        
        # 初始化triggers API
        triggers_api.initialize(db)

        # 初始化pipedream API
        pipedream_api.initialize(db)
        
        yield
        
        # 清理agent资源
        logger.info("Cleaning up agent resources")
        await agent_api.cleanup()
        
        # 关闭Redis连接
        try:
            logger.info("Closing Redis connection")
            await redis.close()
            logger.info("Redis connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
        
        # 断开数据库连接
        logger.info("Disconnecting from database")
        await db.disconnect()
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)

# 请求日志中间件
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    # 清除上下文变量
    structlog.contextvars.clear_contextvars()

    # 生成请求ID
    request_id = str(uuid.uuid4())
    # 记录开始时间
    start_time = time.time()
    # 获取客户端IP
    client_ip = request.client.host if request.client else "unknown"
    # 获取请求方法和路径
    method = request.method
    path = request.url.path
    # 获取查询参数
    query_params = str(request.query_params)

    # 绑定上下文变量
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        client_ip=client_ip,
        method=method,
        path=path,
        query_params=query_params
    )

    # 记录请求开始
    logger.info(f"Request started: {method} {path} from {client_ip} | Query: {query_params}")
    
    try:
        # 处理请求
        response = await call_next(request)
        # 计算处理时间
        process_time = time.time() - start_time
        # 记录请求完成
        logger.debug(f"Request completed: {method} {path} | Status: {response.status_code} | Time: {process_time:.2f}s")
        return response
    except Exception as e:
        # 记录请求失败
        process_time = time.time() - start_time
        logger.error(f"Request failed: {method} {path} | Error: {str(e)} | Time: {process_time:.2f}s")
        raise

# 设置允许的跨域来源
allowed_origins = ["https://www.suna.so", "https://suna.so"]
allow_origin_regex = None

# 本地开发环境添加localhost
if config.ENV_MODE == EnvMode.LOCAL:
    allowed_origins.append("http://localhost:3000")

# 测试环境添加额外来源
if config.ENV_MODE == EnvMode.STAGING:
    allowed_origins.append("https://staging.suna.so")
    allowed_origins.append("http://localhost:3000")
    allow_origin_regex = r"https://suna-.*-prjcts\.vercel\.app"

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Project-Id", "X-MCP-URL", "X-MCP-Type", "X-MCP-Headers"],
)

# 创建主API路由器
api_router = APIRouter()

# 包含所有子模块的路由
api_router.include_router(workflows_api.router)
api_router.include_router(agent_api.router)
api_router.include_router(sandbox_api.router)
api_router.include_router(billing_api.router)
api_router.include_router(feature_flags_api.router)

# 导入并包含MCP相关路由
from mcp_service import api as mcp_api
from mcp_service import secure_api as secure_mcp_api
from mcp_service import template_api as template_api

api_router.include_router(mcp_api.router)
api_router.include_router(secure_mcp_api.router, prefix="/secure-mcp")
api_router.include_router(template_api.router, prefix="/templates")

# 包含其他服务路由
api_router.include_router(transcription_api.router)
api_router.include_router(email_api.router)

# 包含知识库路由
from knowledge_base import api as knowledge_base_api
api_router.include_router(knowledge_base_api.router)

# 包含触发器路由
api_router.include_router(triggers_api.router)

# 包含pipedream路由
from pipedream import api as pipedream_api
api_router.include_router(pipedream_api.router)

# 健康检查端点
@api_router.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {
        "status": "ok", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instance_id": instance_id
    }

# Docker健康检查端点
@api_router.get("/health-docker")
async def health_check():
    logger.info("Health docker check endpoint called")
    try:
        # 检查Redis连接
        client = await redis.get_client()
        await client.ping()
        # 检查数据库连接
        db = DBConnection()
        await db.initialize()
        db_client = await db.client
        await db_client.table("threads").select("thread_id").limit(1).execute()
        logger.info("Health docker check complete")
        return {
            "status": "ok", 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instance_id": instance_id
        }
    except Exception as e:
        logger.error(f"Failed health docker check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# 将主路由器挂载到应用
app.include_router(api_router, prefix="/api")

# 主程序入口
if __name__ == "__main__":
    import uvicorn
    
    # Windows平台特殊设置
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 设置worker数量
    workers = 4
    
    # 启动服务器
    logger.info(f"Starting server on 0.0.0.0:8000 with {workers} workers")
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000,
        workers=workers,
        loop="asyncio"
    )