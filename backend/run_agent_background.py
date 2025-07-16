import dotenv
dotenv.load_dotenv(".env")

import sentry
import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Optional
from services import redis
from agent.run import run_agent
from utils.logger import logger, structlog
import dramatiq
import uuid
from agentpress.thread_manager import ThreadManager
from services.supabase import DBConnection
from services import redis
from dramatiq.brokers.rabbitmq import RabbitmqBroker
import os
from services.langfuse import langfuse
from utils.retry import retry

import sentry_sdk
from typing import Dict, Any

rabbitmq_host = os.getenv('RABBITMQ_HOST', 'rabbitmq')
rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
rabbitmq_broker = RabbitmqBroker(host=rabbitmq_host, port=rabbitmq_port, middleware=[dramatiq.middleware.AsyncIO()])
dramatiq.set_broker(rabbitmq_broker)


_initialized = False
db = DBConnection()
instance_id = "single"

async def initialize():
    """
    初始化agent API，从主API获取资源
    功能：
    1. 设置全局变量(db, instance_id, _initialized)
    2. 如果instance_id未设置，生成一个8字符的UUID作为实例ID
    3. 初始化Redis异步连接
    4. 初始化数据库连接
    5. 标记初始化完成状态
    """
    """Initialize the agent API with resources from the main API."""
    global db, instance_id, _initialized

    if not instance_id:
        instance_id = str(uuid.uuid4())[:8]
    await retry(lambda: redis.initialize_async())
    await db.initialize()

    _initialized = True
    logger.info(f"Initialized agent API with instance ID: {instance_id}")

@dramatiq.actor
async def check_health(key: str):
    """Run the agent in the background using Redis for state."""
    structlog.contextvars.clear_contextvars()
    await redis.set(key, "healthy", ex=redis.REDIS_KEY_TTL)

@dramatiq.actor
async def run_agent_background(
    agent_run_id: str,
    thread_id: str,
    instance_id: str, # Use the global instance ID passed during initialization
    project_id: str,
    model_name: str,
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
    stream: bool,
    enable_context_manager: bool,
    agent_config: Optional[dict] = None,
    is_agent_builder: Optional[bool] = False,
    target_agent_id: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """Run the agent in the background using Redis for state."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
        thread_id=thread_id,
        request_id=request_id,
    )

    try:
        await initialize()
    except Exception as e:
        logger.critical(f"Failed to initialize Redis connection: {e}")
        raise e

    # 幂等性检查：防止重复运行
    # 创建Redis锁键，格式为"agent_run_lock:{agent_run_id}"
    run_lock_key = f"agent_run_lock:{agent_run_id}"
    
    # 尝试获取当前代理运行的锁
    # 使用NX参数确保只有锁不存在时才能设置成功
    # 设置过期时间为redis.REDIS_KEY_TTL
    lock_acquired = await redis.set(run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL)
    
    if not lock_acquired:
        # 如果获取锁失败，检查是否已有其他实例在处理此运行
        existing_instance = await redis.get(run_lock_key)
        if existing_instance:
            # 如果锁已存在且有值，说明其他实例正在处理
            # 处理字节类型和字符串类型的existing_instance
            logger.info(f"Agent run {agent_run_id} is already being processed by instance {existing_instance.decode() if isinstance(existing_instance, bytes) else existing_instance}. Skipping duplicate execution.")
            return
        else:
            # 锁存在但没有值，可能是异常情况，再次尝试获取锁
            lock_acquired = await redis.set(run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL)
            if not lock_acquired:
                # 如果第二次尝试仍然失败，记录日志并跳过执行
                logger.info(f"Agent run {agent_run_id} is already being processed by another instance. Skipping duplicate execution.")
                return

    # 成功获取锁后，设置Sentry的线程ID标签
    # 用于错误追踪和日志关联
    sentry.sentry.set_tag("thread_id", thread_id)

    logger.info(f"Starting background agent run: {agent_run_id} for thread: {thread_id} (Instance: {instance_id})")
    logger.info({
        "model_name": model_name,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "enable_context_manager": enable_context_manager,
        "agent_config": agent_config,
        "is_agent_builder": is_agent_builder,
        "target_agent_id": target_agent_id,
    })
    logger.info(f"🚀 Using model: {model_name} (thinking: {enable_thinking}, reasoning_effort: {reasoning_effort})")
    if agent_config:
        logger.info(f"Using custom agent: {agent_config.get('name', 'Unknown')}")

    client = await db.client
    start_time = datetime.now(timezone.utc)
    total_responses = 0
    pubsub = None
    stop_checker = None
    stop_signal_received = False

    # 定义Redis键和通道名称
    response_list_key = f"agent_run:{agent_run_id}:responses"
    response_channel = f"agent_run:{agent_run_id}:new_response"
    instance_control_channel = f"agent_run:{agent_run_id}:control:{instance_id}"
    global_control_channel = f"agent_run:{agent_run_id}:control"
    instance_active_key = f"active_run:{instance_id}:{agent_run_id}"

    async def check_for_stop_signal():
        """
        检查停止信号的异步函数
        功能：
        1. 监听Redis控制通道的STOP信号
        2. 定期刷新活动键的TTL
        3. 处理取消和异常情况
        """
        nonlocal stop_signal_received  # 声明要修改的外部变量
        if not pubsub: return  # 如果pubsub未初始化则直接返回
        
        try:
            while not stop_signal_received:
                # 从pubsub获取消息，忽略订阅消息，超时0.5秒
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                if message and message.get("type") == "message":
                    data = message.get("data")
                    # 处理字节类型的数据
                    if isinstance(data, bytes): data = data.decode('utf-8')
                    if data == "STOP":
                        # 收到STOP信号，设置标志并跳出循环
                        logger.info(f"Received STOP signal for agent run {agent_run_id} (Instance: {instance_id})")
                        stop_signal_received = True
                        break
                # 每处理50个响应后刷新活动键的TTL
                if total_responses % 50 == 0:
                    try: 
                        await redis.expire(instance_active_key, redis.REDIS_KEY_TTL)
                    except Exception as ttl_err: 
                        logger.warning(f"Failed to refresh TTL for {instance_active_key}: {ttl_err}")
                # 短暂休眠避免CPU占用过高
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # 任务被取消时的处理
            logger.info(f"Stop signal checker cancelled for {agent_run_id} (Instance: {instance_id})")
        except Exception as e:
            # 其他异常处理
            logger.error(f"Error in stop signal checker for {agent_run_id}: {e}", exc_info=True)
            stop_signal_received = True  # 发生异常时停止运行

    # 创建Langfuse跟踪记录
    # 包含运行名称、ID、会话ID和元数据(项目ID和实例ID)
    trace = langfuse.trace(
        name="agent_run", 
        id=agent_run_id, 
        session_id=thread_id, 
        metadata={"project_id": project_id, "instance_id": instance_id}
    )
    try:
        # 设置Pub/Sub监听器用于接收控制信号
        pubsub = await redis.create_pubsub()
        try:
            # 重试订阅控制频道（实例控制频道和全局控制频道）
            await retry(lambda: pubsub.subscribe(instance_control_channel, global_control_channel))
        except Exception as e:
            # 订阅失败时记录错误日志
            logger.error(f"Redis failed to subscribe to control channels: {e}", exc_info=True)
            raise e

        # 记录成功订阅的调试信息
        logger.debug(f"Subscribed to control channels: {instance_control_channel}, {global_control_channel}")
        # 创建异步任务检查停止信号
        stop_checker = asyncio.create_task(check_for_stop_signal())

        # 设置实例活跃键并添加TTL（生存时间）
        await redis.set(instance_active_key, "running", ex=redis.REDIS_KEY_TTL)


        # 初始化agent生成器
        agent_gen = run_agent(
            thread_id=thread_id, project_id=project_id, stream=stream,
            model_name=model_name,
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
            enable_context_manager=enable_context_manager,
            agent_config=agent_config,
            trace=trace,
            is_agent_builder=is_agent_builder,
            target_agent_id=target_agent_id
        )

        final_status = "running"
        error_message = None

        pending_redis_operations = []

        async for response in agent_gen:
            if stop_signal_received:
                logger.info(f"Agent run {agent_run_id} stopped by signal.")
                final_status = "stopped"
                trace.span(name="agent_run_stopped").end(status_message="agent_run_stopped", level="WARNING")
                break

            # Store response in Redis list and publish notification
            response_json = json.dumps(response)
            pending_redis_operations.append(asyncio.create_task(redis.rpush(response_list_key, response_json)))
            pending_redis_operations.append(asyncio.create_task(redis.publish(response_channel, "new")))
            total_responses += 1

            # Check for agent-signaled completion or error
            if response.get('type') == 'status':
                 status_val = response.get('status')
                 if status_val in ['completed', 'failed', 'stopped']:
                     logger.info(f"Agent run {agent_run_id} finished via status message: {status_val}")
                     final_status = status_val
                     if status_val == 'failed' or status_val == 'stopped':
                         error_message = response.get('message', f"Run ended with status: {status_val}")
                     break

        # If loop finished without explicit completion/error/stop signal, mark as completed
        if final_status == "running":
             final_status = "completed"
             duration = (datetime.now(timezone.utc) - start_time).total_seconds()
             logger.info(f"Agent run {agent_run_id} completed normally (duration: {duration:.2f}s, responses: {total_responses})")
             completion_message = {"type": "status", "status": "completed", "message": "Agent run completed successfully"}
             trace.span(name="agent_run_completed").end(status_message="agent_run_completed")
             await redis.rpush(response_list_key, json.dumps(completion_message))
             await redis.publish(response_channel, "new") # Notify about the completion message

        # Fetch final responses from Redis for DB update
        all_responses_json = await redis.lrange(response_list_key, 0, -1)
        all_responses = [json.loads(r) for r in all_responses_json]

        # Update DB status
        await update_agent_run_status(client, agent_run_id, final_status, error=error_message, responses=all_responses)

        # Publish final control signal (END_STREAM or ERROR)
        control_signal = "END_STREAM" if final_status == "completed" else "ERROR" if final_status == "failed" else "STOP"
        try:
            await redis.publish(global_control_channel, control_signal)
            # No need to publish to instance channel as the run is ending on this instance
            logger.debug(f"Published final control signal '{control_signal}' to {global_control_channel}")
        except Exception as e:
            logger.warning(f"Failed to publish final control signal {control_signal}: {str(e)}")

    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.error(f"Error in agent run {agent_run_id} after {duration:.2f}s: {error_message}\n{traceback_str} (Instance: {instance_id})")
        final_status = "failed"
        trace.span(name="agent_run_failed").end(status_message=error_message, level="ERROR")

        # Push error message to Redis list
        error_response = {"type": "status", "status": "error", "message": error_message}
        try:
            await redis.rpush(response_list_key, json.dumps(error_response))
            await redis.publish(response_channel, "new")
        except Exception as redis_err:
             logger.error(f"Failed to push error response to Redis for {agent_run_id}: {redis_err}")

        # Fetch final responses (including the error)
        all_responses = []
        try:
             all_responses_json = await redis.lrange(response_list_key, 0, -1)
             all_responses = [json.loads(r) for r in all_responses_json]
        except Exception as fetch_err:
             logger.error(f"Failed to fetch responses from Redis after error for {agent_run_id}: {fetch_err}")
             all_responses = [error_response] # Use the error message we tried to push

        # Update DB status
        await update_agent_run_status(client, agent_run_id, "failed", error=f"{error_message}\n{traceback_str}", responses=all_responses)

        # Publish ERROR signal
        try:
            await redis.publish(global_control_channel, "ERROR")
            logger.debug(f"Published ERROR signal to {global_control_channel}")
        except Exception as e:
            logger.warning(f"Failed to publish ERROR signal: {str(e)}")

    finally:
        # Cleanup stop checker task
        if stop_checker and not stop_checker.done():
            stop_checker.cancel()
            try: await stop_checker
            except asyncio.CancelledError: pass
            except Exception as e: logger.warning(f"Error during stop_checker cancellation: {e}")

        # Close pubsub connection
        if pubsub:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
                logger.debug(f"Closed pubsub connection for {agent_run_id}")
            except Exception as e:
                logger.warning(f"Error closing pubsub for {agent_run_id}: {str(e)}")

        # Set TTL on the response list in Redis
        await _cleanup_redis_response_list(agent_run_id)

        # Remove the instance-specific active run key
        await _cleanup_redis_instance_key(agent_run_id)

        # Clean up the run lock
        await _cleanup_redis_run_lock(agent_run_id)

        # Wait for all pending redis operations to complete, with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*pending_redis_operations), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for pending Redis operations for {agent_run_id}")

        logger.info(f"Agent run background task fully completed for: {agent_run_id} (Instance: {instance_id}) with final status: {final_status}")

async def _cleanup_redis_instance_key(agent_run_id: str):
    """Clean up the instance-specific Redis key for an agent run."""
    if not instance_id:
        logger.warning("Instance ID not set, cannot clean up instance key.")
        return
    key = f"active_run:{instance_id}:{agent_run_id}"
    logger.debug(f"Cleaning up Redis instance key: {key}")
    try:
        await redis.delete(key)
        logger.debug(f"Successfully cleaned up Redis key: {key}")
    except Exception as e:
        logger.warning(f"Failed to clean up Redis key {key}: {str(e)}")

async def _cleanup_redis_run_lock(agent_run_id: str):
    """Clean up the run lock Redis key for an agent run."""
    run_lock_key = f"agent_run_lock:{agent_run_id}"
    logger.debug(f"Cleaning up Redis run lock key: {run_lock_key}")
    try:
        await redis.delete(run_lock_key)
        logger.debug(f"Successfully cleaned up Redis run lock key: {run_lock_key}")
    except Exception as e:
        logger.warning(f"Failed to clean up Redis run lock key {run_lock_key}: {str(e)}")

# TTL for Redis response lists (24 hours)
REDIS_RESPONSE_LIST_TTL = 3600 * 24

async def _cleanup_redis_response_list(agent_run_id: str):
    """Set TTL on the Redis response list."""
    response_list_key = f"agent_run:{agent_run_id}:responses"
    try:
        await redis.expire(response_list_key, REDIS_RESPONSE_LIST_TTL)
        logger.debug(f"Set TTL ({REDIS_RESPONSE_LIST_TTL}s) on response list: {response_list_key}")
    except Exception as e:
        logger.warning(f"Failed to set TTL on response list {response_list_key}: {str(e)}")

async def update_agent_run_status(
    client,
    agent_run_id: str,
    status: str,
    error: Optional[str] = None,
    responses: Optional[list[any]] = None # Expects parsed list of dicts
) -> bool:
    """
    Centralized function to update agent run status.
    Returns True if update was successful.
    """
    try:
        update_data = {
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }

        if error:
            update_data["error"] = error

        if responses:
            # Ensure responses are stored correctly as JSONB
            update_data["responses"] = responses

        # Retry up to 3 times
        for retry in range(3):
            try:
                update_result = await client.table('agent_runs').update(update_data).eq("id", agent_run_id).execute()

                if hasattr(update_result, 'data') and update_result.data:
                    logger.info(f"Successfully updated agent run {agent_run_id} status to '{status}' (retry {retry})")

                    # Verify the update
                    verify_result = await client.table('agent_runs').select('status', 'completed_at').eq("id", agent_run_id).execute()
                    if verify_result.data:
                        actual_status = verify_result.data[0].get('status')
                        completed_at = verify_result.data[0].get('completed_at')
                        logger.info(f"Verified agent run update: status={actual_status}, completed_at={completed_at}")
                    return True
                else:
                    logger.warning(f"Database update returned no data for agent run {agent_run_id} on retry {retry}: {update_result}")
                    if retry == 2:  # Last retry
                        logger.error(f"Failed to update agent run status after all retries: {agent_run_id}")
                        return False
            except Exception as db_error:
                logger.error(f"Database error on retry {retry} updating status for {agent_run_id}: {str(db_error)}")
                if retry < 2:  # Not the last retry yet
                    await asyncio.sleep(0.5 * (2 ** retry))  # Exponential backoff
                else:
                    logger.error(f"Failed to update agent run status after all retries: {agent_run_id}", exc_info=True)
                    return False
    except Exception as e:
        logger.error(f"Unexpected error updating agent run status for {agent_run_id}: {str(e)}", exc_info=True)
        return False

    return False
