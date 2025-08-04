"""
Response processing module for AgentPress.

This module handles the processing of LLM responses, including:
- Streaming and non-streaming response handling
- XML and native tool call detection and parsing
- Tool execution orchestration
- Message formatting and persistence
"""

import json
import re
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union, Callable, Literal
from dataclasses import dataclass
from utils.logger import logger
from agentpress.tool import ToolResult
from agentpress.tool_registry import ToolRegistry
from agentpress.xml_tool_parser import XMLToolParser
from langfuse.client import StatefulTraceClient
from services.langfuse import langfuse
from agentpress.utils.json_helpers import (
    ensure_dict, ensure_list, safe_json_parse, 
    to_json_string, format_for_yield
)
from litellm.utils import token_counter

# Type alias for XML result adding strategy
XmlAddingStrategy = Literal["user_message", "assistant_message", "inline_edit"]

# Type alias for tool execution strategy
ToolExecutionStrategy = Literal["sequential", "parallel"]

@dataclass
class ToolExecutionContext:
    """Context for a tool execution including call details, result, and display info."""
    tool_call: Dict[str, Any]
    tool_index: int
    result: Optional[ToolResult] = None
    function_name: Optional[str] = None
    xml_tag_name: Optional[str] = None
    error: Optional[Exception] = None
    assistant_message_id: Optional[str] = None
    parsing_details: Optional[Dict[str, Any]] = None

@dataclass
class ProcessorConfig:
    """
    Configuration for response processing and tool execution.
    
    This class controls how the LLM's responses are processed, including how tool calls
    are detected, executed, and their results handled.
    
    Attributes:
        xml_tool_calling: Enable XML-based tool call detection (<tool>...</tool>)
        native_tool_calling: Enable OpenAI-style function calling format
        execute_tools: Whether to automatically execute detected tool calls
        execute_on_stream: For streaming, execute tools as they appear vs. at the end
        tool_execution_strategy: How to execute multiple tools ("sequential" or "parallel")
        xml_adding_strategy: How to add XML tool results to the conversation
        max_xml_tool_calls: Maximum number of XML tool calls to process (0 = no limit)
    """

    xml_tool_calling: bool = True  
    native_tool_calling: bool = False

    execute_tools: bool = True
    execute_on_stream: bool = False
    tool_execution_strategy: ToolExecutionStrategy = "sequential"
    xml_adding_strategy: XmlAddingStrategy = "assistant_message"
    max_xml_tool_calls: int = 0  # 0 means no limit
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.xml_tool_calling is False and self.native_tool_calling is False and self.execute_tools:
            raise ValueError("At least one tool calling format (XML or native) must be enabled if execute_tools is True")
            
        if self.xml_adding_strategy not in ["user_message", "assistant_message", "inline_edit"]:
            raise ValueError("xml_adding_strategy must be 'user_message', 'assistant_message', or 'inline_edit'")
        
        if self.max_xml_tool_calls < 0:
            raise ValueError("max_xml_tool_calls must be a non-negative integer (0 = no limit)")

class ResponseProcessor:
    """Processes LLM responses, extracting and executing tool calls."""
    
    def __init__(self, tool_registry: ToolRegistry, add_message_callback: Callable, trace: Optional[StatefulTraceClient] = None, is_agent_builder: bool = False, target_agent_id: Optional[str] = None, agent_config: Optional[dict] = None):
        """Initialize the ResponseProcessor.
        
        Args:
            tool_registry: Registry of available tools
            add_message_callback: Callback function to add messages to the thread.
                MUST return the full saved message object (dict) or None.
            agent_config: Optional agent configuration with version information
        """
        self.tool_registry = tool_registry
        self.add_message = add_message_callback
        self.trace = trace or langfuse.trace(name="anonymous:response_processor")
        # Initialize the XML parser with backwards compatibility
        self.xml_parser = XMLToolParser(strict_mode=False)
        self.is_agent_builder = is_agent_builder
        self.target_agent_id = target_agent_id
        self.agent_config = agent_config

    async def _yield_message(self, message_obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Helper to yield a message with proper formatting.
        
        Ensures that content and metadata are JSON strings for client compatibility.
        """
        if message_obj:
            return format_for_yield(message_obj)
        return None

    async def _add_message_with_agent_info(
        self,
        thread_id: str,
        type: str,
        content: Union[Dict[str, Any], List[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Helper to add a message with agent version information if available."""
        agent_id = None
        agent_version_id = None
        
        if self.agent_config:
            agent_id = self.agent_config.get('agent_id')
            agent_version_id = self.agent_config.get('current_version_id')
            
        return await self.add_message(
            thread_id=thread_id,
            type=type,
            content=content,
            is_llm_message=is_llm_message,
            metadata=metadata,
            agent_id=agent_id,
            agent_version_id=agent_version_id
        )

    async def process_streaming_response(
        self,
        llm_response: AsyncGenerator,
        thread_id: str,
        prompt_messages: List[Dict[str, Any]],
        llm_model: str,
        config: ProcessorConfig = ProcessorConfig(),
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """处理流式LLM响应，处理工具调用和执行。
        
        参数:
            llm_response: 来自LLM的流式响应
            thread_id: 对话线程的ID
            prompt_messages: 发送给LLM的消息列表（提示）
            llm_model: 使用的LLM模型名称
            config: 解析和执行的配置
            
        生成:
            符合数据库模式的完整消息对象，但内容块除外。
        """
        # 初始化累积内容和各种缓冲区
        accumulated_content = "" # 累积的文本内容
        tool_calls_buffer = {} # 工具调用缓冲区
        current_xml_content = "" # 当前XML内容
        xml_chunks_buffer = [] # XML块缓冲区
        pending_tool_executions = [] # 待执行的工具
        yielded_tool_indices = set() # 已生成状态的工具索引
        tool_index = 0 # 工具索引计数器
        xml_tool_call_count = 0 # XML工具调用计数
        finish_reason = None  # 完成原因
        last_assistant_message_object = None # 最终保存的助手消息对象
        tool_result_message_objects = {} # 工具索引到完整消息对象的映射
        has_printed_thinking_prefix = False # 打印思考前缀的标志
        agent_should_terminate = False # 代理是否应该终止的标志
        complete_native_tool_calls = [] # 完整的原生工具调用列表

        # 收集用于重建LiteLLM响应对象的元数据
        streaming_metadata = {
            "model": llm_model,
            "created": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "response_ms": None,
            "first_chunk_time": None,
            "last_chunk_time": None
        }

        logger.info(f"Streaming Config: XML={config.xml_tool_calling}, Native={config.native_tool_calling}, "
                   f"Execute on stream={config.execute_on_stream}, Strategy={config.tool_execution_strategy}")

        thread_run_id = str(uuid.uuid4())

        try:
            # --- 保存并生成开始事件 ---
            # 创建线程运行开始状态消息
            start_content = {
                "status_type": "thread_run_start",  # 状态类型：线程运行开始
                "thread_run_id": thread_run_id  # 线程运行ID
            }
            # 添加状态消息到数据库
            start_msg_obj = await self.add_message(
                thread_id=thread_id,  # 线程ID
                type="status",  # 消息类型为状态
                content=start_content,  # 消息内容
                is_llm_message=False,  # 不是LLM消息
                metadata={"thread_run_id": thread_run_id}  # 元数据包含线程运行ID
            )
            # 如果消息对象创建成功，则生成该消息
            if start_msg_obj:
                yield format_for_yield(start_msg_obj)

            # 创建助手响应开始状态消息
            assist_start_content = {
                "status_type": "assistant_response_start"  # 状态类型：助手响应开始
            }
            # 添加状态消息到数据库
            assist_start_msg_obj = await self.add_message(
                thread_id=thread_id,  # 线程ID
                type="status",  # 消息类型为状态
                content=assist_start_content,  # 消息内容
                is_llm_message=False,  # 不是LLM消息
                metadata={"thread_run_id": thread_run_id}  # 元数据包含线程运行ID
            )
            # 如果消息对象创建成功，则生成该消息
            if assist_start_msg_obj:
                yield format_for_yield(assist_start_msg_obj)
            # --- 结束开始事件 ---

            __sequence = 0

            # 异步遍历LLM响应流中的每个数据块
            async for chunk in llm_response:
                # 从数据块中提取流式元数据
                # 记录当前时间戳用于计算延迟
                current_time = datetime.now(timezone.utc).timestamp()
                if streaming_metadata["first_chunk_time"] is None:
                    streaming_metadata["first_chunk_time"] = current_time
                streaming_metadata["last_chunk_time"] = current_time
                
                # 从响应数据块中提取核心元数据信息
                # 这些数据用于构建完整的响应记录和监控分析

                # 提取响应创建时间戳（Unix时间戳格式）
                if hasattr(chunk, 'created') and chunk.created:
                    streaming_metadata["created"] = chunk.created

                # 提取使用的AI模型名称（如gpt-4, claude-3等）
                if hasattr(chunk, 'model') and chunk.model:
                    streaming_metadata["model"] = chunk.model

                # 提取token使用量统计（如果提供商返回了使用数据）
                if hasattr(chunk, 'usage') and chunk.usage:
                    # 更新token使用信息（包括零值情况）
                    # prompt_tokens: 输入提示消耗的token数量
                    if hasattr(chunk.usage, 'prompt_tokens') and chunk.usage.prompt_tokens is not None:
                        streaming_metadata["usage"]["prompt_tokens"] = chunk.usage.prompt_tokens
                    # completion_tokens: 生成回复消耗的token数量
                    if hasattr(chunk.usage, 'completion_tokens') and chunk.usage.completion_tokens is not None:
                        streaming_metadata["usage"]["completion_tokens"] = chunk.usage.completion_tokens
                    # total_tokens: 总共消耗的token数量（输入+输出）
                    if hasattr(chunk.usage, 'total_tokens') and chunk.usage.total_tokens is not None:
                        streaming_metadata["usage"]["total_tokens"] = chunk.usage.total_tokens

                # 检测AI响应的结束状态
                # 这一连串条件检查确保我们能够安全地获取响应结束的原因
                if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                    # 提取响应结束的原因，用于判断对话是否正常完成
                    # 可能的值：'stop'（正常结束）、'length'（达到长度限制）、'content_filter'（内容被过滤）等
                    finish_reason = chunk.choices[0].finish_reason
                    logger.debug(f"Detected finish_reason: {finish_reason}")

                if hasattr(chunk, 'choices') and chunk.choices:
                    # 安全提取响应数据块中的增量内容
                    # delta 包含本次数据块中的内容变化，可能是文本内容或工具调用信息
                    # 使用 hasattr 进行防御式编程，避免属性不存在导致的异常
                    delta = chunk.choices[0].delta if hasattr(chunk.choices[0], 'delta') else None
                    
                    # 检查并处理 Anthropic 模型的思考过程内容
                    # Anthropic 的 Claude 模型会在 reasoning_content 中返回模型的思考链
                    # 这部分内容对用户透明，但会被记录到完整响应中用于调试和分析
                    if delta and hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        # 首次发现思考内容时打印标识（已注释掉）··
                        if not has_printed_thinking_prefix:
                            # print("[THINKING]: ", end='', flush=True)
                            has_printed_thinking_prefix = True

                        # 实际打印思考内容（已注释掉）
                        # print(delta.reasoning_content, end='', flush=True)

                        # 将思考内容追加到累积内容中，确保完整保存响应
                        accumulated_content += delta.reasoning_content

                    # Process content chunk
                    if delta and hasattr(delta, 'content') and delta.content:
                        chunk_content = delta.content
                        # print(chunk_content, end='', flush=True)
                        accumulated_content += chunk_content
                        current_xml_content += chunk_content

                        # 检查是否未达到XML工具调用数量限制
                        # 条件：当max_xml_tool_calls <= 0（无限制）或当前计数小于限制时，才允许传输内容块
                        if not (config.max_xml_tool_calls > 0 and xml_tool_call_count >= config.max_xml_tool_calls):
                            # 仅传输内容块（不保存到数据库）
                            # 用于实时向客户端推送助手的回复内容片段
                            now_chunk = datetime.now(timezone.utc).isoformat()
                            yield {
                                "sequence": __sequence,
                                "message_id": None, "thread_id": thread_id, "type": "assistant",
                                "is_llm_message": True,
                                "content": to_json_string({"role": "assistant", "content": chunk_content}),
                                "metadata": to_json_string({"stream_status": "chunk", "thread_run_id": thread_run_id}),
                                "created_at": now_chunk, "updated_at": now_chunk
                            }
                            __sequence += 1
                        else:
                            logger.info("XML tool call limit reached - not yielding more content chunks")
                            self.trace.event(name="xml_tool_call_limit_reached", level="DEFAULT", status_message=(f"XML tool call limit reached - not yielding more content chunks"))

                        # --- 处理XML工具调用（如果启用且未达到数量限制）---
                        # 双重条件检查：确保XML工具调用功能已启用，且未达到最大调用次数限制
                        if config.xml_tool_calling and not (config.max_xml_tool_calls > 0 and xml_tool_call_count >= config.max_xml_tool_calls):
                            # 从当前累积的XML内容中提取所有完整的XML块
                            xml_chunks = self._extract_xml_chunks(current_xml_content)
                            for xml_chunk in xml_chunks:
                                # 从当前内容中移除已处理的XML块，避免重复处理
                                current_xml_content = current_xml_content.replace(xml_chunk, "", 1)
                                # 将处理过的XML块存入缓冲区，用于调试和日志记录
                                xml_chunks_buffer.append(xml_chunk)

                                # 解析XML块为工具调用对象，返回工具调用信息和解析详情
                                result = self._parse_xml_tool_call(xml_chunk)
                                if result:
                                    tool_call, parsing_details = result
                                    xml_tool_call_count += 1  # 递增XML工具调用计数器

                                    # 获取当前助手消息的ID，用于关联工具调用上下文
                                    current_assistant_id = last_assistant_message_object['message_id'] if last_assistant_message_object else None
                                    # 创建工具调用上下文，包含执行所需的所有信息
                                    context = self._create_tool_context(
                                        tool_call, tool_index, current_assistant_id, parsing_details
                                    )

                                    # 如果启用了工具执行且允许流式执行，则启动异步任务
                                    if config.execute_tools and config.execute_on_stream:
                                        # 保存并传输工具开始执行的状态消息
                                        started_msg_obj = await self._yield_and_save_tool_started(context, thread_id, thread_run_id)
                                        if started_msg_obj: yield format_for_yield(started_msg_obj)
                                        yielded_tool_indices.add(tool_index)  # 标记该工具状态已传输

                                        # 创建异步任务执行工具调用，实现并发处理
                                        execution_task = asyncio.create_task(self._execute_tool(tool_call))
                                        pending_tool_executions.append({
                                            "task": execution_task, "tool_call": tool_call,
                                            "tool_index": tool_index, "context": context
                                        })
                                        tool_index += 1  # 为下一个工具调用准备索引

                                    # 检查是否达到XML工具调用限制，如果是则停止处理更多块
                                    if config.max_xml_tool_calls > 0 and xml_tool_call_count >= config.max_xml_tool_calls:
                                        logger.debug(f"已达到XML工具调用限制 ({config.max_xml_tool_calls})")
                                        finish_reason = "xml_tool_limit_reached"  # 设置特殊的完成原因
                                        break  # 停止处理当前delta中的更多XML块

                    # --- 处理原生工具调用块 ---
                    # 检查是否启用原生工具调用，并且当前块包含工具调用信息
                    if config.native_tool_calling and delta and hasattr(delta, 'tool_calls') and delta.tool_calls:
                        # 遍历所有工具调用块
                        for tool_call_chunk in delta.tool_calls:
                            # 生成原生工具调用块状态（临时状态，不保存）
                            # ... (安全提取 tool_call_data_chunk 的逻辑) ...
                            tool_call_data_chunk = {} # 提取数据的占位符
                            # 尝试使用 model_dump 方法提取数据，否则手动提取
                            if hasattr(tool_call_chunk, 'model_dump'):
                                tool_call_data_chunk = tool_call_chunk.model_dump()
                            else:
                                # 手动提取工具调用块的各个属性
                                if hasattr(tool_call_chunk, 'id'):
                                    tool_call_data_chunk['id'] = tool_call_chunk.id
                                if hasattr(tool_call_chunk, 'index'):
                                    tool_call_data_chunk['index'] = tool_call_chunk.index
                                if hasattr(tool_call_chunk, 'type'):
                                    tool_call_data_chunk['type'] = tool_call_chunk.type
                                if hasattr(tool_call_chunk, 'function'):
                                    tool_call_data_chunk['function'] = {}
                                    if hasattr(tool_call_chunk.function, 'name'):
                                        tool_call_data_chunk['function']['name'] = tool_call_chunk.function.name
                                    if hasattr(tool_call_chunk.function, 'arguments'):
                                        # 如果参数是字符串则直接使用，否则转换为 JSON 字符串
                                        tool_call_data_chunk['function']['arguments'] = tool_call_chunk.function.arguments if isinstance(tool_call_chunk.function.arguments, str) else to_json_string(tool_call_chunk.function.arguments)

                            # 记录当前时间并生成工具调用块状态
                            now_tool_chunk = datetime.now(timezone.utc).isoformat()
                            yield {
                                "message_id": None,
                                "thread_id": thread_id,
                                "type": "status",
                                "is_llm_message": True,
                                "content": to_json_string({
                                    "role": "assistant",
                                    "status_type": "tool_call_chunk",
                                    "tool_call_chunk": tool_call_data_chunk
                                }),
                                "metadata": to_json_string({"thread_run_id": thread_run_id}),
                                "created_at": now_tool_chunk,
                                "updated_at": now_tool_chunk
                            }

                            # --- 缓冲并执行完整的原生工具调用 ---
                            # 如果工具调用块没有函数属性，则跳过
                            if not hasattr(tool_call_chunk, 'function'):
                                continue
                            # 获取工具调用块的索引，如果没有则默认为 0
                            idx = tool_call_chunk.index if hasattr(tool_call_chunk, 'index') else 0
                            # ... (缓冲区更新逻辑保持不变) ...
                            # ... (检查完整性的逻辑保持不变) ...
                            has_complete_tool_call = False # 完整工具调用的占位符
                            # 检查缓冲区中是否有完整的工具调用（包含 ID、函数名和参数）
                            if (tool_calls_buffer.get(idx) and
                                tool_calls_buffer[idx]['id'] and
                                tool_calls_buffer[idx]['function']['name'] and
                                tool_calls_buffer[idx]['function']['arguments']):
                                try:
                                    # 尝试解析参数为 JSON，如果成功则标记为完整工具调用
                                    safe_json_parse(tool_calls_buffer[idx]['function']['arguments'])
                                    has_complete_tool_call = True
                                except json.JSONDecodeError:
                                    # 如果解析失败则忽略
                                    pass

                            # 如果有完整工具调用，并且配置允许执行工具，并且在流式传输时执行
                            if has_complete_tool_call and config.execute_tools and config.execute_on_stream:
                                # 获取当前工具调用数据
                                current_tool = tool_calls_buffer[idx]
                                tool_call_data = {
                                    "function_name": current_tool['function']['name'],
                                    "arguments": safe_json_parse(current_tool['function']['arguments']),
                                    "id": current_tool['id']
                                }
                                # 获取当前助手消息 ID
                                current_assistant_id = last_assistant_message_object['message_id'] if last_assistant_message_object else None
                                # 创建工具上下文
                                context = self._create_tool_context(
                                    tool_call_data, tool_index, current_assistant_id
                                )

                                # 保存并生成工具开始状态
                                started_msg_obj = await self._yield_and_save_tool_started(context, thread_id, thread_run_id)
                                if started_msg_obj:
                                    yield format_for_yield(started_msg_obj)
                                # 标记状态已生成
                                yielded_tool_indices.add(tool_index)

                                # 创建异步任务执行工具
                                execution_task = asyncio.create_task(self._execute_tool(tool_call_data))
                                # 将任务添加到待处理执行列表中
                                pending_tool_executions.append({
                                    "task": execution_task,
                                    "tool_call": tool_call_data,
                                    "tool_index": tool_index,
                                    "context": context
                                })
                                # 增加工具索引
                                tool_index += 1

                if finish_reason == "xml_tool_limit_reached":
                    logger.info("Stopping stream processing after loop due to XML tool call limit")
                    self.trace.event(name="stopping_stream_processing_after_loop_due_to_xml_tool_call_limit", level="DEFAULT", status_message=(f"Stopping stream processing after loop due to XML tool call limit"))
                    break

            # print() # Add a final newline after the streaming loop finishes

            # --- After Streaming Loop ---
            
            # --- 流处理循环结束后的处理 ---

            # 检查是否从LLM提供商获得了token使用数据
            # 如果没有获得，则使用litellm.token_counter来估算token使用量
            if (
                streaming_metadata["usage"]["total_tokens"] == 0
            ):
                logger.info("🔥 No usage data from provider, counting with litellm.token_counter")
                
                try:
                    # 计算提示部分的token数量
                    prompt_tokens = token_counter(
                        model=llm_model,
                        messages=prompt_messages               # chat or plain; token_counter handles both
                    )

                    # 计算完成部分的token数量
                    completion_tokens = token_counter(
                        model=llm_model,
                        text=accumulated_content or ""         # empty string safe
                    )

                    # 更新流元数据中的token使用信息
                    streaming_metadata["usage"]["prompt_tokens"]      = prompt_tokens
                    streaming_metadata["usage"]["completion_tokens"]  = completion_tokens
                    streaming_metadata["usage"]["total_tokens"]       = prompt_tokens + completion_tokens

                    # 记录估算的token使用量
                    logger.info(
                        f"🔥 Estimated tokens – prompt: {prompt_tokens}, "
                        f"completion: {completion_tokens}, total: {prompt_tokens + completion_tokens}"
                    )
                    self.trace.event(name="usage_calculated_with_litellm_token_counter", level="DEFAULT", status_message=(f"Usage calculated with litellm.token_counter"))
                except Exception as e:
                    # 如果计算token使用量失败，记录警告信息
                    logger.warning(f"Failed to calculate usage: {str(e)}")
                    self.trace.event(name="failed_to_calculate_usage", level="WARNING", status_message=(f"Failed to calculate usage: {str(e)}"))


            # 等待流式阶段中待处理的工具执行完成
            # 创建一个缓冲区来存储工具执行结果，每个元素包含(工具调用, 执行结果, 工具索引, 上下文)
            tool_results_buffer = []

            # 如果存在待处理的工具执行任务
            if pending_tool_executions:
                logger.info(f"等待 {len(pending_tool_executions)} 个流式工具执行完成")
                self.trace.event(name="waiting_for_pending_streamed_tool_executions", level="DEFAULT",
                               status_message=(f"等待 {len(pending_tool_executions)} 个流式工具执行完成"))

                # 提取所有待执行的异步任务
                pending_tasks = [execution["task"] for execution in pending_tool_executions]
                # 等待所有任务完成，done包含已完成的任务，_包含未完成的任务（理论上应该都已完成）
                done, _ = await asyncio.wait(pending_tasks)

                # 遍历所有待处理的工具执行
                for execution in pending_tool_executions:
                    tool_idx = execution.get("tool_index", -1)  # 获取工具索引，默认为-1
                    context = execution["context"]  # 获取工具执行的上下文信息
                    tool_name = context.function_name  # 获取工具名称
                    
                    # 检查该工具的状态是否已在流式阶段返回
                    if tool_idx in yielded_tool_indices:
                        logger.debug(f"工具索引 {tool_idx} 的状态已在流式阶段返回")

                        # 即使状态已返回，仍需处理结果以填充缓冲区
                        try:
                            # 检查任务是否确实已完成
                            if execution["task"].done():
                                # 获取工具执行结果
                                result = execution["task"].result()
                                context.result = result  # 将结果保存到上下文中
                                # 将结果添加到缓冲区
                                tool_results_buffer.append((execution["tool_call"], result, tool_idx, context))

                                # 检查是否为终止工具（ask或complete），如果是则设置终止标志
                                if tool_name in ['ask', 'complete']:
                                    logger.info(f"终止工具 '{tool_name}' 在流式阶段完成，设置终止标志")
                                    self.trace.event(name="terminating_tool_completed_during_streaming", level="DEFAULT",
                                                   status_message=(f"终止工具 '{tool_name}' 在流式阶段完成，设置终止标志"))
                                    agent_should_terminate = True
                                     
                            else:  # 理论上不应该发生，因为asyncio.wait应该等待所有任务完成
                                logger.warning(f"工具索引 {tool_idx} 的任务在等待后仍未完成")
                                self.trace.event(name="task_for_tool_index_not_done_after_wait", level="WARNING",
                                               status_message=(f"工具索引 {tool_idx} 的任务在等待后仍未完成"))

                        except Exception as e:
                            # 处理工具执行异常
                            logger.error(f"获取待处理工具执行 {tool_idx} 的结果时出错: {str(e)}")
                            self.trace.event(name="error_getting_result_for_pending_tool_execution", level="ERROR",
                                           status_message=(f"获取待处理工具执行 {tool_idx} 的结果时出错: {str(e)}"))
                            context.error = e
                            # 保存并返回工具错误状态消息（即使已开始状态已返回）
                            error_msg_obj = await self._yield_and_save_tool_error(context, thread_id, thread_run_id)
                            if error_msg_obj: yield format_for_yield(error_msg_obj)

                        continue  # 跳过该工具索引的进一步状态返回

                    # 如果状态之前未返回（理论上不应该发生），现在返回
                    try:
                        if execution["task"].done():
                            # 获取工具执行结果
                            result = execution["task"].result()
                            context.result = result
                            tool_results_buffer.append((execution["tool_call"], result, tool_idx, context))
                            
                            # 检查是否为终止工具
                            if tool_name in ['ask', 'complete']:
                                logger.info(f"终止工具 '{tool_name}' 在流式阶段完成，设置终止标志")
                                self.trace.event(name="terminating_tool_completed_during_streaming", level="DEFAULT",
                                               status_message=(f"终止工具 '{tool_name}' 在流式阶段完成，设置终止标志"))
                                agent_should_terminate = True
                                
                            # 保存并返回工具完成/失败状态
                            completed_msg_obj = await self._yield_and_save_tool_completed(
                                context, None, thread_id, thread_run_id
                            )
                            if completed_msg_obj: yield format_for_yield(completed_msg_obj)
                            yielded_tool_indices.add(tool_idx)
                    except Exception as e:
                        # 处理工具执行异常
                        logger.error(f"获取待处理工具执行 {tool_idx} 的结果/返回状态时出错: {str(e)}")
                        self.trace.event(name="error_getting_result_yielding_status_for_pending_tool_execution", level="ERROR",
                                       status_message=(f"获取待处理工具执行 {tool_idx} 的结果/返回状态时出错: {str(e)}"))
                        context.error = e
                        # 保存并返回工具错误状态
                        error_msg_obj = await self._yield_and_save_tool_error(context, thread_id, thread_run_id)
                        if error_msg_obj: yield format_for_yield(error_msg_obj)
                        yielded_tool_indices.add(tool_idx)


            # 如果达到限制，则保存并产生完成状态
            if finish_reason == "xml_tool_limit_reached":
                finish_content = {"status_type": "finish", "finish_reason": "xml_tool_limit_reached"}
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id, type="status", content=finish_content, 
                    is_llm_message=False, metadata={"thread_run_id": thread_run_id}
                )
                if finish_msg_obj: yield format_for_yield(finish_msg_obj)
                logger.info(f"Stream finished with reason: xml_tool_limit_reached after {xml_tool_call_count} XML tool calls")
                self.trace.event(name="stream_finished_with_reason_xml_tool_limit_reached_after_xml_tool_calls", level="DEFAULT", status_message=(f"Stream finished with reason: xml_tool_limit_reached after {xml_tool_call_count} XML tool calls"))

            # --- 保存并输出最终助手消息 ---
            # 如果累积了内容，则处理并保存为完整的助手消息
            if accumulated_content:
                # ... (截断累积内容的逻辑) ...
                # 如果XML工具调用达到上限且存在缓冲区内容，截断内容至最后一个XML块
                if config.max_xml_tool_calls > 0 and xml_tool_call_count >= config.max_xml_tool_calls and xml_chunks_buffer:
                    last_xml_chunk = xml_chunks_buffer[-1]
                    last_chunk_end_pos = accumulated_content.find(last_xml_chunk) + len(last_xml_chunk)
                    if last_chunk_end_pos > 0:
                        accumulated_content = accumulated_content[:last_chunk_end_pos]

                # ... (提取完整原生工具调用的逻辑) ...
                # 从缓冲区更新完整的原生工具调用列表（之前已初始化）
                if config.native_tool_calling:
                    for idx, tc_buf in tool_calls_buffer.items():
                        # 确保工具调用ID、函数名和参数都存在
                        if tc_buf['id'] and tc_buf['function']['name'] and tc_buf['function']['arguments']:
                            try:
                                # 安全解析JSON参数
                                args = safe_json_parse(tc_buf['function']['arguments'])
                                complete_native_tool_calls.append({
                                    "id": tc_buf['id'], "type": "function",
                                    "function": {"name": tc_buf['function']['name'],"arguments": args}
                                })
                            except json.JSONDecodeError:
                                # JSON解析失败时跳过此工具调用
                                continue

                # 构建要保存的消息数据结构
                message_data = { # 要保存在'content'字段中的字典
                    "role": "assistant", "content": accumulated_content,  # 助手角色和累积内容
                    "tool_calls": complete_native_tool_calls or None  # 原生工具调用列表（可为空）
                }

                # 调用内部方法保存助手消息，包含代理信息
                last_assistant_message_object = await self._add_message_with_agent_info(
                    thread_id=thread_id, type="assistant", content=message_data,
                    is_llm_message=True, metadata={"thread_run_id": thread_run_id}
                )

                # 如果成功保存了助手消息
                if last_assistant_message_object:
                    # 为输出准备完整的已保存对象，仅用于输出的添加stream_status元数据
                    yield_metadata = ensure_dict(last_assistant_message_object.get('metadata'), {})
                    yield_metadata['stream_status'] = 'complete'  # 标记流状态为完成
                    # 格式化消息用于输出
                    yield_message = last_assistant_message_object.copy()
                    yield_message['metadata'] = yield_metadata
                    yield format_for_yield(yield_message)
                else:
                    # 保存助手消息失败时的错误处理
                    logger.error(f"无法为线程 {thread_id} 保存最终助手消息")
                    self.trace.event(name="failed_to_save_final_assistant_message_for_thread", level="ERROR", status_message=(f"无法为线程 {thread_id} 保存最终助手消息"))
                    # 保存并输出错误状态
                    err_content = {"role": "system", "status_type": "error", "message": "无法保存最终助手消息"}
                    err_msg_obj = await self.add_message(
                        thread_id=thread_id, type="status", content=err_content, 
                        is_llm_message=False, metadata={"thread_run_id": thread_run_id}
                    )
                    if err_msg_obj: yield format_for_yield(err_msg_obj)

            # --- Process All Tool Results Now ---
            if config.execute_tools:
                final_tool_calls_to_process = []
                # ... (Gather final_tool_calls_to_process from native and XML buffers) ...
                 # Gather native tool calls from buffer
                if config.native_tool_calling and complete_native_tool_calls:
                    for tc in complete_native_tool_calls:
                        final_tool_calls_to_process.append({
                            "function_name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"], # Already parsed object
                            "id": tc["id"]
                        })
                 # Gather XML tool calls from buffer (up to limit)
                parsed_xml_data = []
                if config.xml_tool_calling:
                    # Reparse remaining content just in case (should be empty if processed correctly)
                    xml_chunks = self._extract_xml_chunks(current_xml_content)
                    xml_chunks_buffer.extend(xml_chunks)
                    # Process only chunks not already handled in the stream loop
                    remaining_limit = config.max_xml_tool_calls - xml_tool_call_count if config.max_xml_tool_calls > 0 else len(xml_chunks_buffer)
                    xml_chunks_to_process = xml_chunks_buffer[:remaining_limit] # Ensure limit is respected

                    for chunk in xml_chunks_to_process:
                         parsed_result = self._parse_xml_tool_call(chunk)
                         if parsed_result:
                             tool_call, parsing_details = parsed_result
                             # Avoid adding if already processed during streaming
                             if not any(exec['tool_call'] == tool_call for exec in pending_tool_executions):
                                 final_tool_calls_to_process.append(tool_call)
                                 parsed_xml_data.append({'tool_call': tool_call, 'parsing_details': parsing_details})


                all_tool_data_map = {} # tool_index -> {'tool_call': ..., 'parsing_details': ...}
                 # Add native tool data
                native_tool_index = 0
                if config.native_tool_calling and complete_native_tool_calls:
                     for tc in complete_native_tool_calls:
                         # Find the corresponding entry in final_tool_calls_to_process if needed
                         # For now, assume order matches if only native used
                         exec_tool_call = {
                             "function_name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"],
                             "id": tc["id"]
                         }
                         all_tool_data_map[native_tool_index] = {"tool_call": exec_tool_call, "parsing_details": None}
                         native_tool_index += 1

                 # Add XML tool data
                xml_tool_index_start = native_tool_index
                for idx, item in enumerate(parsed_xml_data):
                    all_tool_data_map[xml_tool_index_start + idx] = item


                tool_results_map = {} # tool_index -> (tool_call, result, context)

                # Populate from buffer if executed on stream
                if config.execute_on_stream and tool_results_buffer:
                    logger.info(f"Processing {len(tool_results_buffer)} buffered tool results")
                    self.trace.event(name="processing_buffered_tool_results", level="DEFAULT", status_message=(f"Processing {len(tool_results_buffer)} buffered tool results"))
                    for tool_call, result, tool_idx, context in tool_results_buffer:
                        if last_assistant_message_object: context.assistant_message_id = last_assistant_message_object['message_id']
                        tool_results_map[tool_idx] = (tool_call, result, context)

                # Or execute now if not streamed
                elif final_tool_calls_to_process and not config.execute_on_stream:
                    logger.info(f"Executing {len(final_tool_calls_to_process)} tools ({config.tool_execution_strategy}) after stream")
                    self.trace.event(name="executing_tools_after_stream", level="DEFAULT", status_message=(f"Executing {len(final_tool_calls_to_process)} tools ({config.tool_execution_strategy}) after stream"))
                    results_list = await self._execute_tools(final_tool_calls_to_process, config.tool_execution_strategy)
                    current_tool_idx = 0
                    for tc, res in results_list:
                       # Map back using all_tool_data_map which has correct indices
                       if current_tool_idx in all_tool_data_map:
                           tool_data = all_tool_data_map[current_tool_idx]
                           context = self._create_tool_context(
                               tc, current_tool_idx,
                               last_assistant_message_object['message_id'] if last_assistant_message_object else None,
                               tool_data.get('parsing_details')
                           )
                           context.result = res
                           tool_results_map[current_tool_idx] = (tc, res, context)
                       else:
                           logger.warning(f"Could not map result for tool index {current_tool_idx}")
                           self.trace.event(name="could_not_map_result_for_tool_index", level="WARNING", status_message=(f"Could not map result for tool index {current_tool_idx}"))
                       current_tool_idx += 1

                # Save and Yield each result message
                if tool_results_map:
                    logger.info(f"Saving and yielding {len(tool_results_map)} final tool result messages")
                    self.trace.event(name="saving_and_yielding_final_tool_result_messages", level="DEFAULT", status_message=(f"Saving and yielding {len(tool_results_map)} final tool result messages"))
                    for tool_idx in sorted(tool_results_map.keys()):
                        tool_call, result, context = tool_results_map[tool_idx]
                        context.result = result
                        if not context.assistant_message_id and last_assistant_message_object:
                            context.assistant_message_id = last_assistant_message_object['message_id']

                        # Yield start status ONLY IF executing non-streamed (already yielded if streamed)
                        if not config.execute_on_stream and tool_idx not in yielded_tool_indices:
                            started_msg_obj = await self._yield_and_save_tool_started(context, thread_id, thread_run_id)
                            if started_msg_obj: yield format_for_yield(started_msg_obj)
                            yielded_tool_indices.add(tool_idx) # Mark status yielded

                        # Save the tool result message to DB
                        saved_tool_result_object = await self._add_tool_result( # Returns full object or None
                            thread_id, tool_call, result, config.xml_adding_strategy,
                            context.assistant_message_id, context.parsing_details
                        )

                        # Yield completed/failed status (linked to saved result ID if available)
                        completed_msg_obj = await self._yield_and_save_tool_completed(
                            context,
                            saved_tool_result_object['message_id'] if saved_tool_result_object else None,
                            thread_id, thread_run_id
                        )
                        if completed_msg_obj: yield format_for_yield(completed_msg_obj)
                        # Don't add to yielded_tool_indices here, completion status is separate yield

                        # Yield the saved tool result object
                        if saved_tool_result_object:
                            tool_result_message_objects[tool_idx] = saved_tool_result_object
                            yield format_for_yield(saved_tool_result_object)
                        else:
                             logger.error(f"Failed to save tool result for index {tool_idx}, not yielding result message.")
                             self.trace.event(name="failed_to_save_tool_result_for_index", level="ERROR", status_message=(f"Failed to save tool result for index {tool_idx}, not yielding result message."))
                             # Optionally yield error status for saving failure?

            # --- Final Finish Status ---
            if finish_reason and finish_reason != "xml_tool_limit_reached":
                finish_content = {"status_type": "finish", "finish_reason": finish_reason}
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id, type="status", content=finish_content, 
                    is_llm_message=False, metadata={"thread_run_id": thread_run_id}
                )
                if finish_msg_obj: yield format_for_yield(finish_msg_obj)

            # Check if agent should terminate after processing pending tools
            if agent_should_terminate:
                logger.info("Agent termination requested after executing ask/complete tool. Stopping further processing.")
                self.trace.event(name="agent_termination_requested", level="DEFAULT", status_message="Agent termination requested after executing ask/complete tool. Stopping further processing.")
                
                # Set finish reason to indicate termination
                finish_reason = "agent_terminated"
                
                # Save and yield termination status
                finish_content = {"status_type": "finish", "finish_reason": "agent_terminated"}
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id, type="status", content=finish_content, 
                    is_llm_message=False, metadata={"thread_run_id": thread_run_id}
                )
                if finish_msg_obj: yield format_for_yield(finish_msg_obj)
                
                # Save assistant_response_end BEFORE terminating
                if last_assistant_message_object:
                    try:
                        # Calculate response time if we have timing data
                        if streaming_metadata["first_chunk_time"] and streaming_metadata["last_chunk_time"]:
                            streaming_metadata["response_ms"] = (streaming_metadata["last_chunk_time"] - streaming_metadata["first_chunk_time"]) * 1000

                        # Create a LiteLLM-like response object for streaming (before termination)
                        # Check if we have any actual usage data
                        has_usage_data = (
                            streaming_metadata["usage"]["prompt_tokens"] > 0 or
                            streaming_metadata["usage"]["completion_tokens"] > 0 or
                            streaming_metadata["usage"]["total_tokens"] > 0
                        )
                        
                        assistant_end_content = {
                            "choices": [
                                {
                                    "finish_reason": finish_reason or "stop",
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": accumulated_content,
                                        "tool_calls": complete_native_tool_calls or None
                                    }
                                }
                            ],
                            "created": streaming_metadata.get("created"),
                            "model": streaming_metadata.get("model", llm_model),
                            "usage": streaming_metadata["usage"],  # Always include usage like LiteLLM does
                            "streaming": True,  # Add flag to indicate this was reconstructed from streaming
                        }
                        
                        # Only include response_ms if we have timing data
                        if streaming_metadata.get("response_ms"):
                            assistant_end_content["response_ms"] = streaming_metadata["response_ms"]
                        
                        await self.add_message(
                            thread_id=thread_id,
                            type="assistant_response_end",
                            content=assistant_end_content,
                            is_llm_message=False,
                            metadata={"thread_run_id": thread_run_id}
                        )
                        logger.info("Assistant response end saved for stream (before termination)")
                    except Exception as e:
                        logger.error(f"Error saving assistant response end for stream (before termination): {str(e)}")
                        self.trace.event(name="error_saving_assistant_response_end_for_stream_before_termination", level="ERROR", status_message=(f"Error saving assistant response end for stream (before termination): {str(e)}"))
                
                # Skip all remaining processing and go to finally block
                return

            # --- Save and Yield assistant_response_end ---
            if last_assistant_message_object: # Only save if assistant message was saved
                try:
                    # Calculate response time if we have timing data
                    if streaming_metadata["first_chunk_time"] and streaming_metadata["last_chunk_time"]:
                        streaming_metadata["response_ms"] = (streaming_metadata["last_chunk_time"] - streaming_metadata["first_chunk_time"]) * 1000

                    # Create a LiteLLM-like response object for streaming
                    # Check if we have any actual usage data
                    has_usage_data = (
                        streaming_metadata["usage"]["prompt_tokens"] > 0 or
                        streaming_metadata["usage"]["completion_tokens"] > 0 or
                        streaming_metadata["usage"]["total_tokens"] > 0
                    )
                    
                    assistant_end_content = {
                        "choices": [
                            {
                                "finish_reason": finish_reason or "stop",
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": accumulated_content,
                                    "tool_calls": complete_native_tool_calls or None
                                }
                            }
                        ],
                        "created": streaming_metadata.get("created"),
                        "model": streaming_metadata.get("model", llm_model),
                        "usage": streaming_metadata["usage"],  # Always include usage like LiteLLM does
                        "streaming": True,  # Add flag to indicate this was reconstructed from streaming
                    }
                    
                    # Only include response_ms if we have timing data
                    if streaming_metadata.get("response_ms"):
                        assistant_end_content["response_ms"] = streaming_metadata["response_ms"]
                    
                    await self.add_message(
                        thread_id=thread_id,
                        type="assistant_response_end",
                        content=assistant_end_content,
                        is_llm_message=False,
                        metadata={"thread_run_id": thread_run_id}
                    )
                    logger.info("Assistant response end saved for stream")
                except Exception as e:
                    logger.error(f"Error saving assistant response end for stream: {str(e)}")
                    self.trace.event(name="error_saving_assistant_response_end_for_stream", level="ERROR", status_message=(f"Error saving assistant response end for stream: {str(e)}"))

        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}", exc_info=True)
            self.trace.event(name="error_processing_stream", level="ERROR", status_message=(f"Error processing stream: {str(e)}"))
            # Save and yield error status message
            
            err_content = {"role": "system", "status_type": "error", "message": str(e)}
            if (not "AnthropicException - Overloaded" in str(e)):
                err_msg_obj = await self.add_message(
                    thread_id=thread_id, type="status", content=err_content, 
                    is_llm_message=False, metadata={"thread_run_id": thread_run_id if 'thread_run_id' in locals() else None}
                )
                if err_msg_obj: yield format_for_yield(err_msg_obj) # Yield the saved error message
                # Re-raise the same exception (not a new one) to ensure proper error propagation
                logger.critical(f"Re-raising error to stop further processing: {str(e)}")
                self.trace.event(name="re_raising_error_to_stop_further_processing", level="ERROR", status_message=(f"Re-raising error to stop further processing: {str(e)}"))
            else:
                logger.error(f"AnthropicException - Overloaded detected - Falling back to OpenRouter: {str(e)}", exc_info=True)
                self.trace.event(name="anthropic_exception_overloaded_detected", level="ERROR", status_message=(f"AnthropicException - Overloaded detected - Falling back to OpenRouter: {str(e)}"))
            raise # Use bare 'raise' to preserve the original exception with its traceback

        finally:
            # Save and Yield the final thread_run_end status
            try:
                end_content = {"status_type": "thread_run_end"}
                end_msg_obj = await self.add_message(
                    thread_id=thread_id, type="status", content=end_content, 
                    is_llm_message=False, metadata={"thread_run_id": thread_run_id if 'thread_run_id' in locals() else None}
                )
                if end_msg_obj: yield format_for_yield(end_msg_obj)
            except Exception as final_e:
                logger.error(f"Error in finally block: {str(final_e)}", exc_info=True)
                self.trace.event(name="error_in_finally_block", level="ERROR", status_message=(f"Error in finally block: {str(final_e)}"))

    async def process_non_streaming_response(
        self,
        llm_response: Any,
        thread_id: str,
        prompt_messages: List[Dict[str, Any]],
        llm_model: str,
        config: ProcessorConfig = ProcessorConfig(),
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a non-streaming LLM response, handling tool calls and execution.
        
        Args:
            llm_response: Response from the LLM
            thread_id: ID of the conversation thread
            prompt_messages: List of messages sent to the LLM (the prompt)
            llm_model: The name of the LLM model used
            config: Configuration for parsing and execution
            
        Yields:
            Complete message objects matching the DB schema.
        """
        content = ""
        thread_run_id = str(uuid.uuid4())
        all_tool_data = [] # Stores {'tool_call': ..., 'parsing_details': ...}
        tool_index = 0
        assistant_message_object = None
        tool_result_message_objects = {}
        finish_reason = None
        native_tool_calls_for_message = []

        try:
            # Save and Yield thread_run_start status message
            start_content = {"status_type": "thread_run_start", "thread_run_id": thread_run_id}
            start_msg_obj = await self.add_message(
                thread_id=thread_id, type="status", content=start_content,
                is_llm_message=False, metadata={"thread_run_id": thread_run_id}
            )
            if start_msg_obj: yield format_for_yield(start_msg_obj)

            # Extract finish_reason, content, tool calls
            if hasattr(llm_response, 'choices') and llm_response.choices:
                 if hasattr(llm_response.choices[0], 'finish_reason'):
                     finish_reason = llm_response.choices[0].finish_reason
                     logger.info(f"Non-streaming finish_reason: {finish_reason}")
                     self.trace.event(name="non_streaming_finish_reason", level="DEFAULT", status_message=(f"Non-streaming finish_reason: {finish_reason}"))
                 response_message = llm_response.choices[0].message if hasattr(llm_response.choices[0], 'message') else None
                 if response_message:
                     if hasattr(response_message, 'content') and response_message.content:
                         content = response_message.content
                         if config.xml_tool_calling:
                             parsed_xml_data = self._parse_xml_tool_calls(content)
                             if config.max_xml_tool_calls > 0 and len(parsed_xml_data) > config.max_xml_tool_calls:
                                 # Truncate content and tool data if limit exceeded
                                 # ... (Truncation logic similar to streaming) ...
                                 if parsed_xml_data:
                                     xml_chunks = self._extract_xml_chunks(content)[:config.max_xml_tool_calls]
                                     if xml_chunks:
                                         last_chunk = xml_chunks[-1]
                                         last_chunk_pos = content.find(last_chunk)
                                         if last_chunk_pos >= 0: content = content[:last_chunk_pos + len(last_chunk)]
                                 parsed_xml_data = parsed_xml_data[:config.max_xml_tool_calls]
                                 finish_reason = "xml_tool_limit_reached"
                             all_tool_data.extend(parsed_xml_data)

                     if config.native_tool_calling and hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                          for tool_call in response_message.tool_calls:
                             if hasattr(tool_call, 'function'):
                                 exec_tool_call = {
                                     "function_name": tool_call.function.name,
                                     "arguments": safe_json_parse(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments,
                                     "id": tool_call.id if hasattr(tool_call, 'id') else str(uuid.uuid4())
                                 }
                                 all_tool_data.append({"tool_call": exec_tool_call, "parsing_details": None})
                                 native_tool_calls_for_message.append({
                                     "id": exec_tool_call["id"], "type": "function",
                                     "function": {
                                         "name": tool_call.function.name,
                                         "arguments": tool_call.function.arguments if isinstance(tool_call.function.arguments, str) else to_json_string(tool_call.function.arguments)
                                     }
                                 })


            # --- SAVE and YIELD Final Assistant Message ---
            message_data = {"role": "assistant", "content": content, "tool_calls": native_tool_calls_for_message or None}
            assistant_message_object = await self._add_message_with_agent_info(
                thread_id=thread_id, type="assistant", content=message_data,
                is_llm_message=True, metadata={"thread_run_id": thread_run_id}
            )
            if assistant_message_object:
                 yield assistant_message_object
            else:
                 logger.error(f"Failed to save non-streaming assistant message for thread {thread_id}")
                 self.trace.event(name="failed_to_save_non_streaming_assistant_message_for_thread", level="ERROR", status_message=(f"Failed to save non-streaming assistant message for thread {thread_id}"))
                 err_content = {"role": "system", "status_type": "error", "message": "Failed to save assistant message"}
                 err_msg_obj = await self.add_message(
                     thread_id=thread_id, type="status", content=err_content, 
                     is_llm_message=False, metadata={"thread_run_id": thread_run_id}
                 )
                 if err_msg_obj: yield format_for_yield(err_msg_obj)

       # --- Execute Tools and Yield Results ---
            tool_calls_to_execute = [item['tool_call'] for item in all_tool_data]
            if config.execute_tools and tool_calls_to_execute:
                logger.info(f"Executing {len(tool_calls_to_execute)} tools with strategy: {config.tool_execution_strategy}")
                self.trace.event(name="executing_tools_with_strategy", level="DEFAULT", status_message=(f"Executing {len(tool_calls_to_execute)} tools with strategy: {config.tool_execution_strategy}"))
                tool_results = await self._execute_tools(tool_calls_to_execute, config.tool_execution_strategy)

                for i, (returned_tool_call, result) in enumerate(tool_results):
                    original_data = all_tool_data[i]
                    tool_call_from_data = original_data['tool_call']
                    parsing_details = original_data['parsing_details']
                    current_assistant_id = assistant_message_object['message_id'] if assistant_message_object else None

                    context = self._create_tool_context(
                        tool_call_from_data, tool_index, current_assistant_id, parsing_details
                    )
                    context.result = result

                    # Save and Yield start status
                    started_msg_obj = await self._yield_and_save_tool_started(context, thread_id, thread_run_id)
                    if started_msg_obj: yield format_for_yield(started_msg_obj)

                    # Save tool result
                    saved_tool_result_object = await self._add_tool_result(
                        thread_id, tool_call_from_data, result, config.xml_adding_strategy,
                        current_assistant_id, parsing_details
                    )

                    # Save and Yield completed/failed status
                    completed_msg_obj = await self._yield_and_save_tool_completed(
                        context,
                        saved_tool_result_object['message_id'] if saved_tool_result_object else None,
                        thread_id, thread_run_id
                    )
                    if completed_msg_obj: yield format_for_yield(completed_msg_obj)

                    # Yield the saved tool result object
                    if saved_tool_result_object:
                        tool_result_message_objects[tool_index] = saved_tool_result_object
                        yield format_for_yield(saved_tool_result_object)
                    else:
                         logger.error(f"Failed to save tool result for index {tool_index}")
                         self.trace.event(name="failed_to_save_tool_result_for_index", level="ERROR", status_message=(f"Failed to save tool result for index {tool_index}"))

                    tool_index += 1

            # --- Save and Yield Final Status ---
            if finish_reason:
                finish_content = {"status_type": "finish", "finish_reason": finish_reason}
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id, type="status", content=finish_content, 
                    is_llm_message=False, metadata={"thread_run_id": thread_run_id}
                )
                if finish_msg_obj: yield format_for_yield(finish_msg_obj)

            # --- Save and Yield assistant_response_end ---
            if assistant_message_object: # Only save if assistant message was saved
                try:
                    # Save the full LiteLLM response object directly in content
                    await self.add_message(
                        thread_id=thread_id,
                        type="assistant_response_end",
                        content=llm_response,
                        is_llm_message=False,
                        metadata={"thread_run_id": thread_run_id}
                    )
                    logger.info("Assistant response end saved for non-stream")
                except Exception as e:
                    logger.error(f"Error saving assistant response end for non-stream: {str(e)}")
                    self.trace.event(name="error_saving_assistant_response_end_for_non_stream", level="ERROR", status_message=(f"Error saving assistant response end for non-stream: {str(e)}"))

        except Exception as e:
             logger.error(f"Error processing non-streaming response: {str(e)}", exc_info=True)
             self.trace.event(name="error_processing_non_streaming_response", level="ERROR", status_message=(f"Error processing non-streaming response: {str(e)}"))
             # Save and yield error status
             err_content = {"role": "system", "status_type": "error", "message": str(e)}
             err_msg_obj = await self.add_message(
                 thread_id=thread_id, type="status", content=err_content, 
                 is_llm_message=False, metadata={"thread_run_id": thread_run_id if 'thread_run_id' in locals() else None}
             )
             if err_msg_obj: yield format_for_yield(err_msg_obj)
             
             # Re-raise the same exception (not a new one) to ensure proper error propagation
             logger.critical(f"Re-raising error to stop further processing: {str(e)}")
             self.trace.event(name="re_raising_error_to_stop_further_processing", level="CRITICAL", status_message=(f"Re-raising error to stop further processing: {str(e)}"))
             raise # Use bare 'raise' to preserve the original exception with its traceback

        finally:
             # Save and Yield the final thread_run_end status
            end_content = {"status_type": "thread_run_end"}
            end_msg_obj = await self.add_message(
                thread_id=thread_id, type="status", content=end_content, 
                is_llm_message=False, metadata={"thread_run_id": thread_run_id if 'thread_run_id' in locals() else None}
            )
            if end_msg_obj: yield format_for_yield(end_msg_obj)

    # XML parsing methods
    def _extract_tag_content(self, xml_chunk: str, tag_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract content between opening and closing tags, handling nested tags."""
        start_tag = f'<{tag_name}'
        end_tag = f'</{tag_name}>'
        
        try:
            # Find start tag position
            start_pos = xml_chunk.find(start_tag)
            if start_pos == -1:
                return None, xml_chunk
                
            # Find end of opening tag
            tag_end = xml_chunk.find('>', start_pos)
            if tag_end == -1:
                return None, xml_chunk
                
            # Find matching closing tag
            content_start = tag_end + 1
            nesting_level = 1
            pos = content_start
            
            while nesting_level > 0 and pos < len(xml_chunk):
                next_start = xml_chunk.find(start_tag, pos)
                next_end = xml_chunk.find(end_tag, pos)
                
                if next_end == -1:
                    return None, xml_chunk
                    
                if next_start != -1 and next_start < next_end:
                    nesting_level += 1
                    pos = next_start + len(start_tag)
                else:
                    nesting_level -= 1
                    if nesting_level == 0:
                        content = xml_chunk[content_start:next_end]
                        remaining = xml_chunk[next_end + len(end_tag):]
                        return content, remaining
                    else:
                        pos = next_end + len(end_tag)
            
            return None, xml_chunk
            
        except Exception as e:
            logger.error(f"Error extracting tag content: {e}")
            self.trace.event(name="error_extracting_tag_content", level="ERROR", status_message=(f"Error extracting tag content: {e}"))
            return None, xml_chunk

    def _extract_attribute(self, opening_tag: str, attr_name: str) -> Optional[str]:
        """Extract attribute value from opening tag."""
        try:
            # Handle both single and double quotes with raw strings
            patterns = [
                fr'{attr_name}="([^"]*)"',  # Double quotes
                fr"{attr_name}='([^']*)'",  # Single quotes
                fr'{attr_name}=([^\s/>;]+)'  # No quotes - fixed escape sequence
            ]
            
            for pattern in patterns:
                match = re.search(pattern, opening_tag)
                if match:
                    value = match.group(1)
                    # Unescape common XML entities
                    value = value.replace('&quot;', '"').replace('&apos;', "'")
                    value = value.replace('&lt;', '<').replace('&gt;', '>')
                    value = value.replace('&amp;', '&')
                    return value
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting attribute: {e}")
            self.trace.event(name="error_extracting_attribute", level="ERROR", status_message=(f"Error extracting attribute: {e}"))
            return None

    def _extract_xml_chunks(self, content: str) -> List[str]:
        """使用起始和结束模式匹配提取完整的XML块。"""
        chunks = []  # 存储提取到的所有完整XML块
        pos = 0  # 当前搜索位置的指针
        
        try:
            # 首先查找新格式的<function_calls>块
            start_pattern = '<function_calls>'
            end_pattern = '</function_calls>'
            
            while pos < len(content):
                # 查找下一个function_calls块的起始位置
                start_pos = content.find(start_pattern, pos)
                if start_pos == -1:  # 未找到起始标签，退出循环
                    break
                
                # 查找匹配的结束标签
                end_pos = content.find(end_pattern, start_pos)
                if end_pos == -1:  # 未找到结束标签，退出循环
                    break
                
                # 提取包括标签在内的完整块
                chunk_end = end_pos + len(end_pattern)
                chunk = content[start_pos:chunk_end]
                chunks.append(chunk)  # 将完整块添加到结果列表
                
                # 移动位置指针，继续查找后续块
                pos = chunk_end
            
            # 如果没有找到新格式，回退到旧格式以保持向后兼容性
            if not chunks:
                pos = 0  # 重置搜索位置
                while pos < len(content):
                    # 查找下一个工具标签
                    next_tag_start = -1
                    current_tag = None
                    
                    # 查找所有已注册标签中最早出现的那个
                    for tag_name in self.tool_registry.xml_tools.keys():
                        start_pattern = f'<{tag_name}'
                        tag_pos = content.find(start_pattern, pos)
                        
                        # 选择最早出现的标签（位置最小的那个）
                        if tag_pos != -1 and (next_tag_start == -1 or tag_pos < next_tag_start):
                            next_tag_start = tag_pos
                            current_tag = tag_name
                    
                    if next_tag_start == -1 or not current_tag:
                        break  # 没有找到更多标签，退出循环
                    
                    # 查找匹配的结束标签
                    end_pattern = f'</{current_tag}>'
                    tag_stack = []  # 用于处理嵌套标签的栈
                    chunk_start = next_tag_start
                    current_pos = next_tag_start
                    
                    while current_pos < len(content):
                        # 查找下一个起始或结束标签
                        next_start = content.find(f'<{current_tag}', current_pos + 1)
                        next_end = content.find(end_pattern, current_pos)
                        
                        if next_end == -1:  # 未找到结束标签
                            break
                        
                        if next_start != -1 and next_start < next_end:
                            # 发现嵌套的起始标签，压入栈中
                            tag_stack.append(next_start)
                            current_pos = next_start + 1
                        else:
                            # 发现结束标签
                            if not tag_stack:  # 栈为空，这是匹配的结束标签
                                chunk_end = next_end + len(end_pattern)
                                chunk = content[chunk_start:chunk_end]
                                chunks.append(chunk)  # 添加完整的嵌套标签块
                                pos = chunk_end
                                break
                            else:
                                # 弹出嵌套标签，继续处理
                                tag_stack.pop()
                                current_pos = next_end + 1
                    
                    if current_pos >= len(content):  # 到达内容末尾但未找到结束标签
                        break
                    
                    pos = max(pos + 1, current_pos)  # 确保向前移动指针
        
        except Exception as e:
            # 记录提取过程中的错误，包括原始内容用于调试
            logger.error(f"提取XML块时出错: {e}")
            logger.error(f"原始内容为: {content}")
            self.trace.event(name="error_extracting_xml_chunks", level="ERROR", status_message=(f"Error extracting XML chunks: {e}"), metadata={"content": content})
        
        return chunks  # 返回提取到的所有XML块列表

    def _parse_xml_tool_call(self, xml_chunk: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """将XML片段解析为工具调用格式并返回解析详情。
        
        Returns:
            元组 (tool_call, parsing_details) 或在解析失败时返回 None。
            - tool_call: 包含 'function_name', 'xml_tag_name', 'arguments' 的字典
            - parsing_details: 包含 'attributes', 'elements', 'text_content', 'root_content' 的字典
        """
        try:
            # 检查是否为新格式（包含 <function_calls>）
            if '<function_calls>' in xml_chunk and '<invoke' in xml_chunk:
                # Use the new XML parser
                parsed_calls = self.xml_parser.parse_content(xml_chunk)
                
                if not parsed_calls:
                    logger.error(f"No tool calls found in XML chunk: {xml_chunk}")
                    return None
                
                # Take the first tool call (should only be one per chunk)
                xml_tool_call = parsed_calls[0]
                
                # Convert to the expected format
                tool_call = {
                    "function_name": xml_tool_call.function_name,
                    "xml_tag_name": xml_tool_call.function_name.replace('_', '-'),  # For backwards compatibility
                    "arguments": xml_tool_call.parameters
                }
                
                # Include the parsing details
                parsing_details = xml_tool_call.parsing_details
                parsing_details["raw_xml"] = xml_tool_call.raw_xml
                
                logger.debug(f"Parsed new format tool call: {tool_call}")
                return tool_call, parsing_details
            
            # 回退到旧格式解析
            # 提取标签名并验证
            tag_match = re.match(r'<([^\s>]+)', xml_chunk)
            if not tag_match:
                logger.error(f"No tag found in XML chunk: {xml_chunk}")
                self.trace.event(name="no_tag_found_in_xml_chunk", level="ERROR", status_message=(f"No tag found in XML chunk: {xml_chunk}"))
                return None
            
            # This is the XML tag as it appears in the text (e.g., "create-file")
            xml_tag_name = tag_match.group(1)
            logger.info(f"Found XML tag: {xml_tag_name}")
            self.trace.event(name="found_xml_tag", level="DEFAULT", status_message=(f"Found XML tag: {xml_tag_name}"))
            
            # Get tool info and schema from registry
            tool_info = self.tool_registry.get_xml_tool(xml_tag_name)
            if not tool_info or not tool_info['schema'].xml_schema:
                logger.error(f"No tool or schema found for tag: {xml_tag_name}")
                self.trace.event(name="no_tool_or_schema_found_for_tag", level="ERROR", status_message=(f"No tool or schema found for tag: {xml_tag_name}"))
                return None
            
            # This is the actual function name to call (e.g., "create_file")
            function_name = tool_info['method']
            
            schema = tool_info['schema'].xml_schema
            params = {}
            remaining_chunk: str = xml_chunk
            
            # --- Store detailed parsing info ---
            parsing_details = {
                "attributes": {},
                "elements": {},
                "text_content": None,
                "root_content": None,
                "raw_chunk": xml_chunk # Store the original chunk for reference
            }
            # ---
            
            # Process each mapping
            for mapping in schema.mappings:
                try:
                    if mapping.node_type == "attribute":
                        # Extract attribute from opening tag
                        opening_tag = remaining_chunk.split('>', 1)[0]
                        value = self._extract_attribute(opening_tag, mapping.param_name)
                        if value is not None:
                            params[mapping.param_name] = value
                            parsing_details["attributes"][mapping.param_name] = value # Store raw attribute
                            # logger.info(f"Found attribute {mapping.param_name}: {value}")
                
                    elif mapping.node_type == "element":
                        # Extract element content
                        content, new_remaining_chunk = self._extract_tag_content(remaining_chunk, mapping.path)
                        if new_remaining_chunk is not None:
                            remaining_chunk = new_remaining_chunk
                        if content is not None:
                            params[mapping.param_name] = content.strip()
                            parsing_details["elements"][mapping.param_name] = content.strip() # Store raw element content
                            # logger.info(f"Found element {mapping.param_name}: {content.strip()}")
                
                    elif mapping.node_type == "text":
                        # Extract text content
                        content, _ = self._extract_tag_content(remaining_chunk, xml_tag_name)
                        if content is not None:
                            params[mapping.param_name] = content.strip()
                            parsing_details["text_content"] = content.strip() # Store raw text content
                            # logger.info(f"Found text content for {mapping.param_name}: {content.strip()}")
                
                    elif mapping.node_type == "content":
                        # Extract root content
                        content, _ = self._extract_tag_content(remaining_chunk, xml_tag_name)
                        if content is not None:
                            params[mapping.param_name] = content.strip()
                            parsing_details["root_content"] = content.strip() # Store raw root content
                            # logger.info(f"Found root content for {mapping.param_name}")
                
                except Exception as e:
                    logger.error(f"Error processing mapping {mapping}: {e}")
                    self.trace.event(name="error_processing_mapping", level="ERROR", status_message=(f"Error processing mapping {mapping}: {e}"))
                    continue

            # Create tool call with clear separation between function_name and xml_tag_name
            tool_call = {
                "function_name": function_name,  # The actual method to call (e.g., create_file)
                "xml_tag_name": xml_tag_name,    # The original XML tag (e.g., create-file)
                "arguments": params              # The extracted parameters
            }
            
            # logger.debug(f"Parsed old format tool call: {tool_call["function_name"]}")
            return tool_call, parsing_details # Return both dicts
            
        except Exception as e:
            logger.error(f"Error parsing XML chunk: {e}")
            logger.error(f"XML chunk was: {xml_chunk}")
            self.trace.event(name="error_parsing_xml_chunk", level="ERROR", status_message=(f"Error parsing XML chunk: {e}"), metadata={"xml_chunk": xml_chunk})
            return None

    def _parse_xml_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse XML tool calls from content string.
        
        Returns:
            List of dictionaries, each containing {'tool_call': ..., 'parsing_details': ...}
        """
        parsed_data = []
        
        try:
            xml_chunks = self._extract_xml_chunks(content)
            
            for xml_chunk in xml_chunks:
                result = self._parse_xml_tool_call(xml_chunk)
                if result:
                    tool_call, parsing_details = result
                    parsed_data.append({
                        "tool_call": tool_call,
                        "parsing_details": parsing_details
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing XML tool calls: {e}", exc_info=True)
            self.trace.event(name="error_parsing_xml_tool_calls", level="ERROR", status_message=(f"Error parsing XML tool calls: {e}"), metadata={"content": content})
        
        return parsed_data

    # Tool execution methods
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """
        执行单个工具调用并返回结果。
        该方法负责实际执行一个工具调用，包括参数解析、函数查找和执行，并处理可能出现的异常。
        Returns:
            ToolResult: 工具执行结果，包含成功状态和输出内容
        """
        # 创建跟踪跨度，用于监控工具执行过程
        span = self.trace.span(name=f"execute_tool.{tool_call['function_name']}", input=tool_call["arguments"])            
        try:
            # 提取函数名和参数
            function_name = tool_call["function_name"]
            arguments = tool_call["arguments"]

            # 记录日志，显示正在执行的工具及其参数
            logger.info(f"Executing tool: {function_name} with arguments: {arguments}")
            # 记录跟踪事件
            self.trace.event(name="executing_tool", level="DEFAULT", status_message=(f"Executing tool: {function_name} with arguments: {arguments}"))
            
            # 如果参数是字符串，尝试解析为JSON对象
            if isinstance(arguments, str):
                try:
                    arguments = safe_json_parse(arguments)
                except json.JSONDecodeError:
                    # 如果解析失败，将参数作为文本处理
                    arguments = {"text": arguments}
            
            # 从工具注册表获取可用函数
            available_functions = self.tool_registry.get_available_functions()
            
            # 根据函数名查找对应的函数
            tool_fn = available_functions.get(function_name)
            if not tool_fn:
                # 如果找不到对应函数，记录错误并返回失败结果
                logger.error(f"Tool function '{function_name}' not found in registry")
                span.end(status_message="tool_not_found", level="ERROR")
                return ToolResult(success=False, output=f"Tool function '{function_name}' not found")
            
            # 记录日志，显示找到函数并开始执行
            logger.debug(f"Found tool function for '{function_name}', executing...")
            # 执行工具函数
            result = await tool_fn(**arguments)
            # 记录日志，显示工具执行完成及其结果
            logger.info(f"Tool execution complete: {function_name} -> {result}")
            # 结束跟踪跨度
            span.end(status_message="tool_executed", output=result)
            # 返回执行结果
            return result
        except Exception as e:
            # 捕获并记录执行过程中出现的异常
            logger.error(f"Error executing tool {tool_call['function_name']}: {str(e)}", exc_info=True)
            # 结束跟踪跨度并标记错误
            span.end(status_message="tool_execution_error", output=f"Error executing tool: {str(e)}", level="ERROR")
            # 返回包含错误信息的失败结果
            return ToolResult(success=False, output=f"Error executing tool: {str(e)}")

    async def _execute_tools(
        self, 
        tool_calls: List[Dict[str, Any]], 
        execution_strategy: ToolExecutionStrategy = "sequential"
    ) -> List[Tuple[Dict[str, Any], ToolResult]]:
        """Execute tool calls with the specified strategy.
        
        This is the main entry point for tool execution. It dispatches to the appropriate
        execution method based on the provided strategy.
        
        Args:
            tool_calls: List of tool calls to execute
            execution_strategy: Strategy for executing tools:
                - "sequential": Execute tools one after another, waiting for each to complete
                - "parallel": Execute all tools simultaneously for better performance 
                
        Returns:
            List of tuples containing the original tool call and its result
        """
        logger.info(f"Executing {len(tool_calls)} tools with strategy: {execution_strategy}")
        self.trace.event(name="executing_tools_with_strategy", level="DEFAULT", status_message=(f"Executing {len(tool_calls)} tools with strategy: {execution_strategy}"))
            
        if execution_strategy == "sequential":
            return await self._execute_tools_sequentially(tool_calls)
        elif execution_strategy == "parallel":
            return await self._execute_tools_in_parallel(tool_calls)
        else:
            logger.warning(f"Unknown execution strategy: {execution_strategy}, falling back to sequential")
            return await self._execute_tools_sequentially(tool_calls)

    async def _execute_tools_sequentially(self, tool_calls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], ToolResult]]:
        """Execute tool calls sequentially and return results.
        
        This method executes tool calls one after another, waiting for each tool to complete
        before starting the next one. This is useful when tools have dependencies on each other.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tuples containing the original tool call and its result
        """
        if not tool_calls:
            return []
            
        try:
            tool_names = [t.get('function_name', 'unknown') for t in tool_calls]
            logger.info(f"Executing {len(tool_calls)} tools sequentially: {tool_names}")
            self.trace.event(name="executing_tools_sequentially", level="DEFAULT", status_message=(f"Executing {len(tool_calls)} tools sequentially: {tool_names}"))
            
            results = []
            for index, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get('function_name', 'unknown')
                logger.debug(f"Executing tool {index+1}/{len(tool_calls)}: {tool_name}")
                
                try:
                    result = await self._execute_tool(tool_call)
                    results.append((tool_call, result))
                    logger.debug(f"Completed tool {tool_name} with success={result.success}")
                    
                    # Check if this is a terminating tool (ask or complete)
                    if tool_name in ['ask', 'complete']:
                        logger.info(f"Terminating tool '{tool_name}' executed. Stopping further tool execution.")
                        self.trace.event(name="terminating_tool_executed", level="DEFAULT", status_message=(f"Terminating tool '{tool_name}' executed. Stopping further tool execution."))
                        break  # Stop executing remaining tools
                        
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    self.trace.event(name="error_executing_tool", level="ERROR", status_message=(f"Error executing tool {tool_name}: {str(e)}"))
                    error_result = ToolResult(success=False, output=f"Error executing tool: {str(e)}")
                    results.append((tool_call, error_result))
            
            logger.info(f"Sequential execution completed for {len(results)} tools (out of {len(tool_calls)} total)")
            self.trace.event(name="sequential_execution_completed", level="DEFAULT", status_message=(f"Sequential execution completed for {len(results)} tools (out of {len(tool_calls)} total)"))
            return results
            
        except Exception as e:
            logger.error(f"Error in sequential tool execution: {str(e)}", exc_info=True)
            # Return partial results plus error results for remaining tools
            completed_results = results if 'results' in locals() else []
            completed_tool_names = [r[0].get('function_name', 'unknown') for r in completed_results]
            remaining_tools = [t for t in tool_calls if t.get('function_name', 'unknown') not in completed_tool_names]
            
            # Add error results for remaining tools
            error_results = [(tool, ToolResult(success=False, output=f"Execution error: {str(e)}")) 
                            for tool in remaining_tools]
                            
            return completed_results + error_results

    async def _execute_tools_in_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], ToolResult]]:
        """Execute tool calls in parallel and return results.
        
        This method executes all tool calls simultaneously using asyncio.gather, which
        can significantly improve performance when executing multiple independent tools.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tuples containing the original tool call and its result
        """
        if not tool_calls:
            return []
            
        try:
            tool_names = [t.get('function_name', 'unknown') for t in tool_calls]
            logger.info(f"Executing {len(tool_calls)} tools in parallel: {tool_names}")
            self.trace.event(name="executing_tools_in_parallel", level="DEFAULT", status_message=(f"Executing {len(tool_calls)} tools in parallel: {tool_names}"))
            
            # Create tasks for all tool calls
            tasks = [self._execute_tool(tool_call) for tool_call in tool_calls]
            
            # Execute all tasks concurrently with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            processed_results = []
            for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
                if isinstance(result, Exception):
                    logger.error(f"Error executing tool {tool_call.get('function_name', 'unknown')}: {str(result)}")
                    self.trace.event(name="error_executing_tool", level="ERROR", status_message=(f"Error executing tool {tool_call.get('function_name', 'unknown')}: {str(result)}"))
                    # Create error result
                    error_result = ToolResult(success=False, output=f"Error executing tool: {str(result)}")
                    processed_results.append((tool_call, error_result))
                else:
                    processed_results.append((tool_call, result))
            
            logger.info(f"Parallel execution completed for {len(tool_calls)} tools")
            self.trace.event(name="parallel_execution_completed", level="DEFAULT", status_message=(f"Parallel execution completed for {len(tool_calls)} tools"))
            return processed_results
        
        except Exception as e:
            logger.error(f"Error in parallel tool execution: {str(e)}", exc_info=True)
            self.trace.event(name="error_in_parallel_tool_execution", level="ERROR", status_message=(f"Error in parallel tool execution: {str(e)}"))
            # Return error results for all tools if the gather itself fails
            return [(tool_call, ToolResult(success=False, output=f"Execution error: {str(e)}")) 
                    for tool_call in tool_calls]

    async def _add_tool_result(
        self, 
        thread_id: str, 
        tool_call: Dict[str, Any], 
        result: ToolResult,
        strategy: Union[XmlAddingStrategy, str] = "assistant_message",
        assistant_message_id: Optional[str] = None,
        parsing_details: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]: # Return the full message object
        """Add a tool result to the conversation thread based on the specified format.
        
        This method formats tool results and adds them to the conversation history,
        making them visible to the LLM in subsequent interactions. Results can be 
        added either as native tool messages (OpenAI format) or as XML-wrapped content
        with a specified role (user or assistant).
        
        Args:
            thread_id: ID of the conversation thread
            tool_call: The original tool call that produced this result
            result: The result from the tool execution
            strategy: How to add XML tool results to the conversation
                     ("user_message", "assistant_message", or "inline_edit")
            assistant_message_id: ID of the assistant message that generated this tool call
            parsing_details: Detailed parsing info for XML calls (attributes, elements, etc.)
        """
        try:
            message_obj = None # Initialize message_obj
            
            # Create metadata with assistant_message_id if provided
            metadata = {}
            if assistant_message_id:
                metadata["assistant_message_id"] = assistant_message_id
                logger.info(f"Linking tool result to assistant message: {assistant_message_id}")
                self.trace.event(name="linking_tool_result_to_assistant_message", level="DEFAULT", status_message=(f"Linking tool result to assistant message: {assistant_message_id}"))
            
            # --- Add parsing details to metadata if available ---
            if parsing_details:
                metadata["parsing_details"] = parsing_details
                logger.info("Adding parsing_details to tool result metadata")
                self.trace.event(name="adding_parsing_details_to_tool_result_metadata", level="DEFAULT", status_message=(f"Adding parsing_details to tool result metadata"), metadata={"parsing_details": parsing_details})
            # ---
            
            # Check if this is a native function call (has id field)
            if "id" in tool_call:
                # Format as a proper tool message according to OpenAI spec
                function_name = tool_call.get("function_name", "")
                
                # Format the tool result content - tool role needs string content
                if isinstance(result, str):
                    content = result
                elif hasattr(result, 'output'):
                    # If it's a ToolResult object
                    if isinstance(result.output, dict) or isinstance(result.output, list):
                        # If output is already a dict or list, convert to JSON string
                        content = json.dumps(result.output)
                    else:
                        # Otherwise just use the string representation
                        content = str(result.output)
                else:
                    # Fallback to string representation of the whole result
                    content = str(result)
                
                logger.info(f"Formatted tool result content: {content[:100]}...")
                self.trace.event(name="formatted_tool_result_content", level="DEFAULT", status_message=(f"Formatted tool result content: {content[:100]}..."))
                
                # Create the tool response message with proper format
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": content
                }
                
                logger.info(f"Adding native tool result for tool_call_id={tool_call['id']} with role=tool")
                self.trace.event(name="adding_native_tool_result_for_tool_call_id", level="DEFAULT", status_message=(f"Adding native tool result for tool_call_id={tool_call['id']} with role=tool"))
                
                # Add as a tool message to the conversation history
                # This makes the result visible to the LLM in the next turn
                message_obj = await self.add_message(
                    thread_id=thread_id,
                    type="tool",  # Special type for tool responses
                    content=tool_message,
                    is_llm_message=True,
                    metadata=metadata
                )
                return message_obj # Return the full message object
            
            # For XML and other non-native tools, use the new structured format
            # Determine message role based on strategy
            result_role = "user" if strategy == "user_message" else "assistant"
            
            # Create the new structured tool result format
            structured_result = self._create_structured_tool_result(tool_call, result, parsing_details)
            
            # Add the message with the appropriate role to the conversation history
            # This allows the LLM to see the tool result in subsequent interactions
            result_message = {
                "role": result_role,
                "content":  json.dumps(structured_result)
            }
            message_obj = await self.add_message(
                thread_id=thread_id, 
                type="tool",
                content=result_message,
                is_llm_message=True,
                metadata=metadata
            )
            return message_obj # Return the full message object
        except Exception as e:
            logger.error(f"Error adding tool result: {str(e)}", exc_info=True)
            self.trace.event(name="error_adding_tool_result", level="ERROR", status_message=(f"Error adding tool result: {str(e)}"), metadata={"tool_call": tool_call, "result": result, "strategy": strategy, "assistant_message_id": assistant_message_id, "parsing_details": parsing_details})
            # Fallback to a simple message
            try:
                fallback_message = {
                    "role": "user",
                    "content": str(result)
                }
                message_obj = await self.add_message(
                    thread_id=thread_id, 
                    type="tool", 
                    content=fallback_message,
                    is_llm_message=True,
                    metadata={"assistant_message_id": assistant_message_id} if assistant_message_id else {}
                )
                return message_obj # Return the full message object
            except Exception as e2:
                logger.error(f"Failed even with fallback message: {str(e2)}", exc_info=True)
                self.trace.event(name="failed_even_with_fallback_message", level="ERROR", status_message=(f"Failed even with fallback message: {str(e2)}"), metadata={"tool_call": tool_call, "result": result, "strategy": strategy, "assistant_message_id": assistant_message_id, "parsing_details": parsing_details})
                return None # Return None on error

    def _create_structured_tool_result(self, tool_call: Dict[str, Any], result: ToolResult, parsing_details: Optional[Dict[str, Any]] = None):
        """Create a structured tool result format that's tool-agnostic and provides rich information.
        
        Args:
            tool_call: The original tool call that was executed
            result: The result from the tool execution
            parsing_details: Optional parsing details for XML calls
            
        Returns:
            Structured dictionary containing tool execution information
        """
        # Extract tool information
        function_name = tool_call.get("function_name", "unknown")
        xml_tag_name = tool_call.get("xml_tag_name")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id")
        logger.info(f"Creating structured tool result for tool_call: {tool_call}")
        
        # Process the output - if it's a JSON string, parse it back to an object
        output = result.output if hasattr(result, 'output') else str(result)
        if isinstance(output, str):
            try:
                # Try to parse as JSON to provide structured data to frontend
                parsed_output = safe_json_parse(output)
                # If parsing succeeded and we got a dict/list, use the parsed version
                if isinstance(parsed_output, (dict, list)):
                    output = parsed_output
                # Otherwise keep the original string
            except Exception:
                # If parsing fails, keep the original string
                pass
        
        # Create the structured result
        structured_result_v1 = {
            "tool_execution": {
                "function_name": function_name,
                "xml_tag_name": xml_tag_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "result": {
                    "success": result.success if hasattr(result, 'success') else True,
                    "output": output,  # Now properly structured for frontend
                    "error": getattr(result, 'error', None) if hasattr(result, 'error') else None
                },
                # "execution_details": {
                #     "timestamp": datetime.now(timezone.utc).isoformat(),
                #     "parsing_details": parsing_details
                # }
            }
        } 

        # STRUCTURED_OUTPUT_TOOLS = {
        #     "str_replace", 
        #     "get_data_provider_endpoints",
        # }

        # summary_output = result.output if hasattr(result, 'output') else str(result)
        
        # if xml_tag_name:
        #     status = "completed successfully" if structured_result_v1["tool_execution"]["result"]["success"] else "failed"
        #     summary = f"Tool '{xml_tag_name}' {status}. Output: {summary_output}"
        # else:
        #     status = "completed successfully" if structured_result_v1["tool_execution"]["result"]["success"] else "failed"
        #     summary = f"Function '{function_name}' {status}. Output: {summary_output}"
        
        # if self.is_agent_builder:
        #     return summary
        # if function_name in STRUCTURED_OUTPUT_TOOLS:
        #     return structured_result_v1
        # else:
        #     return summary

        summary_output = result.output if hasattr(result, 'output') else str(result)
        success_status = structured_result_v1["tool_execution"]["result"]["success"]
        
        # # Create a more comprehensive summary for the LLM
        # if xml_tag_name:
        #     status = "completed successfully" if structured_result_v1["tool_execution"]["result"]["success"] else "failed"
        #     summary = f"Tool '{xml_tag_name}' {status}. Output: {summary_output}"
        # else:
        #     status = "completed successfully" if structured_result_v1["tool_execution"]["result"]["success"] else "failed"
        #     summary = f"Function '{function_name}' {status}. Output: {summary_output}"
        
        # if self.is_agent_builder:
        #     return summary
        # elif function_name == "get_data_provider_endpoints":
        #     logger.info(f"Returning sumnary for data provider call: {summary}")
        #     return summary
            
        return structured_result_v1

    def _create_tool_context(self, tool_call: Dict[str, Any], tool_index: int, assistant_message_id: Optional[str] = None, parsing_details: Optional[Dict[str, Any]] = None) -> ToolExecutionContext:
        """
        创建一个工具执行上下文，包含显示名称和解析详情。
        Returns:
            ToolExecutionContext: 包含完整工具执行上下文信息的对象
        """
        # 初始化工具执行上下文对象
        context = ToolExecutionContext(
            tool_call=tool_call,           # 原始工具调用信息
            tool_index=tool_index,         # 工具索引
            assistant_message_id=assistant_message_id,  # 关联的助手消息ID
            parsing_details=parsing_details  # 解析详情（主要用于XML工具）
        )
        
        # 设置函数名和XML标签名字段
        if "xml_tag_name" in tool_call:
            # 对于XML工具调用，设置XML标签名和函数名
            context.xml_tag_name = tool_call["xml_tag_name"]
            context.function_name = tool_call.get("function_name", tool_call["xml_tag_name"])
        else:
            # 对于非XML工具，直接使用函数名
            context.function_name = tool_call.get("function_name", "unknown")
            context.xml_tag_name = None
        
        return context
        
    async def _yield_and_save_tool_started(self, context: ToolExecutionContext, thread_id: str, thread_run_id: str) -> Optional[Dict[str, Any]]:
        """
        格式化、保存并返回工具开始执行的状态消息。
        该方法用于通知系统某个工具已经开始执行，创建一个状态消息并保存到数据库中。
        Returns:
            Optional[Dict[str, Any]]: 保存后的完整消息对象，如果保存失败则返回None
        """

        tool_name = context.xml_tag_name or context.function_name
        content = {
            "role": "assistant", "status_type": "tool_started",
            "function_name": context.function_name, "xml_tag_name": context.xml_tag_name,
            "message": f"Starting execution of {tool_name}", "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id") # Include tool_call ID if native
        }
        metadata = {"thread_run_id": thread_run_id}
        saved_message_obj = await self.add_message(
            thread_id=thread_id, type="status", content=content, is_llm_message=False, metadata=metadata
        )
        return saved_message_obj # Return the full object (or None if saving failed)

    async def _yield_and_save_tool_completed(self, context: ToolExecutionContext, tool_message_id: Optional[str], thread_id: str, thread_run_id: str) -> Optional[Dict[str, Any]]:
        """Formats, saves, and returns a tool completed/failed status message."""
        if not context.result:
            # Delegate to error saving if result is missing (e.g., execution failed)
            return await self._yield_and_save_tool_error(context, thread_id, thread_run_id)

        tool_name = context.xml_tag_name or context.function_name
        status_type = "tool_completed" if context.result.success else "tool_failed"
        message_text = f"Tool {tool_name} {'completed successfully' if context.result.success else 'failed'}"

        content = {
            "role": "assistant", "status_type": status_type,
            "function_name": context.function_name, "xml_tag_name": context.xml_tag_name,
            "message": message_text, "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id")
        }
        metadata = {"thread_run_id": thread_run_id}
        # Add the *actual* tool result message ID to the metadata if available and successful
        if context.result.success and tool_message_id:
            metadata["linked_tool_result_message_id"] = tool_message_id
            
        # <<< ADDED: Signal if this is a terminating tool >>>
        if context.function_name in ['ask', 'complete']:
            metadata["agent_should_terminate"] = "true"
            logger.info(f"Marking tool status for '{context.function_name}' with termination signal.")
            self.trace.event(name="marking_tool_status_for_termination", level="DEFAULT", status_message=(f"Marking tool status for '{context.function_name}' with termination signal."))
        # <<< END ADDED >>>

        saved_message_obj = await self.add_message(
            thread_id=thread_id, type="status", content=content, is_llm_message=False, metadata=metadata
        )
        return saved_message_obj

    async def _yield_and_save_tool_error(self, context: ToolExecutionContext, thread_id: str, thread_run_id: str) -> Optional[Dict[str, Any]]:
        """Formats, saves, and returns a tool error status message."""
        error_msg = str(context.error) if context.error else "Unknown error during tool execution"
        tool_name = context.xml_tag_name or context.function_name
        content = {
            "role": "assistant", "status_type": "tool_error",
            "function_name": context.function_name, "xml_tag_name": context.xml_tag_name,
            "message": f"Error executing tool {tool_name}: {error_msg}",
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id")
        }
        metadata = {"thread_run_id": thread_run_id}
        # Save the status message with is_llm_message=False
        saved_message_obj = await self.add_message(
            thread_id=thread_id, type="status", content=content, is_llm_message=False, metadata=metadata
        )
        return saved_message_obj
