"""
Conversation thread management system for AgentPress.

This module provides comprehensive conversation management, including:
- Thread creation and persistence
- Message handling with support for text and images
- Tool registration and execution
- LLM interaction with streaming support
- Error handling and cleanup
- Context summarization to manage token limits
"""

import json
from typing import List, Dict, Any, Optional, Type, Union, AsyncGenerator, Literal, cast
from services.llm import make_llm_api_call
from agentpress.tool import Tool
from agentpress.tool_registry import ToolRegistry
from agentpress.context_manager import ContextManager
from agentpress.response_processor import (
    ResponseProcessor,
    ProcessorConfig
)
from services.supabase import DBConnection
from utils.logger import logger
from langfuse.client import StatefulGenerationClient, StatefulTraceClient
from services.langfuse import langfuse
import datetime
from litellm.utils import token_counter

# Type alias for tool choice
ToolChoice = Literal["auto", "required", "none"]

class ThreadManager:
    """Manages conversation threads with LLM models and tool execution.

    Provides comprehensive conversation management, handling message threading,
    tool registration, and LLM interactions with support for both standard and
    XML-based tool execution patterns.
    """

    def __init__(self, trace: Optional[StatefulTraceClient] = None, is_agent_builder: bool = False, target_agent_id: Optional[str] = None, agent_config: Optional[dict] = None):
        """Initialize ThreadManager.

        Args:
            trace: Optional trace client for logging
            is_agent_builder: Whether this is an agent builder session
            target_agent_id: ID of the agent being built (if in agent builder mode)
            agent_config: Optional agent configuration with version information
        """
        self.db = DBConnection()
        self.tool_registry = ToolRegistry()
        self.trace = trace
        self.is_agent_builder = is_agent_builder
        self.target_agent_id = target_agent_id
        self.agent_config = agent_config
        if not self.trace:
            self.trace = langfuse.trace(name="anonymous:thread_manager")
        self.response_processor = ResponseProcessor(
            tool_registry=self.tool_registry,
            add_message_callback=self.add_message,
            trace=self.trace,
            is_agent_builder=self.is_agent_builder,
            target_agent_id=self.target_agent_id,
            agent_config=self.agent_config
        )
        self.context_manager = ContextManager()

    def add_tool(self, tool_class: Type[Tool], function_names: Optional[List[str]] = None, **kwargs):
        """Add a tool to the ThreadManager."""
        self.tool_registry.register_tool(tool_class, function_names, **kwargs)

    async def add_message(
        self,
        thread_id: str,
        type: str,
        content: Union[Dict[str, Any], List[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_version_id: Optional[str] = None
    ):
        """向线程的数据库中添加一条消息。
        
        参数说明:
        thread_id: 要添加消息的线程ID
        type: 消息类型 (例如: 'text', 'image_url', 'tool_call', 'tool', 'user', 'assistant')
        content: 消息内容，可以是字典、列表或字符串，将作为JSONB存储在数据库中
        is_llm_message: 标记消息是否来自LLM，默认为False(用户消息)
        metadata: 可选的附加消息元数据，默认为None，如果为None则存储为空的JSONB对象
            agent_id: 与此消息关联的代理ID(可选)
            agent_version_id: 使用的特定代理版本ID(可选)
        """
        logger.debug(f"Adding message of type '{type}' to thread {thread_id} (agent: {agent_id}, version: {agent_version_id})")
        # 获取数据库客户端
        client = await self.db.client

        # 准备要插入的数据
        data_to_insert = {
            'thread_id': thread_id,
            'type': type,
            'content': content,
            'is_llm_message': is_llm_message,
            'metadata': metadata or {},
        }
        
        # 如果提供了代理信息，则添加到数据中
        if agent_id:
            data_to_insert['agent_id'] = agent_id
        if agent_version_id:
            data_to_insert['agent_version_id'] = agent_version_id

        try:
            # 插入消息并获取插入的行数据(包括ID)
            result = await client.table('messages').insert(data_to_insert).execute()
            logger.info(f"Successfully added message to thread {thread_id}")

            # 检查结果并返回插入的数据
            if result.data and len(result.data) > 0 and isinstance(result.data[0], dict) and 'message_id' in result.data[0]:
                return result.data[0]
            else:
                logger.error(f"Insert operation failed or did not return expected data structure for thread {thread_id}. Result data: {result.data}")
                return None
        except Exception as e:
            logger.error(f"Failed to add message to thread {thread_id}: {str(e)}", exc_info=True)
            raise

    async def get_llm_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """获取线程的所有消息。

        该方法使用SQL函数处理上下文截断，
        通过考虑摘要消息来实现。

        参数:
            thread_id: 要获取消息的线程ID

        返回:
            消息对象列表
        """
        logger.debug(f"Getting messages for thread {thread_id}")
        client = await self.db.client

        try:
            # 原始SQL函数调用方式(已注释)
            # result = await client.rpc('get_llm_formatted_messages', {'p_thread_id': thread_id}).execute()
            
            # 分批获取消息(每批1000条)以避免数据库过载
            all_messages = []
            batch_size = 1000
            offset = 0
            
            while True:
                # 从messages表查询指定thread_id的消息
                result = await client.table('messages')
                    .select('message_id, content')
                    .eq('thread_id', thread_id)
                    .eq('is_llm_message', True)
                    .order('created_at')
                    .range(offset, offset + batch_size - 1)
                    .execute()
                
                # 如果没有数据或数据为空则终止循环
                if not result.data or len(result.data) == 0:
                    break
                    
                # 将当前批次数据添加到总列表
                all_messages.extend(result.data)
                
                # 如果获取的记录数小于批次大小，说明已到达末尾
                if len(result.data) < batch_size:
                    break
                    
                offset += batch_size
            
            # 使用all_messages代替result.data
            result_data = all_messages

            # 解析返回的数据(可能是字符串化的JSON)
            if not result_data:
                return []

            # 返回正确解析的JSON对象
            messages = []
            for item in result_data:
                if isinstance(item['content'], str):
                    try:
                        # 尝试解析字符串内容为JSON
                        parsed_item = json.loads(item['content'])
                        parsed_item['message_id'] = item['message_id']
                        messages.append(parsed_item)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse message: {item['content']}")
                else:
                    # 如果内容已经是字典格式，直接使用
                    content = item['content']
                    content['message_id'] = item['message_id']
                    messages.append(content)

            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for thread {thread_id}: {str(e)}", exc_info=True)
            return []

    async def run_thread(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        stream: bool = True,
        temporary_message: Optional[Dict[str, Any]] = None,
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0,
        llm_max_tokens: Optional[int] = None,
        processor_config: Optional[ProcessorConfig] = None,
        tool_choice: ToolChoice = "auto",
        native_max_auto_continues: int = 25,
        max_xml_tool_calls: int = 0,
        include_xml_examples: bool = False,
        enable_thinking: Optional[bool] = False,
        reasoning_effort: Optional[str] = 'low',
        enable_context_manager: bool = True,
        generation: Optional[StatefulGenerationClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """运行一个包含LLM集成和工具执行的对话线程。

        参数:
            thread_id: 要运行的线程ID
            system_prompt: 设置助手行为的系统消息
            stream: 是否使用LLM的流式API
            temporary_message: 仅本次运行可选的临时用户消息
            llm_model: 使用的LLM模型名称
            llm_temperature: 响应随机性的温度参数(0-1)
            llm_max_tokens: LLM响应的最大token数
            processor_config: 响应处理器的配置
            tool_choice: 工具选择偏好("auto", "required", "none")
            native_max_auto_continues: 当finish_reason="tool_calls"时的最大自动继续次数(0表示禁用自动继续)
            max_xml_tool_calls: 允许的最大XML工具调用次数(0表示无限制)
            include_xml_examples: 是否在系统提示中包含XML工具示例
            enable_thinking: 是否在决策前启用思考
            reasoning_effort: 推理努力级别
            enable_context_manager: 是否启用自动上下文摘要

        返回:
            一个异步生成器，产生响应块或错误字典
        """

        logger.info(f"Starting thread execution for thread {thread_id}")
        logger.info(f"Using model: {llm_model}")
        # Log parameters
        logger.info(f"Parameters: model={llm_model}, temperature={llm_temperature}, max_tokens={llm_max_tokens}")
        logger.info(f"Auto-continue: max={native_max_auto_continues}, XML tool limit={max_xml_tool_calls}")

        # Log model info
        logger.info(f"🤖 Thread {thread_id}: Using model {llm_model}")

        # 确保processor_config不为None，否则使用默认配置
        config = processor_config or ProcessorConfig()

        # 如果指定了max_xml_tool_calls且配置中未设置，则应用该值
        if max_xml_tool_calls > 0 and not config.max_xml_tool_calls:
            config.max_xml_tool_calls = max_xml_tool_calls

        # 创建系统提示的工作副本以便修改
        working_system_prompt = system_prompt.copy()

        # 如果需要添加XML示例且配置启用了XML工具调用
        if include_xml_examples and config.xml_tool_calling:
            # 从工具注册表获取XML示例
            xml_examples = self.tool_registry.get_xml_examples()
            if xml_examples:
                # 构建XML工具调用说明内容
                examples_content = """
 --- XML TOOL CALLING ---

 In this environment you have access to a set of tools you can use to answer the user's question. The tools are specified in XML format.
 Format your tool calls using the specified XML tags. Place parameters marked as 'attribute' within the opening tag (e.g., `<tag attribute='value'>`). Place parameters marked as 'content' between the opening and closing tags. Place parameters marked as 'element' within their own child tags (e.g., `<tag><element>value</element></tag>`). Refer to the examples provided below for the exact structure of each tool.
 String and scalar parameters should be specified as attributes, while content goes between tags.
 Note that spaces for string values are not stripped. The output is parsed with regular expressions.

 Here are the XML tools available with examples:
                """
                # 添加每个工具的示例
                for tag_name, example in xml_examples.items():
                    examples_content += f"<{tag_name}> Example: {example}\n"

                # 获取系统提示内容
                system_content = working_system_prompt.get('content')

                # 根据系统提示内容的类型进行处理
                if isinstance(system_content, str):
                    # 字符串类型直接追加
                    working_system_prompt['content'] += examples_content
                    logger.debug("Appended XML examples to string system prompt content.")
                elif isinstance(system_content, list):
                    # 列表类型查找第一个文本块追加
                    appended = False
                    for item in working_system_prompt['content']:
                        if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                            item['text'] += examples_content
                            logger.debug("Appended XML examples to the first text block in list system prompt content.")
                            appended = True
                            break
                    if not appended:
                        logger.warning("System prompt content is a list but no text block found to append XML examples.")
                else:
                    logger.warning(f"System prompt content is of unexpected type ({type(system_content)}), cannot add XML examples.")
        
        # 控制是否因tool_calls完成原因需要自动继续
        auto_continue = True
        auto_continue_count = 0
        
        # 定义内部函数处理单次运行
        async def _run_once(temp_msg=None):
            try:
                # 确保config在此作用域可用
                nonlocal config
                # 注意：由于上面的检查，config现在保证存在
        
                # 1. 从线程获取消息用于LLM调用
                messages = await self.get_llm_messages(thread_id)
        
                # 2. 继续前检查token计数
                token_count = 0
                try:
                    # 使用可能修改过的working_system_prompt进行token计数
                    token_count = token_counter(model=llm_model, messages=[working_system_prompt] + messages)
                    token_threshold = self.context_manager.token_threshold
                    logger.info(f"Thread {thread_id} token count: {token_count}/{token_threshold} ({(token_count/token_threshold)*100:.1f}%)")
        
                except Exception as e:
                    logger.error(f"Error counting tokens or summarizing: {str(e)}")
        
                # 3. 准备LLM调用的消息 + 添加临时消息(如果存在)
                # 使用可能包含XML示例的working_system_prompt
                prepared_messages = [working_system_prompt]
        
                # 查找最后一个用户消息的索引
                last_user_index = -1
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'user':
                        last_user_index = i
        
                # 如果存在临时消息且找到用户消息，将其插入到最后一个用户消息前
                if temp_msg and last_user_index >= 0:
                    prepared_messages.extend(messages[:last_user_index])
                    prepared_messages.append(temp_msg)
                    prepared_messages.extend(messages[last_user_index:])
                    logger.debug("Added temporary message before the last user message")
                else:
                    # 如果没有用户消息或临时消息，直接添加所有消息
                    prepared_messages.extend(messages)
                    if temp_msg:
                        prepared_messages.append(temp_msg)
                        logger.debug("Added temporary message to the end of prepared messages")
        
                # 4. 准备LLM调用的工具
                openapi_tool_schemas = None
                if config.native_tool_calling:
                    openapi_tool_schemas = self.tool_registry.get_openapi_schemas()
                    logger.debug(f"Retrieved {len(openapi_tool_schemas) if openapi_tool_schemas else 0} OpenAPI tool schemas")
        
                prepared_messages = self.context_manager.compress_messages(prepared_messages, llm_model)
        
                # 5. 进行LLM API调用
                logger.debug("Making LLM API call")
                try:
                    if generation:
                        generation.update(
                            input=prepared_messages,
                            start_time=datetime.datetime.now(datetime.timezone.utc),
                            model=llm_model,
                            model_parameters={
                              "max_tokens": llm_max_tokens,
                              "temperature": llm_temperature,
                              "enable_thinking": enable_thinking,
                              "reasoning_effort": reasoning_effort,
                              "tool_choice": tool_choice,
                              "tools": openapi_tool_schemas,
                            }
                        )
                    llm_response = await make_llm_api_call(
                        prepared_messages, # 传递可能修改过的消息
                        llm_model,
                        temperature=llm_temperature,
                        max_tokens=llm_max_tokens,
                        tools=openapi_tool_schemas,
                        tool_choice=tool_choice if config.native_tool_calling else "none",
                        stream=stream,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort
                    )
                    logger.debug("Successfully received raw LLM API response stream/object")
        
                except Exception as e:
                    logger.error(f"Failed to make LLM API call: {str(e)}", exc_info=True)
                    raise
        
                # 6. 使用ResponseProcessor处理LLM响应
                if stream:
                    logger.debug("Processing streaming response")
                    # 确保我们有异步生成器用于流式传输
                    if hasattr(llm_response, '__aiter__'):
                        response_generator = self.response_processor.process_streaming_response(
                            llm_response=cast(AsyncGenerator, llm_response),
                            thread_id=thread_id,
                            config=config,
                            prompt_messages=prepared_messages,
                            llm_model=llm_model,
                        )
                    else:
                        # 如果响应不可迭代，回退到非流式处理
                        response_generator = self.response_processor.process_non_streaming_response(
                            llm_response=llm_response,
                            thread_id=thread_id,
                            config=config,
                            prompt_messages=prepared_messages,
                            llm_model=llm_model,
                        )
        
                    return response_generator
                else:
                    logger.debug("Processing non-streaming response")
                    # 直接传递响应生成器而不使用try/except，让错误向上传播
                    response_generator = self.response_processor.process_non_streaming_response(
                        llm_response=llm_response,
                        thread_id=thread_id,
                        config=config,
                        prompt_messages=prepared_messages,
                        llm_model=llm_model,
                    )
                    return response_generator # 返回生成器
        
            except Exception as e:
                logger.error(f"Error in run_thread: {str(e)}", exc_info=True)
                # 返回错误字典供调用者处理
                return {
                    "type": "status",
                    "status": "error",
                    "message": str(e)
                }

        # 定义自动继续包装器函数
        async def auto_continue_wrapper():
            nonlocal auto_continue, auto_continue_count

            # 当需要自动继续且未达到最大次数时循环
            while auto_continue and (native_max_auto_continues == 0 or auto_continue_count < native_max_auto_continues):
                # 重置自动继续标志
                auto_continue = False

                # 运行线程一次，传递可能修改的系统提示
                # 仅在第一次迭代时传递临时消息
                try:
                    response_gen = await _run_once(temporary_message if auto_continue_count == 0 else None)

                    # 处理错误响应
                    if isinstance(response_gen, dict) and "status" in response_gen and response_gen["status"] == "error":
                        logger.error(f"Error in auto_continue_wrapper: {response_gen.get('message', 'Unknown error')}")
                        yield response_gen
                        return  # 错误时退出生成器

                    # 处理每个数据块
                    try:
                        if hasattr(response_gen, '__aiter__'):
                            async for chunk in cast(AsyncGenerator, response_gen):
                                # 检查是否是带有tool_calls或xml_tool_limit_reached的完成原因块
                                if chunk.get('type') == 'finish':
                                    if chunk.get('finish_reason') == 'tool_calls':
                                        # 仅在启用时自动继续(max > 0)
                                        if native_max_auto_continues > 0:
                                            logger.info(f"Detected finish_reason='tool_calls', auto-continuing ({auto_continue_count + 1}/{native_max_auto_continues})")
                                            auto_continue = True
                                            auto_continue_count += 1
                                            # 不返回完成块以避免混淆客户端
                                            continue
                                    elif chunk.get('finish_reason') == 'xml_tool_limit_reached':
                                        # 如果达到XML工具限制则不自动继续
                                        logger.info(f"Detected finish_reason='xml_tool_limit_reached', stopping auto-continue")
                                        auto_continue = False
                                        # 仍然返回块以通知客户端

                                # 否则正常返回块
                                yield chunk
                        else:
                            # response_gen不可迭代(可能是错误字典)，直接返回
                            yield response_gen

                        # 如果不自动继续，则完成
                        if not auto_continue:
                            break
                    except Exception as e:
                        if ("AnthropicException - Overloaded" in str(e)):
                            # 处理Anthropic过载异常，回退到OpenRouter
                            logger.error(f"AnthropicException - Overloaded detected - Falling back to OpenRouter: {str(e)}", exc_info=True)
                            nonlocal llm_model
                            llm_model = f"openrouter/{llm_model}"
                            auto_continue = True
                            continue # 继续循环
                        else:
                            # 其他异常，记录错误并返回错误状态
                            logger.error(f"Error in auto_continue_wrapper generator: {str(e)}", exc_info=True)
                            yield {
                                "type": "status",
                                "status": "error",
                                "message": f"Error in thread processing: {str(e)}"
                            }
                        return  # 任何错误时退出生成器
                except Exception as outer_e:
                    # 捕获_run_once本身的异常
                    logger.error(f"Error executing thread: {str(outer_e)}", exc_info=True)
                    yield {
                        "type": "status",
                        "status": "error",
                        "message": f"Error executing thread: {str(outer_e)}"
                    }
                    return  # _run_once异常时立即退出

            # 如果达到最大自动继续次数，记录警告
            if auto_continue and auto_continue_count >= native_max_auto_continues:
                logger.warning(f"Reached maximum auto-continue limit ({native_max_auto_continues}), stopping.")
                yield {
                    "type": "content",
                    "content": f"\n[Agent reached maximum auto-continue limit of {native_max_auto_continues}]"
                }

        #如果禁用自动继续（max=0），只需运行一次 
        if native_max_auto_continues == 0:
            logger.info("Auto-continue is disabled (native_max_auto_continues=0)")
            #传递可能修改的系统提示和临时消息 
            return await _run_once(temporary_message)

        #否则返回自动继续包装生成器
        return auto_continue_wrapper()
