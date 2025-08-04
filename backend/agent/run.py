import os
import json
import asyncio
from typing import Optional

# from agent.tools.message_tool import MessageTool
from agent.tools.message_tool import MessageTool
from agent.tools.sb_deploy_tool import SandboxDeployTool
from agent.tools.sb_expose_tool import SandboxExposeTool
from agent.tools.web_search_tool import SandboxWebSearchTool
from dotenv import load_dotenv
from utils.config import config
from flags.flags import is_enabled
from agent.agent_builder_prompt import get_agent_builder_prompt
from agentpress.thread_manager import ThreadManager
from agentpress.response_processor import ProcessorConfig
from agent.tools.sb_shell_tool import SandboxShellTool
from agent.tools.sb_files_tool import SandboxFilesTool
from agent.tools.sb_browser_tool import SandboxBrowserTool
from agent.tools.data_providers_tool import DataProvidersTool
from agent.tools.expand_msg_tool import ExpandMessageTool
from agent.prompt import get_system_prompt
from utils.logger import logger
from utils.auth_utils import get_account_id_from_thread
from services.billing import check_billing_status
from agent.tools.sb_vision_tool import SandboxVisionTool
from agent.tools.sb_image_edit_tool import SandboxImageEditTool
from services.langfuse import langfuse
from langfuse.client import StatefulTraceClient
from services.langfuse import langfuse
from agent.gemini_prompt import get_gemini_system_prompt
from agent.tools.mcp_tool_wrapper import MCPToolWrapper
from agentpress.tool import SchemaType

load_dotenv()

async def run_agent(
    thread_id: str,
    project_id: str,
    stream: bool,
    thread_manager: Optional[ThreadManager] = None,
    native_max_auto_continues: int = 25,
    max_iterations: int = 100,
    model_name: str = "anthropic/claude-sonnet-4-20250514",
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low',
    enable_context_manager: bool = True,
    agent_config: Optional[dict] = None,    
    trace: Optional[StatefulTraceClient] = None,
    is_agent_builder: Optional[bool] = False,
    target_agent_id: Optional[str] = None
):
    """运行开发代理并指定配置。"""
    # 记录启动日志，显示使用的模型名称
    logger.info(f"🚀 Starting agent with model: {model_name}")
    # 如果有自定义代理配置，记录代理名称
    if agent_config:
        logger.info(f"Using custom agent: {agent_config.get('name', 'Unknown')}")
 
    # 如果没有跟踪信息，创建一个新的跟踪
    if not trace:
        trace = langfuse.trace(name="run_agent", session_id=thread_id, metadata={"project_id": project_id})
    # 初始化线程管理器
    thread_manager = ThreadManager(trace=trace, is_agent_builder=is_agent_builder or False, target_agent_id=target_agent_id, agent_config=agent_config)

    # 获取数据库客户端
    client = await thread_manager.db.client

    # 从线程中获取账户ID用于计费检查
    account_id = await get_account_id_from_thread(client, thread_id)
    if not account_id:
        raise ValueError("Could not determine account ID for thread")

    # 从项目中获取沙盒信息
    project = await client.table('projects').select('*').eq('project_id', project_id).execute()
    if not project.data or len(project.data) == 0:
        raise ValueError(f"Project {project_id} not found")

    # 获取项目数据并检查沙盒配置
    project_data = project.data[0]
    sandbox_info = project_data.get('sandbox', {})
    if not sandbox_info.get('id'):
        raise ValueError(f"No sandbox found for project {project_id}")

    # 使用project_id而不是sandbox对象初始化工具
    # 这样可以确保每个工具都能独立验证它是否在正确的项目上运行
    
    # 从代理配置中获取启用的工具，如果没有则使用默认配置
    enabled_tools = None
    # 检查是否有代理配置且包含agentpress_tools字段
    if agent_config and 'agentpress_tools' in agent_config:
        # 从配置中获取工具列表
        enabled_tools = agent_config['agentpress_tools']
        # 记录日志，表示使用了自定义工具配置
        logger.info(f"Using custom tool configuration from agent")
    

    # 检查是否是代理构建器模式
    if is_agent_builder:
        # 导入代理构建器所需的各种工具类
        from agent.tools.agent_builder_tools.agent_config_tool import AgentConfigTool
        from agent.tools.agent_builder_tools.mcp_search_tool import MCPSearchTool
        from agent.tools.agent_builder_tools.credential_profile_tool import CredentialProfileTool
        from agent.tools.agent_builder_tools.workflow_tool import WorkflowTool
        from agent.tools.agent_builder_tools.trigger_tool import TriggerTool
        
        # 导入数据库连接并初始化
        from services.supabase import DBConnection
        db = DBConnection()
         
        # 向线程管理器添加各种工具
        thread_manager.add_tool(AgentConfigTool, thread_manager=thread_manager, db_connection=db, agent_id=target_agent_id)
        thread_manager.add_tool(MCPSearchTool, thread_manager=thread_manager, db_connection=db, agent_id=target_agent_id)
        thread_manager.add_tool(CredentialProfileTool, thread_manager=thread_manager, db_connection=db, agent_id=target_agent_id)
        thread_manager.add_tool(WorkflowTool, thread_manager=thread_manager, db_connection=db, agent_id=target_agent_id)
        thread_manager.add_tool(TriggerTool, thread_manager=thread_manager, db_connection=db, agent_id=target_agent_id)
        

    # 检查是否没有指定启用的工具
    if enabled_tools is None:
        # 记录日志，表示将注册所有工具以获得完整的Suna功能
        logger.info("No agent specified - registering all tools for full Suna capabilities")
        
        # 注册所有可用的沙盒工具
        thread_manager.add_tool(SandboxShellTool, project_id=project_id, thread_manager=thread_manager)
        thread_manager.add_tool(SandboxFilesTool, project_id=project_id, thread_manager=thread_manager)
        thread_manager.add_tool(SandboxBrowserTool, project_id=project_id, thread_id=thread_id, thread_manager=thread_manager)
        thread_manager.add_tool(SandboxDeployTool, project_id=project_id, thread_manager=thread_manager)
        thread_manager.add_tool(SandboxExposeTool, project_id=project_id, thread_manager=thread_manager)
        
        # 注册消息相关工具
        thread_manager.add_tool(ExpandMessageTool, thread_id=thread_id, thread_manager=thread_manager)
        thread_manager.add_tool(MessageTool)
        
        # 注册其他功能工具
        thread_manager.add_tool(SandboxWebSearchTool, project_id=project_id, thread_manager=thread_manager)
        thread_manager.add_tool(SandboxVisionTool, project_id=project_id, thread_id=thread_id, thread_manager=thread_manager)
        thread_manager.add_tool(SandboxImageEditTool, project_id=project_id, thread_id=thread_id, thread_manager=thread_manager)
        
        # 如果有RapidAPI密钥，注册数据提供者工具
        if config.RAPID_API_KEY:
            thread_manager.add_tool(DataProvidersTool)
    else:
        # 记录日志，表示将只注册启用的工具
        logger.info("Custom agent specified - registering only enabled tools")
        
        # 注册基础消息工具
        thread_manager.add_tool(ExpandMessageTool, thread_id=thread_id, thread_manager=thread_manager)
        thread_manager.add_tool(MessageTool)
        
        # 根据配置逐个检查并注册启用的工具
        if enabled_tools.get('sb_shell_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxShellTool, project_id=project_id, thread_manager=thread_manager)
        if enabled_tools.get('sb_files_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxFilesTool, project_id=project_id, thread_manager=thread_manager)
        if enabled_tools.get('sb_browser_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxBrowserTool, project_id=project_id, thread_id=thread_id, thread_manager=thread_manager)
        if enabled_tools.get('sb_deploy_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxDeployTool, project_id=project_id, thread_manager=thread_manager)
        if enabled_tools.get('sb_expose_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxExposeTool, project_id=project_id, thread_manager=thread_manager)
        if enabled_tools.get('web_search_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxWebSearchTool, project_id=project_id, thread_manager=thread_manager)
        if enabled_tools.get('sb_vision_tool', {}).get('enabled', False):
            thread_manager.add_tool(SandboxVisionTool, project_id=project_id, thread_id=thread_id, thread_manager=thread_manager)
        
        # 检查是否有RapidAPI密钥且数据提供者工具被启用
        if config.RAPID_API_KEY and enabled_tools.get('data_providers_tool', {}).get('enabled', False):
            thread_manager.add_tool(DataProvidersTool)

    # 注册MCP工具包装器，如果代理配置了MCP或自定义MCP
    mcp_wrapper_instance = None
    if agent_config:
        # 合并配置的MCP和自定义MCP
        all_mcps = []
        
        # 添加标准配置的MCP
        if agent_config.get('configured_mcps'):
            all_mcps.extend(agent_config['configured_mcps'])
        
        # 添加自定义MCP
        if agent_config.get('custom_mcps'):
            for custom_mcp in agent_config['custom_mcps']:
                # 将自定义MCP转换为标准格式
                custom_type = custom_mcp.get('customType', custom_mcp.get('type', 'sse'))
                
                # 对于Pipedream MCP，确保我们有用户ID和正确的配置
                if custom_type == 'pipedream':
                    # 从线程获取用户ID
                    if 'config' not in custom_mcp:
                        custom_mcp['config'] = {}
                    
                    # 如果不存在，从配置文件中获取external_user_id
                    if not custom_mcp['config'].get('external_user_id'):
                        profile_id = custom_mcp['config'].get('profile_id')
                        if profile_id:
                            try:
                                from pipedream.profiles import get_profile_manager
                                from services.supabase import DBConnection
                                profile_db = DBConnection()
                                profile_manager = get_profile_manager(profile_db)
                                
                                # 获取配置文件以检索external_user_id
                                profile = await profile_manager.get_profile(account_id, profile_id)
                                if profile:
                                    custom_mcp['config']['external_user_id'] = profile.external_user_id
                                    logger.info(f"Retrieved external_user_id from profile {profile_id} for Pipedream MCP")
                                else:
                                    logger.error(f"Could not find profile {profile_id} for Pipedream MCP")
                            except Exception as e:
                                logger.error(f"Error retrieving external_user_id from profile {profile_id}: {e}")
                    
                    # 处理headers中的x-pd-app-slug
                    if 'headers' in custom_mcp['config'] and 'x-pd-app-slug' in custom_mcp['config']['headers']:
                        custom_mcp['config']['app_slug'] = custom_mcp['config']['headers']['x-pd-app-slug']
                
                # 构建MCP配置对象
                mcp_config = {
                    'name': custom_mcp['name'],
                    'qualifiedName': f"custom_{custom_type}_{custom_mcp['name'].replace(' ', '_').lower()}",
                    'config': custom_mcp['config'],
                    'enabledTools': custom_mcp.get('enabledTools', []),
                    'instructions': custom_mcp.get('instructions', ''),
                    'isCustom': True,
                    'customType': custom_type
                }
                all_mcps.append(mcp_config)
        
        # 如果有MCP配置，注册MCP工具包装器
        if all_mcps:
            logger.info(f"Registering MCP tool wrapper for {len(all_mcps)} MCP servers (including {len(agent_config.get('custom_mcps', []))} custom)")
            thread_manager.add_tool(MCPToolWrapper, mcp_configs=all_mcps)
            
            # 查找MCP包装器实例
            for tool_name, tool_info in thread_manager.tool_registry.tools.items():
                if isinstance(tool_info['instance'], MCPToolWrapper):
                    mcp_wrapper_instance = tool_info['instance']
                    break
            
            # 初始化并注册MCP工具
            if mcp_wrapper_instance:
                try:
                    await mcp_wrapper_instance.initialize_and_register_tools()
                    logger.info("MCP tools initialized successfully")
                    
                    # 获取并注册所有schema
                    updated_schemas = mcp_wrapper_instance.get_schemas()
                    logger.info(f"MCP wrapper has {len(updated_schemas)} schemas available")
                    for method_name, schema_list in updated_schemas.items():
                        if method_name != 'call_mcp_tool':
                            for schema in schema_list:
                                if schema.schema_type == SchemaType.OPENAPI:
                                    thread_manager.tool_registry.tools[method_name] = {
                                        "instance": mcp_wrapper_instance,
                                        "schema": schema
                                    }
                                    logger.info(f"Registered dynamic MCP tool: {method_name}")
                    
                    # 记录所有已注册的工具用于调试
                    all_tools = list(thread_manager.tool_registry.tools.keys())
                    logger.info(f"All registered tools after MCP initialization: {all_tools}")
                    
                    # 过滤出MCP工具
                    mcp_tools = [tool for tool in all_tools if tool not in ['call_mcp_tool', 'sb_files_tool', 'message_tool', 'expand_msg_tool', 'web_search_tool', 'sb_shell_tool', 'sb_vision_tool', 'sb_browser_tool', 'computer_use_tool', 'data_providers_tool', 'sb_deploy_tool', 'sb_expose_tool', 'update_agent_tool']]
                    logger.info(f"MCP tools registered: {mcp_tools}")
                
                except Exception as e:
                    logger.error(f"Failed to initialize MCP tools: {e}")
                    # 如果初始化失败，继续运行但不使用MCP工具

    # 准备系统提示
    # 首先获取默认系统提示
    if "gemini-2.5-flash" in model_name.lower() and "gemini-2.5-pro" not in model_name.lower():
        # 如果是Gemini 2.5 Flash模型，使用专门的系统提示
        default_system_content = get_gemini_system_prompt()
    else:
        # 其他模型使用原始系统提示 - LLM只能使用已注册的工具
        default_system_content = get_system_prompt()
        
    # 为非Anthropic模型添加示例响应
    if "anthropic" not in model_name.lower():
        # 从sample_responses目录读取示例响应
        sample_response_path = os.path.join(os.path.dirname(__file__), 'sample_responses/1.txt')
        with open(sample_response_path, 'r') as file:
            sample_response = file.read()
        # 将示例响应附加到默认系统提示中
        default_system_content = default_system_content + "\n\n <sample_assistant_response>" + sample_response + "</sample_assistant_response>"
    
    # 处理自定义agent系统提示
    if agent_config and agent_config.get('system_prompt'):
        # 如果有自定义系统提示，完全替换默认提示
        custom_system_prompt = agent_config['system_prompt'].strip()
        
        # 这可以防止混淆和工具幻觉
        system_content = custom_system_prompt
        logger.info(f"Using ONLY custom agent system prompt for: {agent_config.get('name', 'Unknown')}")
    elif is_agent_builder:
        # 如果是agent builder，使用专门的提示
        system_content = get_agent_builder_prompt()
        logger.info("Using agent builder system prompt")
    else:
        # 否则只使用默认系统提示
        system_content = default_system_content
        logger.info("Using default system prompt only")
    
    # 检查知识库功能是否启用
    if await is_enabled("knowledge_base"):
        try:
            # 初始化Supabase数据库连接
            from services.supabase import DBConnection
            kb_db = DBConnection()
            kb_client = await kb_db.client
            
            # 获取当前agent ID（如果有）
            current_agent_id = agent_config.get('agent_id') if agent_config else None
            
            # 调用Supabase存储过程获取知识库上下文
            kb_result = await kb_client.rpc('get_combined_knowledge_base_context', {
                'p_thread_id': thread_id,  # 当前线程ID
                'p_agent_id': current_agent_id,  # 当前agent ID
                'p_max_tokens': 4000  # 最大token限制
            }).execute()
            
            # 如果有有效的知识库上下文数据
            if kb_result.data and kb_result.data.strip():
                logger.info(f"Adding combined knowledge base context to system prompt for thread {thread_id}, agent {current_agent_id}")
                # 将知识库上下文附加到系统提示中
                system_content += "\n\n" + kb_result.data
            else:
                logger.debug(f"No knowledge base context found for thread {thread_id}, agent {current_agent_id}")
                
        except Exception as e:
            # 捕获并记录知识库上下文加载过程中的任何错误
            logger.error(f"Error retrieving knowledge base context for thread {thread_id}: {e}")


    # 检查是否存在配置的MCP工具且MCP包装器已初始化
    if agent_config and (agent_config.get('configured_mcps') or agent_config.get('custom_mcps')) and mcp_wrapper_instance and mcp_wrapper_instance._initialized:
        # 初始化MCP工具信息头部
        mcp_info = "\n\n--- MCP Tools Available ---\n"
        mcp_info += "You have access to external MCP (Model Context Protocol) server tools.\n"
        mcp_info += "MCP tools can be called directly using their native function names in the standard function calling format:\n"
        mcp_info += '<function_calls>\n'
        mcp_info += '<invoke name="{tool_name}">\n'
        mcp_info += '<parameter name="param1">value1</parameter>\n'
        mcp_info += '<parameter name="param2">value2</parameter>\n'
        mcp_info += '</invoke>\n'
        mcp_info += '</function_calls>\n\n'
        
        # 列出可用的MCP工具
        mcp_info += "Available MCP tools:\n"
        try:
            # 从包装器获取实际注册的OpenAPI模式
            registered_schemas = mcp_wrapper_instance.get_schemas()
            for method_name, schema_list in registered_schemas.items():
                if method_name == 'call_mcp_tool':
                    continue  # 跳过回退方法
                    
                # 解析每个模式的详细信息
                for schema in schema_list:
                    if schema.schema_type == SchemaType.OPENAPI:
                        func_info = schema.schema.get('function', {})
                        description = func_info.get('description', 'No description available')
                        # 从描述中提取服务器信息
                        server_match = description.find('(MCP Server: ')
                        if server_match != -1:
                            server_end = description.find(')', server_match)
                            server_info = description[server_match:server_end+1]
                        else:
                            server_info = ''
                        
                        # 添加工具名称和描述
                        mcp_info += f"- **{method_name}**: {description}\n"
                        
                        # 显示参数信息
                        params = func_info.get('parameters', {})
                        props = params.get('properties', {})
                        if props:
                            mcp_info += f"  Parameters: {', '.join(props.keys())}\n"
                            
        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")
            mcp_info += "- Error loading MCP tool list\n"
        
        # 添加关键使用说明
        mcp_info += "\n🚨 CRITICAL MCP TOOL RESULT INSTRUCTIONS 🚨\n"
        mcp_info += "When you use ANY MCP (Model Context Protocol) tools:\n"
        mcp_info += "1. ALWAYS read and use the EXACT results returned by the MCP tool\n"
        mcp_info += "2. For search tools: ONLY cite URLs, sources, and information from the actual search results\n"
        mcp_info += "3. For any tool: Base your response entirely on the tool's output - do NOT add external information\n"
        mcp_info += "4. DO NOT fabricate, invent, hallucinate, or make up any sources, URLs, or data\n"
        mcp_info += "5. If you need more information, call the MCP tool again with different parameters\n"
        mcp_info += "6. When writing reports/summaries: Reference ONLY the data from MCP tool results\n"
        mcp_info += "7. If the MCP tool doesn't return enough information, explicitly state this limitation\n"
        mcp_info += "8. Always double-check that every fact, URL, and reference comes from the MCP tool output\n"
        mcp_info += "\nIMPORTANT: MCP tool results are your PRIMARY and ONLY source of truth for external data!\n"
        mcp_info += "NEVER supplement MCP results with your training data or make assumptions beyond what the tools provide.\n"
        
        # 将MCP信息添加到系统提示中
        system_content += mcp_info
    
    system_message = { "role": "system", "content": system_content }

    iteration_count = 0
    continue_execution = True

    # 取得数据库最新用户消息
    latest_user_message = await client.table('messages').select('*').eq('thread_id', thread_id).eq('type', 'user').order('created_at', desc=True).limit(1).execute()
    if latest_user_message.data and len(latest_user_message.data) > 0:
        data = latest_user_message.data[0]['content']
        if isinstance(data, str):
            data = json.loads(data)
        if trace:
            trace.update(input=data['content'])

    while continue_execution and iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"🔄 Running iteration {iteration_count} of {max_iterations}...")
        
        # 每次迭代都进行计费状态检查 - 在迭代过程中仍需检查
        can_run, message, subscription = await check_billing_status(client, account_id)
        if not can_run:
            # 构建计费限制错误消息
            error_msg = f"Billing limit reached: {message}"
            # 如果有跟踪器，记录计费限制事件
            if trace:
                trace.event(name="billing_limit_reached", level="ERROR", status_message=(f"{error_msg}"))
            # 生成一个特殊消息表示达到计费限制
            yield {
                "type": "status",  # 消息类型为状态
                "status": "stopped",  # 状态为停止
                "message": error_msg  # 包含错误消息
            }
            break
        # 检查最后一条消息是否来自助手 - 使用Supabase直接查询
        latest_message = await client.table('messages').select('*').eq('thread_id', thread_id).in_('type', ['assistant', 'tool', 'user']).order('created_at', desc=True).limit(1).execute()
        
        # 如果存在消息数据且不为空
        if latest_message.data and len(latest_message.data) > 0:
            message_type = latest_message.data[0].get('type')
            
            # 如果最后一条消息来自助手，则停止执行
            if message_type == 'assistant':
                logger.info(f"Last message was from assistant, stopping execution")
                
                # 如果有跟踪功能，记录事件
                if trace:
                    trace.event(name="last_message_from_assistant", level="DEFAULT", status_message=(f"Last message was from assistant, stopping execution"))
                
                continue_execution = False
                break

        # ---- 临时消息处理（浏览器状态和图像上下文） ----
        temporary_message = None  # 初始化临时消息
        temp_message_content_list = [] # 用于保存文本/图像块的列表

        # 获取最新的浏览器状态消息
        latest_browser_state_msg = await client.table('messages').select('*').eq('thread_id', thread_id).eq('type', 'browser_state').order('created_at', desc=True).limit(1).execute()
        
        # 如果存在浏览器状态消息
        if latest_browser_state_msg.data and len(latest_browser_state_msg.data) > 0:
            try:
                # 解析浏览器状态内容
                browser_content = latest_browser_state_msg.data[0]["content"]
                # 如果内容是字符串格式，转换为JSON对象
                if isinstance(browser_content, str):
                    browser_content = json.loads(browser_content)
                
                # 获取截图数据（base64格式和URL格式）
                screenshot_base64 = browser_content.get("screenshot_base64")
                screenshot_url = browser_content.get("image_url")
                
                # 创建不包含截图数据的浏览器状态文本副本
                browser_state_text = browser_content.copy()
                browser_state_text.pop('screenshot_base64', None)
                browser_state_text.pop('image_url', None)

                # 如果有浏览器状态文本，添加到临时消息列表
                if browser_state_text:
                    temp_message_content_list.append({
                        "type": "text",
                        "text": f"The following is the current state of the browser:\n{json.dumps(browser_state_text, indent=2)}"
                    })
                
                # 仅当模型不是Gemini、Anthropic或OpenAI时添加截图
                if 'gemini' in model_name.lower() or 'anthropic' in model_name.lower() or 'openai' in model_name.lower():
                    # 优先使用截图URL（如果可用）
                    if screenshot_url:
                        temp_message_content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": screenshot_url,
                                "format": "image/jpeg"
                            }
                        })
                        # 如果有追踪器，记录事件
                        if trace:
                            trace.event(name="screenshot_url_added_to_temporary_message", level="DEFAULT", status_message=(f"Screenshot URL added to temporary message."))
                    # 如果没有URL但有base64数据，使用base64格式
                    elif screenshot_base64:
                        temp_message_content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{screenshot_base64}",
                            }
                        })
                        if trace:
                            trace.event(name="screenshot_base64_added_to_temporary_message", level="WARNING", status_message=(f"Screenshot base64 added to temporary message. Prefer screenshot_url if available."))
                    else:
                        # 没有截图数据时记录警告
                        logger.warning("Browser state found but no screenshot data.")
                        if trace:
                            trace.event(name="browser_state_found_but_no_screenshot_data", level="WARNING", status_message=(f"Browser state found but no screenshot data."))
                else:
                    # 不支持的模型类型记录警告
                    logger.warning("Model is Gemini, Anthropic, or OpenAI, so not adding screenshot to temporary message.")
                    if trace:
                        trace.event(name="model_is_gemini_anthropic_or_openai", level="WARNING", status_message=(f"Model is Gemini, Anthropic, or OpenAI, so not adding screenshot to temporary message."))

            except Exception as e:
                # 浏览器状态解析错误处理
                logger.error(f"Error parsing browser state: {e}")
                if trace:
                    trace.event(name="error_parsing_browser_state", level="ERROR", status_message=(f"{e}"))

        # 获取最新的image_context类型消息
        latest_image_context_msg = await client.table('messages').select('*').eq('thread_id', thread_id).eq('type', 'image_context').order('created_at', desc=True).limit(1).execute()
        
        # 如果存在image_context消息数据
        if latest_image_context_msg.data and len(latest_image_context_msg.data) > 0:
            try:
                # 解析消息内容（可能是字典或JSON字符串）
                image_context_content = latest_image_context_msg.data[0]["content"] if isinstance(latest_image_context_msg.data[0]["content"], dict) else json.loads(latest_image_context_msg.data[0]["content"])
                
                # 提取图像数据
                base64_image = image_context_content.get("base64")
                mime_type = image_context_content.get("mime_type")
                file_path = image_context_content.get("file_path", "unknown file")

                # 如果有有效的图像数据，添加到临时消息列表
                if base64_image and mime_type:
                    temp_message_content_list.append({
                        "type": "text",
                        "text": f"Here is the image you requested to see: '{file_path}'"
                    })
                    temp_message_content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                        }
                    })
                else:
                    logger.warning(f"Image context found for '{file_path}' but missing base64 or mime_type.")

                # 删除已处理的image_context消息
                await client.table('messages').delete().eq('message_id', latest_image_context_msg.data[0]["message_id"]).execute()
            except Exception as e:
                logger.error(f"Error parsing image context: {e}")
                if trace:
                    trace.event(name="error_parsing_image_context", level="ERROR", status_message=(f"{e}"))

        # 如果有临时消息内容，构建临时消息
        if temp_message_content_list:
            temporary_message = {
                "role": "user",  # 消息角色设置为用户
                "content": temp_message_content_list  # 包含文本/图像块的内容列表
            }
            # logger.debug(f"Constructed temporary message with {len(temp_message_content_list)} content blocks.")
        # ---- 临时消息处理结束 ----

        # 根据模型设置max_tokens
        max_tokens = None
        
        # Claude 3.5 Sonnet模型限制8192 tokens
        if "sonnet" in model_name.lower():
            max_tokens = 8192
        # GPT-4模型限制4096 tokens    
        elif "gpt-4" in model_name.lower():
            max_tokens = 4096
        # Gemini 2.5 Pro模型限制64000 tokens
        elif "gemini-2.5-pro" in model_name.lower():
            max_tokens = 64000
            
        # 如果有跟踪功能，创建线程运行的生成记录
        generation = trace.generation(name="thread_manager.run_thread") if trace else None
        try:
            # Make the LLM call and process the response
            response = await thread_manager.run_thread(
                thread_id=thread_id,
                system_prompt=system_message,
                stream=stream,
                llm_model=model_name,
                llm_temperature=0,
                llm_max_tokens=max_tokens,
                tool_choice="auto",
                max_xml_tool_calls=1,
                temporary_message=temporary_message,
                processor_config=ProcessorConfig(
                    xml_tool_calling=True,
                    native_tool_calling=False,
                    execute_tools=True,
                    execute_on_stream=True,
                    tool_execution_strategy="parallel",
                    xml_adding_strategy="user_message"
                ),
                native_max_auto_continues=native_max_auto_continues,
                include_xml_examples=True,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                enable_context_manager=enable_context_manager,
                generation=generation
            )

            # 检查run_thread返回的响应是否为错误响应
            if isinstance(response, dict) and "status" in response and response["status"] == "error":
                logger.error(f"Error response from run_thread: {response.get('message', 'Unknown error')}")
                if trace:
                    trace.event(name="error_response_from_run_thread", level="ERROR", status_message=(f"{response.get('message', 'Unknown error')}"))
                yield response
                break

            # 跟踪是否看到ask、complete或web-browser-takeover工具调用
            last_tool_call = None
            agent_should_terminate = False

            # 处理响应
            error_detected = False
            full_response = ""
            try:
                # 检查响应是否为可迭代对象(async generator)或字典(错误情况)
                if hasattr(response, '__aiter__') and not isinstance(response, dict):
                    async for chunk in response:
                        # 如果接收到错误块，应在本次迭代后停止
                        if isinstance(chunk, dict) and chunk.get('type') == 'status' and chunk.get('status') == 'error':
                            logger.error(f"Error chunk detected: {chunk.get('message', 'Unknown error')}")
                            if trace:
                                trace.event(name="error_chunk_detected", level="ERROR", status_message=(f"{chunk.get('message', 'Unknown error')}"))
                            error_detected = True
                            yield chunk  # 转发错误块
                            continue     # 继续处理其他块但不中断
                        
                        # 检查状态消息中的终止信号
                        if chunk.get('type') == 'status':
                            try:
                                # 解析元数据以检查终止信号
                                metadata = chunk.get('metadata', {})
                                if isinstance(metadata, str):
                                    metadata = json.loads(metadata)
                                
                                if metadata.get('agent_should_terminate'):
                                    agent_should_terminate = True
                                    logger.info("Agent termination signal detected in status message")
                                    if trace:
                                        trace.event(name="agent_termination_signal_detected", level="DEFAULT", status_message="Agent termination signal detected in status message")
                                    
                                    # 如果可用，从状态内容中提取工具名称
                                    content = chunk.get('content', {})
                                    if isinstance(content, str):
                                        content = json.loads(content)
                                    
                                    if content.get('function_name'):
                                        last_tool_call = content['function_name']
                                    elif content.get('xml_tag_name'):
                                        last_tool_call = content['xml_tag_name']
                                        
                            except Exception as e:
                                logger.debug(f"Error parsing status message for termination check: {e}")
                            
                        # 检查assistant内容块中的XML版本如<ask>、<complete>或<web-browser-takeover>
                        if chunk.get('type') == 'assistant' and 'content' in chunk:
                            try:
                                # 内容字段可能是JSON字符串或对象
                                content = chunk.get('content', '{}')
                                if isinstance(content, str):
                                    assistant_content_json = json.loads(content)
                                else:
                                    assistant_content_json = content

                                # 实际文本内容嵌套在内部
                                assistant_text = assistant_content_json.get('content', '')
                                full_response += assistant_text
                                if isinstance(assistant_text, str):
                                    if '</ask>' in assistant_text or '</complete>' in assistant_text or '</web-browser-takeover>' in assistant_text:
                                       if '</ask>' in assistant_text:
                                           xml_tool = 'ask'
                                       elif '</complete>' in assistant_text:
                                           xml_tool = 'complete'
                                       elif '</web-browser-takeover>' in assistant_text:
                                           xml_tool = 'web-browser-takeover'

                                       last_tool_call = xml_tool
                                       logger.info(f"Agent used XML tool: {xml_tool}")
                                       if trace:
                                           trace.event(name="agent_used_xml_tool", level="DEFAULT", status_message=(f"Agent used XML tool: {xml_tool}"))
                            
                            except json.JSONDecodeError:
                                # 处理内容可能不是有效JSON的情况
                                logger.warning(f"Warning: Could not parse assistant content JSON: {chunk.get('content')}")
                                if trace:
                                    trace.event(name="warning_could_not_parse_assistant_content_json", level="WARNING", status_message=(f"Warning: Could not parse assistant content JSON: {chunk.get('content')}"))
                            except Exception as e:
                                logger.error(f"Error processing assistant chunk: {e}")
                                if trace:
                                    trace.event(name="error_processing_assistant_chunk", level="ERROR", status_message=(f"Error processing assistant chunk: {e}"))

                        yield chunk
                else:
                    # 响应不可迭代，可能是错误字典
                    logger.error(f"Response is not iterable: {response}")
                    error_detected = True

                # 根据最后一个工具调用或错误检查是否应停止
                if error_detected:
                    logger.info(f"Stopping due to error detected in response")
                    if trace:
                        trace.event(name="stopping_due_to_error_detected_in_response", level="DEFAULT", status_message=(f"Stopping due to error detected in response"))
                    if generation:
                        generation.end(output=full_response, status_message="error_detected", level="ERROR")
                    break
                    
                if agent_should_terminate or last_tool_call in ['ask', 'complete', 'web-browser-takeover']:
                    logger.info(f"Agent decided to stop with tool: {last_tool_call}")
                    if trace:
                        trace.event(name="agent_decided_to_stop_with_tool", level="DEFAULT", status_message=(f"Agent decided to stop with tool: {last_tool_call}"))
                    if generation:
                        generation.end(output=full_response, status_message="agent_stopped")
                    continue_execution = False

            except Exception as e:
                # 记录错误并重新抛出以停止所有迭代
                error_msg = f"Error during response streaming: {str(e)}"
                logger.error(f"Error: {error_msg}")
                if trace:
                    trace.event(name="error_during_response_streaming", level="ERROR", status_message=(f"Error during response streaming: {str(e)}"))
                if generation:
                    generation.end(output=full_response, status_message=error_msg, level="ERROR")
                yield {
                    "type": "status",
                    "status": "error",
                    "message": error_msg
                }
                # 遇到任何错误立即停止执行
                break
                
        except Exception as e:
            # Just log the error and re-raise to stop all iterations
            error_msg = f"Error running thread: {str(e)}"
            logger.error(f"Error: {error_msg}")
            if trace:
                trace.event(name="error_running_thread", level="ERROR", status_message=(f"Error running thread: {str(e)}"))
            yield {
                "type": "status",
                "status": "error",
                "message": error_msg
            }
            # Stop execution immediately on any error
            break
        if generation:
            generation.end(output=full_response)

    asyncio.create_task(asyncio.to_thread(lambda: langfuse.flush()))
