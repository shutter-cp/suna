"""
Context Management for AgentPress Threads.

This module handles token counting and thread summarization to prevent
reaching the context window limitations of LLM models.
"""

import json
from typing import List, Dict, Any, Optional, Union

from litellm.utils import token_counter
from services.supabase import DBConnection
from utils.logger import logger

DEFAULT_TOKEN_THRESHOLD = 120000

class ContextManager:
    """Manages thread context including token counting and summarization."""
    
    def __init__(self, token_threshold: int = DEFAULT_TOKEN_THRESHOLD):
        """Initialize the ContextManager.
        
        Args:
            token_threshold: Token count threshold to trigger summarization
        """
        self.db = DBConnection()
        self.token_threshold = token_threshold

    def is_tool_result_message(self, msg: Dict[str, Any]) -> bool:
        """
        检查给定的消息是否是工具执行结果消息。
        
        参数:
            msg: 要检查的消息字典，包含消息内容和元数据
            
        返回:
            bool: 如果是工具结果消息返回True，否则返回False
            
        处理逻辑:
            1. 首先检查消息是否有内容
            2. 检查字符串内容中是否包含'ToolResult'标记
            3. 检查字典内容中是否包含'tool_execution'或'interactive_elements'字段
            4. 尝试解析字符串内容为JSON，再次检查上述字段
        """
        if not ("content" in msg and msg['content']):  # 检查消息是否有内容
            return False
        content = msg['content']
        
        # 检查字符串内容中是否包含工具结果标记
        if isinstance(content, str) and "ToolResult" in content: 
            return True
            
        # 检查字典内容中是否包含工具执行或交互元素
        if isinstance(content, dict) and "tool_execution" in content: 
            return True
        if isinstance(content, dict) and "interactive_elements" in content: 
            return True
            
        # 尝试解析字符串内容为JSON格式
        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict) and "tool_execution" in parsed_content: 
                    return True
                if isinstance(parsed_content, dict) and "interactive_elements" in content: 
                    return True
            except (json.JSONDecodeError, TypeError):  # 捕获JSON解析错误
                pass
                
        return False  # 所有检查都不匹配时返回False
    
    def compress_message(self, msg_content: Union[str, dict], message_id: Optional[str] = None, max_length: int = 3000) -> Union[str, dict]:
        """对消息内容进行简单截断压缩
        该方法实现基础的消息压缩功能，对超过长度阈值的文本内容进行截断处理，
        同时保留消息ID以便后续可能的消息展开操作。支持字符串和字典两种内容类型。
        Args:
            msg_content: 待压缩的消息内容，可以是字符串或字典
            message_id: 消息唯一标识符，用于截断提示中
            max_length: 最大允许长度阈值，默认3000字符
        Returns:
            压缩后的消息内容（保持原类型）
        """
        if isinstance(msg_content, str):
            if len(msg_content) > max_length:
                return msg_content[:max_length] + "... (truncated)" + f"\n\nmessage_id \"{message_id}\"\nUse expand-message tool to see contents"
            else:
                return msg_content
        elif isinstance(msg_content, dict):
            if len(json.dumps(msg_content)) > max_length:
                return json.dumps(msg_content)[:max_length] + "... (truncated)" + f"\n\nmessage_id \"{message_id}\"\nUse expand-message tool to see contents"
            else:
                return msg_content
        
    def safe_truncate(self, msg_content: Union[str, dict], max_length: int = 100000) -> Union[str, dict]:
        """安全截断消息内容，保留消息首尾部分
        该方法实现智能的消息截断策略，通过保留消息开头和结尾部分来维持上下文完整性，
        同时添加明确的截断标记和用户提示。支持字符串和字典两种内容类型。
        Args:
            msg_content: 待截断的消息内容，可以是字符串或字典
            max_length: 最大允许长度阈值，默认100000字符（自动限制上限）
        Returns:
            截断后的消息内容（保持原类型）
        """
        max_length = min(max_length, 100000)
        if isinstance(msg_content, str):
            if len(msg_content) > max_length:
                # Calculate how much to keep from start and end
                keep_length = max_length - 150  # Reserve space for truncation message
                start_length = keep_length // 2
                end_length = keep_length - start_length
                
                start_part = msg_content[:start_length]
                end_part = msg_content[-end_length:] if end_length > 0 else ""
                
                return start_part + f"\n\n... (middle truncated) ...\n\n" + end_part + f"\n\nThis message is too long, repeat relevant information in your response to remember it"
            else:
                return msg_content
        elif isinstance(msg_content, dict):
            json_str = json.dumps(msg_content)
            if len(json_str) > max_length:
                # Calculate how much to keep from start and end
                keep_length = max_length - 150  # Reserve space for truncation message
                start_length = keep_length // 2
                end_length = keep_length - start_length
                
                start_part = json_str[:start_length]
                end_part = json_str[-end_length:] if end_length > 0 else ""
                
                return start_part + f"\n\n... (middle truncated) ...\n\n" + end_part + f"\n\nThis message is too long, repeat relevant information in your response to remember it"
            else:
                return msg_content
  
    def compress_tool_result_messages(self, messages: List[Dict[str, Any]], llm_model: str, max_tokens: Optional[int], token_threshold: int = 1000) -> List[Dict[str, Any]]:
        """压缩工具结果消息，保留最新的工具结果，压缩历史工具结果
        该方法通过选择性压缩历史工具结果消息来控制上下文窗口大小，同时确保最新的工具结果保持完整
        Args:
            messages: 待处理的消息列表
            llm_model: 使用的LLM模型名称（用于令牌计数）
            max_tokens: 允许的最大令牌数（None时使用默认值100,000）
            token_threshold: 单条消息的令牌阈值，超过此值将被压缩
        Returns:
            压缩后的消息列表
        """
        # 计算未压缩消息的总令牌数
        uncompressed_total_token_count = token_counter(model=llm_model, messages=messages)
        # 设置最大令牌值，未提供时使用100,000作为默认值
        max_tokens_value = max_tokens or (100 * 1000)

        # 仅当未压缩令牌数超过最大令牌值时执行压缩
        if uncompressed_total_token_count > max_tokens_value:
            _i = 0  # 工具结果消息计数器，用于识别最新的工具结果
            # 从消息列表末尾开始反向迭代（最新消息优先处理）
            for msg in reversed(messages):
                # 仅处理工具结果类型的消息
                if self.is_tool_result_message(msg):
                    _i += 1  # 增加工具结果消息计数
                    # 计算当前消息的令牌数
                    msg_token_count = token_counter(messages=[msg])
                    # 如果消息令牌数超过阈值，则需要压缩
                    if msg_token_count > token_threshold:
                        # _i > 1表示不是最新的工具结果消息（因为是反向迭代）
                        if _i > 1:
                            # 获取消息ID用于压缩标记
                            message_id = msg.get('message_id')
                            if message_id:
                                # 压缩非最新工具结果消息，使用3倍阈值作为压缩目标
                                msg["content"] = self.compress_message(msg["content"], message_id, token_threshold * 3)
                            else:
                                # 记录无消息ID的异常情况
                                logger.warning(f"UNEXPECTED: Message has no message_id {str(msg)[:100]}")
                        else:
                            # 最新的工具结果消息使用安全截断，保留更多内容（2倍最大令牌值）
                            msg["content"] = self.safe_truncate(msg["content"], int(max_tokens_value * 2))
        # 返回处理后的消息列表
        return messages

    def compress_user_messages(self, messages: List[Dict[str, Any]], llm_model: str, max_tokens: Optional[int], token_threshold: int = 1000) -> List[Dict[str, Any]]:
        """压缩用户消息，保留最新的用户消息
        该方法通过选择性压缩历史用户消息来控制上下文窗口大小，同时确保最新的用户消息保持完整
        Args:
            messages: 待处理的消息列表
            llm_model: 使用的LLM模型名称（用于令牌计数）
            max_tokens: 允许的最大令牌数（None时使用默认值100,000）
            token_threshold: 单条消息的令牌阈值，超过此值将被压缩
        Returns:
            压缩后的消息列表
        """
        # 计算未压缩消息的总令牌数
        uncompressed_total_token_count = token_counter(model=llm_model, messages=messages)
        # 设置最大令牌值，未提供时使用100,000作为默认值
        max_tokens_value = max_tokens or (100 * 1000)

        # 仅当未压缩令牌数超过最大令牌值时执行压缩
        if uncompressed_total_token_count > max_tokens_value:
            _i = 0  # 用户消息计数器，用于识别最新的用户消息
            # 从消息列表末尾开始反向迭代（最新消息优先处理）
            for msg in reversed(messages):
                # 仅处理用户角色消息
                if msg.get('role') == 'user':
                    _i += 1  # 增加用户消息计数
                    # 计算当前消息的令牌数
                    msg_token_count = token_counter(messages=[msg])
                    # 如果消息令牌数超过阈值，则需要压缩
                    if msg_token_count > token_threshold:
                        # _i > 1表示不是最新的用户消息（因为是反向迭代）
                        if _i > 1:
                            # 获取消息ID用于压缩标记
                            message_id = msg.get('message_id')
                            if message_id:
                                # 压缩非最新用户消息，使用3倍阈值作为压缩目标
                                msg["content"] = self.compress_message(msg["content"], message_id, token_threshold * 3)
                            else:
                                # 记录无消息ID的异常情况
                                logger.warning(f"UNEXPECTED: Message has no message_id {str(msg)[:100]}")
                        else:
                            # 最新的用户消息使用安全截断，保留更多内容（2倍最大令牌值）
                            msg["content"] = self.safe_truncate(msg["content"], int(max_tokens_value * 2))
        # 返回处理后的消息列表
        return messages

    def compress_assistant_messages(self, messages: List[Dict[str, Any]], llm_model: str, max_tokens: Optional[int], token_threshold: int = 1000) -> List[Dict[str, Any]]:
        """压缩助手消息，保留最新的助手消息
        该方法通过选择性压缩历史助手消息来控制上下文窗口大小，同时确保最新的助手消息保持完整
        Args:
            messages: 待处理的消息列表
            llm_model: 使用的LLM模型名称（用于令牌计数）
            max_tokens: 允许的最大令牌数（None时使用默认值100,000）
            token_threshold: 单条消息的令牌阈值，超过此值将被压缩
        Returns:
            压缩后的消息列表
        """
        # 计算未压缩消息的总令牌数
        uncompressed_total_token_count = token_counter(model=llm_model, messages=messages)
        # 设置最大令牌值，未提供时使用100,000作为默认值
        max_tokens_value = max_tokens or (100 * 1000)
        
        # 仅当未压缩令牌数超过最大令牌值时执行压缩
        if uncompressed_total_token_count > max_tokens_value:
            _i = 0  # 助手消息计数器，用于识别最新的助手消息
            # 从消息列表末尾开始反向迭代（最新消息优先处理）
            for msg in reversed(messages):
                # 仅处理助手角色消息
                if msg.get('role') == 'assistant':
                    _i += 1  # 增加助手消息计数
                    # 计算当前消息的令牌数
                    msg_token_count = token_counter(messages=[msg])
                    # 如果消息令牌数超过阈值，则需要压缩
                    if msg_token_count > token_threshold:
                        # _i > 1表示不是最新的助手消息（因为是反向迭代）
                        if _i > 1:
                            # 获取消息ID用于压缩标记
                            message_id = msg.get('message_id')
                            if message_id:
                                # 压缩非最新助手消息，使用3倍阈值作为压缩目标
                                msg["content"] = self.compress_message(msg["content"], message_id, token_threshold * 3)
                            else:
                                # 记录无消息ID的异常情况
                                logger.warning(f"UNEXPECTED: Message has no message_id {str(msg)[:100]}")
                        else:
                            # 最新的助手消息使用安全截断，保留更多内容（2倍最大令牌值）
                            msg["content"] = self.safe_truncate(msg["content"], int(max_tokens_value * 2))
                            
        # 返回处理后的消息列表
        return messages

    def remove_meta_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove meta messages from the messages."""
        result: List[Dict[str, Any]] = []
        for msg in messages:
            msg_content = msg.get('content')
            # Try to parse msg_content as JSON if it's a string
            if isinstance(msg_content, str):
                try: 
                    msg_content = json.loads(msg_content)
                except json.JSONDecodeError: 
                    pass
            if isinstance(msg_content, dict):
                # Create a copy to avoid modifying the original
                msg_content_copy = msg_content.copy()
                if "tool_execution" in msg_content_copy:
                    tool_execution = msg_content_copy["tool_execution"].copy()
                    if "arguments" in tool_execution:
                        del tool_execution["arguments"]
                    msg_content_copy["tool_execution"] = tool_execution
                # Create a new message dict with the modified content
                new_msg = msg.copy()
                new_msg["content"] = json.dumps(msg_content_copy)
                result.append(new_msg)
            else:
                result.append(msg)
        return result

    def compress_messages(self, messages: List[Dict[str, Any]], llm_model: str, max_tokens: Optional[int] = 41000, token_threshold: int = 4096, max_iterations: int = 5) -> List[Dict[str, Any]]:
        """压缩消息列表以避免超出LLM模型的上下文窗口限制
        
        Args:
            messages: 要压缩的消息列表
            llm_model: 用于token计数的模型名称
            max_tokens: 允许的最大token数
            token_threshold: 单个消息压缩的token阈值(必须是2的幂)
            max_iterations: 最大压缩迭代次数
        """
        # 根据不同的LLM模型设置特定的token限制
        if 'sonnet' in llm_model.lower():
            max_tokens = 200 * 1000 - 64000 - 28000  # Claude 3.5 Sonnet模型限制
        elif 'gpt' in llm_model.lower():
            max_tokens = 128 * 1000 - 28000  # GPT-4模型限制
        elif 'gemini' in llm_model.lower():
            max_tokens = 1000 * 1000 - 300000  # Gemini 2.5 Pro模型限制
        elif 'deepseek' in llm_model.lower():
            max_tokens = 128 * 1000 - 28000  # DeepSeek模型限制
        else:
            max_tokens = 41 * 1000 - 10000  # 默认模型限制
    
        result = messages
        # 第一步: 移除消息中的元数据
        result = self.remove_meta_messages(result)
    
        # 计算未压缩前的总token数
        uncompressed_total_token_count = token_counter(model=llm_model, messages=result)
    
        # 第二步: 分别压缩不同类型的消息
        result = self.compress_tool_result_messages(result, llm_model, max_tokens, token_threshold)  # 压缩工具结果消息
        result = self.compress_user_messages(result, llm_model, max_tokens, token_threshold)  # 压缩用户消息
        result = self.compress_assistant_messages(result, llm_model, max_tokens, token_threshold)  # 压缩助手消息
    
        # 计算压缩后的token数
        compressed_token_count = token_counter(model=llm_model, messages=result)
    
        # 记录压缩前后的token数对比
        logger.info(f"compress_messages: {uncompressed_total_token_count} -> {compressed_token_count}")
    
        # 如果达到最大迭代次数但仍超过限制，则直接省略部分消息
        if max_iterations <= 0:
            logger.warning(f"compress_messages: Max iterations reached, omitting messages")
            result = self.compress_messages_by_omitting_messages(messages, llm_model, max_tokens)
            return result
    
        # 如果压缩后仍超过token限制，则进行更激进的压缩(阈值减半)
        if compressed_token_count > max_tokens:
            logger.warning(f"Further token compression is needed: {compressed_token_count} > {max_tokens}")
            result = self.compress_messages(messages, llm_model, max_tokens, token_threshold // 2, max_iterations - 1)
    
        # 最后采用"中间删除"策略保留关键消息
        return self.middle_out_messages(result)
    
    def compress_messages_by_omitting_messages(
            self, 
            messages: List[Dict[str, Any]], 
            llm_model: str, 
            max_tokens: Optional[int] = 41000,
            removal_batch_size: int = 10,
            min_messages_to_keep: int = 10
        ) -> List[Dict[str, Any]]:
        """Compress the messages by omitting messages from the middle.
        
        Args:
            messages: List of messages to compress
            llm_model: Model name for token counting
            max_tokens: Maximum allowed tokens
            removal_batch_size: Number of messages to remove per iteration
            min_messages_to_keep: Minimum number of messages to preserve
        """
        if not messages:
            return messages
            
        result = messages
        result = self.remove_meta_messages(result)

        # Early exit if no compression needed
        initial_token_count = token_counter(model=llm_model, messages=result)
        max_allowed_tokens = max_tokens or (100 * 1000)
        
        if initial_token_count <= max_allowed_tokens:
            return result

        # Separate system message (assumed to be first) from conversation messages
        system_message = messages[0] if messages and messages[0].get('role') == 'system' else None
        conversation_messages = result[1:] if system_message else result
        
        safety_limit = 500
        current_token_count = initial_token_count
        
        while current_token_count > max_allowed_tokens and safety_limit > 0:
            safety_limit -= 1
            
            if len(conversation_messages) <= min_messages_to_keep:
                logger.warning(f"Cannot compress further: only {len(conversation_messages)} messages remain (min: {min_messages_to_keep})")
                break

            # Calculate removal strategy based on current message count
            if len(conversation_messages) > (removal_batch_size * 2):
                # Remove from middle, keeping recent and early context
                middle_start = len(conversation_messages) // 2 - (removal_batch_size // 2)
                middle_end = middle_start + removal_batch_size
                conversation_messages = conversation_messages[:middle_start] + conversation_messages[middle_end:]
            else:
                # Remove from earlier messages, preserving recent context
                messages_to_remove = min(removal_batch_size, len(conversation_messages) // 2)
                if messages_to_remove > 0:
                    conversation_messages = conversation_messages[messages_to_remove:]
                else:
                    # Can't remove any more messages
                    break

            # Recalculate token count
            messages_to_count = ([system_message] + conversation_messages) if system_message else conversation_messages
            current_token_count = token_counter(model=llm_model, messages=messages_to_count)

        # Prepare final result
        final_messages = ([system_message] + conversation_messages) if system_message else conversation_messages
        final_token_count = token_counter(model=llm_model, messages=final_messages)
        
        logger.info(f"compress_messages_by_omitting_messages: {initial_token_count} -> {final_token_count} tokens ({len(messages)} -> {len(final_messages)} messages)")
            
        return final_messages
    
    def middle_out_messages(self, messages: List[Dict[str, Any]], max_messages: int = 320) -> List[Dict[str, Any]]:
        """Remove messages from the middle of the list, keeping max_messages total."""
        if len(messages) <= max_messages:
            return messages
        
        # Keep half from the beginning and half from the end
        keep_start = max_messages // 2
        keep_end = max_messages - keep_start
        
        return messages[:keep_start] + messages[-keep_end:]