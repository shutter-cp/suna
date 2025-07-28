"""
JSON helper utilities for handling both legacy (string) and new (dict/list) formats.

These utilities help with the transition from storing JSON as strings to storing
them as proper JSONB objects in the database.
"""

import json
from typing import Any, Union, Dict, List


def ensure_dict(value: Union[str, Dict[str, Any], None], default: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Ensure a value is a dictionary.
    
    Handles:
    - None -> returns default or {}
    - Dict -> returns as-is
    - JSON string -> parses and returns dict
    - Other -> returns default or {}
    
    Args:
        value: The value to ensure is a dict
        default: Default value if conversion fails
        
    Returns:
        A dictionary
    """
    if default is None:
        default = {}
        
    if value is None:
        return default
        
    if isinstance(value, dict):
        return value
        
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return default
        except (json.JSONDecodeError, TypeError):
            return default
            
    return default


def ensure_list(value: Union[str, List[Any], None], default: List[Any] = None) -> List[Any]:
    """
    Ensure a value is a list.
    
    Handles:
    - None -> returns default or []
    - List -> returns as-is
    - JSON string -> parses and returns list
    - Other -> returns default or []
    
    Args:
        value: The value to ensure is a list
        default: Default value if conversion fails
        
    Returns:
        A list
    """
    if default is None:
        default = []
        
    if value is None:
        return default
        
    if isinstance(value, list):
        return value
        
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return default
        except (json.JSONDecodeError, TypeError):
            return default
            
    return default


def safe_json_parse(value: Union[str, Dict, List, Any], default: Any = None) -> Any:
    """
    Safely parse a value that might be JSON string or already parsed.
    
    This handles the transition period where some data might be stored as
    JSON strings (old format) and some as proper objects (new format).
    
    Args:
        value: The value to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed value or default
    """
    if value is None:
        return default
        
    # If it's already a dict or list, return as-is
    if isinstance(value, (dict, list)):
        return value
        
    # If it's a string, try to parse it
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, return the string itself
            return value
            
    # For any other type, return as-is
    return value


def to_json_string(value: Any) -> str:
    """
    Convert a value to a JSON string if needed.
    
    This is used for backwards compatibility when yielding data that
    expects JSON strings.
    
    Args:
        value: The value to convert
        
    Returns:
        JSON string representation
    """
    if isinstance(value, str):
        # If it's already a string, check if it's valid JSON
        try:
            json.loads(value)
            return value  # It's already a JSON string
        except (json.JSONDecodeError, TypeError):
            # It's a plain string, encode it as JSON
            return json.dumps(value)
    
    # For all other types, convert to JSON
    return json.dumps(value)


def format_for_yield(message_object: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化消息对象以供生成器yield使用，确保内容和元数据是JSON字符串格式。
    
    此函数用于维护与客户端的向后兼容性，客户端期望接收JSON字符串格式的内容和元数据，
    而数据库现在存储的是适当的对象格式。
    
    在异步流式响应处理中，这个函数特别重要，因为它将数据库中的对象转换为适合通过
    Server-Sent Events (SSE) 发送到前端的格式。
    
    Args:
        message_object: 来自数据库的消息对象，可能包含以下字段：
            - content: 消息内容，可能是字符串或字典/列表格式
            - metadata: 消息元数据，可能是字符串或字典/列表格式
            - type: 消息类型（如 'status', 'tool_call', 'assistant_response' 等）
            - thread_id: 关联的线程ID
            - 其他标准消息字段
        
    Returns:
        Dict[str, Any]: 格式化后的消息对象，其中：
            - content 字段保证为JSON字符串格式
            - metadata 字段保证为JSON字符串格式
            - 其他字段保持不变
            
    Example:
        输入: {
            'type': 'assistant_response',
            'content': {'text': 'Hello world'},
            'metadata': {'tokens': 10}
        }
        
        输出: {
            'type': 'assistant_response',
            'content': '{"text": "Hello world"}',
            'metadata': '{"tokens": 10}'
        }
    """
    if not message_object:
        return message_object
        
    # 创建副本以避免修改原始对象
    formatted = message_object.copy()
    
    # 确保内容是JSON字符串格式
    # 如果内容不是字符串（即数据库中的新格式对象），则将其转换为JSON字符串
    if 'content' in formatted and not isinstance(formatted['content'], str):
        formatted['content'] = json.dumps(formatted['content'])
        
    # 确保元数据是JSON字符串格式
    # 如果元数据不是字符串（即数据库中的新格式对象），则将其转换为JSON字符串
    if 'metadata' in formatted and not isinstance(formatted['metadata'], str):
        formatted['metadata'] = json.dumps(formatted['metadata'])
        
    return formatted