"""
LLM API interface for making calls to various language models.

This module provides a unified interface for making API calls to different LLM providers
(OpenAI, Anthropic, Groq, xAI, etc.) using LiteLLM. It includes support for:
- Streaming responses
- Tool calls and function calling
- Retry logic with exponential backoff
- Model-specific configurations
- Comprehensive error handling and logging
"""

from typing import Union, Dict, Any, Optional, AsyncGenerator, List
import os
import json
import asyncio
from openai import OpenAIError
import litellm
from litellm.files.main import ModelResponse
from utils.logger import logger
from utils.config import config

# litellm.set_verbose=True
litellm.modify_params=True

# Constants
MAX_RETRIES = 2
RATE_LIMIT_DELAY = 30
RETRY_DELAY = 0.1

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRetryError(LLMError):
    """Exception raised when retries are exhausted."""
    pass

def setup_api_keys() -> None:
    """Set up API keys from environment variables."""
    providers = ['OPENAI', 'ANTHROPIC', 'GROQ', 'OPENROUTER', 'XAI']
    for provider in providers:
        # key = getattr(config, f'{provider}_API_KEY')
        key = getattr(config, 'OPENAI_API_KEY')
        if key:
            logger.debug(f"API key set for provider: {provider}")
        else:
            logger.warning(f"No API key found for provider: {provider}")

    # Set up OpenAI base URL if provided
    if config.OPENAI_API_KEY and config.OPENAI_BASE_URL:
        os.environ['OPENAI_API_BASE'] = config.OPENAI_BASE_URL
        logger.debug(f"Set OPENAI_API_BASE to {config.OPENAI_BASE_URL}")

    # Set up OpenRouter API base if not already set
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_BASE:
        os.environ['OPENROUTER_API_BASE'] = config.OPENROUTER_API_BASE
        logger.debug(f"Set OPENROUTER_API_BASE to {config.OPENROUTER_API_BASE}")

    # Set up AWS Bedrock credentials
    aws_access_key = config.AWS_ACCESS_KEY_ID
    aws_secret_key = config.AWS_SECRET_ACCESS_KEY
    aws_region = config.AWS_REGION_NAME

    if aws_access_key and aws_secret_key and aws_region:
        logger.debug(f"AWS credentials set for Bedrock in region: {aws_region}")
        # Configure LiteLLM to use AWS credentials
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_REGION_NAME'] = aws_region
    else:
        logger.warning(f"Missing AWS credentials for Bedrock integration - access_key: {bool(aws_access_key)}, secret_key: {bool(aws_secret_key)}, region: {aws_region}")

def get_openrouter_fallback(model_name: str) -> Optional[str]:
    """Get OpenRouter fallback model for a given model name."""
    # Skip if already using OpenRouter
    if model_name.startswith("openrouter/"):
        return None
    
    # Map models to their OpenRouter equivalents
    fallback_mapping = {
        "anthropic/claude-3-7-sonnet-latest": "openrouter/anthropic/claude-3.7-sonnet",
        "anthropic/claude-sonnet-4-20250514": "openrouter/anthropic/claude-sonnet-4",
        "xai/grok-4": "openrouter/x-ai/grok-4",
    }
    
    # Check for exact match first
    if model_name in fallback_mapping:
        return fallback_mapping[model_name]
    
    # Check for partial matches (e.g., bedrock models)
    for key, value in fallback_mapping.items():
        if key in model_name:
            return value
    
    # Default fallbacks by provider
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        return "openrouter/anthropic/claude-sonnet-4"
    elif "xai" in model_name.lower() or "grok" in model_name.lower():
        return "openrouter/x-ai/grok-4"
    
    return None

async def handle_error(error: Exception, attempt: int, max_attempts: int) -> None:
    """Handle API errors with appropriate delays and logging."""
    delay = RATE_LIMIT_DELAY if isinstance(error, litellm.exceptions.RateLimitError) else RETRY_DELAY
    logger.warning(f"Error on attempt {attempt + 1}/{max_attempts}: {str(error)}")
    logger.debug(f"Waiting {delay} seconds before retry...")
    await asyncio.sleep(delay)

def prepare_params(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Dict[str, Any]:
    """Prepare parameters for the API call."""
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
    }

    if api_key:
        params["api_key"] = api_key

    # Check if OPENAI_BASE_URL is configured and override api_base for all models
    if config.OPENAI_BASE_URL:
        params["api_base"] = config.OPENAI_BASE_URL
        logger.debug(f"Using OPENAI_BASE_URL for all models: {config.OPENAI_BASE_URL}")
    elif api_base:
        params["api_base"] = api_base

    if model_id:
        params["model_id"] = model_id

    # Handle token limits
    if max_tokens is not None:
        # For Claude 3.7 in Bedrock, do not set max_tokens or max_tokens_to_sample
        # as it causes errors with inference profiles
        if model_name.startswith("bedrock/") and "claude-3-7" in model_name:
            logger.debug(f"Skipping max_tokens for Claude 3.7 model: {model_name}")
            # Do not add any max_tokens parameter for Claude 3.7
        else:
            param_name = "max_completion_tokens" if 'o1' in model_name else "max_tokens"
            params[param_name] = max_tokens

    # Add tools if provided
    if tools:
        params.update({
            "tools": tools,
            "tool_choice": tool_choice
        })
        logger.debug(f"Added {len(tools)} tools to API parameters")

    # # Add Claude-specific headers
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        params["extra_headers"] = {
            # "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
            "anthropic-beta": "output-128k-2025-02-19"
        }
        # params["mock_testing_fallback"] = True
        logger.debug("Added Claude-specific headers")

    # Add OpenRouter-specific parameters
    if model_name.startswith("openrouter/"):
        logger.debug(f"Preparing OpenRouter parameters for model: {model_name}")

        # Add optional site URL and app name from config
        site_url = config.OR_SITE_URL
        app_name = config.OR_APP_NAME
        if site_url or app_name:
            extra_headers = params.get("extra_headers", {})
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name
            params["extra_headers"] = extra_headers
            logger.debug(f"Added OpenRouter site URL and app name to headers")

    # Add Bedrock-specific parameters
    if model_name.startswith("bedrock/"):
        logger.debug(f"Preparing AWS Bedrock parameters for model: {model_name}")

        if not model_id and "anthropic.claude-3-7-sonnet" in model_name:
            params["model_id"] = "arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            logger.debug(f"Auto-set model_id for Claude 3.7 Sonnet: {params['model_id']}")

    fallback_model = get_openrouter_fallback(model_name)
    if fallback_model:
        params["fallbacks"] = [{
            "model": fallback_model,
            "messages": messages,
        }]
        logger.debug(f"Added OpenRouter fallback for model: {model_name} to {fallback_model}")

    # Apply Anthropic prompt caching (minimal implementation)
    # Check model name *after* potential modifications (like adding bedrock/ prefix)
    effective_model_name = params.get("model", model_name) # Use model from params if set, else original
    if "claude" in effective_model_name.lower() or "anthropic" in effective_model_name.lower():
        messages = params["messages"] # Direct reference, modification affects params

        # Ensure messages is a list
        if not isinstance(messages, list):
            return params # Return early if messages format is unexpected

        # Apply cache control to the first 4 text blocks across all messages
        cache_control_count = 0
        max_cache_control_blocks = 3

        for message in messages:
            if cache_control_count >= max_cache_control_blocks:
                break
                
            content = message.get("content")
            
            if isinstance(content, str):
                message["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
                cache_control_count += 1
            elif isinstance(content, list):
                for item in content:
                    if cache_control_count >= max_cache_control_blocks:
                        break
                    if isinstance(item, dict) and item.get("type") == "text" and "cache_control" not in item:
                        item["cache_control"] = {"type": "ephemeral"}
                        cache_control_count += 1

    # Add reasoning_effort for Anthropic models if enabled
    use_thinking = enable_thinking if enable_thinking is not None else False
    is_anthropic = "anthropic" in effective_model_name.lower() or "claude" in effective_model_name.lower()
    is_xai = "xai" in effective_model_name.lower() or model_name.startswith("xai/")

    if is_anthropic and use_thinking:
        effort_level = reasoning_effort if reasoning_effort else 'low'
        params["reasoning_effort"] = effort_level
        params["temperature"] = 1.0 # Required by Anthropic when reasoning_effort is used
        logger.info(f"Anthropic thinking enabled with reasoning_effort='{effort_level}'")

    # Add reasoning_effort for xAI models if enabled
    if is_xai and use_thinking:
        effort_level = reasoning_effort if reasoning_effort else 'low'
        params["reasoning_effort"] = effort_level
        logger.info(f"xAI thinking enabled with reasoning_effort='{effort_level}'")

    # Add xAI-specific parameters
    if model_name.startswith("xai/"):
        logger.debug(f"Preparing xAI parameters for model: {model_name}")
        # xAI models support standard parameters, no special handling needed beyond reasoning_effort

    return params

async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Union[Dict[str, Any], AsyncGenerator, ModelResponse]:
    """
    使用LiteLLM调用语言模型的API接口函数，支持多种模型提供商（OpenAI、Anthropic、Google等）

    Args:
        messages: 对话消息列表，每个元素是一个包含角色和内容的字典
        model_name: 要使用的模型名称（例如："gpt-4", "claude-3", "openrouter/openai/gpt-4"）
        response_format: 响应格式要求（如JSON模式）
        temperature: 采样温度参数，控制输出的随机性（0-1之间）
        max_tokens: 响应中最大生成的token数量
        tools: 可供模型调用的工具函数定义列表
        tool_choice: 工具选择策略（"auto"自动选择或"none"不使用工具）
        api_key: 自定义API密钥（覆盖默认配置）
        api_base: 自定义API基础URL（覆盖默认配置）
        stream: 是否启用流式响应
        top_p: 核采样参数，控制生成多样性
        model_id: AWS Bedrock推理配置的ARN标识符
        enable_thinking: 是否启用模型的深度思考模式
        reasoning_effort: 推理努力程度（'low', 'medium', 'high'）

    Returns:
        Union[Dict[str, Any], AsyncGenerator]: API响应结果或流式生成器

    Raises:
        LLMRetryError: 当API调用在多次重试后仍然失败时抛出
        LLMError: 其他与API相关的错误
    """
    # 记录日志：显示正在调用的模型信息及思考模式设置
    logger.info(f"Making LLM API call to model: {model_name} (Thinking: {enable_thinking}, Effort: {reasoning_effort})")
    logger.info(f"📡 API Call: Using model {model_name}")

    # 准备API调用参数，根据模型类型和配置进行特定处理
    params = prepare_params(
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        top_p=top_p,
        model_id=model_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort
    )
    print(params)

    # 保存最后一次错误信息，用于重试失败后的错误报告
    last_error = None

    # 执行最大重试次数的循环，处理可能的API调用异常
    for attempt in range(MAX_RETRIES):
        try:
            # 记录当前尝试次数
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")

            # 使用LiteLLM异步调用API，传入准备好的参数
            response = await litellm.acompletion(**params)

            # 记录成功获取API响应的日志
            logger.debug(f"Successfully received API response from {model_name}")

            # 返回API响应结果
            return response

        # 处理特定的可重试异常：速率限制、OpenAI错误、JSON解析错误
        except (litellm.exceptions.RateLimitError, OpenAIError, json.JSONDecodeError) as e:
            # 保存当前错误信息
            last_error = e

            # 调用错误处理函数，可能包含延迟重试逻辑
            await handle_error(e, attempt, MAX_RETRIES)

        # 处理其他未预期的异常
        except Exception as e:
            # 记录详细错误日志
            logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)

            # 抛出自定义LLM错误
            raise LLMError(f"API call failed: {str(e)}")

    # 如果所有重试都失败，构建并抛出重试失败错误
    error_msg = f"Failed to make API call after {MAX_RETRIES} attempts"
    if last_error:
        error_msg += f". Last error: {str(last_error)}"
    logger.error(error_msg, exc_info=True)
    raise LLMRetryError(error_msg)

# 在模块导入时初始化API密钥配置
setup_api_keys()
