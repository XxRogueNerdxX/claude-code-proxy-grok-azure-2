"""
Claude Code Proxy - Multi-Provider Support
Supports: Azure OpenAI, Anthropic Claude (with custom header), and Groq
"""
import os
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_CUSTOM_HEADER = os.getenv("ANTHROPIC_CUSTOM_HEADER", "X-Lerck-APIKey")
USE_CUSTOM_HEADER_FOR_CLAUDE = os.getenv("USE_CUSTOM_HEADER_FOR_CLAUDE", "true").lower() == "true"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Default provider
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "anthropic")

# Model configuration
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
AZURE_MODEL = os.getenv("AZURE_MODEL", "gpt-4-turbo")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8082"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", "4096"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# HTTP client
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    global http_client
    http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    logger.info(f"Proxy server starting on {HOST}:{PORT}")
    logger.info(f"Default Provider: {DEFAULT_PROVIDER}")
    logger.info(f"Claude Model: {CLAUDE_MODEL}")
    logger.info(f"Azure Model: {AZURE_MODEL}")
    logger.info(f"Groq Model: {GROQ_MODEL}")
    if USE_CUSTOM_HEADER_FOR_CLAUDE:
        logger.info(f"Claude Custom Header Mode: ENABLED ({ANTHROPIC_CUSTOM_HEADER})")
    yield
    await http_client.aclose()
    logger.info("Proxy server shutting down")


app = FastAPI(title="Claude Code Proxy (Azure/Claude/Groq)", lifespan=lifespan)


def detect_provider(model: str) -> str:
    """Detect which provider to use based on model name"""
    model_lower = model.lower()
    
    if "claude" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower:
        return "azure"
    elif "mixtral" in model_lower or "llama" in model_lower or "gemma" in model_lower:
        return "groq"
    else:
        return DEFAULT_PROVIDER


def convert_claude_to_openai_messages(messages: list) -> list:
    """Convert Claude API messages to OpenAI format"""
    openai_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            converted_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        converted_content.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        image_data = item.get("source", {})
                        if image_data.get("type") == "base64":
                            converted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_data.get('media_type', 'image/png')};base64,{image_data.get('data', '')}"
                                }
                            })
                elif isinstance(item, str):
                    converted_content.append(item)
            
            if converted_content:
                if all(isinstance(c, str) for c in converted_content):
                    openai_messages.append({"role": role, "content": " ".join(converted_content)})
                else:
                    openai_messages.append({"role": role, "content": converted_content})
    
    return openai_messages


def convert_claude_tools_to_openai(tools: list) -> list:
    """Convert Claude tools to OpenAI format"""
    if not tools:
        return []
    
    openai_tools = []
    for tool in tools:
        if tool.get("type") == "function" or "input_schema" in tool:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)
    
    return openai_tools


async def convert_openai_to_claude_response(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI response to Claude format"""
    choice = openai_response.get("choices", [{}])
    message = choice.get("message", {})
    
    content = []
    
    if message.get("content"):
        content.append({
            "type": "text",
            "text": message["content"]
        })
    
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            content.append({
                "type": "tool_use",
                "id": tool_call.get("id"),
                "name": tool_call.get("function", {}).get("name"),
                "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            })
    
    claude_response = {
        "id": openai_response.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": openai_response.get("model", ""),
        "stop_reason": "end_turn" if choice.get("finish_reason") == "stop" else choice.get("finish_reason"),
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0),
        }
    }
    
    return claude_response


async def stream_openai_to_claude(response: httpx.Response) -> AsyncGenerator[str, None]:
    """Stream OpenAI SSE response and convert to Claude format"""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                continue
            
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}]).get("delta", {})
                
                claude_event = {
                    "type": "content_block_delta",
                    "delta": {}
                }
                
                if "content" in delta:
                    claude_event["delta"] = {
                        "type": "text_delta",
                        "text": delta["content"]
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(claude_event)}\n\n"
                
                if "tool_calls" in delta:
                    for tool_call in delta["tool_calls"]:
                        claude_event["delta"] = {
                            "type": "tool_use",
                            "id": tool_call.get("id"),
                            "name": tool_call.get("function", {}).get("name"),
                            "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(claude_event)}\n\n"
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE data: {data}")
                continue


async def stream_anthropic_response(response: httpx.Response) -> AsyncGenerator[str, None]:
    """Stream Anthropic response directly (already in Claude format)"""
    async for line in response.aiter_lines():
        if line:
            yield line + "\n"


async def route_to_provider(provider: str, model: str, messages: list, system: Optional[str],
                           tools: Optional[list], max_tokens: int, temperature: float, stream: bool):
    """Route request to appropriate provider"""
    
    if provider == "anthropic":
        return await route_to_anthropic(model, messages, system, tools, max_tokens, temperature, stream)
    elif provider == "azure":
        return await route_to_azure(model, messages, system, tools, max_tokens, temperature, stream)
    elif provider == "groq":
        return await route_to_groq(model, messages, system, tools, max_tokens, temperature, stream)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


async def route_to_anthropic(model: str, messages: list, system: Optional[str],
                            tools: Optional[list], max_tokens: int, temperature: float, stream: bool):
    """Route request to Anthropic Claude API or custom Claude endpoint"""
    
    headers = {
        "Content-Type": "application/json",
    }
    
    # Use custom header if enabled, otherwise use standard Anthropic auth
    if USE_CUSTOM_HEADER_FOR_CLAUDE:
        headers[ANTHROPIC_CUSTOM_HEADER] = ANTHROPIC_API_KEY
        logger.info(f"Using custom header {ANTHROPIC_CUSTOM_HEADER} for Claude authentication")
    else:
        headers["x-api-key"] = ANTHROPIC_API_KEY
        headers["anthropic-version"] = "2023-06-01"
        logger.info("Using standard x-api-key authentication for Claude")
    
    request_body = {
        "model": model or CLAUDE_MODEL,
        "max_tokens": min(max_tokens, MAX_TOKENS_LIMIT),
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    
    if system:
        request_body["system"] = system
    
    if tools:
        request_body["tools"] = tools
    
    try:
        logger.info(f"Routing to Anthropic: {ANTHROPIC_BASE_URL}/v1/messages")
        response = await http_client.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            json=request_body,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        
        if stream:
            return StreamingResponse(
                stream_anthropic_response(response),
                media_type="text/event-stream",
            )
        else:
            return response.json()
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Anthropic: {e.response.status_code} - {e}")
        error_detail = await e.response.atext() if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=f"Anthropic API error: {error_detail}")
    except Exception as e:
        logger.error(f"Error routing to Anthropic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def route_to_azure(model: str, messages: list, system: Optional[str],
                        tools: Optional[list], max_tokens: int, temperature: float, stream: bool):
    """Route request to Azure OpenAI"""
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    
    # Convert Claude messages to OpenAI format
    openai_messages = convert_claude_to_openai_messages(messages)
    
    if system:
        openai_messages.insert(0, {"role": "system", "content": system})
    
    request_body = {
        "messages": openai_messages,
        "max_tokens": min(max_tokens, MAX_TOKENS_LIMIT),
        "temperature": temperature,
        "stream": stream,
    }
    
    if tools:
        openai_tools = convert_claude_tools_to_openai(tools)
        if openai_tools:
            request_body["tools"] = openai_tools
    
    try:
        # Azure OpenAI URL format
        deployment = AZURE_DEPLOYMENT_NAME or "deployment"
        url = f"{AZURE_OPENAI_BASE_URL}/openai/deployments/{deployment}/chat/completions?api-version=2024-08-01-preview"
        
        logger.info(f"Routing to Azure: {url}")
        response = await http_client.post(
            url,
            json=request_body,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        
        if stream:
            return StreamingResponse(
                stream_openai_to_claude(response),
                media_type="text/event-stream",
            )
        else:
            openai_response = response.json()
            claude_response = await convert_openai_to_claude_response(openai_response)
            return claude_response
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Azure OpenAI: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error routing to Azure OpenAI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def route_to_groq(model: str, messages: list, system: Optional[str],
                       tools: Optional[list], max_tokens: int, temperature: float, stream: bool):
    """Route request to Groq"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    
    # Convert Claude messages to OpenAI format
    openai_messages = convert_claude_to_openai_messages(messages)
    
    if system:
        openai_messages.insert(0, {"role": "system", "content": system})
    
    request_body = {
        "model": model or GROQ_MODEL,
        "messages": openai_messages,
        "max_tokens": min(max_tokens, MAX_TOKENS_LIMIT),
        "temperature": temperature,
        "stream": stream,
    }
    
    if tools:
        openai_tools = convert_claude_tools_to_openai(tools)
        if openai_tools:
            request_body["tools"] = openai_tools
    
    try:
        logger.info(f"Routing to Groq: {GROQ_BASE_URL}/chat/completions")
        response = await http_client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            json=request_body,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        
        if stream:
            return StreamingResponse(
                stream_openai_to_claude(response),
                media_type="text/event-stream",
            )
        else:
            openai_response = response.json()
            claude_response = await convert_openai_to_claude_response(openai_response)
            return claude_response
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Groq: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error routing to Groq: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Claude Code Proxy (Azure OpenAI / Claude / Groq)",
        "status": "running",
        "supported_providers": ["anthropic", "azure", "groq"],
        "claude_custom_header_enabled": USE_CUSTOM_HEADER_FOR_CLAUDE,
        "claude_custom_header_name": ANTHROPIC_CUSTOM_HEADER if USE_CUSTOM_HEADER_FOR_CLAUDE else "x-api-key",
        "config": {
            "default_provider": DEFAULT_PROVIDER,
            "claude_model": CLAUDE_MODEL,
            "azure_model": AZURE_MODEL,
            "groq_model": GROQ_MODEL,
        }
    }


@app.post("/v1/messages")
async def create_message(request: Request):
    """Main endpoint for Claude API messages"""
    
    # Parse request body
    try:
        claude_request = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    
    model = claude_request.get("model", "")
    messages = claude_request.get("messages", [])
    system = claude_request.get("system")
    tools = claude_request.get("tools")
    max_tokens = claude_request.get("max_tokens", MAX_TOKENS_LIMIT)
    temperature = claude_request.get("temperature", 1.0)
    stream = claude_request.get("stream", False)
    
    # Detect provider from model name or use default
    provider = detect_provider(model)
    logger.info(f"Request: model={model}, provider={provider}, stream={stream}")
    
    # Route to appropriate provider
    try:
        result = await route_to_provider(provider, model, messages, system, tools, max_tokens, temperature, stream)
        
        if isinstance(result, StreamingResponse):
            return result
        else:
            return Response(
                content=json.dumps(result),
                media_type="application/json",
            )
    
    except Exception as e:
        logger.error(f"Error in request handling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
