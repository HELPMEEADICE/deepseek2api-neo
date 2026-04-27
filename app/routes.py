import json
import logging
import queue
import re
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import chat, config, constants, session as session_module
from .account import (
    determine_mode_and_token,
    get_auth_headers,
    get_hif_headers,
    release_account,
)
from .converter import (
    build_anthropic_response,
    deepseek_line_to_anthropic_events,
    make_message_delta_event,
    make_message_start_event,
    make_message_stop_event,
)
from .pow import get_pow_response

logger = logging.getLogger(__name__)

router = APIRouter()


def _decode_json_string_prefix(raw: str) -> str:
    chars = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            break
        if ch == "\\":
            if i + 1 >= len(raw):
                break
            nxt = raw[i + 1]
            mapping = {
                '"': '"', "\\": "\\", "/": "/",
                "b": "\b", "f": "\f", "n": "\n",
                "r": "\r", "t": "\t",
            }
            if nxt == "u":
                if i + 6 > len(raw):
                    break
                try:
                    chars.append(chr(int(raw[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    chars.append(nxt)
            else:
                chars.append(mapping.get(nxt, nxt))
            i += 2
            continue
        chars.append(ch)
        i += 1
    return "".join(chars)


def _balanced_json_prefix(raw: str) -> str:
    in_string = False
    escape = False
    stack = []
    started = False
    start = 0
    for idx, ch in enumerate(raw):
        if not started:
            if ch.isspace():
                continue
            if ch not in "[{":
                return ""
            start = idx
            stack = ["}" if ch == "{" else "]"]
            started = True
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                return ""
            stack.pop()
            if not stack:
                return raw[start:idx + 1]
    return ""


def _extract_stream_tool_meta(text: str):
    call_id = None
    name = None
    id_match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
    if id_match:
        call_id = id_match.group(1)
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if name_match:
        name = name_match.group(1)
    if not name:
        xml_match = re.search(r'<tool_call\s+name=["\']([^"\']+)["\']', text, re.I)
        if xml_match:
            name = xml_match.group(1)
    if not name:
        attr_match = re.search(r'\bfunction=["\'](?:name\s+)?([^"\'\s>]+)', text, re.I)
        if attr_match:
            name = attr_match.group(1)
    return call_id, name


def _extract_stream_arguments(text: str) -> str:
    match = re.search(r'"arguments"\s*:\s*', text)
    if match:
        raw = text[match.end():]
        stripped = raw.lstrip()
        if stripped.startswith('"'):
            return _decode_json_string_prefix(stripped[1:])
        json_prefix = _balanced_json_prefix(stripped)
        if json_prefix:
            return json_prefix
        return stripped

    xml_match = re.search(r'<tool_call\s+name=["\'][^"\']+["\']\s*>', text, re.I)
    if xml_match:
        value = text[xml_match.end():]
        end = re.search(r'</tool_call>', value, re.I)
        return value[:end.start()] if end else value

    attr_match = re.search(r'\barguments=["\']', text, re.I)
    if attr_match:
        return _decode_json_string_prefix(text[attr_match.end():])
    return ""


TOOL_ARGUMENT_STREAM_CHUNK_SIZE = 24
TOOL_ARGUMENT_STREAM_DELAY = 0.015


def _iter_text_chunks(text: str, size: int = TOOL_ARGUMENT_STREAM_CHUNK_SIZE):
    for start in range(0, len(text), size):
        yield text[start:start + size]


# ----------------------------------------------------------------------
# 删除 DeepSeek 会话的通用辅助函数
# ----------------------------------------------------------------------
def _delete_deepseek_session(request: Request, session_id: str):
    """响应结束后删除 DeepSeek 会话"""
    try:
        headers = get_auth_headers(request)
        payload = {"chat_session_id": session_id}
        ds_session = session_module.get_request_session(request)
        resp = ds_session.post(
            constants.DEEPSEEK_DELETE_SESSION_URL,
            headers=headers,
            json=payload,
            impersonate="safari15_3",
            timeout=3,
        )
        if resp.status_code == 200:
            logger.info(f"[delete_deepseek_session] 响应结束，已删除会话 session={session_id}")
        else:
            logger.warning(f"[delete_deepseek_session] 删除会话失败: {resp.status_code}")
    except Exception as e:
        logger.warning(f"[delete_deepseek_session] 调用 delete_session 失败: {e}")


# ----------------------------------------------------------------------
# 路由：/v1/models
# ----------------------------------------------------------------------
BASE_MODELS = [
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-pro",
]


@router.get("/v1/models")
def list_models():
    models_list = []
    for base_id in BASE_MODELS:
        models_list.append({
            "id": base_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": [],
        })
        models_list.append({
            "id": f"{base_id}-search",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": [],
        })
    data = {"object": "list", "data": models_list}
    return JSONResponse(content=data, status_code=200)


# ----------------------------------------------------------------------
# 路由：/v1/chat/completions
# ----------------------------------------------------------------------
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        # 处理 token 相关逻辑，若登录失败则直接返回错误响应
        try:
            determine_mode_and_token(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code, content={"error": exc.detail}
            )
        except Exception as exc:
            logger.error(f"[chat_completions] determine_mode_and_token 异常: {exc}")
            return JSONResponse(
                status_code=500, content={"error": "Account login failed."}
            )

        req_data = await request.json()
        model = req_data.get("model")
        messages = req_data.get("messages", [])
        if not model or not messages:
            raise HTTPException(
                status_code=400, detail="Request must include 'model' and 'messages'."
            )
        # 判断模型类型（支持 -search 后缀）
        model_lower = model.lower().strip()
        if model_lower.endswith("-search"):
            model_lower = model_lower[:-7]
            search_by_model = True
        else:
            search_by_model = False

        if model_lower in ["deepseek-v4-flash", "deepseek-chat"]:
            model_type = "default"
            auto_thinking = True
            lock_thinking = False
        elif model_lower in ["deepseek-reasoner"]:
            model_type = "default"
            auto_thinking = True
            lock_thinking = True  # deepseek-reasoner 强制开启思考，不可关闭
        elif model_lower in ["deepseek-v4-pro"]:
            model_type = "expert"
            auto_thinking = True
            lock_thinking = False
        else:
            raise HTTPException(
                status_code=503, detail=f"Model '{model}' is not available."
            )
        # 解析 thinking，支持 thinking_enabled 和 thinking.type 两种格式
        if lock_thinking:
            thinking_enabled = True
        else:
            thinking_obj = req_data.get("thinking")
            if isinstance(thinking_obj, dict):
                thinking_enabled = thinking_obj.get("type") == "enabled"
            elif isinstance(thinking_obj, bool):
                thinking_enabled = thinking_obj
            else:
                thinking_enabled = req_data.get("thinking_enabled", auto_thinking)
                if not isinstance(thinking_enabled, bool):
                    thinking_enabled = auto_thinking
        search_enabled = bool(req_data.get("search_enabled", False)) or search_by_model

        # 处理 tools 参数（OpenAI 格式）
        tools_requested = req_data.get("tools") or []
        tool_system_prompt = chat.build_tool_system_prompt(
            tools_requested, source="openai", tool_choice=req_data.get("tool_choice")
        )
        has_tools = bool(tool_system_prompt)

        # 如果有工具定义，在 messages 前添加工具使用指导的系统消息
        if has_tools:
            tool_schemas = []
            for tool in tools_requested:
                func = tool.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_desc = func.get("description", "No description available")
                params = func.get("parameters", {})

                tool_info = f"Tool: {tool_name}\nDescription: {tool_desc}"
                if "properties" in params:
                    props = []
                    required = params.get("required", [])
                    for prop_name, prop_info in params["properties"].items():
                        prop_type = prop_info.get("type", "string")
                        prop_desc = prop_info.get("description", "")
                        is_req = " (required)" if prop_name in required else ""
                        props.append(f"  - {prop_name}: {prop_type}{is_req} - {prop_desc}")
                    if props:
                        tool_info += f"\nParameters:\n{chr(10).join(props)}"
                tool_schemas.append(tool_info)

            tool_system_prompt = f"""You have access to the following tools:

{chr(10).join(tool_schemas)}

When you need to use a tool, respond with a JSON object in this exact format:
{{"tool_calls": [{{"id": "call_xxx", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}

You can call multiple tools in one response by adding more objects to the tool_calls array.
IMPORTANT: The "arguments" field must be a JSON string, not a JSON object.

Example:
{{"tool_calls": [{{"id": "call_001", "type": "function", "function": {{"name": "get_weather", "arguments": "{{\\"location\\": \\"Beijing\\"}}"}}}}]}}

After calling tools, you will receive the results and can continue the conversation."""

            # 将工具说明添加到第一个 system 消息，或创建新的 system 消息
            system_found = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + "\n\n" + tool_system_prompt
                    system_found = True
                    break

            if not system_found:
                messages.insert(0, {"role": "system", "content": tool_system_prompt})

        # 使用 messages_prepare 函数构造最终 prompt
        final_prompt = chat.messages_prepare(messages)
        session_id = chat.create_session(request)
        if not session_id:
            raise HTTPException(status_code=401, detail="invalid token.")
        pow_resp = get_pow_response(request)
        if not pow_resp:
            raise HTTPException(
                status_code=401,
                detail="Failed to get PoW (invalid token or unknown error).",
            )
        headers = {
            **get_auth_headers(request),
            **get_hif_headers(request),
            "x-ds-pow-response": pow_resp,
        }
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "model_type": model_type,
            "prompt": final_prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }

        deepseek_resp = chat.call_completion_endpoint(payload, headers, session_module.get_request_session(request), max_attempts=3)
        if not deepseek_resp:
            raise HTTPException(status_code=500, detail="Failed to get completion.")
        created_time = int(time.time())
        completion_id = f"{session_id}"

        # 流式响应（SSE）或普通响应
        if bool(req_data.get("stream", False)):
            if deepseek_resp.status_code != 200:
                deepseek_resp.close()
                return JSONResponse(
                    content=deepseek_resp.content, status_code=deepseek_resp.status_code
                )

            def sse_stream():
                client_disconnected = False
                stream_completed = False  # 标记流是否正常完成
                try:
                    final_text = ""
                    final_thinking = ""
                    first_chunk_sent = False
                    tool_call_buffering = False
                    tool_stream_started = False
                    tool_stream_arg_sent = 0
                    tool_stream_id = "call_001"
                    tool_stream_name = ""
                    tool_stream_buffer = ""
                    result_queue = queue.Queue()
                    last_send_time = time.time()

                    def process_data():
                        current_ptype = "text"
                        try:
                            for raw_line in deepseek_resp.iter_lines():
                                try:
                                    line = raw_line.decode("utf-8")
                                except Exception as e:
                                    logger.warning(f"[sse_stream] 解码失败: {e}")
                                    error_type = "thinking" if current_ptype == "thinking" else "text"
                                    busy_content_str = f'{{"choices":[{{"index":0,"delta":{{"content":"解码失败，请稍候再试","type":"{error_type}"}}}}],"model":"","chunk_token_usage":1,"created":0,"message_id":-1,"parent_id":-1}}'
                                    try:
                                        busy_content = json.loads(busy_content_str)
                                        result_queue.put(busy_content)
                                    except json.JSONDecodeError:
                                        result_queue.put({"choices": [{"index": 0, "delta": {"content": "解码失败", "type": "text"}}]})
                                    result_queue.put(None)
                                    break
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        result_queue.put(None)  # 结束信号
                                        break
                                    try:
                                        chunk = json.loads(data_str)

                                        # ---- 新格式：初始片段状态（v.response.fragments） ----
                                        if "v" in chunk and isinstance(chunk["v"], dict) and "response" in chunk["v"]:
                                            fragments = chunk["v"]["response"].get("fragments", [])
                                            if fragments and isinstance(fragments, list) and len(fragments) > 0:
                                                frag_type = fragments[0].get("type", "")
                                                current_ptype = "thinking" if frag_type == "THINK" else "text"
                                                # 处理片段中的初始内容
                                                frag_content = fragments[0].get("content", "")
                                                if frag_content:
                                                    unified_chunk = {
                                                        "choices": [{"index": 0, "delta": {"content": frag_content, "type": current_ptype}}],
                                                        "model": "", "chunk_token_usage": len(frag_content) // 4,
                                                        "created": 0, "message_id": -1, "parent_id": -1,
                                                    }
                                                    result_queue.put(unified_chunk)
                                            continue

                                        # ---- 新格式：fragment 追加（THINK→RESPONSE 切换） ----
                                        if "p" in chunk and chunk.get("p") == "response/fragments" and chunk.get("o") == "APPEND":
                                            new_frags = chunk.get("v", [])
                                            if new_frags and isinstance(new_frags, list) and len(new_frags) > 0:
                                                frag_type = new_frags[0].get("type", "")
                                                current_ptype = "thinking" if frag_type == "THINK" else "text"
                                                frag_content = new_frags[0].get("content", "")
                                                if frag_content:
                                                    unified_chunk = {
                                                        "choices": [{"index": 0, "delta": {"content": frag_content, "type": current_ptype}}],
                                                        "model": "", "chunk_token_usage": len(frag_content) // 4,
                                                        "created": 0, "message_id": -1, "parent_id": -1,
                                                    }
                                                    result_queue.put(unified_chunk)
                                            continue

                                        # ---- 兼容旧格式：thinking_content / content 切换 ----
                                        if "p" in chunk and chunk.get("p") == "response/thinking_content":
                                            current_ptype = "thinking"
                                        elif "p" in chunk and chunk.get("p") == "response/content":
                                            current_ptype = "text"

                                        # ---- status / search_status ----
                                        if "p" in chunk and chunk.get("p") == "response/status":
                                            if chunk.get("v") == 'FINISHED':
                                                result_queue.put(None)
                                                break
                                            continue
                                        if "p" in chunk and chunk.get("p") == "response/search_status":
                                            continue

                                        # ---- 处理 v 字段 ----
                                        if "v" in chunk:
                                            v_value = chunk["v"]

                                            if isinstance(v_value, str):
                                                content = v_value
                                            elif isinstance(v_value, list):
                                                for item in v_value:
                                                    if item.get("p") == "status" and item.get("v") == "FINISHED":
                                                        result_queue.put({"choices": [{"index": 0, "finish_reason": "stop"}]})
                                                        result_queue.put(None)
                                                        return
                                                continue
                                            else:
                                                continue

                                            unified_chunk = {
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"content": content, "type": current_ptype}
                                                }],
                                                "model": "",
                                                "chunk_token_usage": len(content) // 4,
                                                "created": 0,
                                                "message_id": -1,
                                                "parent_id": -1,
                                            }
                                            result_queue.put(unified_chunk)
                                    except Exception as e:
                                        logger.warning(f"[sse_stream] 无法解析: {data_str}, 错误: {e}")
                                        error_type = "thinking" if current_ptype == "thinking" else "text"
                                        busy_content_str = f'{{"choices":[{{"index":0,"delta":{{"content":"解析失败，请稍候再试","type":"{error_type}"}}}}],"model":"","chunk_token_usage":1,"created":0,"message_id":-1,"parent_id":-1}}'
                                        try:
                                            busy_content = json.loads(busy_content_str)
                                            result_queue.put(busy_content)
                                        except json.JSONDecodeError:
                                            result_queue.put({"choices": [{"index": 0, "delta": {"content": "解析失败", "type": "text"}}]})
                                        result_queue.put(None)
                                        break
                        except Exception as e:
                            logger.warning(f"[sse_stream] 错误: {e}")
                            try:
                                error_response = {"choices": [{"index": 0, "delta": {"content": "服务器错误，请稍候再试", "type": "text"}}]}
                                result_queue.put(error_response)
                            except Exception:
                                pass
                            result_queue.put(None)
                        finally:
                            deepseek_resp.close()

                    process_thread = threading.Thread(target=process_data)
                    process_thread.start()

                    def _stream_tool_delta(tool_id=None, tool_name=None, arguments_part=None):
                        nonlocal first_chunk_sent, last_send_time
                        delta_tool = {"index": 0}
                        if tool_id is not None:
                            delta_tool["id"] = tool_id
                            delta_tool["type"] = "function"
                        func_delta = {}
                        if tool_name is not None:
                            func_delta["name"] = tool_name
                        if arguments_part is not None:
                            func_delta["arguments"] = arguments_part
                        if func_delta:
                            delta_tool["function"] = func_delta

                        delta_obj = {"tool_calls": [delta_tool]}
                        if not first_chunk_sent:
                            delta_obj["role"] = "assistant"
                            first_chunk_sent = True

                        out = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": delta_obj,
                                "finish_reason": None,
                            }],
                        }
                        last_send_time = time.time()
                        return f"data: {json.dumps(out, ensure_ascii=False)}\n\n"

                    def _stream_tool_arguments(arguments_part: str):
                        for arg_part in _iter_text_chunks(arguments_part):
                            yield _stream_tool_delta(arguments_part=arg_part)
                            if TOOL_ARGUMENT_STREAM_DELAY:
                                time.sleep(TOOL_ARGUMENT_STREAM_DELAY)

                    try:
                        while True:
                            current_time = time.time()
                            if current_time - last_send_time >= constants.KEEP_ALIVE_TIMEOUT:
                                yield ": keep-alive\n\n"
                                last_send_time = current_time
                                continue
                            try:
                                chunk = result_queue.get(timeout=0.05)
                            except queue.Empty:
                                continue

                            if chunk is None:
                                # 检测 tool_calls（如果启用了 tools）
                                tool_calls_detected = None
                                final_text_content = final_text
                                parse_source = tool_stream_buffer or final_text
                                tool_calls_detected, parsed_remaining = chat.detect_and_parse_tool_calls(parse_source)
                                if tool_stream_buffer:
                                    final_text_content = final_text
                                    if not tool_calls_detected:
                                        detected_name = tool_stream_name
                                        args = _extract_stream_arguments(tool_stream_buffer)
                                        if detected_name:
                                            tool_calls_detected = [{
                                                "id": tool_stream_id,
                                                "type": "function",
                                                "function": {
                                                    "name": detected_name,
                                                    "arguments": args or "{}",
                                                },
                                            }]
                                else:
                                    final_text_content = parsed_remaining

                                if tool_calls_detected and final_text_content != final_text:
                                    # Do not leak the raw JSON instruction payload as assistant text.
                                    final_text = final_text_content

                                if has_tools and final_text and not tool_calls_detected:
                                    text_delta = {"content": final_text}
                                    if not first_chunk_sent:
                                        text_delta["role"] = "assistant"
                                        first_chunk_sent = True
                                    text_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": text_delta,
                                            "finish_reason": None,
                                        }],
                                    }
                                    yield f"data: {json.dumps(text_chunk, ensure_ascii=False)}\n\n"
                                    last_send_time = current_time

                                # 如果检测到 tool_calls，先发送 tool_calls chunk
                                if tool_calls_detected and not tool_stream_started:
                                    for tool_index, tool_call in enumerate(tool_calls_detected):
                                        func = tool_call.get("function", {})
                                        arguments = func.get("arguments", "{}")
                                        delta_obj = {
                                            "tool_calls": [{
                                                "index": tool_index,
                                                "id": tool_call.get("id", f"call_{tool_index + 1:03d}"),
                                                "type": tool_call.get("type", "function"),
                                                "function": {
                                                    "name": func.get("name", ""),
                                                    "arguments": "",
                                                },
                                            }]
                                        }
                                        if not first_chunk_sent:
                                            delta_obj["role"] = "assistant"
                                            first_chunk_sent = True
                                        tool_call_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": delta_obj,
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        yield f"data: {json.dumps(tool_call_chunk, ensure_ascii=False)}\n\n"
                                        last_send_time = current_time

                                        for arg_part in _iter_text_chunks(arguments):
                                            arg_chunk = {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_time,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": [{
                                                            "index": tool_index,
                                                            "function": {
                                                                "arguments": arg_part,
                                                            },
                                                        }]
                                                    },
                                                    "finish_reason": None,
                                                }],
                                            }
                                            yield f"data: {json.dumps(arg_chunk, ensure_ascii=False)}\n\n"
                                            last_send_time = time.time()
                                            if TOOL_ARGUMENT_STREAM_DELAY:
                                                time.sleep(TOOL_ARGUMENT_STREAM_DELAY)

                                # 发送最终统计信息
                                prompt_tokens = len(final_prompt) // 4
                                thinking_tokens = len(final_thinking) // 4
                                completion_tokens = len(final_text) // 4
                                usage = {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": thinking_tokens + completion_tokens,
                                    "total_tokens": prompt_tokens + thinking_tokens + completion_tokens,
                                    "completion_tokens_details": {
                                        "reasoning_tokens": thinking_tokens
                                    },
                                }

                                # 根据是否有 tool_calls 设置 finish_reason
                                finish_reason = "tool_calls" if tool_calls_detected else "stop"

                                finish_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": [
                                        {
                                            "delta": {},
                                            "index": 0,
                                            "finish_reason": finish_reason,
                                        }
                                    ],
                                    "usage": usage,
                                }
                                yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                stream_completed = True
                                last_send_time = current_time
                                break
                            new_choices = []
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                ctype = delta.get("type")
                                ctext = delta.get("content", "")
                                if (
                                    choice
                                    .get("finish_reason")
                                    == "backend_busy"
                                ):
                                    ctext = '服务器繁忙，请稍候再试'
                                if search_enabled and ctext.startswith("[citation:"):
                                    ctext = ""
                                if ctype == "thinking":
                                    if thinking_enabled:
                                        final_thinking += ctext
                                elif ctype == "text":
                                    final_text += ctext

                                if ctype == "text" and not tool_call_buffering:
                                    tool_buffer_started_now = False
                                    stripped_text = chat.strip_partial_tool_call_text(final_text)
                                    if stripped_text != final_text:
                                        tool_call_buffering = True
                                        tool_buffer_started_now = True
                                        tool_stream_buffer = final_text[len(stripped_text):]
                                        final_text = stripped_text
                                else:
                                    tool_buffer_started_now = False

                                if ctype == "text" and tool_call_buffering:
                                    if not tool_buffer_started_now and ctext:
                                        tool_stream_buffer += ctext
                                    full_buffer = tool_stream_buffer
                                    detected_id, detected_name = _extract_stream_tool_meta(full_buffer)
                                    if detected_id:
                                        tool_stream_id = detected_id
                                    if detected_name:
                                        tool_stream_name = detected_name
                                    if tool_stream_name and not tool_stream_started:
                                        tool_stream_started = True
                                        yield _stream_tool_delta(
                                            tool_id=tool_stream_id,
                                            tool_name=tool_stream_name,
                                            arguments_part="",
                                        )
                                    if tool_stream_started:
                                        arg_stream = _extract_stream_arguments(full_buffer)
                                        if len(arg_stream) > tool_stream_arg_sent:
                                            arg_part = arg_stream[tool_stream_arg_sent:]
                                            tool_stream_arg_sent = len(arg_stream)
                                            if arg_part:
                                                yield from _stream_tool_arguments(arg_part)

                                # When tools are enabled (or a tool-call marker appears), buffer
                                # text until completion so raw tool-call markup is not shown.
                                if ctype == "text" and (has_tools or tool_call_buffering):
                                    continue

                                delta_obj = {}
                                if not first_chunk_sent:
                                    delta_obj["role"] = "assistant"
                                    first_chunk_sent = True
                                if ctype == "thinking":
                                    if thinking_enabled:
                                        delta_obj["reasoning_content"] = ctext
                                elif ctype == "text":
                                    delta_obj["content"] = ctext
                                if delta_obj:
                                    new_choices.append(
                                        {
                                            "delta": delta_obj,
                                            "index": choice.get("index", 0),
                                        }
                                    )
                            if new_choices:
                                out_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": new_choices,
                                }
                                yield f"data: {json.dumps(out_chunk, ensure_ascii=False)}\n\n"
                                last_send_time = current_time
                    except GeneratorExit:
                        # 客户端主动断开连接（正常情况）
                        logger.info(f"[sse_stream] 客户端断开连接 session={session_id}")
                        client_disconnected = True
                        raise
                except Exception as e:
                    logger.error(f"[sse_stream] 异常: {e}")
                    client_disconnected = True
                finally:
                    _delete_deepseek_session(request, session_id)

                    if getattr(request.state, "use_config_token", False) and hasattr(
                        request.state, "account"
                    ):
                        release_account(request.state.account)

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream"},
            )
        else:
            # 非流式响应处理
            think_list = []
            text_list = []
            result = None

            data_queue = queue.Queue()

            def collect_data():
                nonlocal result
                current_ptype = "text"
                try:
                    for raw_line in deepseek_resp.iter_lines():
                        try:
                            line = raw_line.decode("utf-8")
                        except Exception as e:
                            logger.warning(f"[chat_completions] 解码失败: {e}")
                            if current_ptype == "thinking":
                                think_list.append("解码失败，请稍候再试")
                            else:
                                text_list.append("解码失败，请稍候再试")
                            data_queue.put(None)
                            break
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                data_queue.put(None)
                                break
                            try:
                                chunk = json.loads(data_str)

                                # ---- 新格式：初始片段状态（v.response.fragments） ----
                                if "v" in chunk and isinstance(chunk["v"], dict) and "response" in chunk["v"]:
                                    fragments = chunk["v"]["response"].get("fragments", [])
                                    if fragments and isinstance(fragments, list) and len(fragments) > 0:
                                        frag_type = fragments[0].get("type", "")
                                        current_ptype = "thinking" if frag_type == "THINK" else "text"
                                        frag_content = fragments[0].get("content", "")
                                        if frag_content:
                                            if current_ptype == "thinking":
                                                think_list.append(frag_content)
                                            else:
                                                text_list.append(frag_content)
                                    continue

                                # ---- 新格式：fragment 追加（THINK→RESPONSE 切换） ----
                                if "p" in chunk and chunk.get("p") == "response/fragments" and chunk.get("o") == "APPEND":
                                    new_frags = chunk.get("v", [])
                                    if new_frags and isinstance(new_frags, list) and len(new_frags) > 0:
                                        frag_type = new_frags[0].get("type", "")
                                        current_ptype = "thinking" if frag_type == "THINK" else "text"
                                        frag_content = new_frags[0].get("content", "")
                                        if frag_content:
                                            if current_ptype == "thinking":
                                                think_list.append(frag_content)
                                            else:
                                                text_list.append(frag_content)
                                    continue

                                # ---- 兼容旧格式：thinking_content / content 切换 ----
                                if "p" in chunk and chunk.get("p") == "response/thinking_content":
                                    current_ptype = "thinking"
                                elif "p" in chunk and chunk.get("p") == "response/content":
                                    current_ptype = "text"

                                # ---- status / search_status ----
                                if "p" in chunk and chunk.get("p") == "response/status":
                                    if chunk.get("v") == "FINISHED":
                                        data_queue.put(None)
                                        break
                                    continue
                                if "p" in chunk and chunk.get("p") == "response/search_status":
                                    continue

                                # ---- 处理 v 字段 ----
                                if "v" in chunk:
                                    v_value = chunk["v"]

                                    if isinstance(v_value, str):
                                        if search_enabled and v_value.startswith("[citation:"):
                                            continue
                                        if current_ptype == "thinking":
                                            think_list.append(v_value)
                                        else:
                                            text_list.append(v_value)

                                    elif isinstance(v_value, list):
                                        for item in v_value:
                                            if item.get("p") == "status" and item.get("v") == "FINISHED":
                                                # 构建最终结果
                                                final_reasoning = "".join(think_list)
                                                final_content = "".join(text_list)

                                                # 检测 tool_calls
                                                tool_calls_detected, final_content = chat.detect_and_parse_tool_calls(final_content)

                                                prompt_tokens = len(final_prompt) // 4
                                                reasoning_tokens = len(final_reasoning) // 4
                                                completion_tokens = len(final_content) // 4

                                                # 构建 message 对象
                                                message_obj = {
                                                    "role": "assistant",
                                                    "content": final_content,
                                                    "reasoning_content": final_reasoning,
                                                }

                                                # 如果检测到 tool_calls，添加到 message
                                                finish_reason = "stop"
                                                if tool_calls_detected:
                                                    message_obj["tool_calls"] = tool_calls_detected
                                                    finish_reason = "tool_calls"

                                                result = {
                                                    "id": completion_id,
                                                    "object": "chat.completion",
                                                    "created": created_time,
                                                    "model": model,
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "message": message_obj,
                                                            "finish_reason": finish_reason,
                                                        }
                                                    ],
                                                    "usage": {
                                                        "prompt_tokens": prompt_tokens,
                                                        "completion_tokens": reasoning_tokens + completion_tokens,
                                                        "total_tokens": prompt_tokens + reasoning_tokens + completion_tokens,
                                                        "completion_tokens_details": {
                                                            "reasoning_tokens": reasoning_tokens
                                                        },
                                                    },
                                                }
                                                data_queue.put("DONE")
                                                return

                            except Exception as e:
                                logger.warning(f"[collect_data] 无法解析: {data_str}, 错误: {e}")
                                if current_ptype == "thinking":
                                    think_list.append("解析失败，请稍候再试")
                                else:
                                    text_list.append("解析失败，请稍候再试")
                                data_queue.put(None)
                                break
                except Exception as e:
                    logger.warning(f"[collect_data] 错误: {e}")
                    if current_ptype == "thinking":
                        think_list.append("处理失败，请稍候再试")
                    else:
                        text_list.append("处理失败，请稍候再试")
                    data_queue.put(None)
                finally:
                    deepseek_resp.close()
                    if result is None:
                        # 如果没有提前构造 result，则构造默认结果
                        final_content = "".join(text_list)
                        final_reasoning = "".join(think_list)

                        # 检测 tool_calls
                        tool_calls_detected, final_content = chat.detect_and_parse_tool_calls(final_content)

                        prompt_tokens = len(final_prompt) // 4
                        reasoning_tokens = len(final_reasoning) // 4
                        completion_tokens = len(final_content) // 4

                        # 构建 message 对象
                        message_obj = {
                            "role": "assistant",
                            "content": final_content,
                            "reasoning_content": final_reasoning,
                        }

                        # 如果检测到 tool_calls，添加到 message
                        finish_reason = "stop"
                        if tool_calls_detected:
                            message_obj["tool_calls"] = tool_calls_detected
                            finish_reason = "tool_calls"

                        result = {
                            "id": completion_id,
                            "object": "chat.completion",
                            "created": created_time,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": message_obj,
                                    "finish_reason": finish_reason,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": reasoning_tokens + completion_tokens,
                                "total_tokens": prompt_tokens + reasoning_tokens + completion_tokens,
                            },
                        }
                    data_queue.put("DONE")

            collect_thread = threading.Thread(target=collect_data)
            collect_thread.start()

            def generate():
                last_send_time = time.time()
                try:
                    while True:
                        current_time = time.time()
                        if current_time - last_send_time >= constants.KEEP_ALIVE_TIMEOUT:
                            yield ""
                            last_send_time = current_time
                        if not collect_thread.is_alive() and result is not None:
                            yield json.dumps(result)
                            break
                        time.sleep(0.1)
                finally:
                    _delete_deepseek_session(request, session_id)

            return StreamingResponse(generate(), media_type="application/json")
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception as exc:
        logger.error(f"[chat_completions] 未知异常: {exc}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
    finally:
        if getattr(request.state, "use_config_token", False) and hasattr(
            request.state, "account"
        ):
            release_account(request.state.account)


# ----------------------------------------------------------------------
# 路由：/v1/messages (Anthropic Messages API)
# ----------------------------------------------------------------------
@router.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Anthropic Messages API 兼容端点。
    接收 Anthropic 格式的请求，内部转换为 DeepSeek 调用，返回 Anthropic 格式响应。
    支持流式和非流式两种模式。
    """
    try:
        # ---------- Auth (支持 Authorization: Bearer 和 x-api-key) ----------
        try:
            determine_mode_and_token(request, allow_x_api_key=True)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code, content={"error": exc.detail}
            )
        except Exception as exc:
            logger.error(f"[anthropic_messages] determine_mode_and_token 异常: {exc}")
            return JSONResponse(
                status_code=500, content={"error": "Account login failed."}
            )

        req_data = await request.json()
        model = req_data.get("model", "deepseek-v4-flash")

        # ---------- 模型类型（支持 -search 后缀） ----------
        model_lower = model.lower().strip()
        if model_lower.endswith("-search"):
            model_lower = model_lower[:-7]
            search_by_model = True
        else:
            search_by_model = False

        if model_lower in [
            "deepseek-v4-flash", "deepseek-chat",
            "claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5",
        ]:
            model_type = "default"
            auto_thinking = True
        elif model_lower in ["deepseek-reasoner"]:
            model_type = "default"
            auto_thinking = True
        elif model_lower in ["deepseek-v4-pro"]:
            model_type = "expert"
            auto_thinking = True
        else:
            raise HTTPException(
                status_code=503, detail=f"Model '{model}' is not available."
            )

        # ---------- Thinking (Anthropic 格式) ----------
        thinking_obj = req_data.get("thinking")
        if isinstance(thinking_obj, dict) and thinking_obj.get("type") == "enabled":
            thinking_enabled = True
        elif isinstance(thinking_obj, bool):
            thinking_enabled = thinking_obj
        else:
            thinking_enabled = auto_thinking

        search_enabled = bool(req_data.get("search_enabled", False)) or search_by_model

        # ---------- System + Messages ----------
        system_text = req_data.get("system", "")
        messages = req_data.get("messages", [])
        if not messages:
            raise HTTPException(
                status_code=400, detail="Request must include 'messages'."
            )

        # 将 Anthropic 消息转为内部格式（system 字段作为首条 system 消息）
        internal_messages: list[dict] = []
        if system_text:
            internal_messages.append({"role": "system", "content": system_text})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = "\n".join(texts)
            internal_messages.append({"role": role, "content": str(content)})

        # ---------- Tools (Anthropic 格式 → system prompt) ----------
        tools = req_data.get("tools", [])
        has_tools = len(tools) > 0

        if has_tools:
            tool_schemas = []
            for tool in tools:
                tool_name = tool.get("name", "unknown")
                tool_desc = tool.get("description", "No description available")
                params = tool.get("input_schema", {})

                tool_info = f"Tool: {tool_name}\nDescription: {tool_desc}"
                if "properties" in params:
                    props = []
                    required = params.get("required", [])
                    for prop_name, prop_info in params["properties"].items():
                        prop_type = prop_info.get("type", "string")
                        prop_desc = prop_info.get("description", "")
                        is_req = " (required)" if prop_name in required else ""
                        props.append(f"  - {prop_name}: {prop_type}{is_req} - {prop_desc}")
                    if props:
                        tool_info += f"\nParameters:\n{chr(10).join(props)}"
                tool_schemas.append(tool_info)

            tool_system_prompt = f"""You have access to the following tools:

{chr(10).join(tool_schemas)}

When you need to use a tool, respond with a JSON object in this exact format:
{{"tool_calls": [{{"id": "call_xxx", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}

You can call multiple tools in one response by adding more objects to the tool_calls array.
IMPORTANT: The "arguments" field must be a JSON string, not a JSON object.

Example:
{{"tool_calls": [{{"id": "call_001", "type": "function", "function": {{"name": "get_weather", "arguments": "{{\\"location\\": \\"Beijing\\"}}"}}}}]}}

After calling tools, you will receive the results and can continue the conversation."""

            # 合并到 system 消息
            system_found = False
            for msg in internal_messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + "\n\n" + tool_system_prompt
                    system_found = True
                    break
            if not system_found:
                internal_messages.insert(0, {"role": "system", "content": tool_system_prompt})

        # ---------- Prompt ----------
        final_prompt = chat.messages_prepare(internal_messages)

        # ---------- Session / PoW / Completion ----------
        session_id = chat.create_session(request)
        if not session_id:
            raise HTTPException(status_code=401, detail="Invalid token.")

        pow_resp = get_pow_response(request)
        if not pow_resp:
            raise HTTPException(
                status_code=401,
                detail="Failed to get PoW (invalid token or unknown error).",
            )

        headers = {
            **get_auth_headers(request),
            **get_hif_headers(request),
            "x-ds-pow-response": pow_resp,
        }
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "model_type": model_type,
            "prompt": final_prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }

        deepseek_resp = chat.call_completion_endpoint(
            payload, headers, session_module.get_request_session(request), max_attempts=3
        )
        if not deepseek_resp:
            raise HTTPException(status_code=500, detail="Failed to get completion.")

        created_time = int(time.time())
        msg_id = f"msg_{session_id}_{created_time}"
        stream = bool(req_data.get("stream", False))

        if stream and deepseek_resp.status_code != 200:
            deepseek_resp.close()
            return JSONResponse(
                content=deepseek_resp.content, status_code=deepseek_resp.status_code
            )

        # ========== 流式响应 ==========
        if stream:
            def anthropic_sse_stream():
                client_disconnected = False
                stream_completed = False
                try:
                    # 收集文本用于 tool_call 检测（最后判断）
                    all_thinking = ""
                    all_text = ""
                    visible_text = ""
                    tool_call_buffering = False
                    tool_stream_buffer = ""
                    tool_stream_id = "call_001"
                    tool_stream_name = ""
                    tool_stream_arg_sent = 0
                    tool_stream_index = None
                    tool_stream_started = False
                    tool_calls_detected = None

                    # Anthropic SSE 状态
                    sse_state: dict = {
                        "ptype": "text",
                        "next_block_index": 0,
                        "active_block_index": None,
                        "block_active": False,
                        "has_thinking": False,
                    }

                    # 先发 message_start
                    msg_start = make_message_start_event(msg_id, model, {
                        "input_tokens": len(final_prompt) // 4,
                        "output_tokens": 0,
                    })
                    yield f"event: {msg_start['event']}\ndata: {json.dumps(msg_start['data'], ensure_ascii=False)}\n\n"

                    result_queue: queue.Queue = queue.Queue()
                    last_send_time = time.time()

                    def process_anthropic_stream():
                        nonlocal all_thinking, all_text
                        try:
                            for raw_line in deepseek_resp.iter_lines():
                                try:
                                    line = raw_line.decode("utf-8")
                                except Exception:
                                    logger.warning("[anthropic_sse_stream] 解码失败")
                                    result_queue.put(None)
                                    return

                                if not line:
                                    continue

                                if not line.startswith("data:"):
                                    continue

                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    result_queue.put(None)
                                    return

                                # 用 converter 解析 DeepSeek → Anthropic events
                                try:
                                    events = deepseek_line_to_anthropic_events(line, sse_state)
                                except Exception as e:
                                    logger.warning(f"[anthropic_sse_stream] deepseek_line_to_anthropic_events 错误: {e}")
                                    continue

                                for ev in events:
                                    if ev["event"] == "__FINISHED__":
                                        result_queue.put("__FINISHED__")
                                        return
                                    result_queue.put(ev)

                                    # 收集文本
                                    ed = ev["data"]
                                    if ev["event"] == "content_block_delta":
                                        delta = ed.get("delta", {})
                                        if "thinking" in delta:
                                            all_thinking += delta["thinking"]
                                        elif "text" in delta:
                                            all_text += delta["text"]
                                    elif ev["event"] == "content_block_start":
                                        cb = ed.get("content_block", {})
                                        if cb.get("type") == "thinking":
                                            all_thinking += cb.get("thinking", "")
                                        elif cb.get("type") == "text":
                                            all_text += cb.get("text", "")
                        except Exception as e:
                            logger.warning(f"[anthropic_sse_stream] 流异常: {e}")
                            result_queue.put(None)
                        finally:
                            deepseek_resp.close()

                    process_thread = threading.Thread(target=process_anthropic_stream)
                    process_thread.start()

                    def _make_tool_input_delta(index: int, partial_json: str):
                        return {
                            "event": "content_block_delta",
                            "data": {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": partial_json,
                                },
                            },
                        }

                    def _make_tool_stop(index: int):
                        return {
                            "event": "content_block_stop",
                            "data": {"type": "content_block_stop", "index": index},
                        }

                    def _close_active_text_block():
                        if sse_state["block_active"]:
                            block_index = sse_state.get("active_block_index", 0)
                            sse_state["block_active"] = False
                            sse_state["active_block_index"] = None
                            return _make_tool_stop(block_index)
                        return None

                    try:
                        while True:
                            current_time = time.time()
                            if current_time - last_send_time >= constants.KEEP_ALIVE_TIMEOUT:
                                yield ": keep-alive\n\n"
                                last_send_time = current_time
                                continue
                            try:
                                item = result_queue.get(timeout=0.05)
                            except queue.Empty:
                                continue

                            if item is None:
                                break

                            if item == "__FINISHED__":
                                break

                            if item["event"] == "content_block_delta":
                                delta = item["data"].get("delta", {})
                                text_part = delta.get("text")
                                if text_part is not None:
                                    tool_buffer_started_now = False
                                    if tool_call_buffering:
                                        if text_part:
                                            tool_stream_buffer += text_part
                                    else:
                                        visible_text += text_part
                                        stripped_text = chat.strip_partial_tool_call_text(visible_text)
                                        if stripped_text == visible_text:
                                            yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                                            last_send_time = current_time
                                            continue

                                        previous_visible_len = len(visible_text) - len(text_part)
                                        if len(stripped_text) > previous_visible_len:
                                            safe_part = stripped_text[previous_visible_len:]
                                            if safe_part:
                                                safe_item = {
                                                    "event": "content_block_delta",
                                                    "data": {
                                                        **item["data"],
                                                        "delta": {**delta, "text": safe_part},
                                                    },
                                                }
                                                yield f"event: {safe_item['event']}\ndata: {json.dumps(safe_item['data'], ensure_ascii=False)}\n\n"
                                                last_send_time = current_time

                                        tool_call_buffering = True
                                        tool_buffer_started_now = True
                                        tool_stream_buffer = visible_text[len(stripped_text):]
                                        visible_text = stripped_text

                                    if not tool_call_buffering:
                                        continue

                                    if tool_call_buffering:
                                        detected_id, detected_name = _extract_stream_tool_meta(tool_stream_buffer)
                                        if detected_id:
                                            tool_stream_id = detected_id
                                        if detected_name:
                                            tool_stream_name = detected_name
                                        if tool_stream_name and not tool_stream_started:
                                            stop_active_ev = _close_active_text_block()
                                            if stop_active_ev:
                                                yield f"event: {stop_active_ev['event']}\ndata: {json.dumps(stop_active_ev['data'], ensure_ascii=False)}\n\n"
                                            tool_stream_index = sse_state.get("next_block_index", 0)
                                            sse_state["next_block_index"] = tool_stream_index + 1
                                            tool_stream_started = True
                                            tool_start = {
                                                "event": "content_block_start",
                                                "data": {
                                                    "type": "content_block_start",
                                                    "index": tool_stream_index,
                                                    "content_block": {
                                                        "type": "tool_use",
                                                        "id": tool_stream_id,
                                                        "name": tool_stream_name,
                                                        "input": {},
                                                    },
                                                },
                                            }
                                            yield f"event: {tool_start['event']}\ndata: {json.dumps(tool_start['data'], ensure_ascii=False)}\n\n"
                                            last_send_time = current_time
                                        if tool_stream_started:
                                            arg_stream = _extract_stream_arguments(tool_stream_buffer)
                                            if len(arg_stream) > tool_stream_arg_sent:
                                                arg_part = arg_stream[tool_stream_arg_sent:]
                                                tool_stream_arg_sent = len(arg_stream)
                                                if arg_part:
                                                    for arg_chunk in _iter_text_chunks(arg_part):
                                                        delta_ev = _make_tool_input_delta(tool_stream_index, arg_chunk)
                                                        yield f"event: {delta_ev['event']}\ndata: {json.dumps(delta_ev['data'], ensure_ascii=False)}\n\n"
                                                        last_send_time = time.time()
                                                        if TOOL_ARGUMENT_STREAM_DELAY:
                                                            time.sleep(TOOL_ARGUMENT_STREAM_DELAY)
                                        continue

                            elif item["event"] == "content_block_start":
                                cb = item["data"].get("content_block", {})
                                if cb.get("type") == "text" and tool_call_buffering:
                                    continue
                            elif item["event"] == "content_block_stop" and tool_call_buffering:
                                continue

                            # 发送 Anthropic SSE event
                            yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                            last_send_time = current_time

                        # --- 流结束：检测 tool_calls（如果启用了 tools）---
                        parse_source = tool_stream_buffer or all_text
                        tool_calls_detected, remaining_text = chat.detect_and_parse_tool_calls(parse_source)
                        if tool_stream_buffer and not tool_calls_detected and tool_stream_name:
                            args = _extract_stream_arguments(tool_stream_buffer)
                            tool_calls_detected = [{
                                "id": tool_stream_id,
                                "type": "function",
                                "function": {"name": tool_stream_name, "arguments": args or "{}"},
                            }]

                        if tool_stream_started and tool_stream_index is not None:
                            stop_ev = _make_tool_stop(tool_stream_index)
                            yield f"event: {stop_ev['event']}\ndata: {json.dumps(stop_ev['data'], ensure_ascii=False)}\n\n"

                        # 如果未能提前流式识别，则兼容地在结束时拆分补发。
                        if tool_calls_detected and not tool_stream_started:
                            tool_index = sse_state.get("next_block_index", 0)
                            for tc in tool_calls_detected:
                                func = tc.get("function", {})
                                try:
                                    arguments_dict = json.loads(func.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    arguments_dict = {}
                                arguments_text = json.dumps(arguments_dict, ensure_ascii=False)
                                tool_use_block = {
                                    "type": "tool_use",
                                    "id": tc.get("id", f"toolu_{tool_index}"),
                                    "name": func.get("name", ""),
                                    "input": {},
                                }
                                tool_start = {
                                    "event": "content_block_start",
                                    "data": {
                                        "type": "content_block_start",
                                        "index": tool_index,
                                        "content_block": tool_use_block,
                                    },
                                }
                                yield f"event: {tool_start['event']}\ndata: {json.dumps(tool_start['data'], ensure_ascii=False)}\n\n"
                                for arg_part in _iter_text_chunks(arguments_text):
                                    delta_ev = _make_tool_input_delta(tool_index, arg_part)
                                    yield f"event: {delta_ev['event']}\ndata: {json.dumps(delta_ev['data'], ensure_ascii=False)}\n\n"
                                    if TOOL_ARGUMENT_STREAM_DELAY:
                                        time.sleep(TOOL_ARGUMENT_STREAM_DELAY)
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index}, ensure_ascii=False)}\n\n"
                                tool_index += 1
                            sse_state["next_block_index"] = tool_index

                        # 关闭残留的 text block
                        stop_active_ev = _close_active_text_block()
                        if stop_active_ev:
                            yield f"event: {stop_active_ev['event']}\ndata: {json.dumps(stop_active_ev['data'], ensure_ascii=False)}\n\n"

                        # stop_reason
                        anth_stop_reason = "end_turn"
                        if tool_calls_detected:
                            anth_stop_reason = "tool_use"

                        # message_delta
                        total_output = len(all_thinking) // 4 + len(all_text) // 4
                        delta_ev = make_message_delta_event(
                            stop_reason=anth_stop_reason,
                            output_tokens=total_output,
                        )
                        yield f"event: {delta_ev['event']}\ndata: {json.dumps(delta_ev['data'], ensure_ascii=False)}\n\n"

                        # message_stop
                        stop_ev = make_message_stop_event()
                        yield f"event: {stop_ev['event']}\ndata: {json.dumps(stop_ev['data'], ensure_ascii=False)}\n\n"

                        stream_completed = True

                    except GeneratorExit:
                        logger.info(f"[anthropic_sse_stream] 客户端断开连接 session={session_id}")
                        client_disconnected = True
                        raise
                    except Exception as e:
                        logger.error(f"[anthropic_sse_stream] 异常: {e}")
                        client_disconnected = True
                finally:
                    _delete_deepseek_session(request, session_id)
                    if getattr(request.state, "use_config_token", False) and hasattr(
                        request.state, "account"
                    ):
                        release_account(request.state.account)

            return StreamingResponse(
                anthropic_sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # ========== 非流式响应 ==========
        else:
            think_list: list[str] = []
            text_list: list[str] = []

            def collect_anthropic_data():
                sse_state: dict = {
                    "ptype": "text",
                    "next_block_index": 0,
                    "active_block_index": None,
                    "block_active": False,
                    "has_thinking": False,
                }
                try:
                    for raw_line in deepseek_resp.iter_lines():
                        try:
                            line = raw_line.decode("utf-8")
                        except Exception:
                            logger.warning("[collect_anthropic_data] 解码失败")
                            return
                        if not line:
                            continue

                        events = deepseek_line_to_anthropic_events(line, sse_state)
                        for ev in events:
                            if ev["event"] == "__FINISHED__":
                                return
                            # 收集文本
                            ed = ev["data"]
                            if ev["event"] == "content_block_delta":
                                delta = ed.get("delta", {})
                                if "thinking" in delta:
                                    think_list.append(delta["thinking"])
                                elif "text" in delta:
                                    text_list.append(delta["text"])
                            elif ev["event"] == "content_block_start":
                                cb = ed.get("content_block", {})
                                if cb.get("type") == "thinking":
                                    think_list.append(cb.get("thinking", ""))
                                elif cb.get("type") == "text":
                                    text_list.append(cb.get("text", ""))

                except Exception as e:
                    logger.warning(f"[collect_anthropic_data] 错误: {e}")
                finally:
                    deepseek_resp.close()

            collect_thread = threading.Thread(target=collect_anthropic_data)
            collect_thread.start()

            # 等待收集完成
            collect_thread.join(timeout=120)

            # 清理会话
            _delete_deepseek_session(request, session_id)

            # 构建 content blocks
            content_blocks: list[dict] = []
            final_thinking = "".join(think_list)
            final_text = "".join(text_list)

            # 检测 tool_calls
            tool_calls_detected, final_text = chat.detect_and_parse_tool_calls(final_text)

            stop_reason = "end_turn"

            if final_thinking:
                content_blocks.append({"type": "thinking", "thinking": final_thinking})
            if tool_calls_detected:
                # 如果还有文本，先发 text block
                if final_text:
                    content_blocks.append({"type": "text", "text": final_text})
                # 再发 tool_use blocks
                for tc in tool_calls_detected:
                    func = tc.get("function", {})
                    try:
                        arguments_dict = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        arguments_dict = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{int(time.time())}"),
                        "name": func.get("name", ""),
                        "input": arguments_dict,
                    })
                stop_reason = "tool_use"
            else:
                content_blocks.append({"type": "text", "text": final_text or ""})

            prompt_tokens = len(final_prompt) // 4
            output_tokens = len(final_thinking) // 4 + len(final_text) // 4

            response = build_anthropic_response(
                msg_id=msg_id,
                model=model,
                content_blocks=content_blocks,
                stop_reason=stop_reason,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            return JSONResponse(content=response, status_code=200)

    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception as exc:
        logger.error(f"[anthropic_messages] 未知异常: {exc}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
    finally:
        if getattr(request.state, "use_config_token", False) and hasattr(
            request.state, "account"
        ):
            release_account(request.state.account)


# ----------------------------------------------------------------------
# 路由：停止流式响应
# ----------------------------------------------------------------------
@router.post("/v1/chat/stop_stream")
async def stop_stream(request: Request):
    """
    停止正在进行的流式对话
    请求体示例:
    {
        "chat_session_id": "85437c2a-acf8-436a-a2ba-a4a110907fe7",
        "message_id": 2
    }
    """
    try:
        # 认证
        determine_mode_and_token(request)

        # 解析请求体
        body = await request.json()
        chat_session_id = body.get("chat_session_id")
        message_id = body.get("message_id")

        if not chat_session_id:
            raise HTTPException(status_code=400, detail="缺少 chat_session_id 参数")

        # 构造请求头
        headers = get_auth_headers(request)
        ds_session = session_module.get_request_session(request)

        # 构造请求体
        payload = {
            "chat_session_id": chat_session_id,
            "message_id": message_id,
        }

        # 发送停止请求
        resp = ds_session.post(
            constants.DEEPSEEK_STOP_STREAM_URL,
            headers=headers,
            json=payload,
            impersonate="safari15_3",
        )

        if resp.status_code == 200:
            logger.info(f"[stop_stream] 成功停止会话 {chat_session_id}")
            return JSONResponse(content={"success": True, "message": "已停止流式响应"})
        else:
            logger.warning(f"[stop_stream] 停止失败，状态码: {resp.status_code}")
            return JSONResponse(
                status_code=resp.status_code,
                content={"success": False, "message": f"停止失败: {resp.text}"},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[stop_stream] 异常: {e}")
        raise HTTPException(status_code=500, detail=f"停止流式响应失败: {str(e)}")


# ----------------------------------------------------------------------
# 路由：/
# ----------------------------------------------------------------------
@router.get("/")
def index():
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>服务已启动</title></head><body><p>DeepSeek2API Neo 已启动！</p></body></html>"
    )
