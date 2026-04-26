import json
import logging
import queue
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
        has_tools = len(tools_requested) > 0

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
                                if has_tools:
                                    tool_calls_detected, final_text_content = chat.detect_and_parse_tool_calls(final_text)

                                # 如果检测到 tool_calls，先发送 tool_calls chunk
                                if tool_calls_detected:
                                    for tool_call in tool_calls_detected:
                                        tool_call_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": [tool_call]
                                                    },
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        yield f"data: {json.dumps(tool_call_chunk, ensure_ascii=False)}\n\n"
                                        last_send_time = current_time

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
                                                tool_calls_detected = None
                                                if has_tools:
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
                        tool_calls_detected = None
                        if has_tools:
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

                            # 发送 Anthropic SSE event
                            yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                            last_send_time = current_time

                        # --- 流结束：检测 tool_calls（如果启用了 tools）---
                        if has_tools:
                            tool_calls_detected, remaining_text = chat.detect_and_parse_tool_calls(all_text)
                        else:
                            tool_calls_detected = None
                            remaining_text = all_text

                        # 如果有 tool_calls，需要先发 tool_use content blocks
                        if tool_calls_detected:
                            tool_index = sse_state.get("next_block_index", 0)
                            for tc in tool_calls_detected:
                                func = tc.get("function", {})
                                try:
                                    arguments_dict = json.loads(func.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    arguments_dict = {}
                                tool_use_block = {
                                    "type": "tool_use",
                                    "id": tc.get("id", f"toolu_{tool_index}"),
                                    "name": func.get("name", ""),
                                    "input": arguments_dict,
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
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index}, ensure_ascii=False)}\n\n"
                                tool_index += 1
                            sse_state["next_block_index"] = tool_index

                        # 关闭残留的 text block
                        if sse_state["block_active"]:
                            block_index = sse_state.get("active_block_index", 0)
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index}, ensure_ascii=False)}\n\n"
                            sse_state["block_active"] = False
                            sse_state["active_block_index"] = None

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
            tool_calls_detected = None
            if has_tools:
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
