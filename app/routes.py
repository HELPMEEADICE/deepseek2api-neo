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
from .files import prepare_prompt_with_upload
from .pow import get_pow_response
from .sse_utils import BufferedResponse, OverloadedError, check_hint_events
from .tokens import count_tokens

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
        completion_prompt, ref_file_ids = prepare_prompt_with_upload(request, final_prompt)
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
            "prompt": completion_prompt,
            "ref_file_ids": ref_file_ids,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }

        deepseek_resp = chat.call_completion_endpoint(payload, headers, session_module.get_request_session(request))
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

            # ---------- Hint 事件前置检测 ----------
            try:
                deepseek_resp = check_hint_events(deepseek_resp)
            except OverloadedError:
                deepseek_resp.close()
                _delete_deepseek_session(request, session_id)
                if getattr(request.state, "use_config_token", False) and hasattr(request.state, "account"):
                    release_account(request.state.account)
                raise HTTPException(status_code=429, detail="Server overloaded. Please retry.")

            def sse_stream():
                client_disconnected = False
                stream_completed = False
                try:
                    final_text = ""
                    final_thinking = ""
                    first_chunk_sent = False
                    has_tools = bool(tools_requested)

                    # 工具调用流式检测器
                    detector = chat.ToolCallStreamDetector()

                    # 基于 Event 的线程通信（消除 queue 轮询）
                    result_queue = queue.Queue()
                    data_event = threading.Event()

                    last_send_time = time.time()

                    def process_data():
                        current_ptype = "text"
                        try:
                            for raw_line in deepseek_resp.iter_lines():
                                try:
                                    line = raw_line.decode("utf-8")
                                except Exception:
                                    logger.warning("[sse_stream] 解码失败")
                                    result_queue.put(None)
                                    data_event.set()
                                    break
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        result_queue.put(None)
                                        data_event.set()
                                        break
                                    try:
                                        chunk = json.loads(data_str)

                                        # 新格式：fragments
                                        if "v" in chunk and isinstance(chunk["v"], dict) and "response" in chunk["v"]:
                                            fragments = chunk["v"]["response"].get("fragments", [])
                                            if fragments and isinstance(fragments, list) and len(fragments) > 0:
                                                frag_type = fragments[0].get("type", "")
                                                current_ptype = "thinking" if frag_type == "THINK" else "text"
                                                frag_content = fragments[0].get("content", "")
                                                if frag_content:
                                                    result_queue.put(("content", current_ptype, frag_content))
                                                    data_event.set()
                                            continue

                                        if "p" in chunk and chunk.get("p") == "response/fragments" and chunk.get("o") == "APPEND":
                                            new_frags = chunk.get("v", [])
                                            if new_frags and isinstance(new_frags, list) and len(new_frags) > 0:
                                                frag_type = new_frags[0].get("type", "")
                                                current_ptype = "thinking" if frag_type == "THINK" else "text"
                                                frag_content = new_frags[0].get("content", "")
                                                if frag_content:
                                                    result_queue.put(("content", current_ptype, frag_content))
                                                    data_event.set()
                                            continue

                                        # 旧格式：路径标记
                                        if "p" in chunk and chunk.get("p") == "response/thinking_content":
                                            current_ptype = "thinking"
                                        elif "p" in chunk and chunk.get("p") == "response/content":
                                            current_ptype = "text"

                                        if "p" in chunk and chunk.get("p") == "response/status":
                                            if chunk.get("v") == "FINISHED":
                                                result_queue.put(None)
                                                data_event.set()
                                                break
                                            continue
                                        if "p" in chunk and chunk.get("p") == "response/search_status":
                                            continue

                                        # v 字段
                                        if "v" in chunk:
                                            v_value = chunk["v"]
                                            if isinstance(v_value, str):
                                                result_queue.put(("content", current_ptype, v_value))
                                                data_event.set()
                                            elif isinstance(v_value, list):
                                                for item in v_value:
                                                    if item.get("p") == "status" and item.get("v") == "FINISHED":
                                                        result_queue.put(None)
                                                        data_event.set()
                                                        return
                                    except Exception as e:
                                        logger.warning(f"[sse_stream] 解析失败: {e}")
                                        result_queue.put(None)
                                        data_event.set()
                                        break
                        except Exception as e:
                            logger.warning(f"[sse_stream] 流错误: {e}")
                            result_queue.put(None)
                            data_event.set()
                        finally:
                            deepseek_resp.close()

                    process_thread = threading.Thread(target=process_data)
                    process_thread.start()

                    def _emit_json(o: dict) -> str:
                        return f"data: {json.dumps(o, ensure_ascii=False)}\n\n"

                    def _emit_delta(delta: dict, finish_reason=None):
                        nonlocal first_chunk_sent, last_send_time
                        if not first_chunk_sent:
                            delta["role"] = "assistant"
                            first_chunk_sent = True
                        out = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                        }
                        last_send_time = time.time()
                        return _emit_json(out)

                    # 流式发送工具调用的辅助函数
                    def _stream_tool_call_events(tool_calls: list):
                        nonlocal first_chunk_sent, last_send_time
                        for ti, tc in enumerate(tool_calls):
                            func = tc.get("function", {})
                            name = func.get("name", "")
                            args = func.get("arguments", "{}")
                            tid = tc.get("id", f"call_{ti + 1:03d}")

                            # 第一个 chunk: 发送 id, type, name
                            yield _emit_delta({
                                "tool_calls": [{
                                    "index": ti,
                                    "id": tid,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }]
                            })
                            # 后续 chunks: 流式发送 arguments
                            yield _emit_delta({
                                "tool_calls": [{
                                    "index": ti,
                                    "function": {"arguments": args},
                                }]
                            })

                    try:
                        while True:
                            current_time = time.time()
                            if current_time - last_send_time >= constants.KEEP_ALIVE_TIMEOUT:
                                yield ": keep-alive\n\n"
                                last_send_time = current_time

                            # drain queue: 每次醒来把队列中所有事件处理完
                            if not data_event.wait(timeout=min(constants.KEEP_ALIVE_TIMEOUT, 0.5)):
                                continue
                            data_event.clear()

                            # 批量处理队列中所有项目
                            while True:
                                try:
                                    item = result_queue.get_nowait()
                                except queue.Empty:
                                    break

                                if item is None:
                                    # 流结束
                                    # 1) 刷新 detect 状态的残余文本
                                    flush = detector.force_flush()
                                    if flush and not detector.has_tool_start():
                                        yield _emit_delta({"content": flush})

                                    # 2) 解析工具调用
                                    tool_calls_detected = None
                                    if detector.has_tool_start() and detector.collected:
                                        parsed, _ = chat.detect_and_parse_tool_calls(detector.collected)
                                        if parsed:
                                            tool_calls_detected = parsed
                                        # 如果解析失败，说明是误判，将 collected 内容当作普通文本输出
                                        if not tool_calls_detected:
                                            yield _emit_delta({"content": detector.collected})
                                    else:
                                        parsed, remaining = chat.detect_and_parse_tool_calls(final_text)
                                        if parsed:
                                            tool_calls_detected = parsed
                                            final_text = remaining

                                    if tool_calls_detected:
                                        yield from _stream_tool_call_events(tool_calls_detected)

                                    usage = {
                                        "prompt_tokens": count_tokens(final_prompt),
                                        "completion_tokens": count_tokens(final_thinking) + count_tokens(final_text),
                                        "total_tokens": count_tokens(final_prompt) + count_tokens(final_thinking) + count_tokens(final_text),
                                        "completion_tokens_details": {"reasoning_tokens": count_tokens(final_thinking)},
                                    }
                                    finish_reason = "tool_calls" if tool_calls_detected else "stop"
                                    yield _emit_delta({}, finish_reason=finish_reason)
                                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}], 'usage': usage}, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
                                    stream_completed = True
                                    return  # 退出 while True

                                ctype, ptype, ctext = item

                                if search_enabled and ctext.startswith("[citation:"):
                                    continue

                                if ctype != "content":
                                    continue

                                if ptype == "thinking":
                                    if thinking_enabled:
                                        final_thinking += ctext
                                        yield _emit_delta({"reasoning_content": ctext})
                                elif ptype == "text":
                                    final_text += ctext

                                    if has_tools and detector.state == "detecting":
                                        safe = detector.feed(ctext)
                                        if safe:
                                            yield _emit_delta({"content": safe})
                                    elif detector.state == "collecting":
                                        detector.feed(ctext)
                                    else:
                                        yield _emit_delta({"content": ctext})

                    except GeneratorExit:
                        logger.info(f"[sse_stream] 客户端断开连接 session={session_id}")
                        client_disconnected = True
                        raise
                except Exception as e:
                    logger.error(f"[sse_stream] 异常: {e}")
                    client_disconnected = True
                finally:
                    _delete_deepseek_session(request, session_id)
                    if getattr(request.state, "use_config_token", False) and hasattr(request.state, "account"):
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

                                                prompt_tokens = count_tokens(final_prompt)
                                                reasoning_tokens = count_tokens(final_reasoning)
                                                completion_tokens = count_tokens(final_content)

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

                        prompt_tokens = count_tokens(final_prompt)
                        reasoning_tokens = count_tokens(final_reasoning)
                        completion_tokens = count_tokens(final_content)

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

        completion_prompt, ref_file_ids = prepare_prompt_with_upload(request, final_prompt)

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
            "prompt": completion_prompt,
            "ref_file_ids": ref_file_ids,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }

        deepseek_resp = chat.call_completion_endpoint(
            payload, headers, session_module.get_request_session(request)
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
            # ---------- Hint 事件前置检测 ----------
            try:
                deepseek_resp = check_hint_events(deepseek_resp)
            except OverloadedError:
                deepseek_resp.close()
                _delete_deepseek_session(request, session_id)
                if getattr(request.state, "use_config_token", False) and hasattr(request.state, "account"):
                    release_account(request.state.account)
                raise HTTPException(status_code=429, detail="Server overloaded. Please retry.")

            def anthropic_sse_stream():
                client_disconnected = False
                stream_completed = False
                try:
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

                    # 工具调用检测器
                    detector = chat.ToolCallStreamDetector()
                    tool_calls_detected = None

                    # 先发 message_start
                    msg_start = make_message_start_event(msg_id, model, {
                        "input_tokens": count_tokens(final_prompt),
                        "output_tokens": 0,
                    })
                    yield f"event: {msg_start['event']}\ndata: {json.dumps(msg_start['data'], ensure_ascii=False)}\n\n"

                    result_queue: queue.Queue = queue.Queue()
                    data_event = threading.Event()
                    last_send_time = time.time()

                    def process_anthropic_stream():
                        nonlocal all_thinking, all_text
                        try:
                            for raw_line in deepseek_resp.iter_lines():
                                try:
                                    line = raw_line.decode("utf-8")
                                except Exception:
                                    result_queue.put(None)
                                    data_event.set()
                                    return
                                if not line:
                                    continue
                                if not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    result_queue.put(None)
                                    data_event.set()
                                    return
                                try:
                                    events = deepseek_line_to_anthropic_events(line, sse_state)
                                except Exception:
                                    continue

                                for ev in events:
                                    if ev["event"] == "__FINISHED__":
                                        result_queue.put("__FINISHED__")
                                        data_event.set()
                                        return
                                    result_queue.put(ev)
                                    data_event.set()

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
                            data_event.set()
                        finally:
                            deepseek_resp.close()

                    process_thread = threading.Thread(target=process_anthropic_stream)
                    process_thread.start()

                    def _emit_anthro(ev: dict):
                        nonlocal last_send_time
                        last_send_time = time.time()
                        return f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

                    try:
                        while True:
                            current_time = time.time()
                            if current_time - last_send_time >= constants.KEEP_ALIVE_TIMEOUT:
                                yield ": keep-alive\n\n"
                                last_send_time = current_time

                            if not data_event.wait(timeout=min(constants.KEEP_ALIVE_TIMEOUT, 0.5)):
                                continue
                            data_event.clear()

                            # 批量处理队列中所有项目
                            while True:
                                try:
                                    item = result_queue.get_nowait()
                                except queue.Empty:
                                    break

                                if item is None:
                                    break
                                if item == "__FINISHED__":
                                    break

                                if item["event"] == "content_block_delta":
                                    delta = item["data"].get("delta", {})
                                    text_part = delta.get("text")
                                    if text_part is not None and has_tools and detector.state == "detecting":
                                        safe = detector.feed(text_part)
                                        if safe:
                                            yield _emit_anthro({
                                                "event": "content_block_delta",
                                                "data": {**item["data"], "delta": {"type": "text_delta", "text": safe}},
                                            })
                                        continue
                                    elif detector.state == "collecting":
                                        detector.feed(text_part)
                                        continue

                                elif item["event"] == "content_block_start":
                                    cb = item["data"].get("content_block", {})
                                    if cb.get("type") == "text" and (detector.state == "collecting" or detector.has_tool_start()):
                                        continue
                                elif item["event"] == "content_block_stop" and detector.state == "collecting":
                                    continue

                                yield _emit_anthro(item)

                            if item is None or item == "__FINISHED__":
                                break  # 退出外层 while True

                        # ---- 流结束：检测和处理 tool_calls ----
                        flush = detector.force_flush()
                        if flush and not detector.has_tool_start() and has_tools:
                            if not sse_state["block_active"]:
                                bi = sse_state["next_block_index"]
                                sse_state["next_block_index"] = bi + 1
                                sse_state["active_block_index"] = bi
                                sse_state["block_active"] = True
                                yield _emit_anthro({
                                    "event": "content_block_start",
                                    "data": {"type": "content_block_start", "index": bi, "content_block": {"type": "text", "text": ""}},
                                })
                            yield _emit_anthro({
                                "event": "content_block_delta",
                                "data": {"type": "content_block_delta", "index": sse_state["active_block_index"], "delta": {"type": "text_delta", "text": flush}},
                            })

                        if detector.has_tool_start() and detector.collected:
                            parsed, _ = chat.detect_and_parse_tool_calls(detector.collected)
                            if parsed:
                                tool_calls_detected = parsed
                            else:
                                # 误判：将 collected 当作普通文本输出
                                if not sse_state["block_active"]:
                                    bi = sse_state["next_block_index"]
                                    sse_state["next_block_index"] = bi + 1
                                    sse_state["active_block_index"] = bi
                                    sse_state["block_active"] = True
                                    yield _emit_anthro({
                                        "event": "content_block_start",
                                        "data": {"type": "content_block_start", "index": bi, "content_block": {"type": "text", "text": ""}},
                                    })
                                yield _emit_anthro({
                                    "event": "content_block_delta",
                                    "data": {"type": "content_block_delta", "index": sse_state["active_block_index"], "delta": {"type": "text_delta", "text": detector.collected}},
                                })
                        elif has_tools:
                            parsed, _ = chat.detect_and_parse_tool_calls(all_text)
                            if parsed:
                                tool_calls_detected = parsed

                        # 发送 tool_use blocks
                        if tool_calls_detected:
                            if sse_state["block_active"]:
                                yield _emit_anthro({
                                    "event": "content_block_stop",
                                    "data": {"type": "content_block_stop", "index": sse_state.get("active_block_index", 0)},
                                })
                                sse_state["block_active"] = False
                                sse_state["active_block_index"] = None

                            ti = sse_state.get("next_block_index", 0)
                            for tc in tool_calls_detected:
                                func = tc.get("function", {})
                                try:
                                    args_dict = json.loads(func.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    args_dict = {}
                                args_str = json.dumps(args_dict, ensure_ascii=False)
                                tu_id = tc.get("id", f"toolu_{ti}")
                                yield _emit_anthro({
                                    "event": "content_block_start",
                                    "data": {"type": "content_block_start", "index": ti, "content_block": {"type": "tool_use", "id": tu_id, "name": func.get("name", ""), "input": {}}},
                                })
                                yield _emit_anthro({
                                    "event": "content_block_delta",
                                    "data": {"type": "content_block_delta", "index": ti, "delta": {"type": "input_json_delta", "partial_json": args_str}},
                                })
                                yield _emit_anthro({
                                    "event": "content_block_stop",
                                    "data": {"type": "content_block_stop", "index": ti},
                                })
                                ti += 1
                            sse_state["next_block_index"] = ti

                        # 关闭残留 block
                        if sse_state["block_active"]:
                            yield _emit_anthro({
                                "event": "content_block_stop",
                                "data": {"type": "content_block_stop", "index": sse_state.get("active_block_index", 0)},
                            })
                            sse_state["block_active"] = False

                        # message_delta + message_stop
                        stop_reason_str = "tool_use" if tool_calls_detected else "end_turn"
                        total_out = count_tokens(all_thinking) + count_tokens(all_text)
                        yield _emit_anthro(make_message_delta_event(stop_reason=stop_reason_str, output_tokens=total_out))
                        yield _emit_anthro(make_message_stop_event())

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
                    if getattr(request.state, "use_config_token", False) and hasattr(request.state, "account"):
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

            prompt_tokens = count_tokens(final_prompt)
            output_tokens = count_tokens(final_thinking) + count_tokens(final_text)

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
# Anthropic /v1/models 端点
# ----------------------------------------------------------------------
@router.get("/anthropic/v1/models")
def anthropic_list_models():
    """返回 Anthropic-compatible models list。"""
    model_list = [
        {
            "type": "model",
            "id": "deepseek-v4-flash",
            "display_name": "DeepSeek V4 Flash",
            "created_at": "2024-07-18T00:00:00Z",
        },
        {
            "type": "model",
            "id": "deepseek-chat",
            "display_name": "DeepSeek Chat",
            "created_at": "2024-07-18T00:00:00Z",
        },
        {
            "type": "model",
            "id": "deepseek-v4-pro",
            "display_name": "DeepSeek V4 Pro",
            "created_at": "2024-07-18T00:00:00Z",
        },
        {
            "type": "model",
            "id": "deepseek-reasoner",
            "display_name": "DeepSeek Reasoner",
            "created_at": "2024-07-18T00:00:00Z",
        },
        {
            "type": "model",
            "id": "claude-sonnet-4-6",
            "display_name": "Claude Sonnet 4.6 (via DeepSeek)",
            "created_at": "2024-07-18T00:00:00Z",
        },
        {
            "type": "model",
            "id": "claude-opus-4-6",
            "display_name": "Claude Opus 4.6 (via DeepSeek)",
            "created_at": "2024-07-18T00:00:00Z",
        },
        {
            "type": "model",
            "id": "claude-haiku-4-5",
            "display_name": "Claude Haiku 4.5 (via DeepSeek)",
            "created_at": "2024-07-18T00:00:00Z",
        },
    ]
    return JSONResponse(
        content={
            "type": "models",
            "data": model_list,
            "has_more": False,
            "first_id": model_list[0]["id"] if model_list else None,
            "last_id": model_list[-1]["id"] if model_list else None,
        },
        status_code=200,
    )


# ----------------------------------------------------------------------
# Anthropic /v1/messages/stop_stream 端点
# ----------------------------------------------------------------------
@router.post("/anthropic/v1/messages/stop_stream")
async def anthropic_stop_stream(request: Request):
    """停止正在进行的 Anthropic 流式对话。请求体与 /v1/chat/stop_stream 兼容。"""
    try:
        determine_mode_and_token(request, allow_x_api_key=True)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Account login failed."})

    try:
        body = await request.json()
        chat_session_id = body.get("chat_session_id")
        message_id = body.get("message_id")

        if not chat_session_id:
            raise HTTPException(status_code=400, detail="缺少 chat_session_id 参数")

        headers = get_auth_headers(request)
        ds_session = session_module.get_request_session(request)

        payload = {"chat_session_id": chat_session_id, "message_id": message_id}
        resp = ds_session.post(
            constants.DEEPSEEK_STOP_STREAM_URL,
            headers=headers,
            json=payload,
            impersonate="safari15_3",
        )

        if resp.status_code == 200:
            logger.info(f"[anthropic_stop_stream] 成功停止会话 {chat_session_id}")
            return JSONResponse(content={"success": True, "message": "已停止流式响应"})
        else:
            logger.warning(f"[anthropic_stop_stream] 停止失败，状态码: {resp.status_code}")
            return JSONResponse(
                status_code=resp.status_code,
                content={"success": False, "message": f"停止失败: {resp.text}"},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[anthropic_stop_stream] 异常: {e}")
        raise HTTPException(status_code=500, detail=f"停止流式响应失败: {str(e)}")


# ----------------------------------------------------------------------
# Anthropic /v1/messages/count_tokens 端点
# ----------------------------------------------------------------------
@router.post("/anthropic/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Anthropic-compatible 令牌计数端点（估算值）。"""
    try:
        determine_mode_and_token(request, allow_x_api_key=True)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Account login failed."})

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body")

    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
        else:
            total_chars += len(str(content))

    system_text = body.get("system", "")
    total_chars += len(system_text)

    tools = body.get("tools", [])
    for tool in tools:
        total_chars += len(json.dumps(tool))

    estimated_tokens = count_tokens(json.dumps(body))

    return JSONResponse(
        content={"input_tokens": estimated_tokens},
        status_code=200,
    )


# ----------------------------------------------------------------------
# 路由：/
# ----------------------------------------------------------------------
@router.get("/")
def index():
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>服务已启动</title></head><body><p>DeepSeek2API Neo 已启动！</p></body></html>"
    )
