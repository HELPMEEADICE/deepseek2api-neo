"""
OpenAI Chat Completions ↔ Anthropic Messages API 格式转换模块。

提供请求/响应格式的双向转换，以及 DeepSeek SSE 中间解析函数，
使 /v1/chat/completions 和 /v1/messages 可以共享同一套下游逻辑，
而 Anthropic Messages API 作为内部"标准格式"。

用法：
    from .converter import openai_to_anthropic, anthropic_to_openai

    # 将 OpenAI 请求转为 Anthropic 内部格式
    anth_body = openai_to_anthropic(openai_body)

    # 将内部 Anthropic 响应转回 OpenAI 格式
    openai_resp = anthropic_to_openai(anth_body, model="deepseek-v4-flash")
"""

import json
import logging
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 请求格式转换：OpenAI Chat Completions → Anthropic Messages API
# ---------------------------------------------------------------------------
def openai_to_anthropic(body: dict) -> dict:
    """将 OpenAI Chat Completions 请求体转换为 Anthropic Messages API 格式。

    - system 消息 → 顶层 ``system`` 字段
    - thinking_enabled / thinking.type → ``thinking`` 对象
    - OpenAI tools → Anthropic tools / tool_choice
    - 确保 messages 以 user 开头（Anthropic 要求）
    """
    result: dict = {}

    # -- model --
    result["model"] = body.get("model", "deepseek-v4-flash")

    # -- system（从 system role 消息提取到顶层） --
    messages = body.get("messages", [])
    system_parts: list[str] = []
    non_system: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = "\n".join(texts)
        if role == "system":
            if content:
                system_parts.append(str(content))
        else:
            non_system.append({**msg, "content": str(content)})
    if system_parts:
        result["system"] = "\n".join(system_parts)

    # -- messages（排除 system，转换 role，确保以 user 开头） --
    anth_messages: list[dict] = []
    for msg in non_system:
        role = msg.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"  # Anthropic 只认 user / assistant
        anth_messages.append({"role": role, "content": msg.get("content", "")})

    if anth_messages and anth_messages[0]["role"] != "user":
        # Anthropic 要求第一条消息必须是 user
        anth_messages.insert(0, {"role": "user", "content": "."})
    result["messages"] = anth_messages

    # -- 标量参数 --
    for key in ("max_tokens", "temperature", "top_p", "stream"):
        if key in body:
            result[key] = body[key]

    stop = body.get("stop")
    if stop:
        result["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    # -- thinking --
    thinking_obj = body.get("thinking")
    if isinstance(thinking_obj, dict) and thinking_obj.get("type") == "enabled":
        thinking: dict = {"type": "enabled"}
        if "budget_tokens" in thinking_obj:
            thinking["budget_tokens"] = thinking_obj["budget_tokens"]
        result["thinking"] = thinking
    elif body.get("thinking_enabled", False):
        result["thinking"] = {"type": "enabled"}

    # -- tools --
    tools = body.get("tools", [])
    if tools:
        anth_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anth_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        if anth_tools:
            result["tools"] = anth_tools

    return result


# ---------------------------------------------------------------------------
# 请求格式转换：Anthropic Messages API → OpenAI Chat Completions
# ---------------------------------------------------------------------------
def anthropic_to_openai_request(anth_body: dict) -> dict:
    """将 Anthropic Messages API 请求体转换为 OpenAI Chat Completions 格式。"""
    result: dict = {}

    result["model"] = anth_body.get("model", "deepseek-v4-flash")

    # system 顶层字段 → system 消息
    messages: list[dict] = []
    system_text = anth_body.get("system", "")
    if system_text:
        messages.append({"role": "system", "content": system_text})

    # messages
    for msg in anth_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = "\n".join(texts)
        messages.append({"role": role, "content": str(content)})

    result["messages"] = messages

    for key in ("max_tokens", "temperature", "top_p", "stream"):
        if key in anth_body:
            result[key] = anth_body[key]

    # thinking → thinking_enabled
    thinking = anth_body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        result["thinking_enabled"] = True

    # Anthropic tools → OpenAI tools
    tools = anth_body.get("tools", [])
    if tools:
        oai_tools = []
        for tool in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        if oai_tools:
            result["tools"] = oai_tools

    return result


# ---------------------------------------------------------------------------
# 响应格式转换：Anthropic Messages API → OpenAI Chat Completions
# ---------------------------------------------------------------------------
def anthropic_to_openai(anth_body: dict, model: str | None = None) -> dict:
    """将 Anthropic Messages API **响应体** 转换为 OpenAI Chat Completions 响应体。

    content blocks 映射：
      type=text       → assistant content
      type=thinking   → reasoning_content
      type=tool_use   → tool_calls
    """
    if model is None:
        model = anth_body.get("model", "deepseek-v4-flash")

    content_blocks = anth_body.get("content", [])
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", "call_001"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })

    content = "".join(text_parts) or None
    reasoning = "".join(reasoning_parts)

    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    # finish_reason 映射
    stop_reason = anth_body.get("stop_reason")
    finish_reason = "stop"
    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "end_turn":
        finish_reason = "stop"

    usage = anth_body.get("usage", {})
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)

    response: dict = {
        "id": anth_body.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        },
    }
    if reasoning:
        response["usage"]["completion_tokens_details"] = {
            "reasoning_tokens": len(reasoning) // 4,
        }

    return response


# ---------------------------------------------------------------------------
# DeepSeek SSE → Anthropic SSE 事件构建器
# ---------------------------------------------------------------------------
def deepseek_line_to_anthropic_events(line: str, state: dict) -> list[dict]:
    """解析 DeepSeek SSE 行，生成 0~N 个 Anthropic SSE event dict。

    每个 event dict 格式：{"event": "<event_name>", "data": <json-serializable>}

    ``state`` 是一个可变字典，调用方初始化为::
        state = {"ptype": None, "next_block_index": 0, "active_block_index": None,
                 "block_active": False, "has_thinking": False}
    """
    events: list[dict] = []

    if not line or not line.startswith("data:"):
        return events

    data_str = line[5:].strip()
    if data_str == "[DONE]":
        return events

    try:
        chunk = json.loads(data_str)
    except json.JSONDecodeError:
        return events

    # ---- 辅助：启动一个 content block ----
    def _start_block(btype: str) -> dict:
        idx = state.get("next_block_index", 0)
        state["next_block_index"] = idx + 1
        state["active_block_index"] = idx
        state["block_active"] = True
        if btype == "thinking":
            state["has_thinking"] = True
        block: dict = {"type": btype}
        if btype == "thinking":
            block["thinking"] = ""
        elif btype == "text":
            block["text"] = ""
        return {
            "event": "content_block_start",
            "data": {"type": "content_block_start", "index": idx, "content_block": block},
        }

    def _delta(btype: str, text: str) -> dict:
        delta_type = "thinking_delta" if btype == "thinking" else "text_delta"
        return {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": state.get("active_block_index", 0),
                "delta": {"type": delta_type, btype: text},
            },
        }

    def _stop_block(_: str) -> dict:
        idx = state.get("active_block_index", 0)
        state["block_active"] = False
        state["active_block_index"] = None
        return {
            "event": "content_block_stop",
            "data": {"type": "content_block_stop", "index": idx},
        }

    # ---- 判断内容类型 ----
    ptype = state.get("ptype", "text")

    # 新格式：fragments 中的 type 决定内容类型
    if "v" in chunk and isinstance(chunk["v"], dict) and "response" in chunk["v"]:
        fragments = chunk["v"]["response"].get("fragments", [])
        if fragments and isinstance(fragments, list) and len(fragments) > 0:
            frag_type = fragments[0].get("type", "")
            new_ptype = "thinking" if frag_type == "THINK" else "text"
            frag_content = fragments[0].get("content", "")

            if new_ptype != ptype and state["block_active"]:
                events.append(_stop_block(ptype))
            if new_ptype != ptype or not state["block_active"]:
                events.append(_start_block(new_ptype))
            state["ptype"] = new_ptype

            if frag_content:
                events.append(_delta(new_ptype, frag_content))
            return events

    if "p" in chunk and chunk.get("p") == "response/fragments" and chunk.get("o") == "APPEND":
        new_frags = chunk.get("v", [])
        if new_frags and isinstance(new_frags, list) and len(new_frags) > 0:
            frag_type = new_frags[0].get("type", "")
            new_ptype = "thinking" if frag_type == "THINK" else "text"
            frag_content = new_frags[0].get("content", "")

            if new_ptype != ptype and state["block_active"]:
                events.append(_stop_block(ptype))
            if new_ptype != ptype or not state["block_active"]:
                events.append(_start_block(new_ptype))
            state["ptype"] = new_ptype

            if frag_content:
                events.append(_delta(new_ptype, frag_content))
        return events

    # ---- 兼容旧格式：路径标记（可能同时包含 v 内容） ----
    # 先处理路径标记以决定内容类型
    if "p" in chunk:
        p_value = chunk["p"]

        if p_value == "response/thinking_content":
            new_ptype = "thinking"
            if new_ptype != ptype and state["block_active"]:
                events.append(_stop_block(ptype))
            if new_ptype != ptype or not state["block_active"]:
                events.append(_start_block(new_ptype))
            state["ptype"] = new_ptype
            # 注意：不 return，继续处理可能存在的 v 字段

        elif p_value == "response/content":
            new_ptype = "text"
            if new_ptype != ptype and state["block_active"]:
                events.append(_stop_block(ptype))
            if new_ptype != ptype or not state["block_active"]:
                events.append(_start_block(new_ptype))
            state["ptype"] = new_ptype
            # 注意：不 return，继续处理可能存在的 v 字段

        elif p_value == "response/status":
            if chunk.get("v") == "FINISHED":
                if state["block_active"]:
                    events.append(_stop_block(state["ptype"]))
                events.append({"event": "__FINISHED__", "data": {}})
            return events

        elif p_value == "response/search_status":
            return events

    # 刷新 ptype（可能在路径标记中被修改）
    ptype = state.get("ptype", "text")

    # ---- v 字段中的文本内容 ----
    if "v" in chunk:
        v_value = chunk["v"]
        if isinstance(v_value, str):
            content = v_value
            if not state["block_active"]:
                events.append(_start_block(ptype))
            events.append(_delta(ptype, content))
        elif isinstance(v_value, list):
            for item in v_value:
                if item.get("p") == "status" and item.get("v") == "FINISHED":
                    if state["block_active"]:
                        events.append(_stop_block(state["ptype"]))
                    events.append({"event": "__FINISHED__", "data": {}})

    return events


# ---------------------------------------------------------------------------
# 构建 Anthropic message_start event
# ---------------------------------------------------------------------------
def make_message_start_event(msg_id: str, model: str, usage: dict | None = None) -> dict:
    """生成 ``message_start`` SSE 事件。"""
    if usage is None:
        usage = {"input_tokens": 0, "output_tokens": 0}
    return {
        "event": "message_start",
        "data": {
            "type": "message_start",
            "message": {
                "id": msg_id or f"msg_{int(time.time())}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    }


# ---------------------------------------------------------------------------
# 构建 Anthropic message_delta + message_stop events
# ---------------------------------------------------------------------------
def make_message_delta_event(stop_reason: str = "end_turn",
                              stop_sequence: str | None = None,
                              output_tokens: int = 0) -> dict:
    """生成 ``message_delta`` SSE 事件。"""
    return {
        "event": "message_delta",
        "data": {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": output_tokens},
        },
    }


def make_message_stop_event() -> dict:
    """生成 ``message_stop`` SSE 事件。"""
    return {"event": "message_stop", "data": {"type": "message_stop"}}


# ---------------------------------------------------------------------------
# 构建非流式 Anthropic 响应体（从已收集的 content blocks）
# ---------------------------------------------------------------------------
def build_anthropic_response(msg_id: str, model: str,
                              content_blocks: list[dict],
                              stop_reason: str = "end_turn",
                              input_tokens: int = 0,
                              output_tokens: int = 0) -> dict:
    """将已收集的 content blocks 组装为 Anthropic Messages API 响应体。"""
    return {
        "id": msg_id or f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
