import json
import logging
import re
import time

from . import constants, session as session_module
from .account import choose_new_account, login_deepseek_via_account
from .constants import get_account_identifier

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 消息预处理函数，将多轮对话合并成最终 prompt
# ----------------------------------------------------------------------
def messages_prepare(messages: list) -> str:
    """处理消息列表，合并连续相同角色的消息，并添加角色标签：
    - 对于 assistant 消息，加上 <｜Assistant｜> 前缀及 结束标签；
    - 对于 user/system 消息（除第一条外）加上 <｜User｜> 前缀；
    - 如果消息 content 为数组，则提取其中 type 为 "text" 的部分；
    - 最后移除 markdown 图片格式的内容。
    """
    processed = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = [
                item.get("text", "") for item in content if item.get("type") == "text"
            ]
            text = "\n".join(texts)
        else:
            text = str(content)
        processed.append({"role": role, "text": text})
    if not processed:
        return ""
    # 合并连续同一角色的消息
    merged = [processed[0]]
    for msg in processed[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["text"] += "\n\n" + msg["text"]
        else:
            merged.append(msg)
    # 添加标签
    parts = []
    for idx, block in enumerate(merged):
        role = block["role"]
        text = block["text"]
        if role == "assistant":
            parts.append(f"<｜Assistant｜>{text}")
        elif role in ("user", "system"):
            if idx > 0:
                parts.append(f"<｜User｜>{text}")
            else:
                parts.append(text)
        else:
            parts.append(text)
    final_prompt = "".join(parts)
    return final_prompt


# ----------------------------------------------------------------------
# 工具调用检测
# ----------------------------------------------------------------------
def detect_and_parse_tool_calls(content: str):
    """
    检测并解析模型返回的 tool_calls JSON
    返回: (tool_calls_list, remaining_content)
    """
    # 尝试匹配 JSON 格式的 tool_calls
    # 支持多种格式：{"tool_calls": [...]} 或直接 [...] 数组
    tool_calls = None
    remaining_content = content

    # 模式1: {"tool_calls": [...]}
    pattern1 = r'\{[\s\n]*"tool_calls"[\s\n]*:[\s\n]*\[(.*?)\][\s\n]*\}'
    match1 = re.search(pattern1, content, re.DOTALL)

    if match1:
        try:
            # 提取完整的 JSON 对象
            json_str = match1.group(0)
            parsed = json.loads(json_str)
            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                tool_calls = parsed["tool_calls"]
                # 移除 tool_calls JSON，保留其他内容
                remaining_content = content[:match1.start()] + content[match1.end():]
                remaining_content = remaining_content.strip()
        except json.JSONDecodeError:
            pass

    # 如果找到了 tool_calls，验证格式并返回
    if tool_calls:
        # 确保每个 tool_call 有必需的字段
        valid_calls = []
        for i, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue

            # 生成 call_id（如果没有）
            call_id = call.get("id", f"call_{i+1:03d}")
            call_type = call.get("type", "function")

            if "function" in call:
                func = call["function"]
                if isinstance(func, dict) and "name" in func:
                    # 确保 arguments 是字符串
                    args = func.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)

                    valid_calls.append({
                        "id": call_id,
                        "type": call_type,
                        "function": {
                            "name": func["name"],
                            "arguments": args
                        }
                    })

        if valid_calls:
            return valid_calls, remaining_content

    return None, content


# ----------------------------------------------------------------------
# 封装对话接口调用的重试机制
# ----------------------------------------------------------------------
def call_completion_endpoint(payload, headers, session, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            deepseek_resp = session.post(
                constants.DEEPSEEK_COMPLETION_URL, headers=headers, json=payload, stream=True, impersonate="safari15_3"
            )
        except Exception as e:
            logger.warning(f"[call_completion_endpoint] 请求异常: {e}")
            time.sleep(1)
            attempts += 1
            continue
        if deepseek_resp.status_code == 200:
            return deepseek_resp
        else:
            logger.warning(
                f"[call_completion_endpoint] 调用对话接口失败, 状态码: {deepseek_resp.status_code}"
            )
            deepseek_resp.close()
            time.sleep(1)
            attempts += 1
    return None


# ----------------------------------------------------------------------
# 创建会话（重试时，配置模式下错误会切换账号；用户自带 token 模式下仅重试）
# ----------------------------------------------------------------------
def create_session(request, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = {
            **constants.BASE_HEADERS,
            "authorization": f"Bearer {request.state.deepseek_token}",
        }
        ds_session = session_module.get_request_session(request)
        try:
            resp = ds_session.post(
                constants.DEEPSEEK_CREATE_SESSION_URL, headers=headers, json={}, impersonate="safari15_3"
            )
        except Exception as e:
            logger.error(f"[create_session] 请求异常: {e}")
            attempts += 1
            continue
        try:
            logger.warning(f"[create_session] {resp.text}")
            data = resp.json()
        except Exception as e:
            logger.error(f"[create_session] JSON解析异常: {e}")
            data = {}
        if resp.status_code == 200 and data.get("code") == 0:
            biz_data = data["data"]["biz_data"]
            # 新响应格式：biz_data.chat_session.id；旧格式：biz_data.id
            if "chat_session" in biz_data:
                session_id = biz_data["chat_session"]["id"]
            else:
                session_id = biz_data["id"]

            resp.close()
            return session_id
        else:
            code = data.get("code")
            logger.warning(
                f"[create_session] 创建会话失败, code={code}, msg={data.get('msg')}"
            )
            resp.close()
            if getattr(request.state, "use_config_token", False):
                current_id = get_account_identifier(request.state.account)
                if not hasattr(request.state, "tried_accounts"):
                    request.state.tried_accounts = []
                if current_id not in request.state.tried_accounts:
                    request.state.tried_accounts.append(current_id)
                new_account = choose_new_account(request.state.tried_accounts)
                if new_account is None:
                    break
                try:
                    login_deepseek_via_account(new_account)
                except Exception as e:
                    logger.error(
                        f"[create_session] 账号 {get_account_identifier(new_account)} 登录失败：{e}"
                    )
                    attempts += 1
                    continue
                request.state.account = new_account
                request.state.deepseek_token = new_account.get("token")
            else:
                attempts += 1
                continue
        attempts += 1
    return None
