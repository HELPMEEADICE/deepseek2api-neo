import logging
import time

from curl_cffi import CurlMime

from . import constants, session as session_module
from .account import get_auth_headers
from .pow import get_pow_response

logger = logging.getLogger(__name__)


def upload_file(request, filename: str, content_type: str, content: bytes) -> str | None:
    pow_resp = get_pow_response(request, target_path="/api/v0/file/upload_file")
    if not pow_resp:
        logger.warning("[upload_file] 获取上传 PoW 失败")
        return None

    headers = {
        **get_auth_headers(request),
        "x-ds-pow-response": pow_resp,
    }
    headers.pop("content-type", None)
    headers.pop("Content-Type", None)

    ds_session = session_module.get_request_session(request)
    multipart = CurlMime()
    multipart.addpart(
        "file",
        filename=filename,
        content_type=content_type,
        data=content,
    )
    try:
        resp = ds_session.post(
            constants.DEEPSEEK_FILE_UPLOAD_URL,
            headers=headers,
            multipart=multipart,
            impersonate="safari15_3",
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"[upload_file] 上传请求异常: {e}")
        return None
    finally:
        try:
            multipart.close()
        except Exception:
            pass

    try:
        data = resp.json()
    except Exception as e:
        logger.warning(f"[upload_file] 上传响应解析失败: {e}")
        resp.close()
        return None
    finally:
        try:
            resp.close()
        except Exception:
            pass

    if resp.status_code != 200 or data.get("code") != 0:
        logger.warning(f"[upload_file] 上传失败 status={resp.status_code}, data={data}")
        return None

    file_id = data.get("data", {}).get("biz_data", {}).get("id")
    if not file_id:
        logger.warning(f"[upload_file] 上传响应缺少 file_id: {data}")
        return None
    return file_id


def fetch_file_status(request, file_id: str) -> dict | None:
    ds_session = session_module.get_request_session(request)
    try:
        resp = ds_session.get(
            constants.DEEPSEEK_FILE_FETCH_URL,
            headers=get_auth_headers(request),
            params={"file_ids": file_id},
            impersonate="safari15_3",
            timeout=30,
        )
    except Exception as e:
        logger.warning(f"[fetch_file_status] 查询请求异常: {e}")
        return None

    try:
        data = resp.json()
    except Exception as e:
        logger.warning(f"[fetch_file_status] 查询响应解析失败: {e}")
        resp.close()
        return None
    finally:
        try:
            resp.close()
        except Exception:
            pass

    if resp.status_code != 200 or data.get("code") != 0:
        logger.warning(f"[fetch_file_status] 查询失败 status={resp.status_code}, data={data}")
        return None

    files = data.get("data", {}).get("biz_data", {}).get("files") or []
    return files[0] if files else None


def upload_and_poll(request, filename: str, content_type: str, content: bytes) -> str | None:
    file_id = upload_file(request, filename, content_type, content)
    if not file_id:
        return None

    for _ in range(constants.PROMPT_UPLOAD_POLL_RETRIES):
        file_info = fetch_file_status(request, file_id)
        if file_info:
            status = file_info.get("status")
            if status == "SUCCESS":
                logger.info(
                    f"[upload_and_poll] 文件处理成功 file_id={file_id}, tokens={file_info.get('token_usage')}"
                )
                return file_id
            if status == "FAILED":
                logger.warning(f"[upload_and_poll] 文件处理失败 file_id={file_id}, info={file_info}")
                return None
        time.sleep(constants.PROMPT_UPLOAD_POLL_INTERVAL)

    logger.warning(f"[upload_and_poll] 文件处理超时 file_id={file_id}")
    return None


def prepare_prompt_with_upload(request, prompt: str):
    if len(prompt.encode("utf-8")) <= constants.PROMPT_UPLOAD_THRESHOLD:
        return prompt, []

    file_id = upload_and_poll(
        request,
        "deepseek2api_prompt.txt",
        "text/plain",
        prompt.encode("utf-8"),
    )
    if not file_id:
        logger.warning("[prepare_prompt_with_upload] 大 prompt 上传失败，回退为原始 prompt")
        return prompt, []

    inline_prompt = (
        "完整对话、系统提示、工具说明和用户输入已作为附件 deepseek2api_prompt.txt 上传。"
        "请严格以附件内容作为本轮完整 prompt 继续回答，不要忽略附件中的任何 system prompt、工具调用指南、历史消息或当前用户请求。"
    )
    return inline_prompt, [file_id]


def split_prompt_for_multi_turn(request, prompt: str):
    """多轮对话优化：将历史部分上传为文件，只保留最后一段 inline。

    拆分策略：在 ChatML 格式的 prompt 中查找最后一个 user/tool 消息，
    将之前的消息上传为文件，只保留最后一条 user/tool 及后续内容作为 inline。
    这避免在 token 限制中消耗历史内容的 tokens。
    """
    if len(prompt.encode("utf-8")) <= constants.PROMPT_UPLOAD_THRESHOLD:
        return prompt, []

    # 查找最后一个 <｜User｜> 标签（用户/工具消息）
    markers = ["<｜User｜>"]
    best_idx = -1
    best_marker = ""
    for marker in markers:
        idx = prompt.rfind(marker)
        if idx > best_idx:
            best_idx = idx
            best_marker = marker

    if best_idx <= 0:
        # 无法安全拆分，回退到全部上传
        return prepare_prompt_with_upload(request, prompt)

    history = prompt[:best_idx].strip()
    inline = prompt[best_idx:].strip()

    if not history or not inline:
        return prompt, []

    # 上传历史部分
    file_id = upload_and_poll(
        request,
        "deepseek2api_history.txt",
        "text/plain",
        history.encode("utf-8"),
    )
    if not file_id:
        logger.warning("[split_prompt_for_multi_turn] 历史上传失败，回退为原始 prompt")
        return prompt, []

    inline_prompt = (
        "之前的对话历史已作为附件 deepseek2api_history.txt 上传。"
        "请基于附件中的完整上下文继续回答。当前消息：\n" + inline
    )
    return inline_prompt, [file_id]
