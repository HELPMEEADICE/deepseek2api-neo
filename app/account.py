import logging
import random

from curl_cffi import requests
from fastapi import HTTPException, Request

from . import config, constants, session as session_module

logger = logging.getLogger(__name__)

# -------------------------- 全局账号队列 --------------------------
account_queue = []  # 维护所有可用账号


def init_account_queue():
    """初始化时从配置加载账号"""
    global account_queue
    account_queue = config.CONFIG.get("accounts", [])[:]  # 深拷贝
    random.shuffle(account_queue)  # 初始随机排序


init_account_queue()


def choose_new_account(exclude_ids=None):
    """选择策略：
    1. 遍历队列，找到第一个未被 exclude_ids 包含的账号
    2. 从队列中移除该账号
    3. 返回该账号（由后续逻辑保证最终会重新入队）
    """
    if exclude_ids is None:
        exclude_ids = []

    for i in range(len(account_queue)):
        acc = account_queue[i]
        acc_id = constants.get_account_identifier(acc)
        if acc_id and acc_id not in exclude_ids:
            logger.info(f"[choose_new_account] 新选择账号: {acc_id}")
            return account_queue.pop(i)

    logger.warning("[choose_new_account] 没有可用的账号或所有账号都在使用中")
    return None


def release_account(account):
    """将账号重新加入队列末尾"""
    account_queue.append(account)


# ----------------------------------------------------------------------
# 登录函数：支持使用 email 或 mobile 登录
# ----------------------------------------------------------------------
def login_deepseek_via_account(account):
    """使用 account 中的 email 或 mobile 登录 DeepSeek，
    成功后将返回的 token 写入 account 并保存至配置文件，返回新 token。
    同时初始化 Session 并获取 HIF 令牌。
    """
    email = account.get("email", "").strip()
    mobile = account.get("mobile", "").strip()
    password = account.get("password", "").strip()
    if not password or (not email and not mobile):
        raise HTTPException(
            status_code=400,
            detail="账号缺少必要的登录信息（必须提供 email 或 mobile 以及 password）",
        )

    # 初始化 Session（先访问主页初始化 Cookie）
    session = session_module.get_account_session(account)

    if email:
        payload = {
            "email": email,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android",
        }
    else:
        payload = {
            "mobile": mobile,
            "area_code": None,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android",
        }
    try:
        resp = session.post(constants.DEEPSEEK_LOGIN_URL, headers=constants.BASE_HEADERS, json=payload, impersonate="safari15_3")
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"[login_deepseek_via_account] 登录请求异常: {e}")
        raise HTTPException(status_code=500, detail="Account login failed: 请求异常")
    try:
        logger.warning(f"[login_deepseek_via_account] {resp.text}")
        data = resp.json()
    except Exception as e:
        logger.error(f"[login_deepseek_via_account] JSON解析失败: {e}")
        raise HTTPException(
            status_code=500, detail="Account login failed: invalid JSON response"
        )
    # 校验响应数据格式是否正确
    if (
        data.get("data") is None
        or data["data"].get("biz_data") is None
        or data["data"]["biz_data"].get("user") is None
    ):
        logger.error(f"[login_deepseek_via_account] 登录响应格式错误: {data}")
        raise HTTPException(
            status_code=500, detail="Account login failed: invalid response format"
        )
    new_token = data["data"]["biz_data"]["user"].get("token")
    if not new_token:
        logger.error(f"[login_deepseek_via_account] 登录响应中缺少 token: {data}")
        raise HTTPException(
            status_code=500, detail="Account login failed: missing token"
        )
    account["token"] = new_token
    config.save_config(config.CONFIG)

    # 登录成功后获取 HIF 令牌
    ensure_hif_tokens(account, force=True)

    return new_token


# ----------------------------------------------------------------------
# HIF 令牌获取
# ----------------------------------------------------------------------
def fetch_hif_token(session, url, token_name):
    """从 HIF 端点获取令牌，返回 token 字符串或 None"""
    try:
        resp = session.get(
            url,
            headers=constants.BASE_HEADERS,
            impersonate="safari15_3",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == 0:
                value = data.get("data", {}).get("biz_data", {}).get("value")
                if value:
                    logger.info(f"[fetch_hif_token] 成功获取 {token_name}")
                    return value
                else:
                    logger.warning(f"[fetch_hif_token] {token_name} 响应中缺少 value: {data}")
            else:
                logger.warning(f"[fetch_hif_token] {token_name} 业务错误 code={data.get('code')}")
        else:
            logger.warning(f"[fetch_hif_token] {token_name} HTTP {resp.status_code}")
        resp.close()
    except Exception as e:
        logger.error(f"[fetch_hif_token] {token_name} 异常: {e}")
    return None


def ensure_hif_tokens(account, force=False):
    """确保账号有 HIF 令牌，缺失（或强制）时从 HIF 端点重新获取"""
    if not force and account.get("hif_dliq") and account.get("hif_leim"):
        return True

    session = session_module.get_account_session(account)

    logger.info(f"[ensure_hif_tokens] 账号 {constants.get_account_identifier(account)} 开始获取 HIF 令牌")
    dliq = fetch_hif_token(session, constants.HIF_DLIQ_URL, "x-hif-dliq")
    leim = fetch_hif_token(session, constants.HIF_LEIM_URL, "x-hif-leim")

    if dliq:
        account["hif_dliq"] = dliq
    if leim:
        account["hif_leim"] = leim
    if dliq or leim:
        config.save_config(config.CONFIG)

    if dliq and leim:
        logger.info(f"[ensure_hif_tokens] 账号 {constants.get_account_identifier(account)} HIF 令牌已就绪")
        return True
    else:
        logger.warning(f"[ensure_hif_tokens] 账号 {constants.get_account_identifier(account)} HIF 令牌不完整")
        return False


# ----------------------------------------------------------------------
# 判断调用模式：配置模式 vs 用户自带 token
# ----------------------------------------------------------------------
def determine_mode_and_token(request: Request):
    """
    根据请求头 Authorization 判断使用哪种模式：
    - 如果 Bearer token 出现在 CONFIG["keys"] 中，则为配置模式，从 CONFIG["accounts"] 中随机选择一个账号（排除已尝试账号），
      检查该账号是否已有 token，否则调用登录接口获取；
    - 否则，直接使用请求中的 Bearer 值作为 DeepSeek token。
    结果存入 request.state.deepseek_token；配置模式下同时存入 request.state.account 与 request.state.tried_accounts。
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Unauthorized: missing Bearer token."
        )
    caller_key = auth_header.replace("Bearer ", "", 1).strip()
    config_keys = config.CONFIG.get("keys", [])
    if caller_key in config_keys:
        request.state.use_config_token = True
        request.state.tried_accounts = []  # 初始化已尝试账号
        selected_account = choose_new_account()
        if not selected_account:
            raise HTTPException(
                status_code=429,
                detail="No accounts configured or all accounts are busy.",
            )
        need_login = not selected_account.get("token", "").strip()
        if need_login:
            try:
                login_deepseek_via_account(selected_account)
            except Exception as e:
                logger.error(
                    f"[determine_mode_and_token] 账号 {constants.get_account_identifier(selected_account)} 登录失败：{e}"
                )
                raise HTTPException(status_code=500, detail="Account login failed.")
        else:
            # 已有 token 但可能缺少 HIF 令牌，确保获取
            ensure_hif_tokens(selected_account)

        request.state.deepseek_token = selected_account.get("token")
        request.state.account = selected_account

    else:
        request.state.use_config_token = False
        request.state.deepseek_token = caller_key


def get_auth_headers(request: Request):
    """返回 DeepSeek 请求所需的公共请求头"""
    return {**constants.BASE_HEADERS, "authorization": f"Bearer {request.state.deepseek_token}"}


def get_hif_headers(request: Request):
    """返回 HIF (x-hif-dliq / x-hif-leim) 认证头"""
    headers = {}
    account = getattr(request.state, "account", None)
    if account:
        if account.get("hif_dliq"):
            headers["x-hif-dliq"] = account["hif_dliq"]
        if account.get("hif_leim"):
            headers["x-hif-leim"] = account["hif_leim"]
    return headers
