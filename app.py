import base64
import ctypes
import json
import logging
import queue
import random
import re
import struct
import threading
import time
import transformers
from curl_cffi import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from wasmtime import Linker, Module, Store

# -------------------------- 初始化 tokenizer --------------------------
chat_tokenizer_dir = "./"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)

# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI()

# 添加 CORS 中间件，允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# ----------------------------------------------------------------------
# (1) 配置文件的读写函数
# ----------------------------------------------------------------------
CONFIG_PATH = "config.json"


def load_config():
    """从 config.json 加载配置，出错则返回空 dict"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[load_config] 无法读取配置文件: {e}")
        return {}


def save_config(cfg):
    """将配置写回 config.json（移除不可序列化的 _session 字段）"""
    try:
        # 深拷贝配置，跳过不可序列化的字段
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if not k.startswith("_")}
            elif isinstance(obj, list):
                return [clean(item) for item in obj]
            return obj
        cfg_clean = clean(cfg)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg_clean, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[save_config] 写入 config.json 失败: {e}")


CONFIG = load_config()

# -------------------------- 全局账号队列 --------------------------
account_queue = []  # 维护所有可用账号


def init_account_queue():
    """初始化时从配置加载账号"""
    global account_queue
    account_queue = CONFIG.get("accounts", [])[:]  # 深拷贝
    random.shuffle(account_queue)  # 初始随机排序


init_account_queue()

# ----------------------------------------------------------------------
# (2) DeepSeek 相关常量
# ----------------------------------------------------------------------
DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_LOGIN_URL = f"https://{DEEPSEEK_HOST}/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create"
DEEPSEEK_CREATE_POW_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/completion"
DEEPSEEK_STOP_STREAM_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/stop_stream"
DEEPSEEK_DELETE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/delete"
HIF_DLIQ_URL = "https://hif-dliq.deepseek.com/query"
HIF_LEIM_URL = "https://hif-leim.deepseek.com/query"
BASE_HEADERS = {
    "accept": "*/*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "pragma": "no-cache",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "x-app-version": "20241129.1",
    "x-client-locale": "zh_CN",
    "x-client-platform": "web",
    "x-client-timezone-offset": "28800",
    "x-client-version": "1.8.0",
}


# WASM 模块文件路径
WASM_PATH = "sha3_wasm_bg.7b9ca65ddd.wasm"


# ----------------------------------------------------------------------
# 辅助函数：获取账号唯一标识（优先 email，否则 mobile）
# ----------------------------------------------------------------------
def get_account_identifier(account):
    """返回账号的唯一标识，优先使用 email，否则使用 mobile"""
    return account.get("email", "").strip() or account.get("mobile", "").strip()


# ----------------------------------------------------------------------
# (3) 登录函数：支持使用 email 或 mobile 登录
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
    session = get_account_session(account)

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
        resp = session.post(DEEPSEEK_LOGIN_URL, headers=BASE_HEADERS, json=payload, impersonate="safari15_3")
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
    save_config(CONFIG)

    # 登录成功后获取 HIF 令牌
    ensure_hif_tokens(account, force=True)

    return new_token


# ----------------------------------------------------------------------
# (4) 从 accounts 中随机选择一个未忙且未尝试过的账号
# ----------------------------------------------------------------------
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
        acc_id = get_account_identifier(acc)
        if acc_id and acc_id not in exclude_ids:
            # 从队列中移除并返回
            logger.info(f"[choose_new_account] 新选择账号: {acc_id}")
            return account_queue.pop(i)

    logger.warning("[choose_new_account] 没有可用的账号或所有账号都在使用中")
    return None


def release_account(account):
    """将账号重新加入队列末尾"""
    account_queue.append(account)


# ----------------------------------------------------------------------
# (5) 判断调用模式：配置模式 vs 用户自带 token
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
    config_keys = CONFIG.get("keys", [])
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
                    f"[determine_mode_and_token] 账号 {get_account_identifier(selected_account)} 登录失败：{e}"
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
    return {**BASE_HEADERS, "authorization": f"Bearer {request.state.deepseek_token}"}


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


# ----------------------------------------------------------------------
# Session 管理：每个账号或 token 模式使用独立的 curl_cffi Session 维护 Cookie
# ----------------------------------------------------------------------
_token_session = None  # token 模式下的全局 session


def init_account_session(account):
    """为账号初始化 curl_cffi Session，先访问主页初始化 Cookie"""
    session = requests.Session()
    try:
        resp = session.get(
            "https://chat.deepseek.com/",
            headers={k: v for k, v in BASE_HEADERS.items() if k != "content-type"},
            impersonate="safari15_3",
            timeout=15,
        )
        logger.info(
            f"[init_account_session] 账号 {get_account_identifier(account)} 主页初始化完成, status={resp.status_code}"
        )
        resp.close()
    except Exception as e:
        logger.warning(f"[init_account_session] 主页初始化失败: {e}")
    account["_session"] = session
    return session


def get_account_session(account):
    """获取账号关联的 Session，不存在则创建"""
    session = account.get("_session")
    if session is None:
        session = init_account_session(account)
    return session


def get_request_session(request: Request):
    """从 request.state 获取 Session（配置模式取 account 的，token 模式用全局）"""
    if getattr(request.state, "use_config_token", False):
        account = getattr(request.state, "account", None)
        if account:
            return get_account_session(account)
    # token 模式 — 全局单例
    global _token_session
    if _token_session is None:
        _token_session = requests.Session()
        try:
            resp = _token_session.get(
                "https://chat.deepseek.com/",
                headers={k: v for k, v in BASE_HEADERS.items() if k != "content-type"},
                impersonate="safari15_3",
                timeout=15,
            )
            resp.close()
        except Exception as e:
            logger.warning(f"[get_request_session] 全局 session 主页初始化失败: {e}")
    return _token_session


def fetch_hif_token(session, url, token_name):
    """从 HIF 端点获取令牌，返回 token 字符串或 None"""
    try:
        resp = session.get(
            url,
            headers=BASE_HEADERS,
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

    session = get_account_session(account)

    logger.info(f"[ensure_hif_tokens] 账号 {get_account_identifier(account)} 开始获取 HIF 令牌")
    dliq = fetch_hif_token(session, HIF_DLIQ_URL, "x-hif-dliq")
    leim = fetch_hif_token(session, HIF_LEIM_URL, "x-hif-leim")

    if dliq:
        account["hif_dliq"] = dliq
    if leim:
        account["hif_leim"] = leim
    if dliq or leim:
        save_config(CONFIG)

    if dliq and leim:
        logger.info(f"[ensure_hif_tokens] 账号 {get_account_identifier(account)} HIF 令牌已就绪")
        return True
    else:
        logger.warning(f"[ensure_hif_tokens] 账号 {get_account_identifier(account)} HIF 令牌不完整")
        return False




# ----------------------------------------------------------------------
# (6) 封装对话接口调用的重试机制
# ----------------------------------------------------------------------
def call_completion_endpoint(payload, headers, session, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            deepseek_resp = session.post(
                DEEPSEEK_COMPLETION_URL, headers=headers, json=payload, stream=True, impersonate="safari15_3"
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
# (7) 创建会话 & 获取 PoW（重试时，配置模式下错误会切换账号；用户自带 token 模式下仅重试）
# ----------------------------------------------------------------------
def create_session(request: Request, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers(request)
        ds_session = get_request_session(request)
        try:
            resp = ds_session.post(
                DEEPSEEK_CREATE_SESSION_URL, headers=headers, json={}, impersonate="safari15_3"
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
            if request.state.use_config_token:
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


# ----------------------------------------------------------------------
# (7.1) 使用 WASM 模块计算 PoW 答案的辅助函数
# ----------------------------------------------------------------------
def compute_pow_answer(
    algorithm: str,
    challenge_str: str,
    salt: str,
    difficulty: int,
    expire_at: int,
    signature: str,
    target_path: str,
    wasm_path: str,
) -> int:
    """
    使用 WASM 模块计算 DeepSeekHash 答案（answer）。
    根据 JS 逻辑：
      - 拼接前缀： "{salt}_{expire_at}_"
      - 将 challenge 与前缀写入 wasm 内存后调用 wasm_solve 进行求解，
      - 从 wasm 内存中读取状态与求解结果，
      - 若状态非 0，则返回整数形式的答案，否则返回 None。
    """
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"不支持的算法：{algorithm}")
    prefix = f"{salt}_{expire_at}_"
    # --- 加载 wasm 模块 ---
    store = Store()
    linker = Linker(store.engine)
    try:
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()
    except Exception as e:
        raise RuntimeError(f"加载 wasm 文件失败: {wasm_path}, 错误: {e}")
    module = Module(store.engine, wasm_bytes)
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    try:
        memory = exports["memory"]
        add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        alloc = exports["__wbindgen_export_0"]
        wasm_solve = exports["wasm_solve"]
    except KeyError as e:
        raise RuntimeError(f"缺少 wasm 导出函数: {e}")

    def write_memory(offset: int, data: bytes):
        size = len(data)
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        ctypes.memmove(base_addr + offset, data, size)

    def read_memory(offset: int, size: int) -> bytes:
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        return ctypes.string_at(base_addr + offset, size)

    def encode_string(text: str):
        data = text.encode("utf-8")
        length = len(data)
        ptr_val = alloc(store, length, 1)
        ptr = int(ptr_val.value) if hasattr(ptr_val, "value") else int(ptr_val)
        write_memory(ptr, data)
        return ptr, length

    # 1. 申请 16 字节栈空间
    retptr = add_to_stack(store, -16)
    # 2. 编码 challenge 与 prefix 到 wasm 内存中
    ptr_challenge, len_challenge = encode_string(challenge_str)
    ptr_prefix, len_prefix = encode_string(prefix)
    # 3. 调用 wasm_solve（注意：difficulty 以 float 形式传入）
    wasm_solve(
        store,
        retptr,
        ptr_challenge,
        len_challenge,
        ptr_prefix,
        len_prefix,
        float(difficulty),
    )
    # 4. 从 retptr 处读取 4 字节状态和 8 字节求解结果
    status_bytes = read_memory(retptr, 4)
    if len(status_bytes) != 4:
        add_to_stack(store, 16)
        raise RuntimeError("读取状态字节失败")
    status = struct.unpack("<i", status_bytes)[0]
    value_bytes = read_memory(retptr + 8, 8)
    if len(value_bytes) != 8:
        add_to_stack(store, 16)
        raise RuntimeError("读取结果字节失败")
    value = struct.unpack("<d", value_bytes)[0]
    # 5. 恢复栈指针
    add_to_stack(store, 16)
    if status == 0:
        return None
    return int(value)


# ----------------------------------------------------------------------
# (7.2) 获取 PoW 响应，融合计算 answer 逻辑
# ----------------------------------------------------------------------
def get_pow_response(request: Request, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers(request)
        ds_session = get_request_session(request)
        try:
            resp = ds_session.post(
                DEEPSEEK_CREATE_POW_URL,
                headers=headers,
                json={"target_path": "/api/v0/chat/completion"},
                timeout=30,
                impersonate="safari15_3",
            )
        except Exception as e:
            logger.error(f"[get_pow_response] 请求异常: {e}")
            attempts += 1
            continue
        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"[get_pow_response] JSON解析异常: {e}")
            data = {}
        if resp.status_code == 200 and data.get("code") == 0:
            challenge = data["data"]["biz_data"]["challenge"]
            difficulty = challenge.get("difficulty", 144000)
            expire_at = challenge.get("expire_at", 1680000000)
            try:
                answer = compute_pow_answer(
                    challenge["algorithm"],
                    challenge["challenge"],
                    challenge["salt"],
                    difficulty,
                    expire_at,
                    challenge["signature"],
                    challenge["target_path"],
                    WASM_PATH,
                )
            except Exception as e:
                logger.error(f"[get_pow_response] PoW 答案计算异常: {e}")
                answer = None
            if answer is None:
                logger.warning("[get_pow_response] PoW 答案计算失败，重试中...")
                resp.close()
                attempts += 1
                continue
            pow_dict = {
                "algorithm": challenge["algorithm"],
                "challenge": challenge["challenge"],
                "salt": challenge["salt"],
                "answer": answer,  # 整数形式答案
                "signature": challenge["signature"],
                "target_path": challenge["target_path"],
            }
            pow_str = json.dumps(pow_dict, separators=(",", ":"), ensure_ascii=False)
            encoded = base64.b64encode(pow_str.encode("utf-8")).decode("utf-8").rstrip()
            resp.close()
            return encoded
        else:
            code = data.get("code")
            logger.warning(
                f"[get_pow_response] 获取 PoW 失败, code={code}, msg={data.get('msg')}"
            )
            resp.close()
            if request.state.use_config_token:
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
                        f"[get_pow_response] 账号 {get_account_identifier(new_account)} 登录失败：{e}"
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


# ----------------------------------------------------------------------
# (8) 路由：/v1/models
# ----------------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    models_list = [
        {
            "id": "deepseek-v4-flash",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": [],
        },
        {
            "id": "deepseek-reasoner",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": [],
        },
        {
            "id": "deepseek-v4-pro",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": [],
        },
        {
            "id": "deepseek-chat",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepseek",
            "permission": [],
        },
    ]
    data = {"object": "list", "data": models_list}
    return JSONResponse(content=data, status_code=200)




# ----------------------------------------------------------------------
# 消息预处理函数，将多轮对话合并成最终 prompt
# ----------------------------------------------------------------------
def messages_prepare(messages: list) -> str:
    """处理消息列表，合并连续相同角色的消息，并添加角色标签：
    - 对于 assistant 消息，加上 <｜Assistant｜> 前缀及 <｜end▁of▁sentence｜> 结束标签；
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
            parts.append(f"<｜Assistant｜>{text}<｜end▁of▁sentence｜>")
        elif role in ("user", "system"):
            if idx > 0:
                parts.append(f"<｜User｜>{text}")
            else:
                parts.append(text)
        else:
            parts.append(text)
    final_prompt = "".join(parts)
    return final_prompt


# 添加保活超时配置（5秒）
KEEP_ALIVE_TIMEOUT = 5


def detect_and_parse_tool_calls(content: str):
    """
    检测并解析模型返回的 tool_calls JSON
    返回: (tool_calls_list, remaining_content)
    """
    import re
    
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
# (10) 路由：/v1/chat/completions
# ----------------------------------------------------------------------
@app.post("/v1/chat/completions")
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
        # 判断模型类型
        model_lower = model.lower()
        if model_lower in ["deepseek-v4-flash", "deepseek-chat"]:
            model_type = "default"
            auto_thinking = False
            lock_thinking = False
        elif model_lower in ["deepseek-reasoner"]:
            model_type = "default"
            auto_thinking = True
            lock_thinking = True  # deepseek-reasoner 强制开启思考，不可关闭
        elif model_lower in ["deepseek-v4-pro"]:
            model_type = "expert"
            auto_thinking = False
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
            else:
                thinking_enabled = req_data.get("thinking_enabled", auto_thinking)
                if not isinstance(thinking_enabled, bool):
                    thinking_enabled = auto_thinking
        search_enabled = bool(req_data.get("search_enabled", False))
        
        # 处理 tools 参数（OpenAI 格式）
        tools_requested = req_data.get("tools") or []
        has_tools = len(tools_requested) > 0
        
        # 如果有工具定义，在 messages 前添加工具使用指导的系统消息
        if has_tools:
            tool_schemas = []
            for tool in tools_requested:
                func = tool.get('function', {})
                tool_name = func.get('name', 'unknown')
                tool_desc = func.get('description', 'No description available')
                params = func.get('parameters', {})
                
                tool_info = f"Tool: {tool_name}\nDescription: {tool_desc}"
                if 'properties' in params:
                    props = []
                    required = params.get('required', [])
                    for prop_name, prop_info in params['properties'].items():
                        prop_type = prop_info.get('type', 'string')
                        prop_desc = prop_info.get('description', '')
                        is_req = ' (required)' if prop_name in required else ''
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
        final_prompt = messages_prepare(messages)
        session_id = create_session(request)
        if not session_id:
            raise HTTPException(status_code=401, detail="invalid token.")
        pow_resp = get_pow_response(request)
        if not pow_resp:
            raise HTTPException(
                status_code=401,
                detail="Failed to get PoW (invalid token or unknown error).",
            )
        headers = {**get_auth_headers(request), **get_hif_headers(request), "x-ds-pow-response": pow_resp}
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

        deepseek_resp = call_completion_endpoint(payload, headers, get_request_session(request), max_attempts=3)
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
                    citation_map = {}  # 用于存储引用链接的字典

                    def delete_deepseek_session():
                        """响应结束后删除 DeepSeek 会话"""
                        try:
                            headers = get_auth_headers(request)
                            payload = {
                                "chat_session_id": session_id
                            }
                            ds_session = get_request_session(request)
                            resp = ds_session.post(
                                DEEPSEEK_DELETE_SESSION_URL,
                                headers=headers,
                                json=payload,
                                impersonate="safari15_3",
                                timeout=3
                            )
                            if resp.status_code == 200:
                                logger.info(f"[sse_stream] 响应结束，已删除会话 session={session_id}")
                            else:
                                logger.warning(f"[sse_stream] 删除会话失败: {resp.status_code}")
                        except Exception as e:
                            logger.warning(f"[sse_stream] 调用 delete_session 失败: {e}")

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
                            # 创建基本的错误响应，不依赖JSON解析
                            try:
                                error_response = {"choices": [{"index": 0, "delta": {"content": "服务器错误，请稍候再试", "type": "text"}}]}
                                result_queue.put(error_response)
                            except Exception:
                                # 最终备选方案
                                pass
                            result_queue.put(None)
                            # raise HTTPException(
                                # status_code=500, detail="Server is error."
                            # )
                        finally:
                            deepseek_resp.close()

                    process_thread = threading.Thread(target=process_data)
                    process_thread.start()

                    try:
                        while True:
                            current_time = time.time()
                            if current_time - last_send_time >= KEEP_ALIVE_TIMEOUT:

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
                                    tool_calls_detected, final_text_content = detect_and_parse_tool_calls(final_text)
                                
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
                                prompt_tokens = len(final_prompt) // 4  # 简单估算token数
                                thinking_tokens = len(final_thinking) // 4  # 简单估算token数
                                completion_tokens = len(final_text) // 4  # 简单估算token数
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
                                stream_completed = True  # 标记流正常完成
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
                        raise  # 重新抛出以正常结束生成器
                except Exception as e:
                    logger.error(f"[sse_stream] 异常: {e}")
                    client_disconnected = True
                finally:                    
                    # 响应结束后删除会话
                    delete_deepseek_session()
                    
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
            citation_map = {}

            data_queue = queue.Queue()

            def delete_deepseek_session():
                """响应结束后删除 DeepSeek 会话"""
                try:
                    headers = get_auth_headers(request)
                    payload = {
                        "chat_session_id": session_id
                    }
                    ds_session = get_request_session(request)
                    resp = ds_session.post(
                        DEEPSEEK_DELETE_SESSION_URL,
                        headers=headers,
                        json=payload,
                        impersonate="safari15_3",
                        timeout=3
                    )
                    if resp.status_code == 200:
                        logger.info(f"[chat_completions] 响应结束，已删除会话 session={session_id}")
                    else:
                        logger.warning(f"[chat_completions] 删除会话失败: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"[chat_completions] 调用 delete_session 失败: {e}")

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
                                think_list.append('解码失败，请稍候再试')
                            else:
                                text_list.append('解码失败，请稍候再试')
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
                                    if chunk.get("v") == 'FINISHED':
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
                                                    tool_calls_detected, final_content = detect_and_parse_tool_calls(final_content)
                                                
                                                prompt_tokens = len(final_prompt) // 4  # 简单估算token数
                                                reasoning_tokens = len(final_reasoning) // 4  # 简单估算token数
                                                completion_tokens = len(final_content) // 4  # 简单估算token数
                                                
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
                                                return  # 提前返回，结束函数
            
                            except Exception as e:
                                logger.warning(f"[collect_data] 无法解析: {data_str}, 错误: {e}")
                                # 根据当前处理类型添加错误消息
                                if ptype == "thinking":
                                    think_list.append('解析失败，请稍候再试')
                                else:
                                    text_list.append('解析失败，请稍候再试')
                                data_queue.put(None)
                                break
                except Exception as e:
                    logger.warning(f"[collect_data] 错误: {e}")
                    # 根据当前处理类型添加错误消息
                    if ptype == "thinking":
                        think_list.append('处理失败，请稍候再试')
                    else:
                        text_list.append('处理失败，请稍候再试')
                    data_queue.put(None)
                finally:
                    deepseek_resp.close()
                    if result is None:
                        # 如果没有提前构造 result，则构造默认结果
                        final_content = "".join(text_list)
                        final_reasoning = "".join(think_list)  # 修复：应该使用think_list而不是text_list
                        
                        # 检测 tool_calls
                        tool_calls_detected = None
                        if has_tools:
                            tool_calls_detected, final_content = detect_and_parse_tool_calls(final_content)
                        
                        prompt_tokens = len(final_prompt) // 4  # 简单估算token数
                        reasoning_tokens = len(final_reasoning) // 4  # 简单估算token数
                        completion_tokens = len(final_content) // 4  # 简单估算token数
                        
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
                        if current_time - last_send_time >= KEEP_ALIVE_TIMEOUT:

                            yield ""
                            last_send_time = current_time
                        if not collect_thread.is_alive() and result is not None:
                            yield json.dumps(result)
                            break
                        time.sleep(0.1)
                finally:
                    # 响应结束后删除会话
                    delete_deepseek_session()

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
# (11) 路由：停止流式响应
# ----------------------------------------------------------------------
@app.post("/v1/chat/stop_stream")
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
        ds_session = get_request_session(request)

        # 构造请求体
        payload = {
            "chat_session_id": chat_session_id,
            "message_id": message_id
        }

        # 发送停止请求
        resp = ds_session.post(
            DEEPSEEK_STOP_STREAM_URL,
            headers=headers,
            json=payload,
            impersonate="safari15_3"
        )

        if resp.status_code == 200:
            logger.info(f"[stop_stream] 成功停止会话 {chat_session_id}")
            return JSONResponse(content={"success": True, "message": "已停止流式响应"})
        else:
            logger.warning(f"[stop_stream] 停止失败，状态码: {resp.status_code}")
            return JSONResponse(
                status_code=resp.status_code,
                content={"success": False, "message": f"停止失败: {resp.text}"}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[stop_stream] 异常: {e}")
        raise HTTPException(status_code=500, detail=f"停止流式响应失败: {str(e)}")




# ----------------------------------------------------------------------
# (12) 路由：/
# ----------------------------------------------------------------------
@app.get("/")
def index():
    return HTMLResponse("<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>服务已启动</title></head><body><p>DeepSeek2API Neo 已启动！</p></body></html>")


# ----------------------------------------------------------------------
# 启动 FastAPI 应用
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
