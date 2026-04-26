"""
DeepSeek2API Neo Visualization — 基于 FastAPI 的 API 代理 + Material Design 3 仪表盘
统计：每日 token 消耗、请求次数、按模型分类、按账号分类
数据：SQLite (./data.db)，仪表盘：./web/

启动: python app_visualization.py
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# 导入原始 app — 包含所有 API 端点
# ---------------------------------------------------------------------------
from app.main import app  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app_visualization")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "data.db"
WEB_DIR = Path(__file__).parent / "web"

MODEL_DISPLAY_MAP = {
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-chat": "DeepSeek-V4-Flash",
    "deepseek-reasoner": "DeepSeek-V4-Flash",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "claude-sonnet-4-6": "DeepSeek-V4-Flash",
    "claude-opus-4-6": "DeepSeek-V4-Flash",
    "claude-haiku-4-5": "DeepSeek-V4-Flash",
}

TRACKED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/messages",
}


# =============================================================================
#  数据库层
# =============================================================================
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def init_db():
    conn = get_db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     REAL    NOT NULL,
                date          TEXT    NOT NULL,
                endpoint      TEXT    NOT NULL DEFAULT '',
                model         TEXT    NOT NULL DEFAULT 'unknown',
                model_display TEXT    NOT NULL DEFAULT 'unknown',
                account_id    TEXT    NOT NULL DEFAULT '',
                api_key       TEXT    NOT NULL DEFAULT '',
                stream        INTEGER NOT NULL DEFAULT 0,
                thinking_enabled INTEGER NOT NULL DEFAULT 0,
                request_count INTEGER NOT NULL DEFAULT 1,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS daily_summary (
                date          TEXT    NOT NULL,
                model_display TEXT    NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens      INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (date, model_display)
            );

            CREATE TABLE IF NOT EXISTS account_summary (
                account_id    TEXT    NOT NULL,
                date          TEXT    NOT NULL,
                model_display TEXT    NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (account_id, date, model_display)
            );

            CREATE INDEX IF NOT EXISTS idx_logs_date    ON usage_logs(date);
            CREATE INDEX IF NOT EXISTS idx_logs_model   ON usage_logs(model_display);
            CREATE INDEX IF NOT EXISTS idx_logs_account ON usage_logs(account_id);
            CREATE INDEX IF NOT EXISTS idx_daily_date   ON daily_summary(date);
            CREATE INDEX IF NOT EXISTS idx_acct_date    ON account_summary(date);
        """)
        # Migrate account_summary if old schema (missing prompt/completion cols)
        try:
            conn.execute("ALTER TABLE account_summary ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE account_summary ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        conn.commit()
        logger.info("[init_db] SQLite 数据库初始化完成")
    finally:
        conn.close()


def get_display_model(model_name: str) -> str:
    base = model_name.lower().strip()
    if base.endswith("-search"):
        base = base[:-7]
    return MODEL_DISPLAY_MAP.get(base, model_name or "unknown")


def record_usage(ts, date, endpoint, model, account_id, api_key,
                 stream, thinking, prompt, completion, total):
    display = get_display_model(model)
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO usage_logs
               (timestamp, date, endpoint, model, model_display, account_id, api_key,
                stream, thinking_enabled, request_count,
                prompt_tokens, completion_tokens, total_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (ts, date, endpoint, model, display, account_id, api_key,
             1 if stream else 0, 1 if thinking else 0,
             prompt, completion, total),
        )
        conn.execute(
            """INSERT INTO daily_summary (date, model_display, request_count, prompt_tokens, completion_tokens, total_tokens)
               VALUES (?, ?, 1, ?, ?, ?)
               ON CONFLICT(date, model_display) DO UPDATE SET
                   request_count     = request_count     + 1,
                   prompt_tokens     = prompt_tokens     + excluded.prompt_tokens,
                   completion_tokens = completion_tokens + excluded.completion_tokens,
                   total_tokens      = total_tokens      + excluded.total_tokens""",
            (date, display, prompt, completion, total),
        )
        if account_id:
            conn.execute(
                """INSERT INTO account_summary (account_id, date, model_display, request_count, prompt_tokens, completion_tokens, total_tokens)
                   VALUES (?, ?, ?, 1, ?, ?, ?)
                   ON CONFLICT(account_id, date, model_display) DO UPDATE SET
                       request_count = request_count + 1,
                       prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                       completion_tokens = completion_tokens + excluded.completion_tokens,
                       total_tokens  = total_tokens  + excluded.total_tokens""",
                (account_id, date, display, prompt, completion, total),
            )
        conn.commit()
    except Exception as e:
        logger.error(f"[record_usage] 写入失败: {e}")
    finally:
        conn.close()


# =============================================================================
#  统计中间件
# =============================================================================
class StatsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path not in TRACKED_ENDPOINTS:
            return await call_next(request)

        start_ts = time.time()
        date_str = datetime.now().strftime("%Y-%m-%d")
        model = "unknown"
        stream = False
        thinking = True
        prompt_est = 0

        # 1) 捕获请求体
        body_bytes = await request.body()
        if body_bytes:
            try:
                data = json.loads(body_bytes)
                model = data.get("model", "unknown")
                stream = bool(data.get("stream", False))

                t_obj = data.get("thinking")
                if isinstance(t_obj, dict):
                    thinking = t_obj.get("type") == "enabled"
                elif isinstance(t_obj, bool):
                    thinking = t_obj
                else:
                    thinking = data.get("thinking_enabled", True)

                msgs = data.get("messages", [])
                sys_ = data.get("system", "")
                prompt_est = len(sys_ + json.dumps(msgs, ensure_ascii=False)) // 4
            except Exception:
                pass

        # 2) 恢复请求体，下游路由可正常 json() 读取
        async def _receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        request._receive = _receive

        # 3) 执行下游
        response = await call_next(request)

        # 4) 提取账号（由 determine_mode_and_token 注入到 request.state）
        account_id = ""
        if hasattr(request.state, "account"):
            acc = getattr(request.state, "account", None)
            if acc and isinstance(acc, dict):
                account_id = acc.get("email") or acc.get("mobile") or ""

        # 5) 非流式：从响应体提取精确 token（覆盖估算值）
        completion = 0
        total = prompt_est
        if not stream:
            try:
                raw = b""
                async for chunk in response.body_iterator:
                    raw += chunk
                if raw:
                    resp_data = json.loads(raw)
                    usage = resp_data.get("usage", {})
                    if usage:
                        prompt_est = usage.get("prompt_tokens", prompt_est)
                        completion = usage.get("completion_tokens", 0)
                        total = usage.get("total_tokens", prompt_est + completion)
                # 重新构建 Response（body_iterator 已消费）
                response = Response(
                    content=raw,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            except Exception:
                pass

        # 6) 写入统计
        try:
            record_usage(start_ts, date_str, path, model,
                         account_id, "", stream, thinking,
                         prompt_est, completion, total)
        except Exception as e:
            logger.error(f"[StatsMiddleware] record_usage 异常: {e}")

        return response


# 将中间件添加到原始 app（在已有 CORS 之后）
app.add_middleware(StatsMiddleware)

# 静态文件服务（仪表盘 CSS/JS）
if WEB_DIR.is_dir():
    app.mount("/dashboard-assets", StaticFiles(directory=str(WEB_DIR)), name="dashboard_assets")


# =============================================================================
#  仪表盘页面
# =============================================================================
@app.get("/dashboard")
async def dashboard_page():
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>仪表盘未找到</h1>", status_code=404)


# =============================================================================
#  统计 API
# =============================================================================
def _query(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


@app.get("/api/stats/overview")
async def stats_overview(days: int = 0):
    conn = get_db()
    try:
        if days <= 0:
            today = datetime.now().strftime("%Y-%m-%d")
            row = conn.execute(
                "SELECT COALESCE(SUM(request_count),0) AS total_requests, "
                "COALESCE(SUM(total_tokens),0) AS total_tokens "
                "FROM daily_summary WHERE date = ?", (today,)
            ).fetchone()
        else:
            start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            row = conn.execute(
                "SELECT COALESCE(SUM(request_count),0) AS total_requests, "
                "COALESCE(SUM(total_tokens),0) AS total_tokens "
                "FROM daily_summary WHERE date >= ?", (start,)
            ).fetchone()
        return JSONResponse(dict(row))
    finally:
        conn.close()


@app.get("/api/stats/models")
async def stats_models(days: int = 0):
    conn = get_db()
    try:
        if days <= 0:
            today = datetime.now().strftime("%Y-%m-%d")
            rows = _query(conn,
                "SELECT model_display, SUM(request_count) AS request_count, "
                "SUM(total_tokens) AS total_tokens, SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens "
                "FROM daily_summary WHERE date = ? GROUP BY model_display", (today,))
        else:
            start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            rows = _query(conn,
                "SELECT model_display, SUM(request_count) AS request_count, "
                "SUM(total_tokens) AS total_tokens, SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens "
                "FROM daily_summary WHERE date >= ? GROUP BY model_display", (start,))
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/stats/daily")
async def stats_daily(days: int = 30):
    conn = get_db()
    try:
        start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        rows = _query(conn,
            "SELECT date, model_display, SUM(request_count) AS request_count, "
            "SUM(total_tokens) AS total_tokens, "
            "SUM(prompt_tokens) AS prompt_tokens, "
            "SUM(completion_tokens) AS completion_tokens "
            "FROM daily_summary WHERE date >= ? "
            "GROUP BY date, model_display ORDER BY date ASC", (start,))
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/stats/accounts")
async def stats_accounts(days: int = 0):
    conn = get_db()
    try:
        if days <= 0:
            today = datetime.now().strftime("%Y-%m-%d")
            rows = _query(conn,
                "SELECT account_id, model_display, SUM(request_count) AS request_count, "
                "SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, "
                "SUM(total_tokens) AS total_tokens "
                "FROM account_summary WHERE date = ? "
                "GROUP BY account_id, model_display ORDER BY account_id, model_display",
                (today,))
        else:
            start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            rows = _query(conn,
                "SELECT account_id, model_display, SUM(request_count) AS request_count, "
                "SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, "
                "SUM(total_tokens) AS total_tokens "
                "FROM account_summary WHERE date >= ? "
                "GROUP BY account_id, model_display ORDER BY account_id, model_display",
                (start,))
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/stats/accounts/aggregate")
async def stats_accounts_aggregate(days: int = 0):
    conn = get_db()
    try:
        if days <= 0:
            today = datetime.now().strftime("%Y-%m-%d")
            rows = _query(conn,
                "SELECT account_id, SUM(request_count) AS request_count, "
                "SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, "
                "SUM(total_tokens) AS total_tokens "
                "FROM account_summary WHERE date = ? "
                "GROUP BY account_id ORDER BY total_tokens DESC", (today,))
        else:
            start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            rows = _query(conn,
                "SELECT account_id, SUM(request_count) AS request_count, "
                "SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, "
                "SUM(total_tokens) AS total_tokens "
                "FROM account_summary WHERE date >= ? "
                "GROUP BY account_id ORDER BY total_tokens DESC", (start,))
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/stats/recent")
async def stats_recent(limit: int = 20):
    conn = get_db()
    try:
        rows = _query(conn,
            "SELECT timestamp, date, endpoint, model_display, account_id, "
            "stream, thinking_enabled, prompt_tokens, completion_tokens, total_tokens "
            "FROM usage_logs ORDER BY id DESC LIMIT ?", (limit,))
        return JSONResponse(rows)
    finally:
        conn.close()


# =============================================================================
#  启动
# =============================================================================
@app.on_event("startup")
async def on_startup():
    init_db()
    logger.info("仪表盘就绪 → http://0.0.0.0:5001/dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
