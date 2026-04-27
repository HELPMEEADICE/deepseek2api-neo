import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models  # noqa: F401
from .routes import router
from .account import init_account_queue

# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# -------------------------- 创建 FastAPI 应用 --------------------------
app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# 注册路由
app.include_router(router)

# 初始化账号队列
init_account_queue()


# -------------------------- 启动事件 --------------------------
@app.on_event("startup")
async def on_startup():
    """启动时执行账号健康检查。"""
    from .account import run_health_checks
    logger.info("[startup] 开始账号健康检查...")
    run_health_checks(max_concurrent=10)
    logger.info("[startup] 健康检查完成")


# -------------------------- 关闭事件 --------------------------
@app.on_event("shutdown")
async def on_shutdown():
    """优雅关闭：记录当前活跃账号数。"""
    from .pool import account_pool
    avail = account_pool.available_count()
    total = len(account_pool.all_accounts())
    logger.info(f"[shutdown] 服务器关闭，空闲账号 {avail}/{total}")


# -------------------------- 启动入口 --------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
