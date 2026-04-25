import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入模块以触发初始化（如 tokenizer、account_queue）
from . import models  # noqa: F401
from .routes import router

# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# -------------------------- 创建 FastAPI 应用 --------------------------
app = FastAPI()

# 添加 CORS 中间件，允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# 注册路由
app.include_router(router)

# -------------------------- 启动入口 --------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
