"""
DeepSeek2API Neo — 已拆分为 app/ 包结构。
此文件保留向后兼容，实际逻辑请参见 app/ 目录。
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.main import app  # noqa: E402, F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
