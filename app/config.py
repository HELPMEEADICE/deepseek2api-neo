import json
import logging

logger = logging.getLogger(__name__)

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
