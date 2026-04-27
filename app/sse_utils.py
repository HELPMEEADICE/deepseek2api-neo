"""
SSE 流处理工具：hint 事件拦截、keep-alive 包装器、BufferedResponse。
"""

import json
import logging

logger = logging.getLogger(__name__)

HINT_CHECK_LINE_COUNT = 6  # 前 N 行用来检测 hint


class BufferedResponse:
    """包装 curl_cffi 响应对象，在前面加入已读取的前缀行。

    用法:
        prefix_lines = []
        line_iter = resp.iter_lines()
        for i, raw_line in enumerate(line_iter):
            text = raw_line.decode("utf-8", errors="ignore")
            if "rate_limit" in text.lower():
                raise OverloadedError()
            prefix_lines.append(raw_line)
            if i >= 5:
                break
        resp = BufferedResponse(prefix_lines, resp)
        # 后续 iter_lines() 会先返回前缀行，再返回剩余行
    """

    def __init__(self, prefix_lines, raw_response):
        self._prefix = list(prefix_lines)
        self._resp = raw_response

    def iter_lines(self):
        for line in self._prefix:
            yield line
        yield from self._resp.iter_lines()

    @property
    def status_code(self):
        return self._resp.status_code

    def close(self):
        try:
            self._resp.close()
        except Exception:
            pass


class OverloadedError(Exception):
    """标记服务器过载，应重试。"""
    pass


def check_hint_events(response, max_peek_lines=HINT_CHECK_LINE_COUNT):
    """从响应 stream 中读取前 N 行，检测 hint（rate_limit / overload）。

    如果有 hint → 抛出 OverloadedError。
    否则 → 返回 BufferedResponse（前缀行 + 剩余 stream）。

    调用方应捕获 OverloadedError 并执行清理 + 重试。
    """
    prefix_lines = []
    line_iter = response.iter_lines()
    for i, raw_line in enumerate(line_iter):
        prefix_lines.append(raw_line)
        try:
            text = raw_line.decode("utf-8", errors="ignore")
            lower = text.lower()
            if "rate_limit" in lower or "rate limit" in lower:
                logger.warning("[check_hint_events] 检测到 rate_limit hint")
                raise OverloadedError("rate_limit detected in hint event")
            if '"hint"' in lower:
                logger.warning("[check_hint_events] 检测到 hint 事件")
                raise OverloadedError("hint event detected")
        except UnicodeDecodeError:
            pass
        if i >= max_peek_lines:
            break

    return BufferedResponse(prefix_lines, response)
