# 🧠 DeepSeek2API

[![](https://img.shields.io/github/license/HELPMEEADICE/deepseek2api-neo.svg)](LICENSE)
![](https://img.shields.io/github/stars/HELPMEEADICE/deepseek2api-neo.svg)
![](https://img.shields.io/github/forks/HELPMEEADICE/deepseek2api-neo.svg)

让 DeepSeek 用起来像 OpenAI。高速流式输出、多轮对话、R1 深度思考、多账号轮换、工具调用——全都给你包好了。

**兼容 OpenAI Chat Completions** 和 **Anthropic Messages API** 两种格式，内部自动转换，所以 Claude 的 SDK 也能直接连。

## 📖 目录

- [🧠 DeepSeek2API](#-deepseek2api)
  - [📖 目录](#-目录)
  - [⚠️ 免责声明](#️-免责声明)
  - [📁 项目结构](#-项目结构)
  - [🔑 接入准备](#-接入准备)
    - [多账号](#多账号)
    - [config.json 配置](#configjson-配置)
  - [🚀 部署方式](#-部署方式)
    - [Docker 部署](#docker-部署)
    - [Docker Compose 部署](#docker-compose-部署)
    - [Vercel 部署](#vercel-部署)
    - [原生部署](#原生部署)
  - [📋 接口列表](#-接口列表)
    - [模型列表](#模型列表)
    - [对话补全](#对话补全)
    - [Anthropic Messages API](#anthropic-messages-api)
  - [📊 仪表盘 Dashboard](#-仪表盘-dashboard)
  - [💭 思考模式](#-思考模式)
  - [🔍 搜索增强](#-搜索增强)
  - [⚡ 注意事项](#-注意事项)
    - [Nginx 反代优化](#nginx-反代优化)
    - [Token 统计](#token-统计)
    - [工具调用 (tool\_calls)](#工具调用-tool_calls)
  - [⭐ Star History](#-star-history)
  - [🙏 鸣谢](#-鸣谢)

## ⚠️ 免责声明

**逆向 API 是不稳定的。** 说人话：随时可能炸。建议去 DeepSeek 官方 https://platform.deepseek.com/ 付费使用，稳，而且支持官方。

本组织和个人不接受任何资金捐助和交易，此项目是纯粹研究交流学习性质！

**仅限自用，禁止对外提供服务或商用，避免对官方造成服务压力，否则风险自担！**

## 📁 项目结构

```
app_visualization.py        # 🚀 **推荐入口** — 原 API + 统计仪表盘二合一，启动这个就行
app.py                      # 向后兼容入口，实际逻辑在 app/ 包中
web/                        # 仪表盘前端（Material Design 3）
├── index.html
├── app.js
└── styles.css
data.db                     # 统计数据库（自动生成）
app/                        # 核心逻辑包
├── main.py                 # FastAPI 应用创建、CORS、启动
├── config.py               # config.json 读写
├── constants.py            # DeepSeek URL、请求头、路径常量
├── models.py               # Tokenizer 加载
├── session.py              # curl_cffi Session 管理
├── account.py              # 账号队列、登录、鉴权
├── pow.py                  # WASM PoW 计算（DeepSeekHashV1）
├── chat.py                 # 消息合并、工具调用检测
├── converter.py            # OpenAI ↔ Anthropic 格式互转
└── routes.py               # 所有 API 路由
config.json                 # 你的配置（密钥 + 账号）
```

## 🔑 接入准备

需要一个或多个 DeepSeek 账号

### 多账号

同一个账号同一时间只能跑**一路**输出。配置多个账号，请求会自动轮换。

### config.json 配置

```json
{
  "keys": ["key1", "key2"],
  "accounts": [
    { "email": "user@example.com", "password": "your-password", "token": "" },
    { "mobile": "13800138000", "password": "your-password", "token": "" }
  ]
}
```

- `keys` - 你的 API 鉴权密钥，请求时放在 `Authorization: Bearer xxx` 里
- `accounts` - DeepSeek 账号列表，支持邮箱或手机登录

## 🚀 部署方式

### Docker 部署

```shell
docker run -d \
  -p 5001:5001 \
  -v "$(pwd)/config.json:/app/config.json" \
  -v "$(pwd)/data.db:/app/data.db" \
  --name deepseek2api \
  ghcr.io/HELPMEEADICE/deepseek2api-neo:latest
```

查看日志：`docker logs -f deepseek2api`

### Docker Compose 部署

```shell
git clone https://github.com/HELPMEEADICE/deepseek2api-neo.git
cd deepseek2api
# 修改 config.json
docker-compose up -d
```

### Vercel 部署

> [!NOTE]
> Vercel 免费版请求超时 10 秒，接口响应时间可能更长，可能会 504。

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/import/project?template=https://github.com/HELPMEEADICE/deepseek2api-neo)

部署完后回 GitHub 仓库配 config.json，**一定要私库！**

### 原生部署

```shell
git clone https://github.com/HELPMEEADICE/deepseek2api-neo.git
cd deepseek2api
pip install -r requirements.txt
# 修改 config.json
python app_visualization.py
```

后台挂起：

```shell
nohup uvicorn app_visualization:app --host 0.0.0.0 --port 5001 > uvicorn.log 2>&1 &
```

## 📋 接口列表

同时支持 OpenAI 兼容的 `/v1/chat/completions` 和 Anthropic 兼容的 `/v1/messages`，用你喜欢的 SDK 就行。

### 模型列表

**GET /v1/models**

```json
{
  "data": [
    { "id": "deepseek-v4-flash",         "object": "model" },
    { "id": "deepseek-v4-flash-search",  "object": "model" },
    { "id": "deepseek-chat",             "object": "model" },
    { "id": "deepseek-chat-search",      "object": "model" },
    { "id": "deepseek-reasoner",         "object": "model" },
    { "id": "deepseek-reasoner-search",  "object": "model" },
    { "id": "deepseek-v4-pro",           "object": "model" },
    { "id": "deepseek-v4-pro-search",    "object": "model" }
  ]
}
```

每个模型都有带 `-search` 后缀的版本，自动开启联网搜索。

### 对话补全

**POST /v1/chat/completions**

完全兼容 [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api)。

```json
{
  "model": "deepseek-v4-flash",
  "messages": [{ "role": "user", "content": "你是谁？" }],
  "stream": false,
  "thinking": {"type": "enabled"},
  "tools": [...]
}
```

### Anthropic Messages API

**POST /v1/messages**

兼容 Claude SDK，认证方式二选一：

```
Authorization: Bearer your-key
x-api-key: your-key
```

```json
{
  "model": "deepseek-v4-flash",
  "system": "You are a helpful assistant.",
  "messages": [{ "role": "user", "content": "你是谁？" }],
  "stream": false,
  "thinking": {"type": "enabled"}
}
```

## 📊 仪表盘 Dashboard

打开浏览器访问 `http://你服务器IP:5001/dashboard`，就能看到：

- 📈 每日请求数、Token 消耗
- 🎯 按模型分（Flash / Pro）的输入输出统计
- 👤 每个账号的用量明细
- 🔄 自动刷新，支持今日 / 近7天 / 近30天切换

技术栈：Material Design 3 + Chart.js，暗色模式自适应。

这就是为什么我们**推荐你启动 `app_visualization.py` 而不是 `app.py`**——API 能力完全一样，多了一个漂亮的仪表盘，不亏。

## 💭 思考模式

模型默认开启思考，支持两种关闭方式：

```json
// OpenAI 格式
{ "model": "deepseek-v4-flash", "thinking_enabled": false }

// Anthropic 格式
{ "model": "deepseek-v4-flash", "thinking": false }
```

- `deepseek-reasoner` **强制开启思考**，不能关
- `deepseek-v4-flash` / `deepseek-v4-pro` 默认开，可关

## 🔍 搜索增强

模型名后面加 `-search` 后缀即可开启联网搜索：

```json
{ "model": "deepseek-v4-flash-search" }
```

也可以在请求体里传 `search_enabled: true` 达到同样的效果。

## ⚡ 注意事项

### Nginx 反代优化

```nginx
proxy_buffering off;
chunked_transfer_encoding on;
tcp_nopush on;
tcp_nodelay on;
keepalive_timeout 120;
```

### Token 统计

符合 OpenAI 规范，思考模式下返回 `reasoning_tokens`：

```json
"usage": {
  "completion_tokens": 37,
  "prompt_tokens": 1,
  "total_tokens": 38,
  "completion_tokens_details": { "reasoning_tokens": 20 }
}
```

### 工具调用 (tool_calls)

支持 OpenAI 标准的 `tools` 参数，返回 `finish_reason: "tool_calls"`。

支持多轮工具对话：assistant 返回 `tool_calls` → 客户端执行工具 → 把结果作为 `tool` 角色发回来 → 模型继续回答。

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HELPMEEADICE/deepseek2api-neo&type=Date)](https://star-history.com/#HELPMEEADICE/deepseek2api-neo&Date)

## 🙏 鸣谢

部分代码参考了 [LLM-Red-Team/deepseek-free-api](https://github.com/LLM-Red-Team/deepseek-free-api)，在此表示感谢。
