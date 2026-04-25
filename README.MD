# DeepSeek2API

[![](https://img.shields.io/github/license/iidamie/deepseek2api.svg)](LICENSE)
![](https://img.shields.io/github/stars/iidamie/deepseek2api.svg)
![](https://img.shields.io/github/forks/iidamie/deepseek2api.svg)

支持高速流式输出、支持多轮对话、支持R1深度思考，多路账号支持，支持对话结束后自动删除 DeepSeek 会话，支持 工具函数调用。

同时兼容 **OpenAI Chat Completions** 和 **Anthropic Messages API** 两种格式，内部自动转换。

## 目录

* [免责声明](#免责声明)
* [接入准备](#接入准备)
  * [多账号接入](#多账号接入)
* [Docker部署](#Docker部署)
* [Docker-compose部署](#Docker-compose部署)
* [Vercel部署](#Vercel部署)
* [原生部署](#原生部署)
* [接口列表](#接口列表)
  * [模型列表](#模型列表)
  * [对话补全](#对话补全)
  * [Anthropic Messages API](#anthropic-messages-api-1)
* [注意事项](#注意事项)
  * [Nginx反代优化](#Nginx反代优化)
  * [Token统计](#Token统计)
* [Star History](#star-history)
* [鸣谢](#鸣谢)

## 免责声明

**逆向API是不稳定的，建议前往DeepSeek官方 https://platform.deepseek.com/ 付费使用API，避免封禁的风险。**

**本组织和个人不接受任何资金捐助和交易，此项目是纯粹研究交流学习性质！**

**仅限自用，禁止对外提供服务或商用，避免对官方造成服务压力，否则风险自担！**

**仅限自用，禁止对外提供服务或商用，避免对官方造成服务压力，否则风险自担！**

**仅限自用，禁止对外提供服务或商用，避免对官方造成服务压力，否则风险自担！**

## 项目结构

```
app.py                    # 向后兼容入口，实际逻辑在 app/ 包中
app/                      # 模块化包结构
├── __init__.py           # 包入口，导出 FastAPI app
├── main.py               # FastAPI 应用创建、CORS、启动入口
├── config.py             # config.json 读写、CONFIG 全局
├── constants.py          # DeepSeek URL、请求头、路径常量
├── models.py             # Tokenizer 初始化
├── session.py            # curl_cffi Session 管理（账号级 + 全局）
├── account.py            # 账号队列、登录、HIF 令牌、鉴权
├── pow.py                # WASM PoW 计算（DeepSeekHashV1）
├── chat.py               # 消息合并、工具调用检测
├── converter.py          # OpenAI ↔ Anthropic 格式转换模块
└── routes.py             # 所有 API 路由（/v1/*, /v1/messages, /）
```

## 接入准备

一个或多个 DeepSeek 账号

### 多账号接入

目前同个账号同时只能有*一路*输出，你可以通过提供多个账号并使用。

每次请求服务会从中挑选一个。

## Docker部署

请准备一台具有公网IP的服务器并将5001端口开放。

配置 config.json

在 `deepseek` 目录下，创建 config.json文件
```json
{
  "keys": [
    "key1",
    "key2"
  ],
  "accounts": [
    {
      "email": "example1@example.com",
      "password": "password1",
      "token": ""
    },
    {
      "email": "example2@example.com",
      "password": "password2",
      "token": ""
    },
    {
      "mobile": "12345678901",
      "password": "password3",
      "token": ""
    },
    {
      "mobile": "12345678901",
      "password": "password4",
      "token": ""
    }
  ]
}
```
 * keys - 你的 API 鉴权密钥
 * accounts - DeepSeek 账号列表，支持多个账号轮换，避免单账号受限

拉取镜像并启动服务。

```shell
docker run -d -p 5001:5001 -v "$(pwd)/config.json:/app/config.json" --name deepseek2api ghcr.io/iidamie/deepseek2api:latest
```

查看服务实时日志

```shell
docker logs -f deepseek2api
```

重启服务

```shell
docker restart deepseek2api
```

停止服务

```shell
docker stop deepseek2api
```

## Docker-compose部署

拉取该项目
```shell
git clone https://github.com/iidamie/deepseek2api.git
cd deepseek2api
```
配置 config.json

在当前目录下，修改 config.json 文件
 * keys - 你的 API 鉴权密钥
 * accounts - DeepSeek 账号列表，支持多个账号轮换，避免单账号受限

启动服务

```shell
docker-compose up -d
```

查看服务实时日志

```shell
docker logs -f deepseek2api
```

重启服务

```shell
docker restart deepseek2api
```

停止服务

```shell
docker stop deepseek2api
```

## Vercel部署

> [!NOTE]
> Vercel免费账户的请求响应超时时间为10秒，但接口响应通常较久，可能会遇到Vercel返回的504超时错误！

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/import/project?template=https://github.com/iidamie/deepseek2api)

部署完后，返回你的 Github 仓库配置 config.json
 * keys - 你的 API 鉴权密钥
 * accounts - DeepSeek 账号列表，支持多个账号轮换，避免单账号受限

> [!WARNING]  
> 一定要是私库，防止信息泄露！

配置完后，返回 Vercel 控制台，重新部署即可

## 原生部署

请准备一台具有公网IP的服务器并将5001端口开放。

请先安装好 Python 环境并且配置好环境变量，确认 python 命令可用。

安装依赖

```shell
git clone https://github.com/iidamie/deepseek2api.git
cd deepseek2api
pip install -r requirements.txt
```

配置 config.json

在当前目录下，修改 config.json 文件
 * keys - 你的 API 鉴权密钥
 * accounts - DeepSeek 账号列表，支持多个账号轮换，避免单账号受限

启动服务

```shell
python app.py
```

使用 nohup 启动

```shell
nohup uvicorn app:app --host 0.0.0.0 --port 5001 > uvicorn.log 2>&1 &
```

## 接口列表

同时支持 OpenAI 兼容的 `/v1/chat/completions` 接口和 Anthropic 兼容的 `/v1/messages` 接口，可自行使用对应的客户端接入。

### 模型列表

获取模型列表接口

**GET /v1/models**

响应数据:
```json
{
  "data": [
    {
      "created": 1677610602,
      "id": "deepseek-v4-flash",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-reasoner",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-v4-pro",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-chat",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    }
  ],
  "object": "list"
}
```

### 对话补全

对话补全接口，与openai的 [chat-completions-api](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) 兼容。

**POST /v1/chat/completions**

header 需要设置 Authorization 头部：

```
Authorization: Bearer [userToken value or keys]
```

请求数据：
```json
{
    // model名称
    // 默认：deepseek-v4-flash（别名 deepseek-chat、deepseek-v3）
    // 深度思考：deepseek-reasoner 或 deepseek-r1（基于 v4-flash，默认开启思考）
    // 专家模型：deepseek-v4-pro
    "model": "deepseek-v4-flash",
    // 多轮对话基于消息合并实现，某些场景可能导致能力下降且受单轮最大token数限制
    "messages": [
        {
            "role": "user",
            "content": "你是谁？"
        }
    ],
    // 如果使用流式响应请设置为true，默认false
    "stream": false,
    // 思考模式（默认由模型决定，可覆盖）
    "thinking": {"type": "enabled"}
}
```

响应数据：
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。",
        "reasoning_content": "您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1。如您有任何任何问题，我会尽我所能为您提供帮助。\n",
        "role": "assistant"
      }
    }
  ],
  "created": 1738598554,
  "id": "bc223da6-f091-4687-9f59-b9f2a917bd49",
  "model": "deepseek-r1",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 37,
    "prompt_tokens": 1,
    "total_tokens": 38
  }
}
```

### Anthropic Messages API

兼容 Anthropic Messages API 格式，支持 Claude SDK 直接接入。

**POST /v1/messages**

认证方式（二选一）：

```
# 方式1：标准 Authorization（OpenAI 兼容）
Authorization: Bearer [userToken value or keys]

# 方式2：x-api-key（Anthropic SDK 兼容）
x-api-key: [userToken value or keys]
```

请求数据：
```json
{
    "model": "deepseek-v4-flash",
    "system": "You are a helpful assistant.",
    "messages": [
        {
            "role": "user",
            "content": "你是谁？"
        }
    ],
    "stream": false,
    "max_tokens": 1024,
    "thinking": {"type": "enabled"}
}
```

响应数据（非流式）：
```json
{
  "id": "msg_xxx",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "用户想了解我的身份..."
    },
    {
      "type": "text",
      "text": "您好！我是智能助手DeepSeek-R1，很高兴为您服务！"
    }
  ],
  "model": "deepseek-v4-flash",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 50
  }
}
```

流式响应（SSE）：
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_xxx","type":"message","role":"assistant","content":[],"model":"deepseek-v4-flash","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"用户想了解我的身份..."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"您好！我是智能助手DeepSeek-R1，很高兴为您服务！"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":50}}

event: message_stop
data: {"type":"message_stop"}
```

## 注意事项

### Nginx反代优化

如果您正在使用Nginx反向代理deepseek2api，请添加以下配置项优化流的输出效果，优化体验感。

```nginx
# 关闭代理缓冲。当设置为off时，Nginx会立即将客户端请求发送到后端服务器，并立即将从后端服务器接收到的响应发送回客户端。
proxy_buffering off;
# 启用分块传输编码。分块传输编码允许服务器为动态生成的内容分块发送数据，而不需要预先知道内容的大小。
chunked_transfer_encoding on;
# 开启TCP_NOPUSH，这告诉Nginx在数据包发送到客户端之前，尽可能地发送数据。这通常在sendfile使用时配合使用，可以提高网络效率。
tcp_nopush on;
# 开启TCP_NODELAY，这告诉Nginx不延迟发送数据，立即发送小数据包。在某些情况下，这可以减少网络的延迟。
tcp_nodelay on;
# 设置保持连接的超时时间，这里设置为120秒。如果在这段时间内，客户端和服务器之间没有进一步的通信，连接将被关闭。
keepalive_timeout 120;
```

### Token统计

符合 OPENAI 接口规范，思考模式下包含 reasoning_tokens 详情。

示例:
```json
"usage": {
  "completion_tokens": 37,
  "prompt_tokens": 1,
  "total_tokens": 38,
  "completion_tokens_details": {
    "reasoning_tokens": 20
  }
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=iidamie/deepseek2api&type=Date)](https://star-history.com/#iidamie/deepseek2api&Date)

## 鸣谢

本项目部分代码参考了 [LLM-Red-Team/deepseek-free-api](https://github.com/LLM-Red-Team/deepseek-free-api)，在此表示感谢
