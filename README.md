# AI 学习路径项目

基于阿里云百炼（Alibaba Bailian）API 的 AI 学习路径项目，从基础 LLM 调用到构建复杂的多智能体系统。

## 项目简介

本项目是一个系统化的 AI 学习路径，使用 **阿里云百炼 API**（Qwen 系列模型）配合 **LangChain** / **LangGraph** 框架，
从零开始学习 AI 应用开发。每个章节都有详细的中文注释和文档。

- **模型**: `qwen-plus`（默认）、`qwen-max`、`qwen-turbo` 等
- **API 兼容**: OpenAI SDK 兼容模式
- **框架**: LangChain、LangGraph
- **Python**: 3.13+

## 🗺️ 学习路径总览

| 章节 | 主题 | 难度 |
|------|------|------|
| 01 | 基础 LLM 调用 | ⭐ |
| 02 | Prompt 工程 | ⭐⭐ |
| 03 | Chain 链式调用 | ⭐⭐ |

继续补充中......

## 📋 目录

### 文档
- [01 - 基础LLM调用](https://mp.weixin.qq.com/s/OtdwOoRj0zNXJtk_-AygaQ)
- [02 - Prompt工程](https://mp.weixin.qq.com/s/18WpF61D63yOYpO829j2rQ)
- [03 - Chain链式调用](https://mp.weixin.qq.com/s/AAsPWyFzwvDBDlMvBGBU7A)


### 代码示例

**第01章 基础LLM调用**
- [简单对话](lessons/01_basic_llm/01_simple_chat.py)
- [流式对话](lessons/01_basic_llm/02_streaming_chat.py)
- [思考模型](lessons/01_basic_llm/03_thinking_model.py)

**第02章 Prompt工程**
- [Prompt模板](lessons/02_prompt_engineering/01_prompt_templates.py)
- [Few-Shot提示](lessons/02_prompt_engineering/02_few_shot_prompts.py)
- [思维链提示](lessons/02_prompt_engineering/03_chain_of_thought.py)

**第03章 Chain链式调用**
- [简单链](lessons/03_chains/01_simple_chain.py)
- [顺序链](lessons/03_chains/02_sequential_chain.py)
- [输出解析器](lessons/03_chains/03_output_parsers.py)


## 🚀 快速开始

### 1. 安装依赖

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装核心依赖（适用所有平台，包含章节 01-05、07-10）
uv sync

```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# 请在 https://bailian.console.aliyun.com/ 获取 API Key
```

或者直接设置环境变量：

```bash
export DASHSCOPE_API_KEY=sk-your-api-key-here
```

### 3. 运行示例

```bash
# 运行基础对话示例
uv run python lessons/01_basic_llm/01_simple_chat.py
```

## 🔧 技术栈

| 库 | 版本 | 用途 |
|---|---|---|
| `openai` | >=2.24.0 | OpenAI SDK（兼容百炼API） |
| `langchain` | >=1.2.10 | LLM应用框架 |
| `langchain-openai` | >=1.1.10 | LangChain OpenAI集成 |
| `langchain-community` | >=0.4.1 | 社区集成 |
| `langgraph` | >=0.4.0 | 图状态机工作流 |
| `tiktoken` | >=0.9.0 | Token计数 |


## uv 常用命令

```bash
# 创建项目
uv init ai-agent-test

# 添加依赖
uv add openai langchain langgraph

# 同步依赖
uv sync

# 运行脚本
uv run python script.py
```

## 阿里云源配置

在 `pyproject.toml` 中已配置阿里云镜像源加速下载：

```toml
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
```

## 更多内容 关注公众号「阿聪谈架构」，获取最新教程和实战案例！
### - [阿聪谈架构 AI专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI3NTc0OTYxOQ==&action=getalbum&album_id=4415703927233511424#wechat_redirect)
![img_1.png](img/关注我.png)
