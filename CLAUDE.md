# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LangChain 1.x 中文教程/演示项目：RUNOOB 编程学习网站的模拟 AI 客服机器人。所有数据（订单、天气、日程、邮件、人工转接）均为**模拟数据**，不调用任何真实服务。注释、提示词、提交信息均使用中文，请遵循此惯例。

仓库按教程章节递进（提交信息如 八/九/十 = 第八/九/十章）。[llm.md](llm.md) 是作者的 LangChain 学习笔记（Agent、中间件、RAG、文档加载器、切分器、向量数据库），属于参考资料，不是操作指令。

## 常用命令

Python 3.13 虚拟环境位于 `.venv`（Windows）。必须在仓库根目录运行——所有数据库/知识库路径都是相对路径（`./customer_service_db/`、`./my_knowledge_db/`）。

```bash
# 交互式对话 REPL（主入口；输入 exit 退出，输入 reset 清空当前线程历史）
.venv/Scripts/python.exe customer_service_bot.py

# 校验依赖和环境变量是否就绪
.venv/Scripts/python.exe check_install.py

# 填充/更新个人知识库（添加演示文本 + 自动导入 docs/ 目录）
.venv/Scripts/python.exe KnowledgeBase.py

# 生成结构化每日摘要（独立的结构化输出 Agent）
.venv/Scripts/python.exe daily_summary.py

# 直接运行单个工具（见 tools.py 的 __main__ 块）
.venv/Scripts/python.exe tools.py
```

没有测试、Linter 和构建系统，也没有 requirements.txt——主要已装依赖：langchain 1.4.x、langgraph 1.2.x、langchain-openai、langchain-deepseek、langchain-chroma、chromadb、python-dotenv。

## 环境配置

需要 `.env` 文件（已 gitignore），由 [env_utils.py](env_utils.py) 加载：
- `ZHIPU_API_KEY` — 对话模型（glm-4-flash，智谱 OpenAI 兼容接口）
- `DASHSCOPE_API_KEY` — 向量模型（text-embedding-v4，阿里云 DashScope）
- `DEEPSEEK_API_KEY` — 可选的备用对话模型（customer_service_bot.py 中有注释掉的 `ChatDeepSeek`）

## 架构

单一 Agent 在 [customer_service_bot.py](customer_service_bot.py) 中用 LangChain 1.x 的 `create_agent()` 组装：

- **模型**：`init_chat_model("glm-4-flash", model_provider="openai", ...)` + 智谱 base_url。`max_tokens=128` 是刻意设置的（控制演示成本），不要擅自"修复"。
- **工具**（[tools.py](tools.py)）：`search_kb`、`query_order`、`transfer_to_human`、`get_weather`、`query_schedule`、`send_email`。需要用户身份的工具接收 `runtime: ToolRuntime[UserContext]`，通过 `runtime.context.user_id` 读取用户（实现按用户的数据隔离，如订单归属校验）。
- **中间件**（[middlewares.py](middlewares.py)）：已接入 Agent 的有三个——`content_guard`（before_model，拦截 `BLOCK_DEMO` 测试输入）、`auto_signature`（after_agent，追加签名尾注）、`inject_date_context`（dynamic_prompt，在系统提示词中注入当前日期/时区）。`SYSTEM_PROMPT` 定义在此文件。`wrap_model_call_middleware` 和 `wrap_tool_call_middleware` 是未接入的演示实验代码（工具包装器会篡改参数并重复调用 handler——作为演示是故意写坏的）。
- **Checkpointer**：`SqliteSaver`，位于 `./customer_service_db/customer_service.db`，按 `thread_id` 保存对话历史。
- **上下文模式**：`UserContext` dataclass，每次 invoke 时通过 `context=` 传入。

### 人工审批（Human-in-the-loop）流程

`transfer_to_human` 和 `send_email` 调用 `interrupt()` 暂停 Agent。流程：`chat()` 返回 `{'status': 'pending_approval', 'actions': [...]}` → `resume_apply()` 用 `Command(resume={'approved': bool})` 恢复 → 工具收到决策字典。控制台 REPL 中由 `recursive_deal_approval_status()` 驱动（提示输入 y/n），连续中断会递归处理。

### 两个知识库实现（大小写陷阱！）

- [KnowledgeBase.py](KnowledgeBase.py)（大写 K/B）——**当前使用**。支持增量更新的 Chroma 知识库，持久化在 `my_knowledge_db/`；文件按内容哈希判重，未变化的跳过、已修改的重建索引。支持 `.txt`/`.md`/`.pdf`。tools.py 的 `search_kb` 使用它。
- [knowledge_base.py](knowledge_base.py)（全小写）——旧版简单检索器，持久化在 `runoob_kb/`，仅被 tools.py 中注释掉的代码引用。

两个文件能共存只是因为 Windows 文件系统不区分大小写。**不要重命名其中任何一个**——导入语句分别是 `from KnowledgeBase import KnowledgeBase` 和 `from knowledge_base import retriever`，这份代码在 Linux/macOS 上检出会直接出错。

### 数据流

[simulated_data.py](simulated_data.py) 是模拟数据的唯一来源（`weather_db`、`calendar_events`、Asia/Shanghai 时区工具函数）。[daily_summary.py](daily_summary.py) 直接调用工具（不经过 Agent），把结果喂给另一个带 `response_format=ToolStrategy(DailySummary)` 的 `create_agent` 做结构化输出，从 `result['structured_response']` 读取结果。

## Git

当前分支是 `master`；PR 目标分支为 `main`。提交信息用中文，保持现有风格（简短、章节前缀，如 `九.增加连续审批`）。
