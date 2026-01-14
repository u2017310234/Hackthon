# Hackthon - 基于 OpenAgents 和 MCP 的综合分析系统

一个使用 [OpenAgents](https://github.com/openagents-org/openagents) 框架和 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) 构建的综合分析系统，集成 LLM 和数据库功能。

## 功能特性

- 🤖 **智能分析代理**：使用 LLM 解析用户输入并提取关键信息
- 📊 **数据库集成**：支持 PostgreSQL 数据库查询和结果存储
- 🌐 **OpenAgents 网络**：支持多代理协作和网络通信
- 🔌 **MCP 协议支持**：通过 MCP 协议暴露分析工具，支持标准化工具调用
- 🔄 **API 密钥轮换**：支持多个 API 密钥的负载均衡
- 💬 **对话服务**：支持多轮对话和上下文管理

## 架构概览

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   用户/客户端    │────▶│  OpenAgents     │────▶│   MCP Server    │
│                 │     │  MCP Agent      │     │  (mcp_server.py)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                        ▼                                ▼                                ▼
                ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
                │   Gemini LLM  │              │  PostgreSQL   │              │   分析工具     │
                │   (解析/分析)  │              │   数据库      │              │   (5个工具)    │
                └───────────────┘              └───────────────┘              └───────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n hackthon python=3.12
conda activate hackthon

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 数据库配置
export DAILY_DATABASE="your_database"
export DAILY_USER="your_user"
export DAILY_PASSWORD="your_password"
export DAILY_HOST="your_host"
export DAILY_PORT="5432"

# API 密钥（用于 Gemini LLM）
# 推荐使用标准命名:
export GEMINI_API_KEY="your_api_key"
# 或者
export GOOGLE_API_KEY="your_api_key"

# 支持多个密钥进行轮换:
export Y1="your_api_key_1"
export Y2="your_api_key_2"
# 或者使用
export MILITAI1="your_api_key"
```

### 3. 启动方式

#### 方式一：MCP 独立对话模式（推荐用于测试）

```bash
# 直接运行 MCP Agent 进行对话
python network/agents/mcp_agent.py
```

#### 方式二：完整 OpenAgents 网络模式

```bash
# 终端 1：启动 OpenAgents 网络
openagents network start ./network

# 终端 2：启动 MCP 分析代理
python network/agents/mcp_agent.py
```

#### 方式三：原始命令行模式（不使用 MCP）

```bash
python main.py
```

### 4. 访问 OpenAgents Studio

在浏览器中打开 http://localhost:8050 即可与分析代理交互。

或者使用独立的 Studio：
```bash
openagents studio -s
```

## 项目结构

```
Hackthon/
├── main.py                    # 原始独立脚本
├── mcp_server.py              # MCP 服务器（暴露分析工具）
├── requirements.txt           # Python 依赖
├── README.md                  # 项目文档
├── LICENSE                    # 许可证
└── network/                   # OpenAgents 网络配置
    ├── network.yaml           # 网络配置文件
    ├── simple_client.py       # 简单客户端示例
    └── agents/
        ├── analyst.yaml       # 分析代理 YAML 配置
        ├── analysis_agent.py  # Python 分析代理（直接集成）
        └── mcp_agent.py       # MCP 连接代理（通过 MCP 协议）
```

## MCP 工具列表

MCP 服务器暴露以下工具：

| 工具名称 | 描述 |
|---------|------|
| `parse_user_input` | 解析用户输入，提取关键信息（is_relevant, relevance_score, subject_name, event_type） |
| `query_financial_data` | 根据主体名称查询财务数据 |
| `query_event_model` | 根据事件类型查询事件模型 |
| `analyze_with_context` | 基于上下文信息进行综合分析 |
| `run_integrated_analysis` | 运行完整的综合分析流程 |

## 使用方式

### MCP 对话模式

启动 MCP Agent 后，可以直接进行对话：

```
👤 您: 分析腾讯最新的财务报告

🤖 助手: 正在分析...

摘要：腾讯控股有限公司近期发布了财务报告...

关键要点：
  • 收入增长稳健
  • 游戏业务表现强劲
  ...

风险评估：监管风险需关注

建议：
  • 持续关注政策动态
  • 多元化业务布局
  ...

置信度：0.85
```

### 特殊命令

在对话模式下支持以下命令：

| 命令 | 说明 |
|-----|------|
| `help` / `帮助` / `?` | 显示帮助信息 |
| `tools` / `工具` / `功能` | 显示可用工具列表 |
| `history` / `历史` | 显示对话历史摘要 |
| `clear` / `清除` / `重置` | 清除对话历史 |
| `exit` / `quit` / `退出` | 退出程序 |

## 分析流程

1. **输入解析**：LLM 分析用户输入，提取 `is_relevant`、`relevance_score`、`subject_name`、`event_type`
2. **数据查询**：根据提取的信息查询数据库获取财务数据和事件模型
3. **综合分析**：LLM 基于上下文进行深度分析
4. **结果输出**：生成结构化的分析报告
5. **结果存储**：将分析结果保存到数据库表 "I"

## 分析报告格式

```
摘要：[简要总结]

关键要点：
  • [要点1]
  • [要点2]
  ...

风险评估：[风险评估内容]

建议：
  • [建议1]
  • [建议2]
  ...

置信度：[0.0-1.0]
```

## 技术栈

- **OpenAgents**：AI 代理网络框架
- **MCP**：Model Context Protocol，标准化工具协议
- **Google Gemini**：大语言模型
- **PostgreSQL**：关系型数据库
- **Python 3.10+**：编程语言

## 许可证

Apache 2.0