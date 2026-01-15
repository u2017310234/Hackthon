# Hackthon - 基于 OpenAgents 的综合分析系统

一个使用 [OpenAgents](https://github.com/openagents-org/openagents) 框架构建的综合分析系统，集成 LLM 和数据库功能。

## 功能特性

- 🤖 **智能分析代理**：使用 LLM 解析用户输入并提取关键信息
- 📊 **数据库集成**：支持 PostgreSQL 数据库查询和结果存储
- 🌐 **OpenAgents 网络**：支持多代理协作和网络通信
- 🔄 **API 密钥轮换**：支持多个 API 密钥的负载均衡
- 🌏 **多 LLM 提供商支持**：支持阿里云通义千问、OpenAI GPT、Google Gemini

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
# 数据库配置（可选）
export DAILY_DATABASE="your_database"
export DAILY_USER="your_user"
export DAILY_PASSWORD="your_password"
export DAILY_HOST="your_host"
export DAILY_PORT="5432"

# LLM API 密钥（选择其中一种）
# 推荐使用阿里云通义千问（国内访问快）
export DASHSCOPE_API_KEY="sk-..."
export ALIBABA_MODEL="qwen-max"  # 可选: qwen-plus, qwen-turbo

# 或者使用 OpenAI
export OPENAI_API_KEY="sk-proj-..."
export OPENAI_MODEL="gpt-4o-mini"  # 可选: gpt-4o, gpt-3.5-turbo

# 或者使用 Google Gemini
export GEMINI_API_KEY="AIza..."
# 或者
export GOOGLE_API_KEY="your_api_key"

# 支持多个密钥进行轮换（仅 Gemini）:
export Y1="your_api_key_1"
export Y2="your_api_key_2"
```

### 3. 启动 OpenAgents 网络

```bash
# 初始化并启动网络
openagents network start ./network
```

### 4. 启动分析代理

**方式一：使用 YAML 配置（推荐）**
```bash
openagents agent start ./network/agents/analyst.yaml
```

**方式二：使用 Python 代理（完整功能）**
```bash
python ./network/agents/analysis_agent.py
```

### 5. 访问 OpenAgents Studio

在浏览器中打开 http://localhost:8050 即可与分析代理交互。

或者使用独立的 Studio：
```bash
openagents studio -s
```

## 🤖 LLM Provider Configuration

本系统支持三种 LLM 提供商，按优先级自动选择：

### 阿里云通义千问 (推荐) - Priority 1

阿里云通义千问是国内访问速度最快的选项，推荐国内用户使用。

**方式1: OpenAI兼容接口 (推荐，无需额外依赖)**
```bash
export DASHSCOPE_API_KEY="sk-xxxxxx"
export ALIBABA_MODEL="qwen-max"  # 可选: qwen-plus, qwen-turbo
python mcp_server.py
```

**方式2: DashScope SDK**
```bash
pip install dashscope
export DASHSCOPE_API_KEY="sk-xxxxxx"
export ALIBABA_USE_DASHSCOPE="true"
python mcp_server.py
```

**获取API密钥:**
1. 访问 https://dashscope.console.aliyun.com/
2. 登录阿里云账号
3. 创建API密钥
4. 开通通义千问服务

**模型选择:**
- `qwen-max`: 最强性能，适合复杂分析
- `qwen-plus`: 平衡性能和成本
- `qwen-turbo`: 快速响应，低成本

### OpenAI GPT - Priority 2

```bash
pip install openai
export OPENAI_API_KEY="sk-proj-xxxxxx"
export OPENAI_MODEL="gpt-4o-mini"  # 或 gpt-4o, gpt-3.5-turbo
python mcp_server.py
```

### Google Gemini - Priority 3

```bash
pip install google-genai
export GEMINI_API_KEY="AIza..."
python mcp_server.py
```

## 📦 Dependencies

**核心依赖:**
```bash
pip install mcp
```

**LLM提供商 (至少选一个):**
```bash
# 阿里云 (OpenAI兼容接口)
pip install openai

# 阿里云 (DashScope SDK)
pip install dashscope

# OpenAI
pip install openai

# Gemini
pip install google-genai
```

**数据库 (可选):**
```bash
pip install psycopg2-binary
```

## 项目结构

```
Hackthon/
├── main.py                    # 原始独立脚本
├── mcp_server.py              # MCP 服务器（多 LLM 提供商支持）
├── requirements.txt           # Python 依赖
├── README.md                  # 项目文档
├── LICENSE                    # 许可证
└── network/                   # OpenAgents 网络配置
    ├── network.yaml           # 网络配置文件
    └── agents/
        ├── analyst.yaml       # 分析代理 YAML 配置
        └── analysis_agent.py  # Python 分析代理（完整功能）
```

## 使用方式

### MCP Server 模式（推荐）

MCP (Model Context Protocol) 服务器支持多个 LLM 提供商，提供以下工具：

```bash
python mcp_server.py
```

**可用工具:**
1. `parse_user_input` - 解析用户输入并提取关键信息
2. `query_financial_data` - 根据主体名称查询财务数据
3. `query_event_model` - 根据事件类型查询事件模型
4. `analyze_with_context` - 基于数据库上下文进行分析
5. `run_integrated_analysis` - 运行完整的分析工作流

### 命令行模式（原始脚本）

如果不使用 OpenAgents 框架，可以直接运行原始脚本：

```bash
python main.py
```

### OpenAgents 模式

启动网络和代理后，可以通过以下方式与分析代理交互：

1. **OpenAgents Studio**：通过 Web 界面发送消息
2. **编程方式**：使用 OpenAgents 客户端连接到网络

```python
from openagents.core.client import AgentClient

client = AgentClient()
client.connect(host="localhost", port=8700)
# 发送消息给分析代理
```

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
- **Google Gemini**：大语言模型
- **PostgreSQL**：关系型数据库
- **Python 3.10+**：编程语言

## 许可证

Apache 2.0