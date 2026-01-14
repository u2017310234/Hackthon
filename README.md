# Hackthon - 基于 OpenAgents 的综合分析系统

一个使用 [OpenAgents](https://github.com/openagents-org/openagents) 框架构建的综合分析系统，集成 LLM 和数据库功能。

## 功能特性

- 🤖 **智能分析代理**：使用 LLM 解析用户输入并提取关键信息
- 📊 **数据库集成**：支持 PostgreSQL 数据库查询和结果存储
- 🌐 **OpenAgents 网络**：支持多代理协作和网络通信
- 🔄 **多 LLM 提供商**：通过 OpenAgents 全局 API 支持多种 LLM 提供商

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n hackthon python=3.12
conda activate hackthon

# 安装依赖
pip install -r requirements.txt

# 如果使用 Gemini，还需安装：
pip install google-generativeai
```

### 2. 配置环境变量

```bash
# 数据库配置
export DAILY_DATABASE="your_database"
export DAILY_USER="your_user"
export DAILY_PASSWORD="your_password"
export DAILY_HOST="your_host"
export DAILY_PORT="5432"

# LLM API 密钥（选择其中一种方式）

# 方式一：OpenAgents 标准（推荐）
export DEFAULT_LLM_API_KEY="your_api_key"

# 方式二：Gemini 特定
export GEMINI_API_KEY="your_api_key"
# 或者
export GOOGLE_API_KEY="your_api_key"

# 方式三：OpenAI 特定
export OPENAI_API_KEY="your_api_key"

# 可选：配置 LLM 提供商和模型
export OPENAGENTS_LLM_PROVIDER="gemini"  # 可选: openai, gemini, claude, deepseek 等
export OPENAGENTS_LLM_MODEL="gemini-2.5-flash"

# 兼容旧版：支持以 Y 或 MILITAI 开头的环境变量
export Y1="your_api_key_1"
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

## 项目结构

```
Hackthon/
├── main.py                    # 独立脚本（使用 OpenAgents 全局 API）
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

### 命令行模式（独立脚本）

直接运行独立脚本进行分析：

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

## 支持的 LLM 提供商

通过 OpenAgents 全局 API，本项目支持以下 LLM 提供商：

| 提供商 | 环境变量 | 示例模型 |
|--------|----------|----------|
| OpenAI | `OPENAI_API_KEY` | gpt-4, gpt-3.5-turbo |
| Google Gemini | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` | gemini-2.5-flash, gemini-pro |
| Anthropic Claude | `ANTHROPIC_API_KEY` | claude-3-opus, claude-3-sonnet |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | gpt-4 (Azure) |
| 更多... | 参见 OpenAgents 文档 | - |

## 技术栈

- **OpenAgents**：AI 代理网络框架（包含统一的 LLM 提供商 API）
- **PostgreSQL**：关系型数据库
- **Python 3.10+**：编程语言

## 许可证

Apache 2.0