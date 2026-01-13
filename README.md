# Hackthon - 基于 OpenAgents 的综合分析系统

一个使用 [OpenAgents](https://github.com/openagents-org/openagents) 框架构建的综合分析系统，集成 LLM 和数据库功能。

## 功能特性

- 🤖 **智能分析代理**：使用 LLM 解析用户输入并提取关键信息
- 📊 **数据库集成**：支持 PostgreSQL 数据库查询和结果存储
- 🌐 **OpenAgents 网络**：支持多代理协作和网络通信
- 🔄 **API 密钥轮换**：支持多个 API 密钥的负载均衡

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
export Y1="your_api_key_1"
export Y2="your_api_key_2"
# 或者使用
export MILITAI1="your_api_key"
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
├── main.py                    # 原始独立脚本
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