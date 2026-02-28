# Hackthon 项目介绍

## 项目概述

Hackthon 是一个基于人工智能的综合分析系统，旨在通过大语言模型（LLM）和数据库集成，为用户提供智能化的数据分析和决策支持。该项目利用 OpenAgents 框架构建了一个可扩展的多代理系统，支持自然语言交互和自动化分析流程。

## 核心价值

### 1. 智能化分析
- 自动理解用户自然语言输入
- 智能提取关键信息（主体名称、事件类型、相关性评分等）
- 基于上下文的深度分析和建议生成

### 2. 灵活的 LLM 集成
支持多种大语言模型提供商，按优先级自动选择：
- **优先级 1**: Google Gemini（gemini-2.5-flash）
- **优先级 2**: OpenAI（gpt-4o-mini, gpt-4o）
- **优先级 3**: 阿里巴巴通义千问（qwen-max, qwen-plus）

### 3. 数据库集成
- PostgreSQL 数据库支持
- 财务数据查询功能
- 事件模型检索
- 分析结果持久化存储

### 4. 可扩展架构
- 基于 OpenAgents 框架的多代理系统
- 支持 YAML 配置和 Python 自定义代理
- MCP (Model Context Protocol) 服务器支持

## 技术架构

### 系统组件

```
┌─────────────────────────────────────────────────────────┐
│                     用户交互层                           │
│         (命令行 / OpenAgents Studio / API)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  OpenAgents 网络层                       │
│              (网络配置 / 代理管理)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   分析代理层                             │
│     ┌──────────────┬──────────────┬──────────────┐     │
│     │ 输入解析     │  数据查询    │  综合分析    │     │
│     │  (LLM)      │ (Database)   │   (LLM)      │     │
│     └──────────────┴──────────────┴──────────────┘     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    数据持久层                            │
│         (PostgreSQL / 结果存储)                          │
└─────────────────────────────────────────────────────────┘
```

### 核心模块

#### 1. main.py - 独立分析脚本
原始的命令行分析工具，包含完整的分析流程：
- LLM 客户端管理（支持 API 密钥轮换）
- 数据库操作（查询、插入）
- 分析工作流编排

#### 2. mcp_server.py - MCP 服务器
Model Context Protocol 服务器实现，提供：
- 多 LLM 提供商支持（Gemini / OpenAI / Qwen）
- 可复用的分析工具（parse_user_input, analyze_with_context 等）
- JSON 提取和解析工具
- 健壮的错误处理

#### 3. OpenAgents 网络
- **network/network.yaml**: 网络配置文件
- **network/agents/analyst.yaml**: 简化的 YAML 代理配置
- **network/agents/analysis_agent.py**: 完整功能的 Python 代理实现

## 分析工作流

系统采用五步分析流程：

```
用户输入
   ↓
1. 输入解析 (LLM)
   - 判断相关性
   - 提取主体名称
   - 识别事件类型
   - 评估相关性评分
   ↓
2. 数据查询 (Database)
   - 查询财务数据
   - 获取事件模型
   ↓
3. 综合分析 (LLM)
   - 基于上下文分析
   - 生成关键要点
   - 评估风险
   - 提供建议
   ↓
4. 结果格式化
   - 生成结构化报告
   - 包含置信度评分
   ↓
5. 结果存储 (Database)
   - 保存到表 "I"
   - 记录时间戳
   ↓
输出结果
```

## 使用场景

### 场景 1：财务分析
```
用户输入: "分析特斯拉2024年第一季度财报"

系统处理:
1. 识别主体: 特斯拉
2. 事件类型: 财务报告
3. 查询财务数据库
4. 生成综合分析报告
5. 提供投资建议和风险评估
```

### 场景 2：事件影响分析
```
用户输入: "某公司收购案对行业的影响"

系统处理:
1. 提取公司名称和事件类型
2. 查询相关事件模型
3. 分析行业影响
4. 评估风险和机会
5. 给出应对建议
```

### 场景 3：风险评估
```
用户输入: "评估某企业的法律诉讼风险"

系统处理:
1. 识别企业主体
2. 事件类型: 法律诉讼
3. 查询历史数据
4. 风险量化分析
5. 提供风险缓解建议
```

## 部署方式

### 方式 1: 独立脚本模式
适合快速测试和单次分析：
```bash
python main.py
```

### 方式 2: MCP 服务器模式
适合集成到其他系统：
```bash
python mcp_server.py
```

### 方式 3: OpenAgents 网络模式
适合生产环境和多用户场景：
```bash
# 启动网络
openagents network start ./network

# 启动代理
openagents agent start ./network/agents/analyst.yaml

# 访问 Web 界面
# 浏览器打开 http://localhost:8050
```

## 配置管理

### 环境变量配置

#### LLM 配置
```bash
# Gemini (推荐)
export GEMINI_API_KEY="your_api_key"

# OpenAI (备选)
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4o-mini"

# 阿里巴巴通义千问 (备选)
export DASHSCOPE_API_KEY="your_api_key"
export ALIBABA_MODEL="qwen-max"
```

#### 数据库配置
```bash
export DAILY_DATABASE="your_database"
export DAILY_USER="your_user"
export DAILY_PASSWORD="your_password"
export DAILY_HOST="your_host"
export DAILY_PORT="5432"
```

## 项目特色

### 1. 多 LLM 提供商支持
- 自动故障转移机制
- 按优先级选择最佳提供商
- API 密钥轮换支持（负载均衡）

### 2. 健壮的 JSON 解析
- 支持直接 JSON 响应
- 支持 Markdown 代码块格式
- 智能提取嵌套 JSON 对象
- 处理格式不规范的 LLM 输出

### 3. 安全性考虑
- 使用参数化查询防止 SQL 注入
- 输入长度验证
- 安全设置配置（内容过滤）

### 4. 可观测性
- 完整的日志记录
- 详细的错误追踪
- 性能监控支持

## 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.12+ |
| AI 框架 | OpenAgents |
| LLM 提供商 | Google Gemini, OpenAI, Alibaba Qwen |
| 数据库 | PostgreSQL |
| 协议 | Model Context Protocol (MCP) |
| 依赖管理 | pip, conda |

## 开发路线图

### 当前功能
- ✅ 多 LLM 提供商集成
- ✅ 数据库查询和存储
- ✅ 基础分析工作流
- ✅ OpenAgents 网络支持
- ✅ MCP 服务器实现

### 未来规划
- 🔄 实现实际的财务数据查询逻辑
- 🔄 添加更多事件类型支持
- 🔄 增强风险评估模型
- 🔄 支持多语言分析
- 🔄 添加可视化报告生成
- 🔄 实现批量分析功能
- 🔄 支持流式输出
- 🔄 添加分析历史查询功能

## 贡献指南

本项目欢迎贡献！请参考以下指南：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 作者: S
- 项目链接: [https://github.com/u2017310234/Hackthon](https://github.com/u2017310234/Hackthon)
- OpenAgents: [https://github.com/openagents-org/openagents](https://github.com/openagents-org/openagents)

## 致谢

- [OpenAgents](https://github.com/openagents-org/openagents) - 强大的 AI 代理框架
- [Google Gemini](https://ai.google.dev/) - 高性能 LLM 服务
- [OpenAI](https://openai.com/) - GPT 系列模型
- [阿里巴巴通义千问](https://tongyi.aliyun.com/) - 中文优化 LLM
