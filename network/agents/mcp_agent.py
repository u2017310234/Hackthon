"""
MCP-Connected Analysis Agent for OpenAgents Framework
Author: S
Version: 1.0

This agent connects to the MCP server and provides conversation/dialog services.
It uses the MCP tools to perform analysis and can maintain conversation context.

Usage:
    # First start the MCP server (in another terminal):
    python mcp_server.py
    
    # Then start this agent:
    python network/agents/mcp_agent.py
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("mcp_agent")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ----------------------------
# Optional imports with graceful fallback
# ----------------------------
try:
    from openagents.agents.worker_agent import WorkerAgent
    from openagents.models.agent_config import AgentConfig
    from openagents.models.event_context import ChannelMessageContext, EventContext
    HAS_OPENAGENTS = True
except ImportError:
    HAS_OPENAGENTS = False
    WorkerAgent = object
    AgentConfig = None
    ChannelMessageContext = None
    EventContext = None

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


# ----------------------------
# MCP Client Manager
# ----------------------------
class MCPClientManager:
    """Manages connection to MCP server and tool calls."""
    
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._client_context = None
        self._session_context = None
        
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        if not HAS_MCP:
            logger.error("MCP package is not installed")
            return False
        
        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_script_path],
                env=os.environ.copy()
            )
            
            self._client_context = stdio_client(server_params)
            self._read, self._write = await self._client_context.__aenter__()
            
            self._session_context = ClientSession(self._read, self._write)
            self.session = await self._session_context.__aenter__()
            
            await self.session.initialize()
            logger.info("Connected to MCP server successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server."""
        if not self.session:
            return []
        
        try:
            result = await self.session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in result.tools
            ]
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        if not self.session:
            return "错误：未连接到 MCP 服务器"
        
        try:
            result = await self.session.call_tool(name, arguments)
            if result.content:
                return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            return "工具执行成功，但没有返回内容"
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return f"工具调用失败：{str(e)}"


# ----------------------------
# Conversation Manager
# ----------------------------
class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only the last max_history messages
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
    
    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])
    
    def clear_history(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]


# ----------------------------
# MCP Analysis Agent
# ----------------------------
class MCPAnalysisAgent(WorkerAgent):
    """OpenAgents-based agent that connects to MCP server for analysis."""
    
    default_agent_id = "mcp-analyst"
    
    def __init__(self, *args, mcp_server_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Determine MCP server path
        if mcp_server_path is None:
            # Default to mcp_server.py in the project root
            self.mcp_server_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "mcp_server.py"
            )
        else:
            self.mcp_server_path = mcp_server_path
        
        self.mcp_client: Optional[MCPClientManager] = None
        self.conversation_manager = ConversationManager()
        self._tools_cache: List[Dict[str, Any]] = []
    
    async def _ensure_mcp_connected(self) -> bool:
        """Ensure MCP client is connected."""
        if self.mcp_client and self.mcp_client.session:
            return True
        
        if not HAS_MCP:
            logger.warning("MCP package not installed")
            return False
        
        self.mcp_client = MCPClientManager(self.mcp_server_path)
        success = await self.mcp_client.connect()
        
        if success:
            self._tools_cache = await self.mcp_client.list_tools()
            logger.info(f"Loaded {len(self._tools_cache)} tools from MCP server")
        
        return success
    
    async def on_startup(self):
        """Called when agent starts up."""
        # Connect to MCP server
        await self._ensure_mcp_connected()
        
        ws = self.workspace()
        await ws.channel("general").post("MCP 分析助手已上线！发送消息开始对话和分析。")
        await ws.channel("analysis").post("分析频道已就绪，可以开始提交分析请求。")
        
        # List available tools
        if self._tools_cache:
            tools_info = "可用工具：\n" + "\n".join([
                f"  • {tool['name']}: {tool['description']}"
                for tool in self._tools_cache
            ])
            await ws.channel("general").post(tools_info)
    
    async def on_direct(self, context: EventContext):
        """Handle direct messages to the agent."""
        ws = self.workspace()
        user_input = context.content if hasattr(context, 'content') else str(context)
        conversation_id = context.source_id if hasattr(context, 'source_id') else "default"
        
        try:
            response = await self._process_message(user_input, conversation_id)
            await ws.agent(context.source_id).send(response)
        except Exception as e:
            error_msg = f"处理消息时出错：{str(e)}"
            logger.error(f"Message processing failed: {e}")
            await ws.agent(context.source_id).send(error_msg)
    
    async def on_channel_post(self, context: ChannelMessageContext):
        """Handle channel messages."""
        # Skip messages from self
        if context.source_id == self.agent_id:
            return
        
        user_input = context.content if hasattr(context, 'content') else str(context)
        conversation_id = f"channel_{context.channel_id}"
        
        try:
            response = await self._process_message(user_input, conversation_id)
            ws = self.workspace()
            await ws.channel(context.channel_id).post(response)
        except Exception as e:
            error_msg = f"处理消息时出错：{str(e)}"
            logger.error(f"Message processing failed: {e}")
            ws = self.workspace()
            await ws.channel(context.channel_id).post(error_msg)
    
    async def _process_message(self, user_input: str, conversation_id: str) -> str:
        """Process user message and generate response."""
        # Add user message to history
        self.conversation_manager.add_message(conversation_id, "user", user_input)
        
        # Check for special commands
        lower_input = user_input.lower().strip()
        
        if lower_input in ["help", "帮助", "?"]:
            return self._get_help_message()
        
        if lower_input in ["clear", "清除", "重置"]:
            self.conversation_manager.clear_history(conversation_id)
            return "对话历史已清除。"
        
        if lower_input in ["tools", "工具", "功能"]:
            return await self._get_tools_info()
        
        if lower_input in ["history", "历史"]:
            return self._get_history_summary(conversation_id)
        
        # Ensure MCP connection
        if not await self._ensure_mcp_connected():
            return "无法连接到 MCP 服务器，请检查服务是否运行。"
        
        # Run integrated analysis via MCP
        try:
            result = await self.mcp_client.call_tool(
                "run_integrated_analysis",
                {"user_input": user_input}
            )
            
            # Parse the result
            try:
                result_data = json.loads(result)
                response = result_data.get("result_string", result)
            except json.JSONDecodeError:
                response = result
            
            # Add response to history
            self.conversation_manager.add_message(conversation_id, "assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return f"分析过程中出错：{str(e)}"
    
    def _get_help_message(self) -> str:
        """Get help message."""
        return """
📊 MCP 分析助手 - 帮助

这是一个基于 MCP (Model Context Protocol) 的智能分析助手。

📝 基本使用：
直接输入您想分析的内容，我会自动进行：
1. 解析输入，提取关键信息
2. 查询相关数据
3. 进行综合分析
4. 返回结构化的分析报告

🔧 特殊命令：
• help / 帮助 / ? - 显示此帮助信息
• tools / 工具 / 功能 - 显示可用工具列表
• history / 历史 - 显示对话历史摘要
• clear / 清除 / 重置 - 清除对话历史

💡 示例输入：
• "分析腾讯最新的财务报告"
• "阿里巴巴的并购事件有哪些风险？"
• "评估京东的股权变动影响"
"""
    
    async def _get_tools_info(self) -> str:
        """Get tools information."""
        if not await self._ensure_mcp_connected():
            return "无法连接到 MCP 服务器获取工具列表"
        
        if not self._tools_cache:
            return "没有可用的工具"
        
        lines = ["🔧 可用工具：", ""]
        for tool in self._tools_cache:
            lines.append(f"• {tool['name']}")
            lines.append(f"  描述：{tool['description']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_history_summary(self, conversation_id: str) -> str:
        """Get conversation history summary."""
        history = self.conversation_manager.get_history(conversation_id)
        
        if not history:
            return "没有对话历史记录"
        
        lines = [f"📜 对话历史 ({len(history)} 条消息)：", ""]
        for i, msg in enumerate(history[-5:], 1):  # Show last 5 messages
            role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            await self.mcp_client.disconnect()


# ----------------------------
# Standalone Chat Interface
# ----------------------------
async def run_standalone_chat():
    """Run a standalone chat interface without OpenAgents."""
    print("=" * 60)
    print("MCP 分析助手 - 独立对话模式")
    print("输入 'exit' 或 'quit' 退出")
    print("输入 'help' 获取帮助信息")
    print("=" * 60)
    
    # Determine MCP server path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(
        os.path.dirname(os.path.dirname(script_dir)),
        "mcp_server.py"
    )
    
    if not os.path.exists(mcp_server_path):
        print(f"错误：找不到 MCP 服务器脚本: {mcp_server_path}")
        return
    
    # Initialize MCP client
    mcp_client = MCPClientManager(mcp_server_path)
    conversation_manager = ConversationManager()
    conversation_id = "standalone"
    
    print("\n正在连接 MCP 服务器...")
    if not await mcp_client.connect():
        print("无法连接到 MCP 服务器")
        return
    
    tools = await mcp_client.list_tools()
    print(f"已加载 {len(tools)} 个工具")
    print("\n准备就绪！请输入您的问题：\n")
    
    try:
        while True:
            try:
                user_input = input("👤 您: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("\n再见！")
                break
            
            if user_input.lower() in ["help", "帮助", "?"]:
                print("""
📝 基本使用：
直接输入您想分析的内容

🔧 特殊命令：
• help / 帮助 - 显示帮助
• tools / 工具 - 显示可用工具
• clear / 清除 - 清除历史
• exit / quit / 退出 - 退出程序
""")
                continue
            
            if user_input.lower() in ["tools", "工具"]:
                print("\n🔧 可用工具：")
                for tool in tools:
                    print(f"  • {tool['name']}: {tool['description']}")
                print()
                continue
            
            if user_input.lower() in ["clear", "清除"]:
                conversation_manager.clear_history(conversation_id)
                print("对话历史已清除。\n")
                continue
            
            # Add to history
            conversation_manager.add_message(conversation_id, "user", user_input)
            
            print("\n🤖 助手: 正在分析...")
            
            try:
                result = await mcp_client.call_tool(
                    "run_integrated_analysis",
                    {"user_input": user_input}
                )
                
                try:
                    result_data = json.loads(result)
                    response = result_data.get("result_string", result)
                except json.JSONDecodeError:
                    response = result
                
                conversation_manager.add_message(conversation_id, "assistant", response)
                print(f"\n{response}\n")
                
            except Exception as e:
                print(f"\n分析失败：{str(e)}\n")
    
    finally:
        await mcp_client.disconnect()


# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """Main entry point."""
    if not HAS_MCP:
        print("错误：mcp 包未安装。请运行：pip install mcp>=1.23.0")
        return
    
    if HAS_OPENAGENTS:
        # Run as OpenAgents agent
        try:
            # Determine MCP server path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mcp_server_path = os.path.join(
                os.path.dirname(os.path.dirname(script_dir)),
                "mcp_server.py"
            )
            
            agent = MCPAnalysisAgent(
                mcp_server_path=mcp_server_path,
                agent_config=AgentConfig(
                    model_name="auto",
                    instruction="""
你是一个专业的金融分析助手，通过 MCP 协议连接到分析工具。

你的职责：
1. 接收用户输入，理解其分析需求
2. 使用 MCP 工具进行深度分析
3. 提供结构化的分析报告
4. 维护对话上下文，支持多轮对话

保持专业、客观的语气，提供有价值的分析见解。
""",
                )
            )
            
            print("=" * 60)
            print("MCP 分析代理 (OpenAgents 模式)")
            print("正在连接到网络...")
            print("=" * 60)
            
            agent.start(network_host="localhost", network_port=8700)
            agent.wait_for_stop()
            
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            print(f"\n无法启动 OpenAgents 代理: {e}")
            print("切换到独立对话模式...\n")
            asyncio.run(run_standalone_chat())
    else:
        # Run standalone chat
        print("OpenAgents 未安装，使用独立对话模式")
        asyncio.run(run_standalone_chat())


if __name__ == "__main__":
    main()
