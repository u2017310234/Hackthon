"""
Simple Analysis Client for OpenAgents Network
This script connects to the OpenAgents network and sends analysis requests.

Usage:
    python network/simple_client.py "你的分析请求"
"""

import asyncio
import sys

def main():
    """Simple command-line client for the analysis network."""
    print("=" * 60)
    print("OpenAgents 分析客户端")
    print("=" * 60)
    
    try:
        from openagents.core.client import AgentClient
        
        # Get user input from command line or interactive mode
        if len(sys.argv) > 1:
            user_input = " ".join(sys.argv[1:])
            print(f"分析请求: {user_input}")
        else:
            print("输入您要分析的内容（输入 'exit' 退出）：")
            user_input = input("\n请输入：").strip()
            if user_input.lower() == "exit":
                print("退出系统。")
                return
        
        if not user_input:
            print("输入不能为空，请重新输入。")
            return

        print("\n正在连接到 OpenAgents 网络...")
        print("请确保网络已启动: openagents network start ./network")
        print("\n提示：完整的分析功能需要启动分析代理：")
        print("  openagents agent start ./network/agents/analyst.yaml")
        print("或者：")
        print("  python ./network/agents/analysis_agent.py")
        
    except ImportError:
        print("\n警告：openagents 包未安装")
        print("请运行：pip install openagents")
        print("\n备用方案：直接运行 main.py 进行分析")
        print("  python main.py")


if __name__ == "__main__":
    main()
