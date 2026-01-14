"""
Integrated Analysis Agent for OpenAgents Framework
Author: S
Version: 4.0

This agent integrates with the OpenAgents framework to provide:
1. User input parsing with LLM (using OpenAgents global API)
2. Database querying based on extracted information
3. Comprehensive analysis with context
4. Result storage in database

Usage:
    python network/agents/analysis_agent.py
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None

try:
    from openagents.agents.worker_agent import WorkerAgent
    from openagents.models.agent_config import AgentConfig
    from openagents.models.event_context import ChannelMessageContext, EventContext
    from openagents.config.llm_configs import create_model_provider
    HAS_OPENAGENTS = True
except ImportError:
    HAS_OPENAGENTS = False
    WorkerAgent = object
    AgentConfig = None
    ChannelMessageContext = None
    EventContext = None
    create_model_provider = None

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("analysis_agent")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# ----------------------------
# Database Configuration
# ----------------------------
def get_db_config() -> Optional[Dict[str, str]]:
    """Load database configuration from environment variables."""
    try:
        return {
            "dbname": os.environ["DAILY_DATABASE"],
            "user": os.environ["DAILY_USER"],
            "password": os.environ["DAILY_PASSWORD"],
            "host": os.environ["DAILY_HOST"],
            "port": os.environ.get("DAILY_PORT", "5432"),
        }
    except KeyError as e:
        logger.warning(f"Missing database environment variable: {e}")
        return None


# ----------------------------
# LLM Configuration
# ----------------------------
@dataclass(frozen=True)
class LlmConfig:
    """LLM configuration for OpenAgents providers."""
    # Default provider and model (can be overridden via environment variables)
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"


def get_llm_config() -> LlmConfig:
    """Get LLM configuration from environment variables."""
    provider = os.environ.get("OPENAGENTS_LLM_PROVIDER", "gemini")
    model = os.environ.get("OPENAGENTS_LLM_MODEL", "gemini-2.5-flash")
    return LlmConfig(provider=provider, model=model)


def get_api_key() -> Optional[str]:
    """
    Get API key from environment variables.
    
    Priority order:
    1. DEFAULT_LLM_API_KEY (OpenAgents standard)
    2. GEMINI_API_KEY or GOOGLE_API_KEY (for Gemini provider)
    3. OPENAI_API_KEY (for OpenAI provider)
    4. Variables starting with 'Y' (legacy)
    5. Variables starting with 'MILITAI' (legacy)
    """
    # OpenAgents standard
    api_key = os.environ.get("DEFAULT_LLM_API_KEY")
    if api_key:
        return api_key
    
    # Gemini-specific
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    
    # OpenAI-specific
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Legacy: variables starting with 'Y'
    for key in os.environ.keys():
        if key.startswith("Y") and os.environ.get(key):
            return os.environ.get(key)
    
    # Legacy: variables starting with 'MILITAI'
    for key in os.environ.keys():
        if key.startswith("MILITAI") and os.environ.get(key):
            return os.environ.get(key)
    
    return None


# ----------------------------
# OpenAgents LLM Runner
# ----------------------------
class OpenAgentsLLMRunner:
    """LLM runner using OpenAgents framework's global API."""

    def __init__(self, llm_config: Optional[LlmConfig] = None, api_key: Optional[str] = None) -> None:
        if not HAS_OPENAGENTS:
            raise RuntimeError("openagents package is not installed")
        
        self._config = llm_config or get_llm_config()
        self._api_key = api_key or get_api_key()
        self._provider = None
        self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Initialize the LLM provider using OpenAgents framework."""
        if not self._api_key:
            raise RuntimeError("No API key found. Set DEFAULT_LLM_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY")
        
        logger.info(f"Initializing LLM provider: {self._config.provider}, model: {self._config.model}")
        self._provider = create_model_provider(
            provider=self._config.provider,
            model_name=self._config.model,
            api_key=self._api_key,
        )

    async def generate_json_async(self, prompt: str) -> Dict[str, Any]:
        """Generate content and parse as JSON (async version)."""
        if not self._provider:
            raise RuntimeError("LLM provider not initialized")
        
        logger.info(f"Calling LLM (provider={self._config.provider}, model={self._config.model})")
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self._provider.chat_completion(messages)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e
        
        content = response.get("content")
        if not content:
            raise RuntimeError("LLM returned empty content.")
        
        # Try to parse as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(f"Failed to parse LLM response as JSON: {content[:200]}...")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate content and parse as JSON (sync wrapper)."""
        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(self.generate_json_async(prompt))
        else:
            # Already in an event loop, use run_until_complete with a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.generate_json_async(prompt))
                return future.result()


# ----------------------------
# Database Operations
# ----------------------------
class DatabaseManager:
    """Database operations manager."""

    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config
        self.conn = None

    def connect(self) -> bool:
        """Establish database connection."""
        if not HAS_PSYCOPG2:
            logger.warning("psycopg2 is not installed, database operations will be skipped")
            return False
        
        if not self.config:
            logger.warning("Database configuration not available")
            return False
        
        try:
            self.conn = psycopg2.connect(**self.config)
            logger.info("Connected to database successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")

    def create_result_table(self) -> None:
        """Create table 'I' for storing analysis results (if not exists)."""
        if not self.conn:
            return
        
        query = """
        CREATE TABLE IF NOT EXISTS "I" (
            id SERIAL PRIMARY KEY,
            user_input TEXT NOT NULL,
            is_relevant BOOLEAN,
            relevance_score FLOAT,
            subject_name TEXT,
            event_type TEXT,
            financial_data JSONB,
            event_model JSONB,
            analysis_result TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(query)
            self.conn.commit()
        logger.info("Table 'I' created or already exists.")

    def query_financial_data(self, subject_name: str) -> Optional[Dict[str, Any]]:
        """
        Query financial data by subject name.
        
        WARNING: This is a PLACEHOLDER implementation. Replace with actual database 
        queries based on your schema. The placeholder data is for demonstration only.
        
        Example implementation:
            query = "SELECT * FROM financial_data WHERE subject_name = %s"
            with self.conn.cursor() as cur:
                cur.execute(query, (subject_name,))
                result = cur.fetchone()
                return dict(result) if result else None
        """
        logger.info(f"Querying financial data for subject: {subject_name}")
        logger.warning("Using PLACEHOLDER financial data - implement actual query for production")
        return {"placeholder": "financial_data", "subject": subject_name}

    def query_event_model(self, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Query event model by event type.
        
        WARNING: This is a PLACEHOLDER implementation. Replace with actual database 
        queries based on your schema. The placeholder data is for demonstration only.
        
        Example implementation:
            query = "SELECT * FROM event_models WHERE event_type = %s"
            with self.conn.cursor() as cur:
                cur.execute(query, (event_type,))
                result = cur.fetchone()
                return dict(result) if result else None
        """
        logger.info(f"Querying event model for type: {event_type}")
        logger.warning("Using PLACEHOLDER event model - implement actual query for production")
        return {"placeholder": "event_model", "type": event_type}

    def insert_analysis_result(
        self,
        user_input: str,
        parsed_data: Dict[str, Any],
        financial_data: Optional[Dict[str, Any]],
        event_model: Optional[Dict[str, Any]],
        analysis_result: str,
    ) -> Optional[int]:
        """Insert analysis result into table 'I' and return the new ID."""
        if not self.conn:
            logger.warning("No database connection, result not stored")
            return None
        
        query = """
        INSERT INTO "I" (
            user_input, is_relevant, relevance_score, subject_name, event_type,
            financial_data, event_model, analysis_result
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (
                user_input,
                parsed_data.get("is_relevant"),
                parsed_data.get("relevance_score"),
                parsed_data.get("subject_name"),
                parsed_data.get("event_type"),
                json.dumps(financial_data) if financial_data else None,
                json.dumps(event_model) if event_model else None,
                analysis_result,
            ))
            result_id = cur.fetchone()[0]
            self.conn.commit()
        logger.info(f"Inserted analysis result with ID: {result_id}")
        return result_id


# ----------------------------
# Analysis Functions
# ----------------------------
def parse_user_input(runner: OpenAgentsLLMRunner, user_input: str) -> Dict[str, Any]:
    """Parse user input and extract key information using LLM."""
    prompt = f"""
请分析以下用户输入，提取关键信息。

用户输入：
"{user_input}"

请以JSON格式返回以下字段：
1. "is_relevant": 布尔值，表示输入是否与财务/事件分析相关
2. "relevance_score": 0.0到1.0之间的浮点数，表示相关程度
3. "subject_name": 字符串，提取的主体名称（如公司名、人名等），如无则为null
4. "event_type": 字符串，事件类型（如"财务报告"、"并购"、"诉讼"等），如无则为null

仅返回JSON对象，不要包含其他内容。
"""
    return runner.generate_json(prompt)


def analyze_with_context(
    runner: OpenAgentsLLMRunner,
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze input with context from database using LLM."""
    context = {
        "user_input": user_input,
        "parsed_info": parsed_data,
        "financial_data": financial_data,
        "event_model": event_model,
    }

    prompt = f"""
请基于以下信息进行综合分析：

{json.dumps(context, ensure_ascii=False, indent=2)}

请提供分析结果，以JSON格式返回，包含以下字段：
1. "summary": 简要总结（1-2句话）
2. "key_points": 关键要点列表（数组形式，每个要点为一个字符串）
3. "risk_assessment": 风险评估（如适用）
4. "recommendations": 建议列表（数组形式）
5. "confidence": 分析置信度（0.0-1.0）

仅返回JSON对象，不要包含其他内容。
"""
    return runner.generate_json(prompt)


def format_result_as_string(analysis_result: Dict[str, Any]) -> str:
    """Format the analysis result as a bullet-point string."""
    lines = []

    if "summary" in analysis_result:
        lines.append(f"摘要：{analysis_result['summary']}")

    if "key_points" in analysis_result and analysis_result["key_points"]:
        lines.append("\n关键要点：")
        for point in analysis_result["key_points"]:
            lines.append(f"  • {point}")

    if "risk_assessment" in analysis_result:
        lines.append(f"\n风险评估：{analysis_result['risk_assessment']}")

    if "recommendations" in analysis_result and analysis_result["recommendations"]:
        lines.append("\n建议：")
        for rec in analysis_result["recommendations"]:
            lines.append(f"  • {rec}")

    if "confidence" in analysis_result:
        lines.append(f"\n置信度：{analysis_result['confidence']}")

    return "\n".join(lines)


# ----------------------------
# OpenAgents Analysis Agent
# ----------------------------
class AnalysisAgent(WorkerAgent):
    """OpenAgents-based Analysis Agent for integrated financial analysis."""

    default_agent_id = "analyst"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_manager = None
        self.llm_runner = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize database and LLM components."""
        # Initialize database manager
        db_config = get_db_config()
        self.db_manager = DatabaseManager(db_config)
        if self.db_manager.connect():
            self.db_manager.create_result_table()

        # Initialize LLM runner using OpenAgents framework
        # API key environment variables (checked in order of priority):
        # 1. DEFAULT_LLM_API_KEY (OpenAgents standard)
        # 2. GEMINI_API_KEY or GOOGLE_API_KEY (for Gemini provider)
        # 3. OPENAI_API_KEY (for OpenAI provider)
        # 4. Variables starting with 'Y' (legacy)
        # 5. Variables starting with 'MILITAI' (legacy)
        if HAS_OPENAGENTS:
            try:
                self.llm_runner = OpenAgentsLLMRunner()
                logger.info("LLM runner initialized successfully using OpenAgents framework")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM runner: {e}")

    async def on_startup(self):
        """Called when agent starts up."""
        ws = self.workspace()
        await ws.channel("general").post("分析助手已上线！发送消息开始分析。")
        await ws.channel("analysis").post("分析频道已就绪，可以开始提交分析请求。")

    async def on_direct(self, context: EventContext):
        """Handle direct messages to the agent."""
        ws = self.workspace()
        user_input = context.content if hasattr(context, 'content') else str(context)
        
        try:
            result = await self._run_analysis(user_input)
            await ws.agent(context.source_id).send(result)
        except Exception as e:
            error_msg = f"分析过程中出错：{str(e)}"
            logger.error(f"Analysis failed: {e}")
            await ws.agent(context.source_id).send(error_msg)

    async def on_channel_post(self, context: ChannelMessageContext):
        """Handle channel messages."""
        # Skip messages from self
        if context.source_id == self.agent_id:
            return
        
        user_input = context.content if hasattr(context, 'content') else str(context)
        
        try:
            result = await self._run_analysis(user_input)
            ws = self.workspace()
            await ws.channel(context.channel_id).post(result)
        except Exception as e:
            error_msg = f"分析过程中出错：{str(e)}"
            logger.error(f"Analysis failed: {e}")
            ws = self.workspace()
            await ws.channel(context.channel_id).post(error_msg)

    async def _run_analysis(self, user_input: str) -> str:
        """Run the complete analysis workflow."""
        if not self.llm_runner:
            return "LLM 服务不可用，请检查 API 密钥配置。"

        # Step 1: Parse user input with LLM
        logger.info("Step 1: Parsing user input with LLM...")
        try:
            parsed_data = parse_user_input(self.llm_runner, user_input)
        except Exception as e:
            logger.error(f"Failed to parse user input: {e}")
            return f"解析用户输入失败：{str(e)}"
        
        logger.info(f"Parsed data: {json.dumps(parsed_data, ensure_ascii=False)}")

        # Step 2: Query database based on extracted information
        logger.info("Step 2: Querying database...")
        financial_data = None
        event_model = None

        subject_name = parsed_data.get("subject_name")
        event_type = parsed_data.get("event_type")

        if subject_name and self.db_manager:
            financial_data = self.db_manager.query_financial_data(subject_name)

        if event_type and self.db_manager:
            event_model = self.db_manager.query_event_model(event_type)

        # Step 3: Analyze with context from database
        logger.info("Step 3: Analyzing with LLM...")
        try:
            analysis_result = analyze_with_context(
                self.llm_runner, user_input, parsed_data, financial_data, event_model
            )
        except Exception as e:
            logger.error(f"Failed to analyze: {e}")
            return f"分析失败：{str(e)}"
        
        logger.info(f"Analysis result: {json.dumps(analysis_result, ensure_ascii=False)}")

        # Step 4: Format result as string
        result_string = format_result_as_string(analysis_result)

        # Step 5: Store result in database table "I"
        logger.info("Step 5: Storing result in database...")
        if self.db_manager:
            result_id = self.db_manager.insert_analysis_result(
                user_input, parsed_data, financial_data, event_model, result_string
            )
            if result_id:
                result_string += f"\n\n[结果已保存，ID: {result_id}]"

        return result_string

    def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()


# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """Main entry point for the Analysis Agent."""
    if not HAS_OPENAGENTS:
        print("错误：openagents 包未安装。请运行：pip install openagents")
        print("\n备用方案：运行原始的 main.py 脚本")
        return

    # Check for API key
    api_key = get_api_key()
    if not api_key:
        print("警告：未找到 API 密钥。请设置以下环境变量之一：")
        print("  - DEFAULT_LLM_API_KEY (OpenAgents 标准)")
        print("  - GEMINI_API_KEY 或 GOOGLE_API_KEY (Gemini)")
        print("  - OPENAI_API_KEY (OpenAI)")

    # Create and start the agent
    try:
        agent = AnalysisAgent(
            agent_config=AgentConfig(
                model_name="auto",
                instruction="""
你是一个专业的金融分析助手。

你的职责：
1. 接收用户输入，分析其是否与财务/事件分析相关
2. 提取关键信息：主体名称、事件类型等
3. 基于提取的信息进行综合分析
4. 提供结构化的分析报告

保持专业、客观的语气，提供有价值的分析见解。
""",
            )
        )

        print("=" * 60)
        print("OpenAgents 分析代理")
        print("正在连接到网络...")
        print("=" * 60)

        agent.start(network_host="localhost", network_port=8700)
        agent.wait_for_stop()

    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise
    finally:
        if 'agent' in locals():
            agent.cleanup()


if __name__ == "__main__":
    main()
