"""
MCP Server for Integrated Analysis
Author: S
Version: 1.0

This MCP server exposes the analysis tools from main.py:
1. parse_user_input - Parse user input and extract key information
2. query_financial_data - Query financial data by subject name
3. query_event_model - Query event model by event type
4. analyze_with_context - Analyze with context from database
5. run_integrated_analysis - Run the complete analysis workflow

Usage:
    # Run as stdio server (for MCP clients)
    python mcp_server.py
    
    # Or use with uvicorn for HTTP transport
    uvicorn mcp_server:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Optional, Sequence, Tuple

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("mcp_analysis_server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ----------------------------
# Optional imports with graceful fallback
# ----------------------------
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None


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
# API Key Rotation (thread-safe)
# ----------------------------
class ApiKeyRotator:
    """Thread-safe round-robin over env var names that store API keys."""

    def __init__(self, env_var_names: Sequence[str]) -> None:
        if not env_var_names:
            raise ValueError("env_var_names must not be empty")
        self._env_names = list(env_var_names)
        self._pool = cycle(self._env_names)
        self._lock = threading.Lock()

    def next_key(self) -> Tuple[str, str]:
        with self._lock:
            name = next(self._pool)
        value = os.environ.get(name, "")
        if not value:
            raise RuntimeError(f"Missing or empty API key env var: {name}")
        return name, value


# ----------------------------
# LLM Configuration
# ----------------------------
if HAS_GENAI:
    DEFAULT_SAFETY_SETTINGS = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]
else:
    DEFAULT_SAFETY_SETTINGS = []


@dataclass(frozen=True)
class LlmConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    safety_settings: Sequence = tuple(DEFAULT_SAFETY_SETTINGS)


# ----------------------------
# Gemini LLM Runner
# ----------------------------
class GeminiRunner:
    """Reliability-first LLM runner with API key rotation."""

    def __init__(self, key_rotator: ApiKeyRotator) -> None:
        self._keys = key_rotator

    def generate_json(
        self,
        prompt: str,
        llm_cfg: Optional[LlmConfig] = None,
    ) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
        if not HAS_GENAI:
            raise RuntimeError("google-genai package is not installed")
        
        llm_cfg = llm_cfg or LlmConfig()
        key_name, api_key = self._keys.next_key()
        logger.info("Calling LLM (key=%s, model=%s)", key_name, llm_cfg.model)

        try:
            with genai.Client(api_key=api_key) as client:
                resp = client.models.generate_content(
                    model=llm_cfg.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=llm_cfg.temperature,
                        safety_settings=list(llm_cfg.safety_settings),
                        response_mime_type="application/json",
                    ),
                )
        except Exception as e:
            raise RuntimeError(f"LLM call failed (model={llm_cfg.model}, key={key_name}): {e}") from e

        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("LLM returned empty text. Possible safety block or no candidates.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {e}") from e


# ----------------------------
# Database Manager
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
        """Query financial data by subject name."""
        logger.info(f"Querying financial data for subject: {subject_name}")
        logger.warning("Using PLACEHOLDER financial data - implement actual query for production")
        return {"placeholder": "financial_data", "subject": subject_name}

    def query_event_model(self, event_type: str) -> Optional[Dict[str, Any]]:
        """Query event model by event type."""
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
# Global instances (initialized lazily)
# ----------------------------
_db_manager: Optional[DatabaseManager] = None
_llm_runner: Optional[GeminiRunner] = None


def get_db_manager() -> Optional[DatabaseManager]:
    """Get or create database manager singleton."""
    global _db_manager
    if _db_manager is None:
        db_config = get_db_config()
        _db_manager = DatabaseManager(db_config)
        if _db_manager.connect():
            _db_manager.create_result_table()
    return _db_manager


def get_llm_runner() -> Optional[GeminiRunner]:
    """Get or create LLM runner singleton."""
    global _llm_runner
    if _llm_runner is None and HAS_GENAI:
        api_key_names = []
        
        if os.environ.get("GEMINI_API_KEY"):
            api_key_names.append("GEMINI_API_KEY")
        if os.environ.get("GOOGLE_API_KEY"):
            api_key_names.append("GOOGLE_API_KEY")
        
        if not api_key_names:
            api_key_names = [key for key in os.environ.keys() if key.startswith("Y")]
        if not api_key_names:
            api_key_names = [key for key in os.environ.keys() if key.startswith("MILITAI")]
        
        if api_key_names:
            try:
                rotator = ApiKeyRotator(env_var_names=api_key_names)
                _llm_runner = GeminiRunner(rotator)
                logger.info("LLM runner initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM runner: {e}")
    return _llm_runner


# ----------------------------
# Analysis Functions
# ----------------------------
def parse_user_input_impl(user_input: str) -> Dict[str, Any]:
    """Parse user input and extract key information using LLM."""
    runner = get_llm_runner()
    if not runner:
        raise RuntimeError("LLM runner not available. Check API key configuration.")
    
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


def analyze_with_context_impl(
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze input with context from database using LLM."""
    runner = get_llm_runner()
    if not runner:
        raise RuntimeError("LLM runner not available. Check API key configuration.")
    
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


def run_integrated_analysis_impl(user_input: str) -> Dict[str, Any]:
    """Run the complete analysis workflow."""
    db_manager = get_db_manager()
    
    # Step 1: Parse user input with LLM
    logger.info("Step 1: Parsing user input with LLM...")
    parsed_data = parse_user_input_impl(user_input)
    logger.info(f"Parsed data: {json.dumps(parsed_data, ensure_ascii=False)}")

    # Step 2: Query database based on extracted information
    logger.info("Step 2: Querying database...")
    financial_data = None
    event_model = None

    subject_name = parsed_data.get("subject_name")
    event_type = parsed_data.get("event_type")

    if subject_name and db_manager:
        financial_data = db_manager.query_financial_data(subject_name)

    if event_type and db_manager:
        event_model = db_manager.query_event_model(event_type)

    # Step 3: Analyze with context from database
    logger.info("Step 3: Analyzing with LLM...")
    analysis_result = analyze_with_context_impl(
        user_input, parsed_data, financial_data, event_model
    )
    logger.info(f"Analysis result: {json.dumps(analysis_result, ensure_ascii=False)}")

    # Step 4: Format result as string
    result_string = format_result_as_string(analysis_result)

    # Step 5: Store result in database table "I"
    logger.info("Step 5: Storing result in database...")
    result_id = None
    if db_manager:
        result_id = db_manager.insert_analysis_result(
            user_input, parsed_data, financial_data, event_model, result_string
        )

    return {
        "id": result_id,
        "parsed_data": parsed_data,
        "financial_data": financial_data,
        "event_model": event_model,
        "analysis_result": analysis_result,
        "result_string": result_string,
    }


# ----------------------------
# MCP Server Setup
# ----------------------------
app = Server("analysis-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="parse_user_input",
            description="解析用户输入，提取关键信息（is_relevant, relevance_score, subject_name, event_type）",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "用户输入的文本"
                    }
                },
                "required": ["user_input"]
            }
        ),
        Tool(
            name="query_financial_data",
            description="根据主体名称查询财务数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject_name": {
                        "type": "string",
                        "description": "主体名称（如公司名）"
                    }
                },
                "required": ["subject_name"]
            }
        ),
        Tool(
            name="query_event_model",
            description="根据事件类型查询事件模型",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "事件类型（如财务报告、并购、诉讼等）"
                    }
                },
                "required": ["event_type"]
            }
        ),
        Tool(
            name="analyze_with_context",
            description="基于上下文信息进行综合分析",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "用户输入的文本"
                    },
                    "parsed_data": {
                        "type": "object",
                        "description": "解析后的数据（来自parse_user_input）"
                    },
                    "financial_data": {
                        "type": "object",
                        "description": "财务数据（可选）"
                    },
                    "event_model": {
                        "type": "object",
                        "description": "事件模型（可选）"
                    }
                },
                "required": ["user_input", "parsed_data"]
            }
        ),
        Tool(
            name="run_integrated_analysis",
            description="运行完整的综合分析流程：解析输入 -> 查询数据库 -> 综合分析 -> 存储结果",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "用户输入的文本"
                    }
                },
                "required": ["user_input"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "parse_user_input":
            user_input = arguments.get("user_input", "")
            if not user_input:
                return [TextContent(type="text", text="错误：user_input 不能为空")]
            
            result = parse_user_input_impl(user_input)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "query_financial_data":
            subject_name = arguments.get("subject_name", "")
            if not subject_name:
                return [TextContent(type="text", text="错误：subject_name 不能为空")]
            
            db_manager = get_db_manager()
            if not db_manager:
                return [TextContent(type="text", text="警告：数据库未连接")]
            
            result = db_manager.query_financial_data(subject_name)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "query_event_model":
            event_type = arguments.get("event_type", "")
            if not event_type:
                return [TextContent(type="text", text="错误：event_type 不能为空")]
            
            db_manager = get_db_manager()
            if not db_manager:
                return [TextContent(type="text", text="警告：数据库未连接")]
            
            result = db_manager.query_event_model(event_type)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "analyze_with_context":
            user_input = arguments.get("user_input", "")
            parsed_data = arguments.get("parsed_data", {})
            financial_data = arguments.get("financial_data")
            event_model = arguments.get("event_model")
            
            if not user_input:
                return [TextContent(type="text", text="错误：user_input 不能为空")]
            if not parsed_data:
                return [TextContent(type="text", text="错误：parsed_data 不能为空")]
            
            result = analyze_with_context_impl(user_input, parsed_data, financial_data, event_model)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "run_integrated_analysis":
            user_input = arguments.get("user_input", "")
            if not user_input:
                return [TextContent(type="text", text="错误：user_input 不能为空")]
            
            result = run_integrated_analysis_impl(user_input)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        else:
            return [TextContent(type="text", text=f"错误：未知工具 '{name}'")]
    
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [TextContent(type="text", text=f"错误：{str(e)}")]


# ----------------------------
# Main Entry Point
# ----------------------------
async def main():
    """Run the MCP server with stdio transport."""
    logger.info("Starting MCP Analysis Server...")
    
    # Pre-initialize components
    get_db_manager()
    get_llm_runner()
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
