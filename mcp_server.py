"""
MCP Server for Integrated Analysis
Author: S
Version: 1.1

This MCP server exposes the analysis tools with multi-LLM provider support.

Supported LLM Providers (Priority Order):
1. Alibaba Cloud 通义千问 (Qwen) - qwen-max, qwen-plus, qwen-turbo
2. OpenAI GPT - gpt-4o-mini, gpt-4o, gpt-3.5-turbo
3. Google Gemini - gemini-2.0-flash-exp

Environment Variables:

Alibaba Cloud (Priority 1):
- DASHSCOPE_API_KEY or ALIBABA_API_KEY: 阿里云API密钥 (required)
- ALIBABA_MODEL: 模型名称 (default: qwen-max)
  Available: qwen-max, qwen-plus, qwen-turbo, qwen-max-longcontext
- ALIBABA_USE_DASHSCOPE: 使用DashScope SDK (default: false, 使用OpenAI兼容接口)

OpenAI (Priority 2):
- OPENAI_API_KEY: OpenAI API key
- OPENAI_MODEL: Model name (default: gpt-4o-mini)

Gemini (Priority 3):
- GEMINI_API_KEY or GOOGLE_API_KEY: Gemini API key

Database Configuration (optional):
- DAILY_DATABASE, DAILY_USER, DAILY_PASSWORD, DAILY_HOST, DAILY_PORT

Usage Examples:

    # 使用阿里云通义千问 (推荐，国内访问快)
    export DASHSCOPE_API_KEY="sk-..."
    export ALIBABA_MODEL="qwen-max"
    python mcp_server.py
    
    # 使用阿里云 + DashScope SDK
    export DASHSCOPE_API_KEY="sk-..."
    export ALIBABA_USE_DASHSCOPE="true"
    pip install dashscope
    python mcp_server.py
    
    # 使用 OpenAI
    export OPENAI_API_KEY="sk-proj-..."
    export OPENAI_MODEL="gpt-4o-mini"
    python mcp_server.py
    
    # 使用 Gemini
    export GEMINI_API_KEY="AIza..."
    python mcp_server.py

Tools:
1. parse_user_input - Parse user input and extract key information
2. query_financial_data - Query financial data by subject name
3. query_event_model - Query event model by event type
4. analyze_with_context - Analyze with context from database
5. run_integrated_analysis - Run the complete analysis workflow
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Try importing database library
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None

# Try importing Gemini
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None

# Try importing OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

# Try importing Alibaba DashScope
try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    dashscope = None

# Try importing MCP
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Server = None
    Tool = None
    TextContent = None

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("mcp_integrated_analysis")
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
# OpenAI LLM Runner
# ----------------------------
class OpenAIRunner:
    """OpenAI LLM Runner with JSON response support."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not HAS_OPENAI:
            raise RuntimeError("openai package is not installed. Run: pip install openai")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"OpenAI runner initialized (model={model})")
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e


# ----------------------------
# Alibaba Cloud (DashScope SDK) LLM Runner
# ----------------------------
class AlibabaRunner:
    """Alibaba Cloud (通义千问/Qwen) LLM Runner with JSON response support."""
    
    def __init__(self, api_key: str, model: str = "qwen-max"):
        if not HAS_DASHSCOPE:
            raise RuntimeError("dashscope package is not installed. Run: pip install dashscope")
        dashscope.api_key = api_key
        self.model = model
        logger.info(f"Alibaba Cloud runner initialized (model={model})")
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
        try:
            from dashscope import Generation
            
            # Add JSON format instruction to prompt
            json_prompt = f"{prompt}\n\n请以JSON格式返回结果。"
            
            response = Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                    {"role": "user", "content": json_prompt}
                ],
                result_format='message',
                temperature=0.0,
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"API returned status {response.status_code}: {response.message}")
            
            content = response.output.choices[0].message.content
            
            # Try to extract JSON if wrapped in markdown code blocks
            if "```json" in content:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Alibaba Cloud response as JSON: {e}\nContent: {content}") from e
        except Exception as e:
            raise RuntimeError(f"Alibaba Cloud API call failed: {e}") from e


# ----------------------------
# Alibaba Cloud (OpenAI-Compatible) LLM Runner
# ----------------------------
class AlibabaOpenAIRunner:
    """Alibaba Cloud using OpenAI-compatible API endpoint."""
    
    def __init__(self, api_key: str, model: str = "qwen-max"):
        if not HAS_OPENAI:
            raise RuntimeError("openai package is not installed. Run: pip install openai")
        from openai import OpenAI
        
        # Alibaba Cloud OpenAI-compatible endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        logger.info(f"Alibaba Cloud runner initialized (OpenAI-compatible, model={model})")
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in content:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            return json.loads(content)
            
        except Exception as e:
            raise RuntimeError(f"Alibaba Cloud API call failed: {e}") from e


# ----------------------------
# Gemini LLM Runner
# ----------------------------
class GeminiRunner:
    """Reliability-first LLM runner with API key rotation."""

    def __init__(self, key_rotator: ApiKeyRotator) -> None:
        if not HAS_GENAI:
            raise RuntimeError("google-genai package is not installed. Run: pip install google-genai")
        self._keys = key_rotator
        logger.info("Gemini runner initialized")

    def generate_json(
        self,
        prompt: str,
        llm_cfg: Optional[LlmConfig] = None,
    ) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
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
# LLM Runner Singleton
# ----------------------------
_llm_runner = None


def get_llm_runner() -> Optional[Any]:
    """Get or create LLM runner singleton - supports multiple LLM providers."""
    global _llm_runner
    
    if _llm_runner is not None:
        return _llm_runner
    
    # Priority 1: Try Alibaba Cloud (通义千问)
    alibaba_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY")
    if alibaba_key:
        model = os.environ.get("ALIBABA_MODEL", "qwen-max")
        use_dashscope = os.environ.get("ALIBABA_USE_DASHSCOPE", "false").lower() == "true"
        
        # Try DashScope SDK first if requested and available
        if use_dashscope and HAS_DASHSCOPE:
            try:
                _llm_runner = AlibabaRunner(api_key=alibaba_key, model=model)
                logger.info("✅ Using Alibaba Cloud (DashScope SDK) as LLM provider")
                return _llm_runner
            except Exception as e:
                logger.warning(f"Failed to initialize Alibaba Cloud (DashScope): {e}")
        
        # Try OpenAI-compatible endpoint
        if HAS_OPENAI:
            try:
                _llm_runner = AlibabaOpenAIRunner(api_key=alibaba_key, model=model)
                logger.info("✅ Using Alibaba Cloud (OpenAI-compatible) as LLM provider")
                return _llm_runner
            except Exception as e:
                logger.warning(f"Failed to initialize Alibaba Cloud (OpenAI-compatible): {e}")
    
    # Priority 2: Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and HAS_OPENAI:
        try:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            _llm_runner = OpenAIRunner(api_key=openai_key, model=model)
            logger.info("✅ Using OpenAI as LLM provider")
            return _llm_runner
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
    
    # Priority 3: Try Gemini (existing logic)
    if HAS_GENAI:
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
                logger.info("✅ Using Gemini as LLM provider")
                return _llm_runner
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
    
    logger.error(
        "❌ No LLM provider available! Set one of:\n"
        "  - DASHSCOPE_API_KEY or ALIBABA_API_KEY (阿里云通义千问)\n"
        "  - OPENAI_API_KEY (OpenAI GPT)\n"
        "  - GEMINI_API_KEY (Google Gemini)"
    )
    return None


# ----------------------------
# Database Operations
# ----------------------------
_db_connection = None


def get_db_connection():
    """Get or create database connection singleton."""
    global _db_connection
    
    if _db_connection is not None:
        return _db_connection
    
    if not HAS_PSYCOPG2:
        logger.warning("psycopg2 is not installed, database operations will be skipped")
        return None
    
    db_config = get_db_config()
    if not db_config:
        logger.warning("Database configuration not available, database operations will be skipped")
        return None
    
    try:
        _db_connection = psycopg2.connect(**db_config)
        logger.info("Connected to database successfully")
        # Create result table if it doesn't exist
        create_result_table(_db_connection)
        return _db_connection
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


def create_result_table(conn) -> None:
    """Create table 'I' for storing analysis results (if not exists)."""
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
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
    logger.info("Table 'I' created or already exists.")


def query_financial_data(subject_name: str) -> Optional[Dict[str, Any]]:
    """
    Query financial data by subject name.
    TODO: Implement actual query logic based on your database schema.
    """
    conn = get_db_connection()
    if not conn:
        logger.warning("No database connection, returning placeholder data")
        return {"placeholder": "financial_data", "subject": subject_name}
    
    logger.info(f"Querying financial data for subject: {subject_name}")
    # Placeholder - replace with actual query
    return {"placeholder": "financial_data", "subject": subject_name}


def query_event_model(event_type: str) -> Optional[Dict[str, Any]]:
    """
    Query event model by event type.
    TODO: Implement actual query logic based on your database schema.
    """
    conn = get_db_connection()
    if not conn:
        logger.warning("No database connection, returning placeholder data")
        return {"placeholder": "event_model", "type": event_type}
    
    logger.info(f"Querying event model for type: {event_type}")
    # Placeholder - replace with actual query
    return {"placeholder": "event_model", "type": event_type}


def insert_analysis_result(
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
    analysis_result: str,
) -> Optional[int]:
    """Insert analysis result into table 'I' and return the new ID."""
    conn = get_db_connection()
    if not conn:
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
    with conn.cursor() as cur:
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
        conn.commit()
    logger.info(f"Inserted analysis result with ID: {result_id}")
    return result_id


# ----------------------------
# Core Analysis Functions
# ----------------------------
def parse_user_input_impl(user_input: str) -> Dict[str, Any]:
    """
    Use LLM to parse user input and extract:
    - is_relevant: whether the input is relevant
    - relevance_score: relevance score (0.0-1.0)
    - subject_name: the subject/entity name mentioned
    - event_type: the type of event
    """
    runner = get_llm_runner()
    if not runner:
        raise RuntimeError("No LLM provider available")
    
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
    """
    Use LLM to analyze the input with context from database.
    Returns analysis result in JSON format.
    """
    runner = get_llm_runner()
    if not runner:
        raise RuntimeError("No LLM provider available")
    
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
    """
    Main workflow that integrates database interaction and LLM analysis.

    Args:
        user_input: User's input text

    Returns:
        Dictionary containing all analysis results
    """
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

    if subject_name:
        financial_data = query_financial_data(subject_name)

    if event_type:
        event_model = query_event_model(event_type)

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
    result_id = insert_analysis_result(
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
if HAS_MCP:
    server = Server("integrated-analysis")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="parse_user_input",
                description="Parse user input and extract key information (is_relevant, relevance_score, subject_name, event_type)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "User input text to parse",
                        }
                    },
                    "required": ["user_input"],
                },
            ),
            Tool(
                name="query_financial_data",
                description="Query financial data by subject name from the database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_name": {
                            "type": "string",
                            "description": "Subject name to query financial data for",
                        }
                    },
                    "required": ["subject_name"],
                },
            ),
            Tool(
                name="query_event_model",
                description="Query event model by event type from the database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "Event type to query model for",
                        }
                    },
                    "required": ["event_type"],
                },
            ),
            Tool(
                name="analyze_with_context",
                description="Analyze user input with context from database using LLM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "User input text to analyze",
                        },
                        "parsed_data": {
                            "type": "object",
                            "description": "Parsed data from parse_user_input",
                        },
                        "financial_data": {
                            "type": "object",
                            "description": "Financial data from database (optional)",
                        },
                        "event_model": {
                            "type": "object",
                            "description": "Event model from database (optional)",
                        },
                    },
                    "required": ["user_input", "parsed_data"],
                },
            ),
            Tool(
                name="run_integrated_analysis",
                description="Run the complete analysis workflow: parse input, query database, analyze, and store results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "User input text to analyze",
                        }
                    },
                    "required": ["user_input"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "parse_user_input":
                result = parse_user_input_impl(arguments["user_input"])
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
            
            elif name == "query_financial_data":
                result = query_financial_data(arguments["subject_name"])
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
            
            elif name == "query_event_model":
                result = query_event_model(arguments["event_type"])
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
            
            elif name == "analyze_with_context":
                result = analyze_with_context_impl(
                    arguments["user_input"],
                    arguments["parsed_data"],
                    arguments.get("financial_data"),
                    arguments.get("event_model"),
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
            
            elif name == "run_integrated_analysis":
                result = run_integrated_analysis_impl(arguments["user_input"])
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )]


# ----------------------------
# Main Entry Point
# ----------------------------
async def main():
    """Main entry point for the MCP server."""
    if not HAS_MCP:
        logger.error("mcp package is not installed. Run: pip install mcp")
        return
    
    # Initialize LLM runner
    runner = get_llm_runner()
    if not runner:
        logger.error("Failed to initialize any LLM provider. Please check your API keys.")
        return
    
    # Initialize database connection (optional)
    get_db_connection()
    
    # Run the MCP server
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
