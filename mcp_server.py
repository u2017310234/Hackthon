"""
MCP Server for Integrated Analysis
Author: S
Version: 1.2

This MCP server exposes analysis tools with multi-LLM provider support.

Supported LLM Providers (Priority Order):
1. Google Gemini (gemini-2.0-flash-exp, gemini-2.5-flash, etc.) - Priority 1
2. OpenAI (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.) - Priority 2
3. Alibaba Qwen (qwen-max, qwen-plus, qwen-turbo) - Priority 3 (Fallback)

Environment Variables - LLM Configuration:

Priority 1 - Gemini:
  GEMINI_API_KEY or GOOGLE_API_KEY: Google Gemini API key
  GEMINI_MODEL: Model name (default from LlmConfig)

Priority 2 - OpenAI:
  OPENAI_API_KEY: OpenAI API key
  OPENAI_MODEL: Model name (default: gpt-4o-mini)

Priority 3 - Alibaba Qwen:
  DASHSCOPE_API_KEY: Alibaba DashScope API key (https://dashscope.console.aliyun.com/)
  ALIBABA_MODEL: Model name (default: qwen-max, options: qwen-plus, qwen-turbo)
  ALIBABA_USE_DASHSCOPE: Use native DashScope SDK instead of OpenAI-compatible API (default: false)

Database Configuration (Optional):
  DAILY_DATABASE, DAILY_USER, DAILY_PASSWORD, DAILY_HOST, DAILY_PORT

Installation:
  # For Gemini support
  pip install google-genai
  
  # For OpenAI support
  pip install openai
  
  # For Alibaba Qwen support
  pip install openai  # For OpenAI-compatible API (recommended)
  # OR
  pip install dashscope  # For native SDK

Usage Examples:
  # With Gemini (Priority 1)
  export GEMINI_API_KEY="AIza..."
  python mcp_server.py
  
  # With OpenAI (Priority 2)
  export OPENAI_API_KEY="sk-proj-..."
  python mcp_server.py
  
  # With Alibaba Qwen (Priority 3)
  export DASHSCOPE_API_KEY="sk-..."
  python mcp_server.py

MCP Tools:
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

# Database support
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None

# Google Gemini support
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None
    types = None

# OpenAI support
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

# Alibaba DashScope support
try:
    import dashscope
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    dashscope = None

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("mcp_server")
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
# JSON Parsing Utility
# ----------------------------
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from text, handling markdown code blocks."""
    
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code block
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text
    json_object_pattern = r'\{.*\}'
    matches = re.findall(json_object_pattern, text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    raise RuntimeError(f"Failed to extract valid JSON from LLM response: {text[:200]}...")


# ----------------------------
# Gemini LLM Runner
# ----------------------------
class GeminiRunner:
    """Reliability-first LLM runner with API key rotation."""

    def __init__(self, key_rotator: ApiKeyRotator) -> None:
        if not HAS_GENAI:
            raise RuntimeError("google-genai package is not installed. Run: pip install google-genai")
        self._keys = key_rotator

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

        return extract_json_from_text(text)


# ----------------------------
# OpenAI LLM Runner
# ----------------------------
class OpenAIRunner:
    """OpenAI LLM Runner with JSON response support."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not HAS_OPENAI:
            raise RuntimeError("openai package is not installed. Run: pip install openai")
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
            return extract_json_from_text(content)
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed (model={self.model}): {e}") from e


# ----------------------------
# Alibaba Qwen LLM Runner
# ----------------------------
class AlibabaRunner:
    """Alibaba Qwen LLM Runner - supports both DashScope SDK and OpenAI-compatible API."""
    
    def __init__(self, api_key: str, model: str = "qwen-max", use_dashscope: bool = False):
        self.api_key = api_key
        self.model = model
        self.use_dashscope = use_dashscope
        
        if use_dashscope:
            if not HAS_DASHSCOPE:
                raise RuntimeError("dashscope package is not installed. Run: pip install dashscope")
            dashscope.api_key = api_key
            logger.info(f"Alibaba runner initialized with DashScope SDK (model={model})")
        else:
            if not HAS_OPENAI:
                raise RuntimeError("openai package is not installed. Run: pip install openai")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            logger.info(f"Alibaba runner initialized with OpenAI-compatible API (model={model})")
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate content and parse as JSON."""
        if self.use_dashscope:
            return self._generate_with_dashscope(prompt)
        else:
            return self._generate_with_openai_api(prompt)
    
    def _generate_with_dashscope(self, prompt: str) -> Dict[str, Any]:
        """Generate using native DashScope SDK."""
        try:
            from dashscope import Generation
            
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                result_format='message',
                temperature=0.0,
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"DashScope API error: {response.message}")
            
            content = response.output.choices[0].message.content
            return extract_json_from_text(content)
            
        except Exception as e:
            raise RuntimeError(f"Alibaba DashScope call failed (model={self.model}): {e}") from e
    
    def _generate_with_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Generate using OpenAI-compatible API."""
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
            return extract_json_from_text(content)
            
        except Exception as e:
            raise RuntimeError(f"Alibaba OpenAI-API call failed (model={self.model}): {e}") from e


# ----------------------------
# LLM Runner Singleton
# ----------------------------
_llm_runner = None


def get_llm_runner() -> Optional[Any]:
    """Get or create LLM runner singleton - supports multiple LLM providers.
    
    Priority order:
    1. Google Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
    2. OpenAI (OPENAI_API_KEY)
    3. Alibaba Qwen (DASHSCOPE_API_KEY)
    """
    global _llm_runner
    
    if _llm_runner is not None:
        return _llm_runner
    
    # Priority 1: Try Gemini first (existing logic preserved)
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
                logger.info("✅ Using Gemini as LLM provider (Priority 1)")
                return _llm_runner
            except Exception as e:
                logger.warning(f"Gemini initialization failed, trying next provider: {e}")
    
    # Priority 2: Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and HAS_OPENAI:
        try:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            _llm_runner = OpenAIRunner(api_key=openai_key, model=model)
            logger.info("✅ Using OpenAI as LLM provider (Priority 2)")
            return _llm_runner
        except Exception as e:
            logger.warning(f"OpenAI initialization failed, trying next provider: {e}")
    
    # Priority 3: Try Alibaba Qwen (fallback)
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    if dashscope_key:
        try:
            model = os.environ.get("ALIBABA_MODEL", "qwen-max")
            use_dashscope_sdk = os.environ.get("ALIBABA_USE_DASHSCOPE", "false").lower() == "true"
            
            # Check if we have required dependencies
            if use_dashscope_sdk and not HAS_DASHSCOPE:
                logger.warning("dashscope not installed, falling back to OpenAI-compatible API")
                use_dashscope_sdk = False
            
            if not use_dashscope_sdk and not HAS_OPENAI:
                raise RuntimeError("Neither dashscope nor openai package is installed for Alibaba Qwen")
            
            _llm_runner = AlibabaRunner(
                api_key=dashscope_key,
                model=model,
                use_dashscope=use_dashscope_sdk
            )
            logger.info("✅ Using Alibaba Qwen as LLM provider (Priority 3)")
            return _llm_runner
        except Exception as e:
            logger.warning(f"Alibaba Qwen initialization failed: {e}")
    
    # No provider available
    logger.error(
        "❌ No LLM provider available! Please set one of:\n"
        "  1. GEMINI_API_KEY or GOOGLE_API_KEY (Priority 1)\n"
        "  2. OPENAI_API_KEY (Priority 2)\n"
        "  3. DASHSCOPE_API_KEY (Priority 3)"
    )
    return None


# ----------------------------
# Database Operations
# ----------------------------
def create_result_table(conn) -> None:
    """Create table 'I' for storing analysis results (if not exists)."""
    if not conn:
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
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
    logger.info("Table 'I' created or already exists.")


def query_financial_data(conn, subject_name: str) -> Optional[Dict[str, Any]]:
    """
    Query financial data by subject name.
    TODO: Implement actual query logic based on your database schema.
    """
    logger.info(f"Querying financial data for subject: {subject_name}")
    return {"placeholder": "financial_data", "subject": subject_name}


def query_event_model(conn, event_type: str) -> Optional[Dict[str, Any]]:
    """
    Query event model by event type.
    TODO: Implement actual query logic based on your database schema.
    """
    logger.info(f"Querying event model for type: {event_type}")
    return {"placeholder": "event_model", "type": event_type}


def insert_analysis_result(
    conn,
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
    analysis_result: str,
) -> Optional[int]:
    """Insert analysis result into table 'I' and return the new ID."""
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
# Core Analysis Functions (MCP Tools)
# ----------------------------
def parse_user_input(user_input: str) -> Dict[str, Any]:
    """
    MCP Tool: Parse user input and extract key information.
    
    Uses LLM to extract:
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


def analyze_with_context(
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    MCP Tool: Analyze input with context from database.
    
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


def run_integrated_analysis(
    user_input: str,
    db_conn=None,
) -> Dict[str, Any]:
    """
    MCP Tool: Run the complete analysis workflow.

    Args:
        user_input: User's input text
        db_conn: Database connection (optional)

    Returns:
        Dictionary containing all analysis results
    """
    # Step 1: Parse user input with LLM
    logger.info("Step 1: Parsing user input with LLM...")
    parsed_data = parse_user_input(user_input)
    logger.info(f"Parsed data: {json.dumps(parsed_data, ensure_ascii=False)}")

    # Step 2: Query database based on extracted information
    logger.info("Step 2: Querying database...")
    financial_data = None
    event_model = None

    subject_name = parsed_data.get("subject_name")
    event_type = parsed_data.get("event_type")

    if subject_name and db_conn:
        financial_data = query_financial_data(db_conn, subject_name)

    if event_type and db_conn:
        event_model = query_event_model(db_conn, event_type)

    # Step 3: Analyze with context from database
    logger.info("Step 3: Analyzing with LLM...")
    analysis_result = analyze_with_context(
        user_input, parsed_data, financial_data, event_model
    )
    logger.info(f"Analysis result: {json.dumps(analysis_result, ensure_ascii=False)}")

    # Step 4: Format result as string
    result_string = format_result_as_string(analysis_result)

    # Step 5: Store result in database table "I"
    result_id = None
    if db_conn:
        logger.info("Step 5: Storing result in database...")
        result_id = insert_analysis_result(
            db_conn, user_input, parsed_data, financial_data, event_model, result_string
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
# Main Entry Point
# ----------------------------
def main():
    """Main entry point for the MCP server."""
    # Initialize LLM runner
    runner = get_llm_runner()
    if not runner:
        logger.error("Cannot start MCP server without LLM provider")
        return

    # Initialize database connection (optional)
    db_config = get_db_config()
    conn = None
    if db_config and HAS_PSYCOPG2:
        try:
            conn = psycopg2.connect(**db_config)
            logger.info("Connected to database successfully.")
            create_result_table(conn)
        except Exception as e:
            logger.warning(f"Failed to connect to database: {e}")
            logger.warning("Running without database support")

    # Interactive mode
    print("=" * 60)
    print("MCP Server - Integrated Analysis System")
    print("Multi-LLM Provider Support (Gemini / OpenAI / Alibaba Qwen)")
    print("输入您要分析的内容（输入 'exit' 退出）：")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入：").strip()
            if user_input.lower() == "exit":
                print("退出系统。")
                break

            if not user_input:
                print("输入不能为空，请重新输入。")
                continue

            try:
                result = run_integrated_analysis(user_input, conn)
                print("\n" + "=" * 60)
                print("分析结果：")
                print("=" * 60)
                print(result["result_string"])
                if result["id"]:
                    print(f"\n结果已保存，ID: {result['id']}")
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                print(f"分析失败：{e}")

    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
