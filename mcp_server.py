"""
MCP Server for Integrated Analysis
Author: S
Version: 1.1

This MCP server exposes the analysis tools with multi-LLM provider support:

Supported LLM Providers:
1. OpenAI (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
2. Google Gemini (gemini-2.0-flash-exp, etc.)

Environment Variables:
- OPENAI_API_KEY: OpenAI API key (priority 1)
- OPENAI_MODEL: Model name (default: gpt-4o-mini)
- GEMINI_API_KEY: Gemini API key (priority 2)
- GOOGLE_API_KEY: Alternative Gemini API key

Database Configuration (optional):
- DAILY_DATABASE, DAILY_USER, DAILY_PASSWORD, DAILY_HOST, DAILY_PORT

Usage:
    # With OpenAI
    export OPENAI_API_KEY="sk-proj-..."
    python mcp_server.py
    
    # With Gemini
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
import threading
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Optional database support
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None

# Gemini support
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
            raise ValueError("At least one environment variable name must be provided for API key rotation")
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
    
    # Priority 1: Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and HAS_OPENAI:
        try:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            _llm_runner = OpenAIRunner(api_key=openai_key, model=model)
            logger.info("✅ Using OpenAI as LLM provider")
            return _llm_runner
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
    
    # Priority 2: Try Gemini (existing logic)
    if HAS_GENAI:
        api_key_names = []
        
        # Check standard Gemini environment variables
        if os.environ.get("GEMINI_API_KEY"):
            api_key_names.append("GEMINI_API_KEY")
        if os.environ.get("GOOGLE_API_KEY"):
            api_key_names.append("GOOGLE_API_KEY")
        
        # Legacy compatibility: Check for custom-named API keys
        # These patterns are for backward compatibility with existing deployments
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
    
    logger.error("❌ No LLM provider available! Set OPENAI_API_KEY or GEMINI_API_KEY")
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
    if not conn:
        logger.warning("No database connection available")
        return {"placeholder": "financial_data", "subject": subject_name}
    
    # Placeholder - replace with actual query
    return {"placeholder": "financial_data", "subject": subject_name}


def query_event_model(conn, event_type: str) -> Optional[Dict[str, Any]]:
    """
    Query event model by event type.
    TODO: Implement actual query logic based on your database schema.
    """
    logger.info(f"Querying event model for type: {event_type}")
    if not conn:
        logger.warning("No database connection available")
        return {"placeholder": "event_model", "type": event_type}
    
    # Placeholder - replace with actual query
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
def parse_user_input(user_input: str, runner=None) -> Dict[str, Any]:
    """
    Parse user input and extract key information.
    
    MCP Tool: parse_user_input
    
    Args:
        user_input: User's input text
        runner: LLM runner instance (optional, will use singleton if not provided)
    
    Returns:
        Dictionary with: is_relevant, relevance_score, subject_name, event_type
    """
    if runner is None:
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
    runner=None,
) -> Dict[str, Any]:
    """
    Analyze input with context from database.
    
    MCP Tool: analyze_with_context
    
    Args:
        user_input: User's input text
        parsed_data: Parsed information from user input
        financial_data: Financial data from database
        event_model: Event model from database
        runner: LLM runner instance (optional, will use singleton if not provided)
    
    Returns:
        Analysis result dictionary with: summary, key_points, risk_assessment, recommendations, confidence
    """
    if runner is None:
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
    runner=None,
    db_conn=None,
) -> Dict[str, Any]:
    """
    Run the complete analysis workflow.
    
    MCP Tool: run_integrated_analysis
    
    Args:
        user_input: User's input text
        runner: LLM runner instance (optional, will use singleton if not provided)
        db_conn: Database connection (optional)
    
    Returns:
        Dictionary containing all analysis results
    """
    if runner is None:
        runner = get_llm_runner()
    
    if not runner:
        raise RuntimeError("No LLM provider available")
    
    # Step 1: Parse user input with LLM
    logger.info("Step 1: Parsing user input with LLM...")
    parsed_data = parse_user_input(user_input, runner)
    logger.info(f"Parsed data: {json.dumps(parsed_data, ensure_ascii=False)}")

    # Step 2: Query database based on extracted information
    logger.info("Step 2: Querying database...")
    financial_data = None
    event_model = None

    subject_name = parsed_data.get("subject_name")
    event_type = parsed_data.get("event_type")

    if subject_name:
        financial_data = query_financial_data(db_conn, subject_name)

    if event_type:
        event_model = query_event_model(db_conn, event_type)

    # Step 3: Analyze with context from database
    logger.info("Step 3: Analyzing with LLM...")
    analysis_result = analyze_with_context(
        user_input, parsed_data, financial_data, event_model, runner
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
    logger.info("=" * 60)
    logger.info("MCP Server for Integrated Analysis")
    logger.info("Version: 1.1")
    logger.info("=" * 60)
    
    # Check LLM availability
    runner = get_llm_runner()
    if not runner:
        logger.error("No LLM provider available. Please set OPENAI_API_KEY or GEMINI_API_KEY")
        return
    
    # Initialize database connection (optional)
    db_config = get_db_config()
    db_conn = None
    if db_config and HAS_PSYCOPG2:
        try:
            db_conn = psycopg2.connect(**db_config)
            logger.info("Connected to database successfully.")
            create_result_table(db_conn)
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.warning("Continuing without database support...")
    else:
        logger.info("Database not configured or psycopg2 not installed. Continuing without database...")
    
    # Interactive mode
    logger.info("\nMCP Server started. Available tools:")
    logger.info("1. parse_user_input")
    logger.info("2. query_financial_data")
    logger.info("3. query_event_model")
    logger.info("4. analyze_with_context")
    logger.info("5. run_integrated_analysis")
    logger.info("\nEnter user input for analysis (or 'exit' to quit):")
    
    try:
        while True:
            user_input = input("\n请输入：").strip()
            if user_input.lower() == "exit":
                logger.info("Exiting MCP Server.")
                break

            if not user_input:
                print("输入不能为空，请重新输入。")
                continue

            try:
                result = run_integrated_analysis(user_input, runner, db_conn)
                print("\n" + "=" * 60)
                print("分析结果：")
                print("=" * 60)
                print(result["result_string"])
                if result.get("id"):
                    print(f"\n结果已保存，ID: {result['id']}")
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                print(f"分析失败：{e}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    finally:
        if db_conn:
            db_conn.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
