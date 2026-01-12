"""
Author: S
Version:2.0

This script:
1. Accepts user input
2. Uses LLM to parse user input and extract: is_relevant, relevance_score, subject_name, event_type as JSON
3. Queries database based on subject_name and event_type (placeholder logic)
4. Sends retrieved data to LLM for analysis and outputs JSON result
5. Stores JSON result to database table "I"
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import psycopg2
from psycopg2.extras import execute_batch

from google import genai
from google.genai import types

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("integrated_analysis")
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
        logger.error(f"Missing database environment variable: {e}")
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


@dataclass(frozen=True)
class LlmConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    safety_settings: Sequence[types.SafetySetting] = tuple(DEFAULT_SAFETY_SETTINGS)


# ----------------------------
# Gemini LLM Runner
# ----------------------------
class GeminiRunner:
    """Reliability-first LLM runner with API key rotation."""

    def __init__(self, key_rotator: ApiKeyRotator) -> None:
        self._keys = key_rotator

    @staticmethod
    def _normalize_content(content: Union[str, Dict[str, Any], List[Any]]) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)

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

    def generate_text(
        self,
        prompt: str,
        llm_cfg: Optional[LlmConfig] = None,
    ) -> str:
        """Generate plain text content."""
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
                    ),
                )
        except Exception as e:
            raise RuntimeError(f"LLM call failed (model={llm_cfg.model}, key={key_name}): {e}") from e

        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("LLM returned empty text. Possible safety block or no candidates.")

        return text


# ----------------------------
# Database Operations
# ----------------------------
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


def query_financial_data(conn, subject_name: str) -> Optional[Dict[str, Any]]:
    """
    Query financial data by subject name.
    TODO: Implement actual query logic based on your database schema.
    """
    # Placeholder - replace with actual query
    logger.info(f"Querying financial data for subject: {subject_name}")
    # Example placeholder query:
    # query = "SELECT * FROM financial_data WHERE subject_name = %s"
    # with conn.cursor() as cur:
    #     cur.execute(query, (subject_name,))
    #     result = cur.fetchone()
    #     return result
    return {"placeholder": "financial_data", "subject": subject_name}


def query_event_model(conn, event_type: str) -> Optional[Dict[str, Any]]:
    """
    Query event model by event type.
    TODO: Implement actual query logic based on your database schema.
    """
    # Placeholder - replace with actual query
    logger.info(f"Querying event model for type: {event_type}")
    # Example placeholder query:
    # query = "SELECT * FROM event_models WHERE event_type = %s"
    # with conn.cursor() as cur:
    #     cur.execute(query, (event_type,))
    #     result = cur.fetchone()
    #     return result
    return {"placeholder": "event_model", "type": event_type}

def insert_analysis_result(
    conn,
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
    analysis_result: str,
) -> int:
    """Insert analysis result into table 'I' and return the new ID."""
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
def parse_user_input(runner: GeminiRunner, user_input: str) -> Dict[str, Any]:
    """
    Use LLM to parse user input and extract:
    - is_relevant: whether the input is relevant
    - relevance_score: relevance score (0.0-1.0)
    - subject_name: the subject/entity name mentioned
    - event_type: the type of event
    """
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
    runner: GeminiRunner,
    user_input: str,
    parsed_data: Dict[str, Any],
    financial_data: Optional[Dict[str, Any]],
    event_model: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Use LLM to analyze the input with context from database.
    Returns analysis result in JSON format.
    """
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
# Main Workflow
# ----------------------------
def run_integrated_analysis(
    user_input: str,
    runner: GeminiRunner,
    db_conn,
) -> Dict[str, Any]:
    """
    Main workflow that integrates database interaction and LLM analysis.

    Args:
        user_input: User's input text
        runner: GeminiRunner instance
        db_conn: Database connection

    Returns:
        Dictionary containing all analysis results
    """
    # Step 1: Parse user input with LLM
    logger.info("Step 1: Parsing user input with LLM...")
    parsed_data = parse_user_input(runner, user_input)
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
        runner, user_input, parsed_data, financial_data, event_model
    )
    logger.info(f"Analysis result: {json.dumps(analysis_result, ensure_ascii=False)}")

    # Step 4: Format result as string
    result_string = format_result_as_string(analysis_result)

    # Step 5: Store result in database table "I"
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
    """Main entry point for the integrated analysis script."""
    # Initialize database connection
    db_config = get_db_config()
    if not db_config:
        logger.error("Database configuration not available. Exiting.")
        return

    # Initialize API key rotator (adjust env var names as needed)
    api_key_names = [key for key in os.environ.keys() if key.startswith("Y")]
    if not api_key_names:
        # Fallback to alternative key names
        api_key_names = [key for key in os.environ.keys() if key.startswith("MILITAI")]

    if not api_key_names:
        logger.error("No API keys found in environment variables. Exiting.")
        return

    try:
        rotator = ApiKeyRotator(env_var_names=api_key_names)
        runner = GeminiRunner(rotator)

        # Connect to database
        conn = psycopg2.connect(**db_config)
        logger.info("Connected to database successfully.")

        # Create table if not exists
        create_result_table(conn)

        # Get user input
        print("=" * 60)
        print("综合分析系统")
        print("输入您要分析的内容（输入 'exit' 退出）：")
        print("=" * 60)

        while True:
            user_input = input("\n请输入：").strip()
            if user_input.lower() == "exit":
                print("退出系统。")
                break

            if not user_input:
                print("输入不能为空，请重新输入。")
                continue

            try:
                result = run_integrated_analysis(user_input, runner, conn)
                print("\n" + "=" * 60)
                print("分析结果：")
                print("=" * 60)
                print(result["result_string"])
                print(f"\n结果已保存，ID: {result['id']}")
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                print(f"分析失败：{e}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if "conn" in locals() and conn:
            conn.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
