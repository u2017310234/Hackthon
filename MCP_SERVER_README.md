# MCP Server for Integrated Analysis

This document provides additional information about the new `mcp_server.py` module that adds multi-LLM provider support to the Hackthon project.

## Overview

The MCP (Model Context Protocol) Server exposes the integrated analysis tools with support for multiple LLM providers and automatic fallback mechanism.

## Features

- ✅ **Multi-LLM Provider Support**: OpenAI and Google Gemini
- ✅ **Automatic Fallback**: Tries OpenAI first, falls back to Gemini if unavailable
- ✅ **Optional Database**: Works with or without database configuration
- ✅ **API Key Rotation**: Supports multiple API keys for load balancing
- ✅ **Graceful Error Handling**: Clear error messages for debugging

## Supported LLM Providers

### Priority 1: OpenAI
- Models: gpt-4o-mini (default), gpt-4o, gpt-3.5-turbo, etc.
- Environment variables:
  - `OPENAI_API_KEY`: Your OpenAI API key (required)
  - `OPENAI_MODEL`: Model name (optional, default: gpt-4o-mini)

### Priority 2: Google Gemini
- Models: gemini-2.5-flash (default), gemini-2.0-flash-exp, etc.
- Environment variables:
  - `GEMINI_API_KEY`: Your Gemini API key (recommended)
  - `GOOGLE_API_KEY`: Alternative Gemini API key
  - Legacy support: `Y*` or `MILITAI*` prefixed variables

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install optional dependencies if needed
pip install openai>=1.0.0          # For OpenAI support
pip install google-genai>=0.5.0    # For Gemini support
pip install psycopg2-binary>=2.9.0 # For database support
```

## Configuration

### Required Environment Variables

At least ONE of the following:
```bash
# Option 1: Use OpenAI
export OPENAI_API_KEY="sk-proj-..."
export OPENAI_MODEL="gpt-4o-mini"  # Optional

# Option 2: Use Gemini
export GEMINI_API_KEY="AIza..."
# or
export GOOGLE_API_KEY="AIza..."
```

### Optional Environment Variables

Database configuration (optional):
```bash
export DAILY_DATABASE="your_database"
export DAILY_USER="your_user"
export DAILY_PASSWORD="your_password"
export DAILY_HOST="your_host"
export DAILY_PORT="5432"
```

## Usage

### Command Line Interface

Run the MCP server in interactive mode:

```bash
python mcp_server.py
```

This will start the server and prompt you for input:

```
MCP Server started. Available tools:
1. parse_user_input
2. query_financial_data
3. query_event_model
4. analyze_with_context
5. run_integrated_analysis

Enter user input for analysis (or 'exit' to quit):
请输入：
```

### Programmatic Usage

Import and use the MCP server functions in your Python code:

```python
import mcp_server

# Get LLM runner (automatically selects best available provider)
runner = mcp_server.get_llm_runner()

if runner:
    # Parse user input
    parsed = mcp_server.parse_user_input("分析特斯拉的财务报告", runner)
    print(parsed)
    
    # Run complete analysis
    result = mcp_server.run_integrated_analysis(
        "分析特斯拉的财务报告",
        runner=runner,
        db_conn=None  # Optional database connection
    )
    print(result["result_string"])
```

## Available Tools

### 1. parse_user_input
Parse user input and extract key information.

**Parameters:**
- `user_input` (str): User's input text
- `runner` (optional): LLM runner instance

**Returns:**
- Dictionary with: `is_relevant`, `relevance_score`, `subject_name`, `event_type`

### 2. query_financial_data
Query financial data by subject name.

**Parameters:**
- `conn`: Database connection
- `subject_name` (str): Subject name to query

**Returns:**
- Dictionary with financial data (placeholder if no DB)

### 3. query_event_model
Query event model by event type.

**Parameters:**
- `conn`: Database connection
- `event_type` (str): Event type to query

**Returns:**
- Dictionary with event model data (placeholder if no DB)

### 4. analyze_with_context
Analyze input with context from database.

**Parameters:**
- `user_input` (str): User's input text
- `parsed_data` (dict): Parsed information
- `financial_data` (dict): Financial data
- `event_model` (dict): Event model data
- `runner` (optional): LLM runner instance

**Returns:**
- Dictionary with: `summary`, `key_points`, `risk_assessment`, `recommendations`, `confidence`

### 5. run_integrated_analysis
Run the complete analysis workflow.

**Parameters:**
- `user_input` (str): User's input text
- `runner` (optional): LLM runner instance
- `db_conn` (optional): Database connection

**Returns:**
- Dictionary with all analysis results

## Provider Selection Logic

The `get_llm_runner()` function follows this priority order:

1. **Check for OpenAI** (`OPENAI_API_KEY`)
   - If found and package installed → Use OpenAI
   - If found but package missing → Log warning, try next

2. **Check for Gemini** (in order):
   - `GEMINI_API_KEY`
   - `GOOGLE_API_KEY`
   - Legacy: `Y*` prefixed variables
   - Legacy: `MILITAI*` prefixed variables
   - If found and package installed → Use Gemini
   - If found but package missing → Log warning

3. **No Provider Available**
   - Log error message
   - Return `None`

## Error Handling

The server provides clear error messages for common issues:

```
❌ No LLM provider available! Set OPENAI_API_KEY or GEMINI_API_KEY
```

This means:
1. No API keys found in environment variables, OR
2. API keys found but required packages not installed

**Solution:**
1. Set at least one API key environment variable
2. Install required package: `pip install openai` or `pip install google-genai`

## Differences from main.py and analysis_agent.py

### mcp_server.py
- **Purpose**: MCP server with multi-LLM support
- **Features**: OpenAI + Gemini, automatic fallback
- **Usage**: Standalone server or programmatic API

### main.py
- **Purpose**: Original standalone script
- **Features**: Gemini only
- **Usage**: Command-line interactive mode

### analysis_agent.py
- **Purpose**: OpenAgents framework integration
- **Features**: Gemini only, network-based
- **Usage**: Part of OpenAgents network

## Migration Guide

### From main.py to mcp_server.py

No changes needed! The MCP server is backward compatible:

```bash
# Before (main.py)
export GEMINI_API_KEY="AIza..."
python main.py

# After (mcp_server.py) - works the same
export GEMINI_API_KEY="AIza..."
python mcp_server.py
```

### Adding OpenAI Support

Just set the OpenAI API key - it takes priority:

```bash
# Add OpenAI (takes priority over Gemini)
export OPENAI_API_KEY="sk-proj-..."
export GEMINI_API_KEY="AIza..."  # Fallback
python mcp_server.py
```

## Testing

Run the implementation tests:

```bash
python -c "import mcp_server; print('✅ MCP Server loads correctly')"
```

## License

Same as the main project (Apache 2.0).
