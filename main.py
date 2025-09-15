import os
import json
import logging
from time import perf_counter
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from openai import OpenAI

# FastMCP server + in-memory client and Context for server-to-client logs
from fastmcp import FastMCP, Client, Context  # pip install fastmcp

# ---------------- Strict system prompt ----------------
SYSTEM_PROMPT = (
    "You must only use the provided function tools to answer.\n"
    "- Always try to call one or more tools; do not solve anything yourself.\n"
    "- Do not write any explanatory text.\n"
    "- After tools return, respond using the tool output. Do not deviate. Just use tool output to formulate your response\n"
    "- If a tool fails or returns an error, return that message exactly.\n"
    "- Never include extra words, formatting, or commentary."
    "- Just respond out of scope if no tools were used. Don't use tools if not needed. Just say out of scope"
)

# ---------------- Logging setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mcp-calculator")

# ---------------- MCP server (calculator tools) ----------------
mcp = FastMCP("calculator-mcp")

@mcp.tool
async def add(a: float, b: float, ctx: Context) -> str:
    """Add two numbers"""
    await ctx.info(f"add called with a={a}, b={b}")
    result = a + b
    logger.info(f"mcp.add({a}, {b}) -> {result}")
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("Hello, world!\n")
    return str(result)

@mcp.tool
async def subtract(a: float, b: float, ctx: Context) -> str:
    """Subtract b from a"""
    await ctx.info(f"subtract called with a={a}, b={b}")
    result = a - b
    logger.info(f"mcp.subtract({a}, {b}) -> {result}")
    return str(result)

@mcp.tool
async def multiply(a: float, b: float, ctx: Context) -> str:
    """Multiply two numbers"""
    await ctx.info(f"multiply called with a={a}, b={b}")
    result = a * b
    logger.info(f"mcp.multiply({a}, {b}) -> {result}")
    return str(result)

@mcp.tool
async def divide(a: float, b: float, ctx: Context) -> str:
    """Divide a by b (error if b == 0)"""
    await ctx.info(f"divide called with a={a}, b={b}")
    if b == 0:
        await ctx.error("division by zero attempted")
        logger.warning(f"mcp.divide({a}, {b}) -> division by zero")
        return "Division by zero is not allowed"
    result = a / b
    logger.info(f"mcp.divide({a}, {b}) -> {result}")
    return str(result)

# ---------------- FastAPI app + logging middleware ----------------
app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = perf_counter()
        method = request.method
        path = request.url.path
        logger.info(f"HTTP {method} {path} -> start")
        try:
            response = await call_next(request)
            duration_ms = (perf_counter() - start) * 1000
            logger.info(f"HTTP {method} {path} -> {response.status_code} in {duration_ms:.1f}ms")
            return response
        except Exception as e:
            duration_ms = (perf_counter() - start) * 1000
            logger.exception(f"HTTP {method} {path} -> EXC {e} after {duration_ms:.1f}ms")
            raise

app.add_middleware(LoggingMiddleware)

# ---------------- Request model ----------------
class QueryBody(BaseModel):
    query: str

# ---------------- Function tool schemas for the model ----------------
def calculator_function_tools() -> List[Dict[str, Any]]:
    params = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"],
        "additionalProperties": False
    }
    return [
        {"type": "function", "name": "add", "description": "Add two numbers", "parameters": params},
        {"type": "function", "name": "subtract", "description": "Subtract b from a", "parameters": params},
        {"type": "function", "name": "multiply", "description": "Multiply two numbers", "parameters": params},
        {"type": "function", "name": "divide", "description": "Divide a by b", "parameters": params},
    ]

def extract_function_calls(resp: Any) -> List[Dict[str, Any]]:
    """Collect function calls from Responses API output."""
    calls: List[Dict[str, Any]] = []
    for item in getattr(resp, "output", []) or []:
        itype = getattr(item, "type", None)
        if itype in ("function_call", "tool_call"):
            calls.append({
                "id": getattr(item, "call_id", getattr(item, "id", None)),
                "name": getattr(item, "name", None),
                "arguments": json.loads(getattr(item, "arguments", "{}") or "{}"),
            })
    return calls

# ---------------- POST /query (single-turn) ----------------
@app.post("/query")
async def query(body: QueryBody):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    tools = calculator_function_tools()

    # Create a response; model may emit function_call items
    response = client.responses.create(
        model=model,
        tools=tools,
        # tool_choice="required",  # optional: force a tool call
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": body.query}],
    )

    calls = extract_function_calls(response)

    # If the model didn’t emit any tool call, return its text; else out of scope
    if not calls:
        ai_text = (getattr(response, "output_text", "") or "").strip()
        return {"text": ai_text or "out of scope"}

    # Execute tools locally via FastMCP and collect outputs
    tool_outputs: List[Dict[str, str]] = []
    async with Client(mcp) as mcp_client:
        for c in calls:
            logger.info(f"tool_call requested by model: name={c['name']} args={c['arguments']} call_id={c['id']}")
            result = await mcp_client.call_tool(c["name"], c["arguments"])

            # Extract text from FastMCP result
            text_out = ""
            content = getattr(result, "content", None)
            if isinstance(content, list) and content:
                first = content
                if getattr(first, "type", "") == "text":
                    text_out = getattr(first, "text", "") or ""

            # Optional: structured content path
            if not text_out and hasattr(result, "structured_content") and result.structured_content is not None:
                try:
                    text_out = json.dumps(result.structured_content)
                except Exception:
                    pass

            # Final fallback to string
            if not text_out:
                try:
                    text_out = json.dumps(result, default=str)
                except Exception:
                    text_out = str(result)

            logger.info(f"tool_result produced: name={c['name']} call_id={c['id']} output={text_out}")

            tool_outputs.append({
                "tool_call_id": c["id"],
                "output": text_out,
            })

    # Submit tool outputs using whichever mechanism the SDK supports
    submit = getattr(client.responses, "submit_tool_outputs", None)
    if callable(submit):
        # Newer SDKs: direct submission
        resumed = submit(
            response_id=response.id,
            tool_outputs=tool_outputs,
        )
    else:
        # Fallback: create a new response with function_call_output + previous_response_id
        input_items = [
            {
                "type": "function_call_output",
                "call_id": t["tool_call_id"],
                "output": t["output"],
            }
            for t in tool_outputs
        ]
        resumed = client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=input_items,
        )

    # Prefer the model's final assistant text; fallback to the last tool output
    ai_text = (getattr(resumed, "output_text", "") or "").strip()
    final_text = ai_text or (tool_outputs[-1]["output"] if tool_outputs else "")
    return {"text": final_text}
