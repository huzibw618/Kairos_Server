# client/client_ollama_mcp.py
import asyncio
import argparse
import json
import os
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

def mcp_tool_to_ollama(tool) -> dict:
    """
    Convert an MCP Tool (name, description/title, inputSchema) to an Ollama tool schema.
    Ollama uses an OpenAI-style tools array with parameters as JSON Schema.
    """
    desc = getattr(tool, "description", None) or getattr(tool, "title", "") or ""
    params = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
    if not isinstance(params, dict) or params.get("type") != "object":
        params = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": desc,
            "parameters": params,
        },
    }

async def run(model: str, server_dir: str):
    # Launch the MCP server via stdio using uv in the server directory
    server_params = StdioServerParameters(
        command="uv",
        args=["--directory", server_dir, "run", "calculator_server.py"],
        env={},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools from the MCP server
            list_result = await session.list_tools()
            ollama_tools = [mcp_tool_to_ollama(t) for t in list_result.tools]

            client = AsyncClient()
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": "Use the provided tools to answer user's query. only use the tools. Use multiple tools if need be. If no tools can be used do not answer. Just say not in your scope. You may list the tools to tell the user what can you them with. DO NOT REPLY FOR ANY OTHER QUESTIONS."
                }
            ]

            print(f"Connected. Using model: {model}. Enter math questions, or 'quit' to exit.")
            while True:
                query = input("\n>>> ").strip()
                if query.lower() in ("quit", "exit"):
                    break

                messages.append(                {
                    "role": "system",
                    "content": "Use the provided tools to answer user's query. only use the tools. Use multiple tools if need be. If no tools can be used do not answer. Just say not in your scope. You may list the tools to tell the user what can you them with. DO NOT REPLY FOR ANY OTHER QUESTIONS. DO NOT USE ANY OTHER TOOL. ONLY GIVE TOOL's OUTPUT OR ITS SUMMARY !!"
                })
                messages.append({"role": "user", "content": query})

                # First pass: let the model decide whether to call tools
                response = await client.chat(
                    model=model,
                    messages=messages,
                    think=False,
                    tools=ollama_tools,  # advertise MCP tools to the model
                )

                tool_calls = getattr(response.message, "tool_calls", None) or []
                print("tool_calls:", tool_calls)

                if tool_calls:
                    # Execute each tool call via MCP and return results to the model
                    for tc in tool_calls:
                        name = tc.function.name
                        args = tc.function.arguments or {}
                        tool_call_id = getattr(tc, "id", None) or getattr(tc, "tool_call_id", None)

                        print(f"Tool requested: {name} with args: {args}")

                        call_result = await session.call_tool(name=name, arguments=args)

                        # Extract text content (MCP returns a list of content items)
                        tool_text = ""
                        for item in (call_result.content or []):
                            if getattr(item, "type", "") == "text":
                                tool_text += (item.text or "")

                        # Fallback to structured content if no text was provided
                        if not tool_text:
                            sc = getattr(call_result, "structuredContent", None)
                            if sc is not None:
                                tool_text = json.dumps(sc)

                        tool_text = tool_text or "(no content)"
                        print(f"Tool output ({name}): {tool_text}")

                        # Provide the tool result back to the model; include tool_call_id to link the result
                        tool_msg = {"role": "tool", "content": tool_text}
                        if tool_call_id:
                            tool_msg["tool_call_id"] = tool_call_id
                        else:
                            # Fallback for older formats if no id is present
                            tool_msg["name"] = name

                        messages.append(tool_msg)

                    # Second pass: get the final answer with tool results provided
                    final = await client.chat(model=model, messages=messages, think=False)
                    print("\nAssistant:", final.message.content)
                    messages.append({"role": "assistant", "content": final.message.content})
                else:
                    # No tools requested; print direct response
                    print("\nAssistant:", response.message.content)
                    messages.append({"role": "assistant", "content": response.message.content})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1", help="Ollama model name that supports tools (e.g., llama3.1)")
    parser.add_argument(
        "--server-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server")),
        help="Absolute path to the server project directory",
    )
    args = parser.parse_args()
    asyncio.run(run(model=args.model, server_dir=args.server_dir))

if __name__ == "__main__":
    main()
