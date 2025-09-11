# client.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
import traceback
import os

from datetime import datetime
from zoneinfo import ZoneInfo

# Import the consolidated prompt builder
from tools.promptGen import assemble_prompt


def _build_allvoicelab_cfg():
    """
    Return a server config dict for AllVoiceLab MCP or None if disabled/misconfigured.
    Supports two transports:
      - stdio (recommended): runs `uvx allvoicelab-mcp` with env vars
      - http: points to ALLVOICELAB_MCP_URL
    """
    enabled = os.getenv("ALLVOICELAB_ENABLED", "false").lower() in ("1", "true", "yes", "y")
    if not enabled:
        print("[AllVoiceLab] Disabled (ALLVOICELAB_ENABLED != true). Skipping.")
        return None

    api_key = os.getenv("ALLVOICELAB_API_KEY", "").strip()
    api_domain = os.getenv("ALLVOICELAB_API_DOMAIN", "").strip()
    if not api_key or not api_domain:
        print("[AllVoiceLab] Missing ALLVOICELAB_API_KEY or ALLVOICELAB_API_DOMAIN. Skipping.")
        return None

    transport = os.getenv("ALLVOICELAB_TRANSPORT", "stdio").lower()

    if transport == "http":
        url = os.getenv("ALLVOICELAB_MCP_URL", "").strip()
        if not url:
            print("[AllVoiceLab] HTTP transport selected but ALLVOICELAB_MCP_URL is empty. Skipping.")
            return None
        return {
            "url": url,
            "transport": "streamable_http",
        }

    # Default: stdio
    cmd = os.getenv("ALLVOICELAB_COMMAND", "uvx")
    entry = os.getenv("ALLVOICELAB_ENTRY", "allvoicelab-mcp")
    base_path = os.getenv("ALLVOICELAB_BASE_PATH", "")

    env = {
        "ALLVOICELAB_API_KEY": api_key,
        "ALLVOICELAB_API_DOMAIN": api_domain,
    }
    if base_path:
        env["ALLVOICELAB_BASE_PATH"] = base_path

    cfg = {
        "transport": "stdio",
        "command": cmd,
        "args": [entry],
        "env": env,
    }
    return cfg


def make_mcp_client() -> MultiServerMCPClient:
    """
    Build a MultiServerMCPClient with your existing MCP servers plus (optionally) AllVoiceLab.
    """
    serp_url = os.getenv("SERP_MCP_URL", "http://localhost:8001/mcp/")
    weather_url = os.getenv("WEATHER_MCP_URL", "http://localhost:8002/mcp/")

    servers = {
        "serpSearch": {"url": serp_url, "transport": "streamable_http"},
        "weather": {"url": weather_url, "transport": "streamable_http"},
    }

    allvoicelab_cfg = _build_allvoicelab_cfg()
    if allvoicelab_cfg:
        servers["allvoicelab"] = allvoicelab_cfg
        print(f"[AllVoiceLab] Added with transport={allvoicelab_cfg.get('transport')}")

    return MultiServerMCPClient(servers)


# ---------- NEW: Tool-based invocation (no call_tool needed) ----------
async def allvoicelab_tts(text: str, voice_id: str = "en_female_1", fmt: str = "mp3") -> dict:
    """
    Look up the AllVoiceLab `text_to_speech` Tool and call it directly.
    Returns {"output_file": <path>, "raw": <tool_result>} where available.
    """
    mcp_client = make_mcp_client()

    try:
        tools = await mcp_client.get_tools()
    except* Exception as eg:
        raise RuntimeError(
            "Failed to load MCP tools. Ensure AllVoiceLab server is running and env vars are set."
        ) from eg
    # Find the TTS tool by name
    tts = None
    for t in tools:
        name = getattr(t, "name", "")
        if name in ("text_to_speech", "allvoicelab_text_to_speech", "AllVoiceLab.text_to_speech"):
            tts = t
            break

    if not tts:
        raise RuntimeError(
            "AllVoiceLab `text_to_speech` tool not found. "
            "Confirm ALLVOICELAB_ENABLED=true and the server exposes this tool."
        )

    payload = {"text": text, "voice_id": voice_id, "format": fmt}

    # Invoke the tool with best-effort API
    try:
        if hasattr(tts, "ainvoke"):
            res = await tts.ainvoke(payload)
        elif hasattr(tts, "arun"):
            res = await tts.arun(payload)
        else:
            import asyncio
            res = await asyncio.to_thread(getattr(tts, "run"), payload)
    except TypeError:
        # Some tools expect string input; fallback to stringified JSON
        try:
            if hasattr(tts, "ainvoke"):
                res = await tts.ainvoke(str(payload))
            elif hasattr(tts, "arun"):
                res = await tts.arun(str(payload))
            else:
                import asyncio
                res = await asyncio.to_thread(getattr(tts, "run"), str(payload))
        except Exception as e:
            raise RuntimeError(f"AllVoiceLab TTS invocation failed: {e}") from e

    # Normalize result to return a file path if present
    out_path = None
    if isinstance(res, dict):
        out_path = res.get("output_file") or res.get("file") or res.get("path")
    elif hasattr(res, "dict"):
        d = res.dict()
        out_path = d.get("output_file") or d.get("file") or d.get("path")
    elif isinstance(res, str):
        s = res.strip()
        if s.lower().endswith((".mp3", ".wav", ".ogg", ".flac", ".m4a")):
            out_path = s

    return {"output_file": out_path, "raw": res}


async def agents(llm_model: str, llm_provider: str, question: str):
    """
    Create and run a ReAct agent with MCP tools, using the consolidated prompt
    defined in tools/promptGen. The system message includes security, context,
    behavior, tool policy, style, and few-shot examples.
    """
    # --- LLM Initialization ---
    try:
        print(f"Setting up MCP Client with model: {llm_model} and provider: {llm_provider}")
        if llm_provider == "aws":
            model = BedrockLLM(llm_model).get_llm()
        elif llm_provider == "ollama":
            model = OllamaLLM(llm_model).get_llm()
        else:
            raise ValueError("Unsupported LLM provider. Choose 'aws' or 'ollama'.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")

    print(f"LLM Model: {model['llm_model']} from {model['llm_provider']} is initialized successfully.")

    # --- MCP Client / Tool Registry ---
    mcp_client = make_mcp_client()
    print("Connecting to MCP tools and agents")

    try:
        tools = await mcp_client.get_tools()
    except* Exception as eg:
        print("ExceptionGroup caught during tool loading:")
        traceback.print_exception(eg)
        raise RuntimeError(
            "Failed to load tools. "
            "If you just enabled AllVoiceLab, verify env vars and that the MCP server is running "
            "(stdio: `uvx allvoicelab-mcp`, or set ALLVOICELAB_TRANSPORT=http and ALLVOICELAB_MCP_URL)."
        ) from eg

    print(f"Loaded Tools: {[tool.name for tool in tools]}")
    agent = create_react_agent(model=model["llm_model"], tools=tools)

    # --- Prompt Assembly ---
    tz_name = os.getenv("AGENT_TZ", "Europe/Oslo")
    now = datetime.now(ZoneInfo(tz_name))
    knowledge_cutoff = os.getenv("ASSISTANT_KNOWLEDGE_CUTOFF", "2024-06-01")
    system_block = assemble_prompt(now=now, knowledge_cutoff=knowledge_cutoff)

    input_messages = [
        {"role": "system", "content": system_block},
        {"role": "user", "content": question},
    ]

    # --- Run agent ---
    response = await agent.ainvoke({"messages": input_messages})
    final_content = response["messages"][-1].content if response.get("messages") else str(response)

    print(f"Agent Response: {final_content}")
    return final_content
