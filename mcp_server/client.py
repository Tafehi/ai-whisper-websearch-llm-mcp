# mcp_server/client.py
import traceback
from typing import Optional, Union
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM


def _coerce_prompt_text(msg: Union[str, dict, list, None]) -> str:
    """
    Convert a variety of possible 'prompt' payloads to a plain string.
    Handles:
      - str
      - LangChain/Anthropic content blocks (lists/dicts)
      - lists of messages (take first)
      - objects with `.content`
    Fallback to empty string if not recognizable.
    """
    if msg is None:
        return ""

    if isinstance(msg, str):
        return msg

    content = getattr(msg, "content", None)
    if content is not None:
        return _coerce_prompt_text(content)

    if isinstance(msg, list) and len(msg) > 0:
        # If list of messages or content blocks, take first
        first = msg[0]
        return _coerce_prompt_text(first)

    if isinstance(msg, dict):
        if "text" in msg and isinstance(msg["text"], str):
            return msg["text"]
        if msg.get("type") == "text" and isinstance(msg.get("text"), str):
            return msg["text"]
        if "content" in msg:
            return _coerce_prompt_text(msg["content"])

    return str(msg)


async def agents(
    llm_model: str,
    llm_provider: str,
    question: str,
    memory: Optional[str] = None,
) -> str:
    """
    Build a ReAct agent over MCP tools using either an Ollama or Bedrock chat model.

    This version *guarantees* web search via the 'search_one' MCP tool before invoking the agent.
    It injects the observation and a Europe/Oslo timestamp into the context and then lets the
    agent craft the final answer (and optionally do more tool calls).
    """

    # 1) Initialize LLM
    try:
        print(f"Setting up MCP Client with model: {llm_model} and provider: {llm_provider}")

        if llm_provider == "aws":
            llm_info = BedrockLLM(llm_model).get_llm()
        elif llm_provider == "ollama":
            llm_info = OllamaLLM(llm_model).get_llm()
        else:
            raise ValueError("Unsupported LLM provider. Choose 'aws' or 'ollama'.")

        if not isinstance(llm_info, dict) or "llm_model" not in llm_info:
            raise TypeError("LLM wrapper must return a dict with key 'llm_model' holding a ChatModel.")
        model = llm_info["llm_model"]
        print("model:", model)

    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}") from e
    print(f"LLM Model: {llm_model} from {llm_provider} is initialized successfully.")

    # 2) Initialize MCP multi-server client
    mcp_client = MultiServerMCPClient(
        {
            "searchOne": {
                "url": "http://localhost:8001/mcp/",
                "transport": "streamable_http",
            },
            "promptgen": {
                "url": "http://localhost:8004/mcp/",
                "transport": "streamable_http",
            },
        }
    )

    print("Connecting to MCP tools and prompts")

    # 3) Load tools and prompts
    try:
        tools = await mcp_client.get_tools()

        security_prompt_msg = await mcp_client.get_prompt("promptgen", "security_prompt")
        system_prompt_msg = await mcp_client.get_prompt("promptgen", "system_prompt")

        # If those APIs return lists, take first
        if isinstance(security_prompt_msg, list) and len(security_prompt_msg) > 0:
            security_prompt_msg = security_prompt_msg[0]
        if isinstance(system_prompt_msg, list) and len(system_prompt_msg) > 0:
            system_prompt_msg = system_prompt_msg[0]

        # Coerce into plain strings
        security_prompt = _coerce_prompt_text(security_prompt_msg)
        system_prompt = _coerce_prompt_text(system_prompt_msg)

    except Exception as eg:
        print("Exception caught during tool/prompt loading:")
        traceback.print_exception(type(eg), eg, eg.__traceback__)
        # Fallback to empty prompts if promptgen server is down, but still proceed
        security_prompt = ""
        system_prompt = ""
        # Still need tools to proceed; re-raise if tools not loaded
        try:
            tools  # noqa: F401
        except NameError as _:
            raise RuntimeError(f"Failed to load tools or prompts: {eg}") from eg

    print(f"Loaded Tools: {[tool.name for tool in tools]}")

    # 3.1) Find the SearchOne tool object (as a LangChain tool)
    search_tool = next((t for t in tools if t.name == "search_one"), None)
    if search_tool is None:
        raise RuntimeError(
            "The 'search_one' tool was not found. Ensure the MCP server at :8001 is running and exposes this tool."
        )

    # 3.2) Perform a deterministic search before handing off to the agent
    # Use strong defaults for time-sensitive queries like weather/news.
    search_args = {
        "query": question,                         # pass the exact question to the search tool
        "search_service": "google",
        "max_results": 5,
        "crawl_results": 2,                        # crawl a couple of results for fresh content
        "image": False,
        "include_sites": "yr.no,met.no,accuweather.com,weather.com",  # prioritize reliable weather sources
        "exclude_sites": "",
        "language": "en",
        "time_range": "day",                       # constrain to today's info
    }

    print("Calling 'search_one' MCP tool with:", search_args)
    try:
        # LangChain tools typically provide .ainvoke for async execution
        search_observation = await search_tool.ainvoke(search_args)
        if not isinstance(search_observation, str):
            search_observation = str(search_observation)
    except Exception as call_err:
        print("Search tool call failed; continuing without observation. Error:", call_err)
        search_observation = "⚠️ Search failed. No observation available."

    collected_at = datetime.now(ZoneInfo("Europe/Oslo")).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Collected at: {collected_at}")

    # Create an observation block the LLM can reliably use
    search_context_msg = (
        "SEARCH RESULTS (do not ignore):\n"
        f"{search_observation}\n\n"
        f"Collected at: {collected_at} (Europe/Oslo)"
    )

    # 4) Bind ReAct agent (still allow additional tool calls if the LLM needs them)
    agent = create_react_agent(model=model, tools=tools)

    # 5) Run the agent with explicit tool-use instructions
    tool_use_instructions = (
        "You MUST use the provided 'SEARCH RESULTS' to answer any time-sensitive or factual question. "
        "Extract the most relevant facts and cite up to 3 sources with title and URL. "
        "Include the 'Collected at' timestamp verbatim in your answer. "
        "If the search results are insufficient, you MAY call 'search_one' again "
        "with improved keywords, time_range='day', crawl_results>=2, and include_sites for authoritative sources."
    )

    input_messages = [
        {"role": "system", "content": security_prompt},
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": tool_use_instructions},
        {"role": "system", "content": search_context_msg},   # <-- Inject observation here
    ]
    if memory:
        input_messages.append({"role": "system", "content": memory})
    input_messages.append({"role": "user", "content": question})

    response = await agent.ainvoke({"messages": input_messages})

    # 6) Extract last assistant message as a string
    try:
        last = response["messages"][-1]
        content = getattr(last, "content", last)
        if isinstance(content, list):
            parts = []
            for part in content:
                text = None
                if isinstance(part, str):
                    text = part
                elif isinstance(part, dict) and "text" in part:
                    text = part["text"]
                elif hasattr(part, "get") and part.get("type") == "text":
                    text = part.get("text")
                elif hasattr(part, "content"):
                    text = str(part.content)
                if text:
                    parts.append(text)
            content = "\n".join(parts) if parts else str(content)
        elif not isinstance(content, str):
            content = str(content)
    except Exception:
        content = "⚠️ Unable to parse agent response."

    print(f"Agent Response: {content}")
    return content
