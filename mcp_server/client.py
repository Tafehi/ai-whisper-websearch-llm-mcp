# client.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
import traceback
import os

from datetime import datetime
from zoneinfo import ZoneInfo

# Import the new prompt assembly function
from tools.promptGen import assemble_prompt

from tools.polly import PollyTTS


async def agents(llm_model: str, llm_provider: str, question: str):
    """
    Create and run a ReAct agent with MCP tools, using the consolidated prompt
    defined in tools.promptGen. The system message includes security, context,
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
    mcp_client = MultiServerMCPClient(
        {
            "serpSearch": {
                "url": "http://localhost:8001/mcp/",
                "transport": "streamable_http",
            },
            "weather": {
                "url": "http://localhost:8002/mcp/",
                "transport": "streamable_http",
            },
            # Add more tools here as needed.
        }
    )

    print("Connecting to MCP tools and agents")

    try:
        tools = await mcp_client.get_tools()
    except* Exception as eg:
        print("ExceptionGroup caught during tool loading:")
        traceback.print_exception(eg)
        raise RuntimeError(f"Failed to load tools: {eg}")

    print(f"Loaded Tools: {[tool.name for tool in tools]}")
    agent = create_react_agent(model=model["llm_model"], tools=tools)

    # --- Prompt Assembly according to tools/promptGen.py ---
    tz_name = os.getenv("AGENT_TZ", "Europe/Oslo")
    now = datetime.now(ZoneInfo(tz_name))
    knowledge_cutoff = os.getenv("ASSISTANT_KNOWLEDGE_CUTOFF", "2024-06-01")
    system_block = assemble_prompt(now=now, knowledge_cutoff=knowledge_cutoff)

    input_messages = [
        {"role": "system", "content": system_block},
        {"role": "user", "content": question},
    ]

    # --- Speak the question via Polly (simple & blocking) ---

    from tools.polly import PollyTTS

    # Create one instance and reuse it
    tts = PollyTTS()

    # Speak the question
    try:
        tts.speak(question, "question")
    except Exception as e:
        print(f"[Polly] Failed to read question: {e}")



    # --- Run agent ---
    response = await agent.ainvoke({"messages": input_messages})
    final_content = response["messages"][-1].content if response.get("messages") else str(response)

    print(f"Agent Response: {final_content}")

    # Speak the answer
    try:
        tts.speak(final_content, "answer")
    except Exception as e:
        print(f"[Polly] Failed to read answer: {e}")
    return final_content
