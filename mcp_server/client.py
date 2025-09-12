from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
import traceback
from tools.promptGen import assemble_prompt
import os
import datetime
from tools.promptGen import system_prompt, tool_use_instructions, security_prompt


async def agents(llm_model, llm_provider, question):
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

    from langchain_mcp_adapters.client import MultiServerMCPClient
    import os

    # --- Sanitize env so we never pass None to subprocess ---

    # If neo4j-data-modeling is hosted remotely, set this in .env:
    # NEO4J_DATA_MODELING_MCP_URL=https://your-remote-host/mcp/
    ndm_url = (os.getenv("NEO4J_DATA_MODELING_MCP_URL") or "").strip()

    # --- Build servers dict safely ---
    serp_url = "http://localhost:8001/mcp/"
    weather_url = "http://localhost:8002/mcp/"

    # 1) Sanitize env to avoid passing None to subprocess (fatal)
    allvoice_env = {
        "ALLVOICELAB_API_KEY": os.getenv("ALLVOICELAB_API_KEY"),  # required
        "ALLVOICELAB_API_DOMAIN": os.getenv("NEO4J_USERNAME")
    }

    servers = {
        "serpSearch": {"url": serp_url, "transport": "streamable_http"},
        "weather": {"url": weather_url, "transport": "streamable_http"},
        "AllVoiceLab": {
            "command": "uvx",
            "args": ["allvoicelab-mcp", "--transport", "stdio"],
            "transport": "stdio",
            "env": {
                "ALLVOICELAB_API_KEY": os.getenv("ALLVOICELAB_API_KEY"),
                "ALLVOICELAB_API_DOMAIN": os.getenv("ALLVOICELAB_API_DOMAIN"),
            }
        },

    }



    # 👉 This is how you add them:
    mcp_client = MultiServerMCPClient(servers)

    # Optional: helpful log
    print("[MCP] Enabling servers:", list(servers.keys()))

    print("Connecting to MCP tools and agents")

    try:
        tools = await mcp_client.get_tools()
    except* Exception as eg:
        print("ExceptionGroup caught during tool loading:")
        traceback.print_exception(eg)
        raise RuntimeError(f"Failed to load tools: {eg}")

    print(f"Loaded Tools: {[tool.name for tool in tools]}")
    agent = create_react_agent(model=model["llm_model"], tools=tools)

    now = datetime.datetime.now()
    knowledge_cutoff = os.getenv("ASSISTANT_KNOWLEDGE_CUTOFF", "2024-06-01")
    system_block = assemble_prompt(now=now, knowledge_cutoff=knowledge_cutoff)
    # Input messages
    input_messages = [
        {"role": "system", "content": system_block},
        {"role": "user", "content": question},
    ]

    # --- Run agent ---
    response = await agent.ainvoke({"messages": input_messages})
    final_content = (
        response["messages"][-1].content if response.get("messages") else str(response)
    )

    print(f"Agent Response: {final_content}")
    return final_content



