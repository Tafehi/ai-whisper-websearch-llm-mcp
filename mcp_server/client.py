import os
import datetime
import traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
from tools.promptGen import assemble_prompt

# Global MCP client and memory
mcp_client = None
temporary_memory = []  # ← Temporary in-session memory

async def agents(llm_model, llm_provider, question):
    global mcp_client, temporary_memory

    # Select model
    if llm_provider == "aws":
        model = BedrockLLM(llm_model).get_llm()
    elif llm_provider == "ollama":
        model = OllamaLLM(llm_model).get_llm()
    else:
        raise ValueError("Unsupported LLM provider")

    # MCP server definitions
    servers = {
        "serpSearch": {"url": "http://localhost:8001/mcp/", "transport": "streamable_http"},
        "weather": {"url": "http://localhost:8002/mcp/", "transport": "streamable_http"},
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

    # Initialize MCP client
    mcp_client = MultiServerMCPClient(servers)
    tools = await mcp_client.get_tools()
    agent = create_react_agent(model=model["llm_model"], tools=tools)

    # Prompt setup
    now = datetime.datetime.now()
    system_block = assemble_prompt(now=now, knowledge_cutoff=os.getenv("ASSISTANT_KNOWLEDGE_CUTOFF", "2024-06-01"))

    # Add system message only once
    if not temporary_memory:
        temporary_memory.append({"role": "system", "content": system_block})

    # Add user message to memory
    temporary_memory.append({"role": "user", "content": question})

    # Run agent with full memory
    response = await agent.ainvoke({"messages": temporary_memory})

    # Extract assistant message and store it
    final_content = response["messages"][-1].content if response.get("messages") else str(response)
    temporary_memory.append({"role": "assistant", "content": final_content})

    return final_content

import asyncio

async def speak_text(text, voice_id, model_id, retries=2):
    global mcp_client
    if not mcp_client:
        return None
    try:
        tools = await mcp_client.get_tools()
        tts_tool = next((t for t in tools if t.name == "text_to_speech"), None)
        if not tts_tool:
            return None

        for attempt in range(retries):
            response = await tts_tool.ainvoke({
                "text": text,
                "voice_id": voice_id,
                "model_id": model_id
            })
            if isinstance(response, str) and response.startswith("http"):
                return response
            await asyncio.sleep(1)  # small delay before retry

        print(f"TTS failed after {retries} attempts: {response}")
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

