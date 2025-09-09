from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
import traceback
import os
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

    # Multi-server MCP client
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

    # Input messages
    input_messages = [
        {"role": "system", "content": security_prompt()},
        {"role": "system", "content": system_prompt()},
        {"role": "system", "content": tool_use_instructions()},
        {"role": "user", "content": question},
    ]

    # Run agent
    response = await agent.ainvoke({"messages": input_messages})
    final_content = response["messages"][-1].content if response.get("messages") else str(response)

    print(f"Agent Response: {final_content}")

    # --- Optional TTS ---
    tts_out = os.getenv("TTS_OUT")  # e.g., "out/reply.wav"
    if tts_out:
        voice = os.getenv("TTS_VOICE")
        rate = os.getenv("TTS_RATE")
        volume = os.getenv("TTS_VOLUME")

        # Find synthesize_speech tool
        tts_tool = next((t for t in tools if t.name == "synthesize_speech"), None)
        if tts_tool:
            args = {"text": final_content, "out_path": tts_out}
            if voice: args["voice"] = voice
            if rate: args["rate"] = int(rate)
            if volume: args["volume"] = float(volume)

            print(f"[audioTools] Synthesizing reply to {tts_out}...")
            try:
                tts_result = await tts_tool.ainvoke(args)
                print(tts_result)
            except Exception as e:
                print(f"⚠️ TTS failed: {e}")
        else:
            print("⚠️ synthesize_speech tool not found in loaded tools.")

    return final_content
