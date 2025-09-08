# main.py
import asyncio
import os
from mcp_server.client import agents
from dotenv import load_dotenv

load_dotenv()

async def run():
    reply = await agents(
        llm_model=os.getenv("OLLAMA_LLM"),       # or OLLAMA_MODEL from your setup
        llm_provider="ollama",
        question="what is current oslo weather? Include when you collected the info.",
    )
    print("Reply:", reply)

if __name__ == "__main__":
    asyncio.run(run())
