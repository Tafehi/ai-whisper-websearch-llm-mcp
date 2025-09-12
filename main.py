# main.py
import os
from dotenv import load_dotenv
import asyncio
import streamlit as st
from mcp_server import client

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit app for MCP Server
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LLM Chatbot", layout="wide")

# Sidebar: LLM Provider Selection
st.sidebar.title("LLM Configuration")

provider = st.sidebar.selectbox("Select LLM Provider", ["aws", "ollama"], index=1)

model_options = {
    "ollama": [
        "llama3.2:latest",
        "orionstar/orion14b-q4:latest",
        "prompt/hermes-2-pro",
    ],
    "aws": [
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ],
}

default_model = (
    "llama3.2:latest"
    if provider == "ollama"
    else "anthropic.claude-3-7-sonnet-20250219-v1:0"
)

model = st.sidebar.selectbox(
    "Select Model",
    model_options[provider],
    index=model_options[provider].index(default_model),
)

# Sidebar: Logs placeholder
st.sidebar.markdown("### Logs for Agents/Tools from MCP Server")
log_placeholder = "Logs related to the agents and tools generated from MCP server will appear here."
st.sidebar.text_area("Logs", log_placeholder, height=150)

# Main area: Chatbot interface
st.title("🧠 Ask Your Question")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# TEXT INPUT → AGENT
#───────────────
if prompt := st.chat_input("Ask me anything..."):
    # Store & show the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    async def run_agent():
        return await client.agents(
            llm_model=model, llm_provider=provider, question=prompt
        )

    response = asyncio.run(run_agent())

    # Store & show the assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

st.divider()
