import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from mcp_server import client

load_dotenv()
st.set_page_config(page_title="LLM Chatbot", layout="wide")

# Sidebar
st.sidebar.title("LLM Configuration")
provider = st.sidebar.selectbox("Select LLM Provider", ["aws", "ollama"], index=1)
voice_id = st.sidebar.text_input("Voice ID", value="1")
model_id = st.sidebar.text_input("Model ID", value="allvoicelab/tts-english-v1")

model_options = {
    "ollama": ["llama3.2:latest", "orionstar/orion14b-q4:latest", "prompt/hermes-2-pro"],
    "aws": ["anthropic.claude-3-7-sonnet-20250219-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", "anthropic.claude-3-haiku-20240307-v1:0"],
}
default_model = model_options[provider][0]
model = st.sidebar.selectbox("Select Model", model_options[provider], index=0)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    async def run_agent():
        return await client.agents(llm_model=model, llm_provider=provider, question=prompt)

    response = asyncio.run(run_agent())

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # Speak assistant response only
    assistant_audio_url = asyncio.run(client.speak_text(response, voice_id=voice_id, model_id=model_id))
    if assistant_audio_url:
        st.audio(assistant_audio_url, format="audio/mp3")

st.divider()
