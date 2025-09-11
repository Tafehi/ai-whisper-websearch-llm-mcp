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

# Sidebar: Loaded Agents/Tools (placeholder)
st.sidebar.markdown("### Logs for Agents/Tools from MCP Server")
log_placeholder = "logs related to the agents and tools generated from MCP server will appear here."
st.sidebar.text_area("Logs", log_placeholder, height=150)

# Sidebar: AllVoiceLab TTS options
st.sidebar.markdown("### AllVoiceLab TTS")
enable_tts = st.sidebar.toggle("Enable AllVoiceLab TTS", value=True)
voice_id = st.sidebar.text_input("Voice ID", value="en_female_1")
audio_fmt = st.sidebar.selectbox("Audio format", ["mp3", "wav", "ogg"], index=0)

# Optional diagnostics (environment)
with st.sidebar.expander("AllVoiceLab MCP Diagnostics"):
    st.write("ALLVOICELAB_MCP_URL:", os.getenv("ALLVOICELAB_MCP_URL", "http://localhost:8003/mcp/"))
    st.write("ALLVOICELAB_API_DOMAIN:", os.getenv("ALLVOICELAB_API_DOMAIN", "(unset)"))
    st.write("ALLVOICELAB_ENABLED:", os.getenv("ALLVOICELAB_ENABLED", "true"))

# Main area: Chatbot interface
st.title("🧠 Type Your Question (and optionally Speak the Answer)")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Helper: get the latest assistant reply (if any)
def get_last_assistant_text():
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            return m["content"]
    return None
st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# TEXT INPUT → AGENT
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# SPEAK THE LATEST ASSISTANT REPLY (AllVoiceLab TTS)
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# SPEAK THE LATEST ASSISTANT REPLY (AllVoiceLab TTS)
# ────────────────────────────────────────────────────────────────────────────────
last_reply = get_last_assistant_text()

# Map to correct MIME types (important; 'audio/mp3' is not a valid MIME)
MIME_MAP = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}
mime = MIME_MAP.get(audio_fmt, "audio/mpeg")

col_tts_btn, _ = st.columns([1, 6])
with col_tts_btn:
    # Keep the button always enabled; gate behavior inside
    if st.button("🔊 Speak with AllVoiceLab", use_container_width=True, key="speak_allvoicelab"):
        if not enable_tts:
            st.warning("Enable TTS in the sidebar first.")
        elif not last_reply:
            st.info("No assistant reply found to speak yet. Ask a question first.")
        else:
            with st.spinner("Synthesizing with AllVoiceLab…"):
                try:
                    result = asyncio.run(
                        client.allvoicelab_tts(last_reply, voice_id=voice_id, fmt=audio_fmt)
                    )
                except Exception as e:
                    st.error(f"AllVoiceLab TTS error: {e}")
                    result = None

            if result:
                # Be flexible about what the MCP returns: path OR bytes OR dict
                out_file = None
                audio_bytes = None

                if isinstance(result, (bytes, bytearray)):
                    audio_bytes = bytes(result)
                elif isinstance(result, str):
                    # assume a file path
                    out_file = result
                elif isinstance(result, dict):
                    out_file = result.get("output_file") or result.get("path") or result.get("file")
                    audio_bytes = result.get("bytes") or result.get("audio_bytes")

                if audio_bytes:
                    st.success("Playing synthesized audio")
                    st.audio(audio_bytes, format=mime)
                elif out_file and os.path.exists(out_file):
                    st.success(f"Playing: {os.path.basename(out_file)}")
                    with open(out_file, "rb") as f:
                        st.audio(f.read(), format=mime)
                else:
                    st.warning(
                        "No playable audio returned by AllVoiceLab MCP. "
                        "If you expected a file path, ensure the path is accessible from this process."
                    )
