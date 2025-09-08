# main.py
import asyncio
import os
from dotenv import load_dotenv

from mcp_server.client import agents, transcribe_via_mcp, synthesize_via_mcp

load_dotenv()

async def run():
    # 1) Optionally transcribe an audio file to use as the question
    audio_in = os.getenv("AUDIO_IN")  # e.g., "samples/oslo_question.m4a"
    language = os.getenv("AUDIO_LANG")  # e.g., "en" or "nb"
    if audio_in:
        print(f"Transcribing AUDIO_IN: {audio_in} (lang={language or 'auto'})")
        question = await transcribe_via_mcp(audio_in, language=language)
        print("Transcribed question:", question)
    else:
        # Fallback to static text question
        question = "Where is Oslo and where is Norway? Include when you collected the info."

    # 2) Run your agent as-is
    reply = await agents(
        llm_model=os.getenv("OLLAMA_LLM"),   # or OLLAMA_MODEL
        llm_provider="ollama",
        question=question,
    )
    print("Reply:", reply)

    # 3) Optionally synthesize the reply to speech
    tts_out = os.getenv("TTS_OUT")  # e.g., "out/reply.wav"
    tts_voice = os.getenv("TTS_VOICE")  # optional name/id substring, e.g., "english" or "nb"
    if tts_out:
        print(f"Synthesizing reply to: {tts_out} (voice={tts_voice or 'default'})")
        tts_result = await synthesize_via_mcp(reply, out_path=tts_out, voice=tts_voice)
        print(tts_result)

if __name__ == "__main__":
    asyncio.run(run())
