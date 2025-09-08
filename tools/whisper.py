## Live Mic → Whisper → LLM → TTS
import asyncio
import os
import sounddevice as sd
import numpy as np
import tempfile
import wave
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from mcp_server.client import agents, synthesize_via_mcp

load_dotenv()

# Whisper config
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds per recording chunk

async def record_audio_to_file() -> str:
    """Record audio from mic and save to a temporary WAV file."""
    print(f"🎤 Recording {DURATION} seconds... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    print(f"✅ Audio saved: {temp_file.name}")
    return temp_file.name

async def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using faster-whisper."""
    print("🔍 Transcribing...")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, info = model.transcribe(file_path, beam_size=5)
    text = " ".join([seg.text.strip() for seg in segments])
    print(f"📝 Transcribed: {text}")
    return text

async def main():
    # 1) Record from mic
    audio_file = await record_audio_to_file()

    # 2) Transcribe with Whisper
    question = await transcribe_audio(audio_file)

    # 3) Send to LLM agent
    reply = await agents(
        llm_model=os.getenv("OLLAMA_LLM"),
        llm_provider="ollama",
        question=question,
    )
    print(f"🤖 LLM Reply: {reply}")

    # 4) Convert reply to speech
    tts_out = "reply.wav"
    print("🔊 Synthesizing speech...")
    tts_result = await synthesize_via_mcp(reply, out_path=tts_out)
    print(tts_result)
    print(f"✅ Play the audio: {tts_out}")

if __name__ == "__main__":
    asyncio.run(main())
