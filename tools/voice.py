# audio_tools_server.py
import os
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from faster_whisper import WhisperModel
import pyttsx3
import traceback

# Initialize MCP server
mcp = FastMCP(name="audioTools", host="localhost", port=8002)

class AudioBackend:
    def __init__(self):
        load_dotenv()
        # Whisper config
        self.model_size = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium/large-v3
        self.device = os.getenv("WHISPER_DEVICE", "cpu")       # "cpu" or "cuda"
        self.compute_type = os.getenv("WHISPER_COMPUTE", "int8")  # "int8"/"int8_float16"/"float16"/"float32"
        self._whisper_model = None

        # TTS config (pyttsx3)
        self._tts_engine = None
        self.default_voice = os.getenv("TTS_VOICE", "")  # try to match voice name/id substring
        self.tts_rate = int(os.getenv("TTS_RATE", "180"))  # words per minute
        self.tts_volume = float(os.getenv("TTS_VOLUME", "1.0"))  # 0.0–1.0

    # ---------- Whisper ----------
    def _ensure_whisper(self):
        if self._whisper_model is None:
            print(f"[audioTools] Loading Whisper model: size={self.model_size}, device={self.device}, compute={self.compute_type}")
            self._whisper_model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    async def transcribe_impl(self, audio_path: str, language: Optional[str] = None) -> str:
        if not os.path.isfile(audio_path):
            return f"❌ File not found: {audio_path}"
        try:
            self._ensure_whisper()
            # faster-whisper returns generator for segments and info
            segments, info = self._whisper_model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=200),
            )
            text_parts: List[str] = []
            for seg in segments:
                text_parts.append(seg.text.strip())
            text = " ".join([t for t in text_parts if t])
            if not text.strip():
                return "⚠️ No speech detected."
            return text.strip()
        except Exception as e:
            traceback.print_exc()
            return f"❌ Transcription error: {str(e)}"

    # ---------- TTS (pyttsx3) ----------
    def _ensure_tts(self):
        if self._tts_engine is None:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', self.tts_rate)
            self._tts_engine.setProperty('volume', self.tts_volume)
            # Try to set default voice if provided
            if self.default_voice:
                self._set_voice(self.default_voice)

    def _set_voice(self, voice_query: str) -> Optional[str]:
        """Attempt to select a voice whose name or id contains voice_query (case-insensitive)."""
        engine = self._tts_engine
        voices = engine.getProperty('voices')
        q = voice_query.lower()
        for v in voices:
            if q in (v.name or "").lower() or q in (v.id or "").lower():
                engine.setProperty('voice', v.id)
                return v.id
        return None

    def list_voices_text(self) -> str:
        self._ensure_tts()
        voices = self._tts_engine.getProperty('voices')
        lines = []
        for v in voices:
            meta = []
            if hasattr(v, "languages") and v.languages:
                meta.append(f"langs={v.languages}")
            if hasattr(v, "age"):
                meta.append(f"age={v.age}")
            if hasattr(v, "gender"):
                meta.append(f"gender={v.gender}")
            meta_str = (", ".join(meta)) if meta else ""
            lines.append(f"- id='{v.id}' | name='{v.name}' {meta_str}")
        return "Available TTS voices:\n" + "\n".join(lines)

    async def tts_impl(self, text: str, out_path: str, voice: Optional[str] = None) -> str:
        if not text or not text.strip():
            return "❌ TTS input text is empty."

        # Ensure directory exists
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        try:
            self._ensure_tts()
            chosen = None
            if voice:
                chosen = self._set_voice(voice)

            # pyttsx3 can save to file with .save_to_file
            self._tts_engine.save_to_file(text, out_path)
            self._tts_engine.runAndWait()
            vo = self._tts_engine.getProperty('voice')
            vinfo = f"voice='{voice}'" if voice else f"voice='{vo}'"
            if chosen is None and voice:
                vinfo += " (requested voice not found; used current/default)"
            return f"✅ Saved audio to: {out_path} ({vinfo}, rate={self.tts_rate}, volume={self.tts_volume})"
        except Exception as e:
            traceback.print_exc()
            return f"❌ TTS error: {str(e)}"


# Instantiate backend
_backend = AudioBackend()

# ----- Register MCP tools (TOP-LEVEL) -----
@mcp.tool(name="transcribe_audio", description="Transcribe an audio file to text using Whisper (faster-whisper).")
async def transcribe_audio(audio_path: str, language: Optional[str] = None) -> str:
    """
    Args:
      audio_path: Path to local audio file (wav, mp3, m4a, etc.)
      language: Optional BCP-47 or ISO code, e.g., 'en', 'nb', 'no', 'en-US' (if omitted, auto-detect)
    """
    return await _backend.transcribe_impl(audio_path=audio_path, language=language)

@mcp.tool(name="synthesize_speech", description="Convert text to speech and save to an audio file (offline pyttsx3).")
async def synthesize_speech(text: str, out_path: str = "out.wav", voice: Optional[str] = None) -> str:
    """
    Args:
      text: The text to synthesize
      out_path: Output audio file path (e.g., out.wav). pyttsx3 writes WAV reliably.
      voice: Optional voice id or name substring to select a specific voice
    """
    return await _backend.tts_impl(text=text, out_path=out_path, voice=voice)

@mcp.tool(name="list_tts_voices", description="List available TTS voices and their IDs.")
async def list_tts_voices() -> str:
    return _backend.list_voices_text()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
