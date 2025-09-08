# audio_tts_server.py
import os
import asyncio
import traceback
from typing import Optional, List
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import pyttsx3
# Initialize MCP server
mcp = FastMCP(name="audioTools", host="localhost", port=8003)

class TTSBackend:
    def __init__(self):
        load_dotenv()
        self._engine = None
        # Defaults (can be overridden via tool args)
        self.default_voice_query = os.getenv("TTS_VOICE", "")        # substring or id/name
        self.default_rate = int(os.getenv("TTS_RATE", "180"))        # words per minute
        self.default_volume = float(os.getenv("TTS_VOLUME", "1.0"))  # 0.0–1.0

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.default_rate)
            self._engine.setProperty("volume", self.default_volume)
            if self.default_voice_query:
                self._set_voice(self.default_voice_query)

    # ---------- Voice helpers ----------
    def _voices(self):
        self._ensure_engine()
        return self._engine.getProperty("voices")

    def _best_match_voice_id(self, voice_query: str) -> Optional[str]:
        """Pick best matching voice id: exact name/id match first, then substring in name/id."""
        if not voice_query:
            return None
        voices = self._voices()
        q = voice_query.strip().lower()

        # exact match on name or id
        for v in voices:
            if (v.name or "").lower() == q or (v.id or "").lower() == q:
                return v.id

        # substring match on name or id
        for v in voices:
            if q in (v.name or "").lower() or q in (v.id or "").lower():
                return v.id

        return None

    def _set_voice(self, voice_query: str) -> Optional[str]:
        """Set engine voice from a query (id or name, exact/substr). Return chosen id or None."""
        voice_id = self._best_match_voice_id(voice_query)
        if voice_id:
            self._engine.setProperty("voice", voice_id)
            return voice_id
        return None

    def list_voices_text(self, query: str = "") -> str:
        self._ensure_engine()
        voices = self._voices()
        lines = ["Available TTS voices:"]
        q = query.strip().lower()
        for v in voices:
            if q and not (q in (v.name or "").lower() or q in (v.id or "").lower()):
                continue
            lines.append(f"- id='{v.id}' | name='{v.name}'")
        if len(lines) == 1:
            return "No voices matched your filter." if query else "No voices found."
        return "\n".join(lines)

    def get_current_voice_text(self) -> str:
        self._ensure_engine()
        vid = self._engine.getProperty("voice")
        # Look up readable name
        name = None
        for v in self._voices():
            if v.id == vid:
                name = v.name
                break
        return f"Current voice id='{vid}'" + (f" | name='{name}'" if name else "")

    def set_default_voice(self, voice_query: str) -> str:
        self._ensure_engine()
        chosen = self._set_voice(voice_query)
        if chosen:
            # Store as the new default for future engine resets (current process)
            self.default_voice_query = voice_query
            # Read back properties
            rate = self._engine.getProperty('rate')
            volume = self._engine.getProperty('volume')
            return f"✅ Default voice set to id='{chosen}' (query='{voice_query}'), rate={rate}, volume={volume}"
        else:
            return f"❌ No voice matched query: '{voice_query}'. Use list_tts_voices(query) to find options."

    # ---------- Synthesis ----------
    def _synthesize_sync(self, text: str, out_path: str, voice: Optional[str], rate: Optional[int], volume: Optional[float]) -> str:
        """Blocking pyttsx3 call executed in a thread."""
        if not text or not text.strip():
            return "❌ TTS input text is empty."

        # Ensure engine up and config set
        self._ensure_engine()

        # Apply per-call overrides
        chosen = None
        if voice:
            chosen = self._set_voice(voice)
        if rate is not None:
            self._engine.setProperty("rate", int(rate))
        if volume is not None:
            self._engine.setProperty("volume", float(volume))

        # Ensure output directory exists
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        try:
            # pyttsx3 writes WAV reliably; keep extension .wav for best results
            self._engine.save_to_file(text, out_path)
            self._engine.runAndWait()
            current_voice = self._engine.getProperty("voice")
            meta = f"(voice='{voice or current_voice}', rate={self._engine.getProperty('rate')}, volume={self._engine.getProperty('volume')})"
            if voice and chosen is None:
                meta += " [requested voice not found; used current/default]"
            return f"✅ Saved audio to: {out_path} {meta}"
        except Exception as e:
            traceback.print_exc()
            return f"❌ TTS error: {str(e)}"

    async def synthesize(self, text: str, out_path: str, voice: Optional[str], rate: Optional[int], volume: Optional[float]) -> str:
        # Run blocking TTS in a thread to avoid blocking the event loop
        return await asyncio.to_thread(self._synthesize_sync, text, out_path, voice, rate, volume)


# Backend instance
_backend = TTSBackend()

# ---------- MCP Tools (TOP-LEVEL, no 'self') ----------
@mcp.tool(name="list_tts_voices", description="List available TTS voices and their IDs/names. Optional filter with 'query'.")
async def list_tts_voices(query: str = "") -> str:
    return _backend.list_voices_text(query=query)

@mcp.tool(name="get_current_tts_voice", description="Get the currently active TTS voice id and name.")
async def get_current_tts_voice() -> str:
    return _backend.get_current_voice_text()

@mcp.tool(name="set_default_tts_voice", description="Set the default TTS voice by id or name (exact or substring).")
async def set_default_tts_voice(voice_query: str) -> str:
    return _backend.set_default_voice(voice_query)

@mcp.tool(name="synthesize_speech", description="Convert text to speech and save to an audio file (WAV).")
async def synthesize_speech(
    text: str,
    out_path: str = "out.wav",
    voice: Optional[str] = None,
    rate: Optional[int] = None,
    volume: Optional[float] = None,
) -> str:
    """
    Args:
      text: Text to synthesize.
      out_path: Output path for WAV file (e.g., 'out/reply.wav').
      voice: Optional voice id or name substring; per-call override (takes priority over default).
      rate: Optional speaking rate (wpm).
      volume: Optional volume (0.0–1.0).
    """
    # Ensure WAV extension is used for best compatibility with pyttsx3
    root, ext = os.path.splitext(out_path)
    if not ext:
        out_path = f"{out_path}.wav"
    elif ext.lower() != ".wav":
        # We still allow custom extensions, but pyttsx3 is most reliable with .wav
        pass

    return await _backend.synthesize(text=text, out_path=out_path, voice=voice, rate=rate, volume=volume)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
