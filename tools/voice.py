# voice.py
# MCP server: Mic → OpenAI Whisper (API) + TTS via pyttsx3

import os
import asyncio
import tempfile
import wave
import traceback
from typing import Optional, List, Union
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# --- New: OpenAI SDK for Whisper API ---
from openai import OpenAI

# --- Mic recording ---
import sounddevice as sd
import numpy as np

# --- TTS ---
import pyttsx3

# Initialize MCP server
mcp = FastMCP(name="audioTools", host="localhost", port=8002)


class AudioBackend:
    def __init__(self):
        load_dotenv()

        # ----- OpenAI Whisper (API) -----
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.whisper_model = os.getenv("WHISPER_OPENAI_MODEL", "whisper-1")
        self.whisper_language = os.getenv(
            "WHISPER_LANGUAGE"
        )  # e.g., "en", "no" (optional)
        self.whisper_prompt = os.getenv(
            "WHISPER_PROMPT"
        )  # optional domain/context hint
        self._openai_client: Optional[OpenAI] = None

        # ----- Audio capture defaults -----
        self.sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
        self.channels = int(os.getenv("CHANNELS", "1"))
        self.default_duration = int(os.getenv("DURATION", "5"))  # seconds
        self.input_device: Optional[Union[int, str]] = os.getenv(
            "AUDIO_INPUT_DEVICE"
        )  # optional

        # ----- TTS (pyttsx3) -----
        self._tts_engine = None
        self.default_voice = os.getenv("TTS_VOICE", "")  # substring match
        self.tts_rate = int(os.getenv("TTS_RATE", "180"))  # words per minute
        self.tts_volume = float(os.getenv("TTS_VOLUME", "1.0"))  # 0.0–1.0

    # ---------- OpenAI client ----------
    def _ensure_openai(self):
        if not self.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.openai_api_key)

    # ---------- Microphone helpers ----------
    def _record_wav_tempfile(
        self,
        duration: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
    ) -> str:
        """
        Records from the microphone and returns path to a temporary WAV file.
        """
        duration = duration or self.default_duration
        sample_rate = sample_rate or self.sample_rate
        channels = channels or self.channels
        device = device if device is not None else self.input_device

        # Configure and record
        print(
            f"[audioTools] Recording {duration}s @ {sample_rate} Hz, channels={channels}, device={device}"
        )
        sd.default.samplerate = sample_rate
        sd.default.channels = channels
        if device is not None:
            sd.default.device = device

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()

        # Save to temp WAV
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        print(f"[audioTools] Saved mic capture → {tmp.name}")
        return tmp.name

    def list_input_devices_text(self) -> str:
        """
        Returns a string listing available audio devices for input selection.
        """
        devices = sd.query_devices()
        lines = ["Available audio devices (input-capable):"]
        for idx, d in enumerate(devices):
            if int(d.get("max_input_channels", 0)) > 0:
                lines.append(
                    f"- index={idx} | name='{d.get('name')}' | inputs={d.get('max_input_channels')} | default_sr={d.get('default_samplerate')}"
                )
        return "\n".join(lines)

    # ---------- Transcription (OpenAI Whisper API) ----------
    async def transcribe_file_impl(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Transcribe an existing audio file via OpenAI Whisper API.
        """
        if not os.path.isfile(audio_path):
            return f"❌ File not found: {audio_path}"
        try:
            self._ensure_openai()
            lang = language or self.whisper_language
            prmpt = prompt or self.whisper_prompt

            print(
                f"[audioTools] Transcribing file via OpenAI: model={self.whisper_model}, lang={lang}"
            )
            with open(audio_path, "rb") as f:
                resp = self._openai_client.audio.transcriptions.create(
                    model=self.whisper_model,
                    file=f,
                    language=lang if lang else None,
                    prompt=prmpt if prmpt else None,
                    # response_format="json"  # default; includes .text
                    # If you prefer raw text: response_format="text"
                )
            text = getattr(resp, "text", None) or str(resp)
            text = (text or "").strip()
            if not text:
                return "⚠️ No speech detected."
            return text
        except Exception as e:
            traceback.print_exc()
            return f"❌ Transcription error: {str(e)}"

    async def transcribe_mic_impl(
        self,
        duration: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        device: Optional[Union[int, str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        delete_tmp: bool = True,
    ) -> str:
        """
        Record from mic, send to OpenAI Whisper API, return transcript.
        """
        tmp_path = None
        try:
            tmp_path = self._record_wav_tempfile(
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                device=device,
            )
            return await self.transcribe_file_impl(
                tmp_path, language=language, prompt=prompt
            )
        finally:
            if delete_tmp and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ---------- TTS (pyttsx3) ----------
    def _ensure_tts(self):
        if self._tts_engine is None:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", self.tts_rate)
            self._tts_engine.setProperty("volume", self.tts_volume)
            if self.default_voice:
                self._set_voice(self.default_voice)

    def _set_voice(self, voice_query: str) -> Optional[str]:
        """Select a pyttsx3 voice whose name/id contains the substring (case-insensitive)."""
        engine = self._tts_engine
        voices = engine.getProperty("voices")
        q = voice_query.lower()
        for v in voices:
            if q in (v.name or "").lower() or q in (v.id or "").lower():
                engine.setProperty("voice", v.id)
                return v.id
        return None

    def list_voices_text(self) -> str:
        self._ensure_tts()
        voices = self._tts_engine.getProperty("voices")
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

    async def tts_impl(
        self, text: str, out_path: str, voice: Optional[str] = None
    ) -> str:
        if not text or not text.strip():
            return "❌ TTS input text is empty."

        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        try:
            self._ensure_tts()
            chosen = None
            if voice:
                chosen = self._set_voice(voice)

            self._tts_engine.save_to_file(text, out_path)
            self._tts_engine.runAndWait()
            vo = self._tts_engine.getProperty("voice")
            vinfo = f"voice='{voice}'" if voice else f"voice='{vo}'"
            if chosen is None and voice:
                vinfo += " (requested voice not found; used current/default)"
            return f"✅ Saved audio to: {out_path} ({vinfo}, rate={self.tts_rate}, volume={self.tts_volume})"
        except Exception as e:
            traceback.print_exc()
            return f"❌ TTS error: {str(e)}"


# Instantiate backend
_backend = AudioBackend()

# ----- MCP tools -----


@mcp.tool(
    name="transcribe_audio",
    description="Transcribe an audio file to text using OpenAI Whisper API.",
)
async def transcribe_audio(
    audio_path: str, language: Optional[str] = None, prompt: Optional[str] = None
) -> str:
    """
    Args:
      audio_path: Path to local audio file (wav, mp3, m4a, etc.)
      language: Optional BCP-47/ISO code, e.g., 'en', 'nb', 'no' (if omitted, auto-detect)
      prompt: Optional transcription hint/context/domain terms
    """
    return await _backend.transcribe_file_impl(
        audio_path=audio_path, language=language, prompt=prompt
    )


@mcp.tool(
    name="transcribe_mic",
    description="Record from microphone for a few seconds and transcribe via OpenAI Whisper API.",
)
async def transcribe_mic(
    duration: Optional[int] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    device: Optional[Union[int, str]] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Args:
      duration: Seconds to record (default from env DURATION or 5)
      sample_rate: e.g., 16000 (default from env)
      channels: 1 for mono, 2 for stereo (default from env)
      device: Optional input device index or name substring (see list_audio_devices)
      language: Optional language code (e.g., 'en', 'no'); if omitted, Whisper will try to auto-detect
      prompt: Optional hint text to bias recognition (domain terms, names, etc.)
    """
    # If device is a name substring, resolve it to index
    if isinstance(device, str) and device.strip():
        # Attempt to find a matching device index
        name_q = device.lower()
        devices = sd.query_devices()
        match_idx = None
        for idx, d in enumerate(devices):
            if (
                int(d.get("max_input_channels", 0)) > 0
                and name_q in (d.get("name") or "").lower()
            ):
                match_idx = idx
                break
        device = match_idx if match_idx is not None else None

    return await _backend.transcribe_mic_impl(
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        device=device,
        language=language,
        prompt=prompt,
    )


@mcp.tool(
    name="synthesize_speech",
    description="Convert text to speech and save to an audio file (offline pyttsx3).",
)
async def synthesize_speech(
    text: str, out_path: str = "out.wav", voice: Optional[str] = None
) -> str:
    """
    Args:
      text: Text to synthesize
      out_path: Output audio file path (WAV recommended)
      voice: Optional voice id or name substring to select a specific voice
    """
    return await _backend.tts_impl(text=text, out_path=out_path, voice=voice)


@mcp.tool(
    name="list_tts_voices", description="List available TTS voices and their IDs."
)
async def list_tts_voices() -> str:
    return _backend.list_voices_text()


@mcp.tool(
    name="list_audio_devices",
    description="List available microphone/input audio devices.",
)
async def list_audio_devices() -> str:
    return _backend.list_input_devices_text()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
