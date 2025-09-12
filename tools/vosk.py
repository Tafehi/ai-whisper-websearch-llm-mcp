# tools/vosk.py
import io
import json
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer


class VoskASR:
    """
    Minimal offline ASR using Vosk.
    - Input: WAV bytes (any sample rate, mono/stereo)
    - Output: text transcript (str)
    """

    def __init__(self, model_path: str, sample_rate: int = 16000):
        if not model_path:
            raise ValueError("VoskASR requires a model_path.")
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._model: Optional[Model] = None

    def _ensure_model(self):
        if self._model is None:
            # Loads the acoustic + graph model from disk (do once / reuse)
            self._model = Model(self.model_path)

    def _wav_to_pcm16_mono(self, wav_bytes: bytes) -> bytes:
        """
        Normalize any WAV to 16 kHz, 16-bit PCM, mono (bytes) for Vosk recognizer.
        """
        data, sr = sf.read(
            io.BytesIO(wav_bytes), dtype="int16", always_2d=True
        )  # (n, ch)
        mono = data.mean(axis=1).astype(np.int16)
        if sr != self.sample_rate:
            # high-quality resample
            mono = resample_poly(mono, up=self.sample_rate, down=sr).astype(np.int16)
        return mono.tobytes()

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        """
        Transcribe a single utterance (push-to-talk) to text.
        """
        self._ensure_model()
        pcm = self._wav_to_pcm16_mono(wav_bytes)

        rec = KaldiRecognizer(self._model, self.sample_rate)
        rec.SetWords(True)
        rec.AcceptWaveform(pcm)
        result = json.loads(rec.Result() or "{}")
        text = result.get("text", "") or ""
        return text.strip()
