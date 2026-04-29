"""
Speech-to-Text using Faster-Whisper (CPU-optimized)
"""
import io
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional
from app.config import settings


class SpeechToText:
    """CPU-optimized speech recognition"""

    def __init__(self):
        # Use CPU-optimized whisper with quantization
        # "tiny" or "base" for CPU, int8 quantization
        self.model_size = settings.stt_model
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Load whisper model (called on first use)"""
        if not self.is_loaded:
            print(f"Loading Whisper {self.model_size} model...")
            # cpu_threads for parallelization on CPU
            # compute_type="int8" for faster CPU inference
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4  # Use all cores of i5-8350U
            )
            self.is_loaded = True
            print(f"Whisper model loaded: {self.model_size}")

    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text"""
        if not self.is_loaded:
            self.load_model()

        segments, info = self.model.transcribe(
            audio_file,
            language=settings.stt_language,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            )
        )

        # Combine all segments
        text = " ".join([segment.text for segment in segments]).strip()
        return text

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe audio from bytes (for WebSocket streaming)"""
        if not self.is_loaded:
            self.load_model()

        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        audio = audio / 32768.0  # Normalize

        segments, info = self.model.transcribe(
            audio,
            language=settings.stt_language,
            vad_filter=True,
            word_timestamps=False
        )

        text = " ".join([segment.text for segment in segments]).strip()
        return text


# Global instance (lazy loaded)
stt_engine: Optional[SpeechToText] = None


def get_stt() -> SpeechToText:
    """Get or create STT engine"""
    global stt_engine
    if stt_engine is None:
        stt_engine = SpeechToText()
    return stt_engine
