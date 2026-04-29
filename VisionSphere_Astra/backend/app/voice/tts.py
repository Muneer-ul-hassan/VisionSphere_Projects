"""
Text-to-Speech using Edge TTS (free, fast, cloud but minimal latency)
Alternative: Piper TTS for fully local operation
"""
import asyncio
import edge_tts
from typing import AsyncGenerator, Optional
from app.config import settings


class TextToSpeech:
    """Fast TTS using Edge TTS (Microsoft's free cloud TTS)"""

    def __init__(self):
        self.voice = settings.tts_voice
        self.rate = settings.tts_rate

    async def synthesize(self, text: str) -> bytes:
        """Generate audio from text, return as bytes"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=f"+{self.rate}%"
        )

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=f"+{self.rate}%"
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    def get_available_voices(self) -> list:
        """Get list of available voices"""
        # Edge TTS has many voices, here are good English ones
        return [
            "en-US-AriaNeural",  # Female, natural
            "en-US-GuyNeural",   # Male, natural
            "en-US-JennyNeural", # Female, friendly
            "en-GB-SoniaNeural", # British female
        ]


# Global instance
tts_engine = TextToSpeech()
