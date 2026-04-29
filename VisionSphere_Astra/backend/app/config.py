from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App Settings
    app_name: str = "VisionSphere Astra"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM Settings (Ollama) - CPU optimized models
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:0.5b"  # Fast and small for CPU
    llm_model_fallback: str = "gemma4:e4b"  # User's high-quality model
    llm_temperature: float = 0.7

    # Voice Settings
    stt_model: str = "tiny"  # Whisper tiny for speed
    stt_language: str = "en"
    tts_voice: str = "en-US-AriaNeural"  # Edge TTS voice
    tts_rate: int = 180  # Faster speech

    # Vision Settings - CPU optimized
    camera_index: int = 0
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    video_fps: int = 5  # Lower FPS for CPU
    video_fps_active: int = 15  # Higher when actively viewing
    
    # Phase 2: Face & Tracking
    face_recognition_enabled: bool = True
    face_data_path: str = "data/faces"
    face_detection_skip_frames: int = 3  # Only recognize every 3rd frame for speed
    face_tolerance: float = 0.6  # Lower is stricter
    tracking_max_age: int = 30  # Frames to remember lost object

    # Memory Settings
    memory_db_path: str = "./data/memory_db"
    memory_collection: str = "astra_memory"

    # Performance Settings
    max_workers: int = 2  # Limit concurrent model inference
    model_cache_size: int = 4  # GB

    # Optional API Keys (for future expansion)
    claude_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
