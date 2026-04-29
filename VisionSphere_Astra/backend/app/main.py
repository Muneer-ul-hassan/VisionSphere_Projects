"""
VisionSphere Astra - Backend Server
A multimodal AI assistant inspired by Google's Project Astra
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from app.config import settings
from app.api.routes import router
from app.vision.detector import detector
from app.llm.ollama_client import llm_client
from app.proactive.manager import proactive_manager


app = FastAPI(
    title=settings.app_name,
    description="A multimodal AI assistant with vision, voice, and memory capabilities",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print(f"🚀 Starting {settings.app_name}...")
    
    # Load vision model
    detector.load_model()
    
    print(f"📷 Camera index: {settings.camera_index}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print(f"🌐 Ollama URL: {settings.ollama_base_url}")

    # Try to start camera
    if detector.start_camera():
        print("✅ Camera initialized")
    else:
        print("⚠️  Camera not available - will retry on first capture")

    # Check LLM connection
    llm_available = await llm_client.check_connection()
    if llm_available:
        print("✅ LLM connected")
    else:
        print("⚠️  LLM not connected - make sure Ollama is running")

    # Create data directory for memory
    os.makedirs(settings.memory_db_path, exist_ok=True)
    print(f"💾 Memory DB path: {settings.memory_db_path}")

    # Start Proactive Manager
    await proactive_manager.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 Shutting down...")
    await proactive_manager.stop()
    detector.stop_camera()
    print("✅ Camera stopped")


@app.get("/")
async def root():
    """Root endpoint - serves frontend"""
    # Try to serve frontend files
    frontend_paths = [
        "../frontend/build/index.html",
        "../frontend/dist/index.html",
        "../frontend/public/index.html"
    ]

    for path in frontend_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            return FileResponse(full_path)

    return {"message": "VisionSphere Astra API", "docs": "/docs"}


def main():
    """Run the server"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
