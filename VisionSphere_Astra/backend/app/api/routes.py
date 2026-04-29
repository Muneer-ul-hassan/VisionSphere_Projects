from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional
import asyncio
import base64
import cv2
import numpy as np
from datetime import datetime
import io
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    context: Optional[List[Dict]] = None

from app.llm.ollama_client import llm_client
from app.vision.detector import detector
from app.vision.screen_capture import screen_capture
from app.vision.ocr import get_ocr
from app.voice.stt import get_stt
from app.voice.tts import tts_engine
from app.memory.store import memory_store
from app.proactive.manager import proactive_manager


router = APIRouter()

# Active WebSocket connections
active_connections: Dict[int, WebSocket] = {}
connection_counter = 0


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    llm_available = await llm_client.check_connection()
    return {
        "status": "healthy",
        "llm_connected": llm_available,
        "model": llm_client.model,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/models")
async def get_available_models():
    """Get list of available Ollama models"""
    models = await llm_client.get_available_models()
    return {"models": models}


@router.get("/vision/detect")
async def detect_objects():
    """Run object detection and return results"""
    frame = await asyncio.to_thread(detector.capture_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    result = await asyncio.to_thread(detector.detect_objects, frame)

    # Convert image to base64 for response
    image_b64 = None
    if result["image"] is not None:
        _, buffer = cv2.imencode('.jpg', result["image"])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "objects": result["objects"],
        "counts": result["counts"],
        "total_count": result["frame_count"],
        "image": image_b64,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    """Send a message to the LLM and get response"""
    try:
        # 1. Get recent conversation history
        recent_convos = memory_store.get_recent_conversations(5)
        history_context = []
        for conv in reversed(recent_convos):
            parts = conv["content"].split("\nAssistant: ")
            if len(parts) == 2:
                user_msg = parts[0].replace("User: ", "")
                ast_msg = parts[1]
                history_context.append({"role": "user", "content": user_msg})
                history_context.append({"role": "assistant", "content": ast_msg})

        # Add any context passed from frontend
        if req.context:
            history_context.extend(req.context)

        # 2. Get current camera view
        frame = await asyncio.to_thread(detector.capture_frame)
        scene_context = ""
        if frame is not None:
            detection = await asyncio.to_thread(detector.detect_objects, frame)
            if detection["objects"]:
                objects_desc = []
                for obj in detection["objects"]:
                    objects_desc.append(f"- {obj['label']}")
                unique_objects = list(set([o['label'] for o in detection['objects']]))
                scene_context = f"\n\n[System Vision: Currently visible on camera: {', '.join(unique_objects)}]"

        # 3. Combine into final message
        final_message = f"{req.message}{scene_context}"

        # Get LLM response
        response = await asyncio.to_thread(llm_client.chat, final_message, history_context)

        # Store in memory
        memory_store.add_conversation(
            user_message=req.message,
            assistant_response=response,
            context=scene_context
        )

        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")


@router.post("/chat/scene")
async def chat_about_scene(question: str = "What do you see?"):
    """Ask about the current camera view"""
    # Get current detection
    frame = await asyncio.to_thread(detector.capture_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    detection = await asyncio.to_thread(detector.detect_objects, frame)

    # Build context from detection
    objects_desc = []
    for obj in detection["objects"]:
        objects_desc.append(
            f"- {obj['label']} (confidence: {obj['confidence']}, "
            f"bbox: {obj['bbox']}, center: {obj['center']})"
        )

    scene_context = f"""Current scene analysis:
- Total objects detected: {detection['frame_count']}
- Object counts: {detection['counts']}
- Detailed detections:
{chr(10).join(objects_desc) if objects_desc else 'No objects detected with sufficient confidence'}
"""

    # Combine with user question
    full_prompt = f"{scene_context}\n\nUser question: {question}"

    try:
        response = await asyncio.to_thread(llm_client.chat, full_prompt)

        # Store visual memory
        memory_store.add_visual_memory(
            description=scene_context,
            objects=list(detection["counts"].keys())
        )

        # Store conversation
        memory_store.add_conversation(
            user_message=question,
            assistant_response=response,
            context=scene_context
        )

        return {
            "response": response,
            "detections": detection["objects"],
            "counts": detection["counts"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")


@router.post("/chat/screen")
async def chat_about_screen(question: str = "What's on my screen?"):
    """Ask about the current screen content"""
    # Capture screen
    screen_img = await asyncio.to_thread(screen_capture.capture_full_screen)
    resolution = await asyncio.to_thread(screen_capture.get_screen_resolution)

    # Run object detection on screen
    detection = await asyncio.to_thread(detector.detect_objects, screen_img)

    # Run OCR for text
    ocr_result = await asyncio.to_thread(get_ocr().read_text, screen_img)

    # Build context
    screen_context = f"""Screen analysis:
- Resolution: {resolution[0]}x{resolution[1]}
- Objects detected: {detection['frame_count']}
- Object counts: {detection['counts']}
- Text found: {ocr_result['line_count']} lines
- OCR text: {ocr_result['text'][:500] if ocr_result['text'] else 'No text detected'}
"""

    full_prompt = f"{screen_context}\n\nUser question: {question}"

    try:
        response = await asyncio.to_thread(llm_client.chat, full_prompt)

        # Capture annotated image for response
        _, buffer = cv2.imencode('.jpg', detection["image"])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "response": response,
            "screen_image": image_b64,
            "objects": detection["objects"],
            "text": ocr_result["text"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")


@router.post("/ocr/read")
async def read_text_from_camera():
    """Read text from camera view"""
    frame = await asyncio.to_thread(detector.capture_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    ocr_result = await asyncio.to_thread(get_ocr().read_text, frame)

    # Store in memory
    if ocr_result["text"]:
        memory_store.add_visual_memory(
            description=f"Text found: {ocr_result['text'][:200]}",
            objects=["text"]
        )

    return {
        "text": ocr_result["text"],
        "lines": ocr_result["regions"],
        "line_count": ocr_result["line_count"],
        "timestamp": datetime.now().isoformat()
    }


@router.post("/ocr/find")
async def find_text(search_term: str):
    """Find specific text in camera view"""
    frame = await asyncio.to_thread(detector.capture_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    matches = await asyncio.to_thread(get_ocr().find_specific_text, frame, search_term)

    return {
        "search_term": search_term,
        "found": len(matches) > 0,
        "matches": matches,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/vision/enroll")
async def enroll_face(name: str):
    """Enroll the current unknown face with a name"""
    frame = await asyncio.to_thread(detector.capture_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    # Find unknown faces in current frame
    faces = await asyncio.to_thread(face_recognizer.recognize, frame)
    unknown_faces = [f for f in faces if f["label"] == "Unknown"]
    
    if not unknown_faces:
        return {"status": "error", "message": "No unknown face detected to enroll"}
    
    # Enroll the first unknown face found
    success = face_recognizer.enroll_face(frame, name, unknown_faces[0]["bbox"])
    
    if success:
        # Reset proactive manager state for this person
        proactive_manager.known_people_seen.add(name)
        return {"status": "success", "name": name}
    else:
        return {"status": "error", "message": "Failed to save face data"}
async def get_screen_info():
    """Get screen information"""
    resolution = await asyncio.to_thread(screen_capture.get_screen_resolution)
    return {
        "resolution": resolution,
        "width": resolution[0],
        "height": resolution[1]
    }


@router.get("/screen/capture")
async def capture_screen():
    """Capture current screen"""
    screen_img = await asyncio.to_thread(screen_capture.capture_full_screen)
    _, buffer = cv2.imencode('.jpg', screen_img)
    image_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "image": image_b64,
        "timestamp": datetime.now().isoformat()
    }


# ==========================================
# Voice Endpoints
# ==========================================

@router.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    try:
        # Save uploaded file temporarily
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        stt = get_stt()
        text = await asyncio.to_thread(stt.transcribe, tmp_path)

        # Cleanup
        os.unlink(tmp_path)

        return {
            "text": text,
            "duration": "unknown",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"STT error: {str(e)}")


@router.get("/voice/synthesize")
async def synthesize_speech(text: str):
    """Synthesize speech from text"""
    try:
        audio_bytes = await tts_engine.synthesize(text)

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"TTS error: {str(e)}")


@router.get("/voice/voices")
async def list_voices():
    """Get available TTS voices"""
    return {
        "voices": tts_engine.get_available_voices()
    }


# ==========================================
# Memory Endpoints
# ==========================================

@router.get("/memory/conversations")
async def get_conversations(limit: int = 10):
    """Get recent conversation history"""
    conversations = memory_store.get_recent_conversations(limit)
    return {"conversations": conversations}


@router.get("/memory/visual")
async def get_visual_memories(limit: int = 20):
    """Get recent visual memories"""
    memories = memory_store.get_visual_memories(limit)
    return {"memories": memories}


@router.post("/memory/search")
async def search_memories(query: str, n_results: int = 5):
    """Search memories by query"""
    results = memory_store.search_memories(query, n_results)
    return {"results": results}


@router.post("/memory/find-object")
async def find_object_in_memory(object_name: str):
    """Search memory for specific object"""
    results = memory_store.search_memories(object_name, n_results=10, memory_type="visual")

    # Filter results that mention the object
    matches = []
    for result in results:
        metadata = result.get("metadata", {})
        objects = metadata.get("objects", [])
        if any(object_name.lower() in obj.lower() for obj in objects):
            matches.append(result)

    return {
        "object": object_name,
        "found": len(matches) > 0,
        "memories": matches
    }


@router.delete("/memory")
async def clear_memory(memory_type: Optional[str] = None):
    """Clear memories (optionally by type)"""
    memory_store.clear_memories(memory_type)
    return {"status": "cleared", "type": memory_type}


# ==========================================
# WebSocket Endpoints
# ==========================================

@router.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming video frames"""
    global connection_counter

    await websocket.accept()
    connection_id = connection_counter
    connection_counter += 1
    active_connections[connection_id] = websocket

    try:
        while True:
            # Check for messages from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                # Handle client commands if needed
                if data == "stop":
                    break
            except asyncio.TimeoutError:
                pass

            # Capture and send frame
            frame = await asyncio.to_thread(detector.capture_frame)
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    "type": "frame",
                    "image": frame_b64,
                    "timestamp": datetime.now().isoformat()
                })

            await asyncio.sleep(0.2)  # 5 FPS for CPU optimization

    except WebSocketDisconnect:
        pass
    finally:
        active_connections.pop(connection_id, None)


@router.websocket("/ws/detections")
async def detections_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming detection results"""
    await websocket.accept()

    try:
        while True:
            frame = await asyncio.to_thread(detector.capture_frame)
            if frame is not None:
                detection = await asyncio.to_thread(detector.detect_objects, frame)
                
                # Base64 encode the image so the frontend can show it
                image_b64 = None
                if detection.get("image") is not None:
                    _, buffer = cv2.imencode('.jpg', detection["image"], [cv2.IMWRITE_JPEG_QUALITY, 70])
                    image_b64 = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    "type": "detection",
                    "objects": detection["objects"],
                    "counts": detection["counts"],
                    "total": detection["frame_count"],
                    "timestamp": datetime.now().isoformat(),
                    "image": image_b64
                })

            await asyncio.sleep(0.1)  # 10 FPS for smoother detections

    except WebSocketDisconnect:
        pass


@router.websocket("/ws/events")
async def events_stream(websocket: WebSocket):
    """WebSocket for proactive events and alerts"""
    await websocket.accept()
    
    async def send_event(event_data):
        try:
            # Optionally synthesize voice here
            if event_data.get("type") == "proactive_alert":
                text = event_data.get("message")
                audio_bytes = await tts_engine.synthesize(text)
                event_data["audio"] = base64.b64encode(audio_bytes).decode('utf-8')
            
            await websocket.send_json(event_data)
        except Exception as e:
            print(f"Error sending event: {e}")

    proactive_manager.set_callback(send_event)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        proactive_manager.set_callback(None)
