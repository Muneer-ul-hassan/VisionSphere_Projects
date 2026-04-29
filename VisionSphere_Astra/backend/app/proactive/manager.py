import asyncio
import time
from typing import Dict, List, Optional, Callable
from app.vision.detector import detector
from app.vision.face_recognizer import face_recognizer
from app.voice.tts import tts_engine

class ProactiveManager:
    """Background monitoring and proactive behavior engine"""
    
    def __init__(self):
        self.is_running = False
        self.known_people_seen = set()
        self.unknown_person_start_time = 0
        self.last_alert_time = 0
        self.event_callback: Optional[Callable] = None
        self.loop_task: Optional[asyncio.Task] = None

    def set_callback(self, callback: Callable):
        self.event_callback = callback

    async def start(self):
        """Start the background monitoring loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.loop_task = asyncio.create_task(self._monitoring_loop())
        print("🚀 Proactive Manager started")

    async def stop(self):
        """Stop the monitoring loop"""
        self.is_running = False
        if self.loop_task:
            self.loop_task.cancel()
            try:
                await self.loop_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Main loop that analyzes the scene periodically"""
        while self.is_running:
            try:
                detection = detector.detect_objects()
                objects = detection["objects"]
                
                # Analyze for events
                await self._process_identities(objects)
                
                # Smooth the loop
                await asyncio.sleep(1.0) # Check every 1 second
                
            except Exception as e:
                print(f"⚠️ Proactive Manager error: {e}")
                await asyncio.sleep(5)

    async def _process_identities(self, objects: List[Dict]):
        """Handle greetings and auto-enrollment logic"""
        now = time.time()
        
        has_unknown = False
        known_names = []
        
        for obj in objects:
            if obj["type"] == "face":
                if obj["label"] == "Unknown":
                    has_unknown = True
                else:
                    known_names.append(obj["label"])
        
        # 1. Handle New/Known People (Greetings)
        for name in known_names:
            if name not in self.known_people_seen:
                self.known_people_seen.add(name)
                await self._trigger_alert(f"Hello {name}, I see you're here.")
        
        # 2. Handle Unknown People (Enrollment Request)
        if has_unknown:
            if self.unknown_person_start_time == 0:
                self.unknown_person_start_time = now
            elif now - self.unknown_person_start_time > 3.0: # 3 seconds persistence
                if now - self.last_alert_time > 30: # Don't annoy the user
                    await self._trigger_alert("I see a new face. Who is this?")
                    self.last_alert_time = now
        else:
            self.unknown_person_start_time = 0

    async def _trigger_alert(self, text: str):
        """Speak alert and send to UI"""
        print(f"📢 Proactive Alert: {text}")
        
        # Trigger TTS (Optional: can send audio or just text for UI to speak)
        if self.event_callback:
            await self.event_callback({
                "type": "proactive_alert",
                "message": text,
                "timestamp": time.time()
            })

# Global instance
proactive_manager = ProactiveManager()
