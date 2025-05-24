import cv2
import threading
import time
from ultralytics import YOLO
from tkinter import Tk, Label, Button, StringVar
from PIL import Image, ImageTk
import pyttsx3
import speech_recognition as sr
from collections import Counter

class SmartCamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("SmartCam Answer Bot")

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # Setup video display
        self.label = Label(self.window)
        self.label.pack()

        # Text feedback label
        self.feedback_var = StringVar()
        self.feedback_label = Label(self.window, textvariable=self.feedback_var, font=("Arial", 12))
        self.feedback_label.pack()

        # Buttons
        self.ask_btn = Button(self.window, text="Ask SmartCam", command=self.manual_query)
        self.ask_btn.pack()

        self.quit_btn = Button(self.window, text="Quit", command=self.quit_app)
        self.quit_btn.pack()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # Voice engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)

        # Start video thread
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

        # Start voice recognition thread
        self.listen_thread = threading.Thread(target=self.listen_for_voice_command, daemon=True)
        self.listen_thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.window.mainloop()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Run YOLO detection
            results = self.model(frame, verbose=False)
            self.latest_results = results  # Store results for query

            # Annotate and display
            annotated = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

    def manual_query(self):
        self.describe_scene()

    def describe_scene(self):
        results = getattr(self, 'latest_results', None)
        if results is None:
            message = "No frame available yet."
        else:
            names = results[0].names
            boxes = results[0].boxes
            detected_ids = boxes.cls.tolist()
            confidences = boxes.conf.tolist()

            # Filter by confidence
            filtered = [
                names[int(cls)]
                for cls, conf in zip(detected_ids, confidences)
                if conf >= 0.4 and names[int(cls)] in ['person', 'laptop', 'cell phone', 'chair']
            ]

            if not filtered:
                message = "I see nothing important."
            else:
                counts = Counter(filtered)
                phrases = [f"{count} {label}" + ("s" if count > 1 else "") for label, count in counts.items()]
                message = "I see " + ", ".join(phrases)

        self.speak(message)
        self.feedback_var.set("SmartCam says: " + message)

    def speak(self, text):
        print("Speaking:", text)
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_for_voice_command(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)

        while self.running:
            with mic as source:
                print("Listening for voice command...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                    command = recognizer.recognize_google(audio).lower()
                    print("Heard:", command)
                    if "what do you see" in command or "what can you see" in command:
                        self.describe_scene()
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print("Speech recognition error:", e)
                    continue

    def quit_app(self):
        self.running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    SmartCamApp(Tk())
