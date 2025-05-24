import cv2
import threading
import time
from ultralytics import YOLO
from tkinter import Tk, Label, Button, StringVar
from PIL import Image, ImageTk
import pyttsx3
from collections import Counter

class SmartCamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("SmartCam Answer Bot")

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # Set up GUI
        self.label = Label(self.window)
        self.label.pack()

        # Label to show detected object counts
        self.obj_text_var = StringVar()
        self.obj_label = Label(self.window, textvariable=self.obj_text_var, font=("Arial", 12))
        self.obj_label.pack()

        self.btn_quit = Button(self.window, text="Quit", command=self.quit_app)
        self.btn_quit.pack()

        self.cap = cv2.VideoCapture(0)
        self.running = True

        # Voice engine
        self.engine = pyttsx3.init()

        # Important objects to announce
        self.important_objects = {"person", "laptop", "cell phone", "kite"}

        self.last_spoken = 0
        self.last_counts = Counter()

        # Mute flag (optional - can be extended)
        self.is_muted = False

        # Start webcam thread
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.window.mainloop()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

            # Process detections with confidence threshold
            names = results[0].names
            boxes = results[0].boxes
            detected_labels = []

            for box in boxes:
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                label = names[cls_id]

                if conf >= 0.4:
                    detected_labels.append(label)

            # Count all detected objects for display (filtered by confidence)
            all_counts = Counter(detected_labels)

            if all_counts:
                phrases_all = [f"{count} {label}" + ("s" if count > 1 else "") for label, count in all_counts.items()]
                display_text = "Detected: " + ", ".join(phrases_all)
            else:
                display_text = "No objects detected."

            self.obj_text_var.set(display_text)

            # Filter important objects for announcements
            important_counts = Counter([label for label in detected_labels if label in self.important_objects])

            # Announce if changed and every 5 seconds, and not muted
            if time.time() - self.last_spoken > 5:
                if important_counts != self.last_counts and not self.is_muted:
                    self.speak_objects(important_counts)
                    self.last_spoken = time.time()
                    self.last_counts = important_counts

    def speak_objects(self, counts):
        if not counts:
            self.speak("No important objects detected.")
            return

        phrases = [f"{count} {label}" + ("s" if count > 1 else "") for label, count in counts.items()]
        message = "I see " + ", ".join(phrases)
        self.speak(message)

    def speak(self, text):
        print("Speaking:", text)
        self.engine.say(text)
        self.engine.runAndWait()

    def quit_app(self):
        self.running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    SmartCamApp(Tk())
