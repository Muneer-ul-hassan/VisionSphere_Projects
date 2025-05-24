# SmartCam Answer Bot

## Overview

SmartCam Answer Bot is an interactive AI assistant that uses a live camera feed to detect and identify objects in real-time.  
Users can ask, either by voice or via a GUI button, "What do you see?" and the assistant will respond with a spoken description of the scene, including object types and counts.  

This project demonstrates a practical implementation of visual question answering (VQA) using computer vision and speech synthesis.

## Features

- Real-time object detection using YOLOv8
- On-demand voice responses triggered by user interaction (voice command or button press)
- Speech synthesis with `pyttsx3` for natural voice feedback
- GUI built with Tkinter displaying live video feed and textual feedback
- Confidence threshold filtering to reduce false positives
- Object count summarization with pluralization support
- Option to mute/unmute voice responses (can be added)
- Easily extendable for further multimodal interaction

## Technologies Used

- Python 3.x
- OpenCV (for webcam video capture and image processing)
- Ultralytics YOLOv8 (for object detection)
- pyttsx3 (for offline text-to-speech)
- Tkinter (for GUI)
- threading (to handle video capture and UI responsiveness)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/smartcam-answer-bot.git
   cd smartcam-answer-bot
