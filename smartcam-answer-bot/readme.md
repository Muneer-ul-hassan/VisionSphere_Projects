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
 
## Quick setup (Windows)

1. (Optional) If you want to use the included virtual environment, activate it:

PowerShell:

```powershell
.\smartcam-env\Scripts\Activate.ps1
```

Command Prompt:

```bat
.\smartcam-env\Scripts\activate.bat
```

2. Install required packages (if not using the included venv or after activation):

```powershell
pip install -r requirements.txt
```

3. Run the GUI:

```powershell
python smartcam_gui.py
```

Or run the on-demand script:

```powershell
python smartcam_on_demand.py
```

## Notes

- A ready-made virtual environment is included at `smartcam-env/`. It's provided for convenience but may contain platform-specific binary wheels. If you see import errors, recreate a fresh venv and install `requirements.txt`.
- The YOLOv8 weights file `yolov8n.pt` is included for immediate testing. Replace it with custom weights by updating the script paths.

## Troubleshooting

- If the camera doesn't open, try changing the camera index (0, 1, ...) in the script.
- If TTS output is silent, ensure `pyttsx3` dependencies are available in the environment.

## Contributing

Contributions are welcome — open an issue or a PR with changes or feature requests.
