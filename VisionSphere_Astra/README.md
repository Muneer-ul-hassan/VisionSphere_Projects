# 🔮 VisionSphere Astra

**Your Personal AI Assistant with Vision** - Inspired by Google's Project Astra

A multimodal AI assistant that can see what you see, remember what you've seen, and converse naturally about the world around you.

> **Note for recruiter:** You can watch the full multimodal AI demonstration video below, featuring real-time vision, memory, and conversation:

https://github.com/Muneer-ul-hassan/VisionSphere_Projects/blob/main/VisionSphere_Astra/demo.mp4

---

## 🌟 Core Features

### 👁️ Always-On Contextual Vision
- **Real-Time Object Detection**: Powered by a custom **YOLOv8** implementation with Non-Maximum Suppression (NMS) for precise, duplicate-free entity tracking.
- **Continuous Scene Awareness**: Astra constantly monitors the camera feed and seamlessly injects real-time visual context into every conversation, allowing it to "see" without you having to prompt it.

### 🧠 Advanced Persistent Memory (ChromaDB)
- **Semantic Conversation History**: Stores and retrieves past conversations using vector embeddings, ensuring Astra remembers your preferences, identity, and past discussions.
- **Visual Memory Logging**: Automatically logs significant visual events (e.g., "saw a person at 10:45 AM") allowing you to query past visual states.

### 🗣️ Natural Multimodal Interaction
- **Seamless Context Blending**: Combines explicit user text, implied visual state, and long-term memory into a unified prompt for the LLM.
- **Instant Audio Feedback**: Built-in Text-to-Speech (TTS) integration allows Astra to speak its answers instantly.
- **Voice Input Ready**: Integration for STT (Speech-to-Text) allowing you to talk to Astra naturally.

### 🔒 Privacy-First Local Architecture
- **100% Local Execution**: Runs entirely on your machine without any cloud dependencies.
- **Ollama + Gemma**: Uses powerful local LLMs (like Google's Gemma) ensuring your camera feed and personal conversations never leave your device.
- **Optimized for CPU/GPU**: Built with asynchronous FastAPI endpoints to ensure smooth video streaming without bottlenecking the LLM.

### 🌐 Modern React Dashboard
- **Live Video Streaming**: Real-time WebSocket connection for low-latency video feed.
- **Dynamic Data Visualization**: Live counters and tracking statistics for detected objects.
- **Unified Interface**: Chat, memory explorer, and settings integrated into a single, sleek dark-mode UI.

---

## 🚀 Coming Soon
- 📝 **OCR integration** for reading text from the camera
- 🔍 **CLIP-based scene understanding** for complex visual reasoning
- ⚡ **Proactive Alerts** (e.g., "You left your keys on the table")

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   VisionSphere Astra                    │
├─────────────────────────────────────────────────────────┤
│  Frontend (React)          │  Backend (FastAPI)        │
│  - Live video display      │  - Vision (YOLOv8)        │
│  - Chat interface          │  - LLM (Ollama/Gemma)     │
│  - Memory viewer           │  - Memory (ChromaDB)      │
│  - Settings panel          │  - WebSocket streaming    │
└─────────────────────────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   Ollama       │
                    │   (Gemma 2B)   │
                    └────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **Node.js 16+** installed
3. **Ollama** installed and running

### Step 1: Install Ollama

```bash
# Download from https://ollama.ai
# Then install the Gemma model:
ollama pull gemma2:2b
```

### Step 2: Setup Backend

```bash
cd VisionSphere_Astra/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m app.main
```

Backend will start at: **http://localhost:8000**

### Step 3: Setup Frontend

```bash
cd VisionSphere_Astra/frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will open at: **http://localhost:3000**

---

## 📁 Project Structure

```
VisionSphere_Astra/
├── backend/
│   ├── app/
│   │   ├── api/          # REST + WebSocket endpoints
│   │   ├── llm/          # Ollama client
│   │   ├── vision/       # YOLO detection
│   │   ├── memory/       # ChromaDB storage
│   │   ├── config.py     # Configuration
│   │   └── main.py       # FastAPI app
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   └── package.json
└── README.md
```

---

## ⚙️ Configuration

Edit `backend/.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `gemma2:2b` | LLM model to use |
| `CAMERA_INDEX` | `0` | Webcam index |
| `CONFIDENCE_THRESHOLD` | `0.4` | Detection confidence |

---

## 🎮 Usage

### Chat Commands

- **"What do you see?"** - Astra describes the current scene
- **"Are there any people?"** - Ask about specific objects
- **"Count the laptops"** - Request object counts
- **"Describe your view"** - Get a detailed description

### Quick Actions

Use the preset buttons for common queries:
- 👁️ What do you see?
- 👥 Any people?
- 💻 Electronics?
- 📝 Read text

---

## 🔧 Troubleshooting

### Camera not working
```bash
# Check camera index in .env
# Try different indices: 0, 1, 2...
CAMERA_INDEX=0
```

### Ollama connection failed
```bash
# Make sure Ollama is running
ollama list

# If not running:
ollama serve
```

### Model not found
```bash
# Pull the required model
ollama pull gemma2:2b
```

---

## 🛣️ Roadmap

### Phase 1: Foundation ✅
- [x] Basic architecture
- [x] YOLOv8 detection
- [x] Ollama LLM integration
- [x] Web UI dashboard
- [x] Memory system

### Phase 2: Enhanced Perception
- [ ] OCR text reading (EasyOCR)
- [ ] CLIP scene understanding
- [ ] Face recognition (optional)
- [ ] Depth estimation

### Phase 3: Voice & Audio
- [ ] Speech-to-text (Whisper)
- [ ] Text-to-speech (Coqui/ElevenLabs)
- [ ] Voice activity detection
- [ ] Wake word detection

### Phase 4: Proactive Features
- [ ] Attention detection
- [ ] Context-aware interruptions
- [ ] Automatic suggestions
- [ ] Anomaly detection

### Phase 5: Multi-Device
- [ ] Screen sharing
- [ ] Multiple cameras
- [ ] Mobile companion app
- [ ] Cross-device sync

---

## 🤝 Contributing

Contributions welcome! Open an issue or PR with:
- Bug reports
- Feature requests
- Code improvements
- Documentation fixes

---

## 📄 License

MIT License - Feel free to use this for your projects!

---

## 🙏 Acknowledgments

- Inspired by **Google's Project Astra**
- Built with **Ultralytics YOLOv8**
- Powered by **Ollama** and **Gemma**
- UI inspired by modern AI assistants

---

## 📬 Contact

Project Link: [VisionSphere_Projects](https://github.com/yourusername/VisionSphere_Projects)

Made with ❤️ by the VisionSphere Team
