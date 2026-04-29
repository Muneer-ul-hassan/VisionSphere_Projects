# 🔮 VisionSphere Astra

**Your Personal AI Assistant with Vision** - Inspired by Google's Project Astra

A multimodal AI assistant that can see what you see, remember what you've seen, and converse naturally about the world around you.

> **Note for recruiter:** You can watch the full multimodal AI demonstration video below, featuring real-time vision, memory, and conversation:
> 
> <video src="demo.mp4" controls width="100%"></video>

---

## 🌟 Features

### Current Capabilities
- 📹 **Live Camera Feed** - Real-time object detection with YOLOv8
- 💬 **Natural Conversation** - Chat with Astra about what it sees
- 🧠 **Visual Memory** - Remembers scenes and conversations
- 🤖 **Local LLM** - Runs on your machine with Ollama (no cloud required)
- 🌐 **Web UI** - Modern, responsive dashboard

### Coming Soon
- 🎤 Voice input/output (Whisper STT + TTS)
- 📝 OCR text reading
- 🔍 CLIP-based scene understanding
- ⚡ Proactive assistance
- 📱 Multi-device support

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
