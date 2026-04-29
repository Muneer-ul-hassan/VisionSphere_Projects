# ⚡ Quick Start - VisionSphere Astra

**5 minutes to your first conversation!**

---

## Prerequisites Check

```bash
# 1. Check Python
python --version
# Should show: Python 3.9+

# 2. Check Node.js
node --version
# Should show: v16+

# 3. Check Ollama
ollama --version
# If not installed, download from: https://ollama.ai
```

---

## Step 1: Install Ollama Model

```bash
# Pull the AI model (CPU-optimized)
ollama pull llama3.2:1b

# Wait for download (~500MB)
# This takes 2-5 minutes depending on internet
```

---

## Step 2: Run Setup Script

```bash
cd D:\projects\VisionSphere_Projects\VisionSphere_Astra
.\setup.bat
```

This will:
- Create Python virtual environment
- Install all Python packages
- Install Node.js dependencies
- Check Ollama is running

**Wait for:** `Setup Complete!`

---

## Step 3: Start Astra

```bash
.\start.bat
```

Two windows will open:

**Window 1 (Backend):**
```
🚀 Starting VisionSphere Astra...
📷 Camera index: 0
🤖 LLM Model: llama3.2:1b
✅ Camera initialized
✅ LLM connected
```

**Window 2 (Frontend):**
```
Compiled successfully!
You can now view Astra in the browser.
Local: http://localhost:3000
```

---

## Step 4: Allow Camera Access

Browser will ask for camera permission → **Allow**

You should see live video feed in the left panel.

---

## Step 5: Talk to Astra!

### Type or click quick actions:

1. **"👁️ What do you see?"** → Astra describes the scene
2. **"👥 Any people?"** → Checks for people
3. **"📝 Read text"** → Points camera at text to read

### Or type anything:
- "Describe what's in front of the camera"
- "How many objects can you see?"
- "Is there a laptop in view?"

---

## Step 6: Try Voice Input

1. Click the **🎤 microphone** button
2. Speak: "What do you see?"
3. Wait for transcription
4. Astra responds automatically!

---

## Step 7: Access from iPhone (Optional)

### Find your Tailscale IP:
```bash
tailscale ip
# Example: 100.108.146.68
```

### On iPhone Safari:
1. Go to: `http://100.108.146.68:8000`
2. (Replace with your IP)
3. You should see Astra!

### Add to Home Screen:
1. Tap **Share** button
2. **Add to Home Screen**
3. Name it "Astra"
4. Now it's like a native app!

---

## Expected First Run

```
[Backend Terminal]
🚀 Starting VisionSphere Astra...
📷 Camera index: 0
🤖 LLM Model: llama3.2:1b
🌐 Ollama URL: http://localhost:11434
✅ Camera initialized
✅ LLM connected
💾 Memory DB path: ./data/memory_db
INFO:     Uvicorn running on http://0.0.0.0:8000
```

```
[Frontend Browser]
http://localhost:3000 opens automatically
```

---

## Troubleshooting

### ❌ "Ollama connection failed"
```bash
# Start Ollama manually
ollama serve

# In another terminal:
ollama pull llama3.2:1b
```

### ❌ "Camera not available"
- Close other apps using camera (Zoom, Teams)
- Try different camera index in `.env`: `CAMERA_INDEX=1`

### ❌ "Port already in use"
```bash
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <number> /F
```

### ❌ Frontend doesn't open
- Manually open browser: http://localhost:3000
- Or check frontend terminal for errors

---

## What's Working After Setup

✅ Live camera feed with object detection
✅ Chat with Astra about what it sees
✅ Voice input (click mic, speak)
✅ Screen sharing (capture your screen)
✅ OCR text reading from camera
✅ Visual memory (remembers what it saw)
✅ iPhone access via Tailscale

---

## Next Steps

1. ✅ Test all features
2. ✅ Read [DEPLOYMENT.md](./DEPLOYMENT.md) for 24/7 setup
3. ✅ Add to iPhone home screen
4. ✅ Customize in `.env` if needed

---

**Need help?** Check the full [README.md](./README.md)
