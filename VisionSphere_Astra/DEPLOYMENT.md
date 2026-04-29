# 🚀 VisionSphere Astra - Deployment Guide

**For: Intel i5-8350U, 16GB RAM, No GPU, Zero Budget**

---

## 📋 Quick Start Checklist

### Step 1: Install Ollama (Required)
```bash
# Download from: https://ollama.ai
# Install and run:
ollama serve

# Pull the CPU-optimized model:
ollama pull llama3.2:1b
```

### Step 2: Run Setup
```bash
cd VisionSphere_Astra
.\setup.bat
```

### Step 3: Start Astra
```bash
.\start.bat
```

This opens two windows:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000

---

## 🏠 Accessing from iPhone (Free Methods)

### Method 1: Tailscale (Recommended - Works Anywhere)

Tailscale is already installed on your laptop! This creates a secure tunnel.

**On Laptop:**
```bash
# Find your Tailscale IP
tailscale ip
# Example: 100.108.146.68
```

**On iPhone:**
1. Install Tailscale from App Store (free)
2. Sign in with same account
3. Enable Tailscale
4. Open Safari → Go to: `http://<laptop-tailscale-ip>:8000`
5. Tap Share → Add to Home Screen

**Result:** Astra accessible from iPhone anywhere in the world!

---

### Method 2: Local Network (Home Only)

**On Laptop:**
```bash
# Find your local IP
ipconfig
# Look for "IPv4 Address" under Wi-Fi, e.g., 192.168.2.68
```

**On iPhone (same Wi-Fi):**
1. Open Safari
2. Go to: `http://192.168.2.68:8000`
3. Tap Share → Add to Home Screen

**Note:** Only works when both devices on same Wi-Fi.

---

### Method 3: Cloudflare Tunnel (Advanced - Custom Domain)

For a real domain like `astra.yourname.com`:

1. Sign up at https://cloudflare.com (free)
2. Install cloudflared:
```bash
winget install cloudflare.cloudflared
```

3. Create tunnel:
```bash
cloudflared tunnel create astra
cloudflared tunnel route dns astra astra.yourname.com
cloudflared tunnel run astra
```

**Result:** Secure HTTPS access from anywhere, no port forwarding!

---

## ⚙️ Optimization for Your Hardware

### CPU Optimization (i5-8350U)

Your CPU has 4 cores / 8 threads. Here's how we optimize:

```env
# .env settings (already configured)
LLM_MODEL=llama3.2:1b      # Small, fast model
STT_MODEL=tiny             # Fastest Whisper
VIDEO_FPS=5                # Low FPS when idle
VIDEO_FPS_ACTIVE=15        # Higher when actively viewing
MAX_WORKERS=2              # Don't overload CPU
```

### Memory Management (16GB RAM)

Models are loaded lazily (on first use) to save RAM:
- LLM: ~2 GB (when first query made)
- Whisper: ~500 MB (when first voice input)
- YOLOv8n: ~50 MB (always loaded)
- EasyOCR: ~1 GB (when first OCR call)

**Total when all active: ~4 GB** (leaves 12 GB for system)

---

## 🔧 Running 24/7

### Option A: Laptop Always On

**Pros:** Free, uses existing hardware
**Cons:** Battery wear, not portable when server running

**Setup:**
1. Keep laptop plugged in
2. Disable sleep: Settings → Power → Never sleep
3. Run Astra in background (see below)

---

### Option B: Run as Windows Service

Run Astra automatically on startup:

**Using NSSM (Non-Sucking Service Manager):**

1. Download NSSM: https://nssm.cc/download
2. Extract to `C:\nssm`

3. Install backend service:
```bash
cd C:\nssm
nssm install AstraBackend
# Set Path: C:\path\to\VisionSphere_Astra\backend\venv\Scripts\python.exe
# Set Args: -m app.main
# Set Startup: C:\path\to\VisionSphere_Astra\backend
```

4. Install frontend (use http-server):
```bash
npm install -g http-server
nssm install AstraFrontend
# Set Path: C:\path\to\node\npm.cmd
# Set Args: run start
# Set Startup: C:\path\to\VisionSphere_Astra\frontend
```

5. Start services:
```bash
nssm start AstraBackend
nssm start AstraFrontend
```

**Result:** Astra starts automatically on boot!

---

### Option C: Future Mini PC (~$150-400)

When you have budget, get a dedicated mini PC:

**Budget Option (~$150-200):**
- Used Dell Optiplex / HP EliteDesk
- Add low-profile GPU if needed (GTX 1650)
- Run 24/7 at home

**Better Option (~$400):**
- Beelink SER5 (Ryzen 7 5700G, 8 core / 16 thread)
- 32GB RAM
- Much faster than your laptop for AI

**Best Option (~$800):**
- Mac Mini M2 (16GB)
- Excellent for ML inference
- Silent, low power

---

## 📊 Performance Expectations

### On Your Dell Latitude (i5-8350U)

| Task | Expected Performance |
|------|---------------------|
| Object Detection | 10-15 FPS |
| LLM Response | 5-10 tokens/sec |
| Voice Transcription | ~1 sec for 5 sec audio |
| TTS Generation | Real-time (no lag) |
| OCR (text reading) | 1-2 sec per frame |
| Screen Capture | Instant |

### Latency Breakdown

```
User asks "What do you see?" (voice):
├─ Voice recording: 2-3 seconds (user speaking)
├─ Whisper STT: ~1 second
├─ LLM processing: ~2 seconds
├─ TTS generation: ~1 second
└─ Audio playback: 2-3 seconds
─────────────────────────────────
Total: ~6-9 seconds for full loop
```

**Optimization:** Stream responses (TTS starts while LLM generating)
**Target:** 4-5 seconds total

---

## 🔒 Security

### Tailscale (Recommended)
✅ Encrypted (WireGuard)
✅ No port forwarding
✅ Authentication required
✅ Free for personal use

### Cloudflare Tunnel
✅ HTTPS encryption
✅ DDoS protection
✅ No open ports
✅ Free tier available

### Port Forwarding (NOT Recommended)
❌ Exposes your laptop to internet
❌ Requires router configuration
❌ Security risks
❌ ISP may block ports

---

## 🐛 Troubleshooting

### "Ollama not found"
```bash
# Make sure Ollama is running:
ollama list

# If not running:
ollama serve

# If model missing:
ollama pull llama3.2:1b
```

### "Camera not available"
```bash
# Check camera index in .env
CAMERA_INDEX=0  # Try 1, 2, etc.

# Test camera:
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### "Port 8000 already in use"
```bash
# Find and kill process:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change port in .env:
PORT=8001
```

### iPhone can't connect
```bash
# Check firewall:
# Windows Defender → Firewall → Allow app through firewall
# Allow Python and Node.js

# Check Tailscale:
tailscale status

# Try local IP first:
http://192.168.2.68:8000
```

---

## 📱 iPhone PWA Setup

Once accessing via Safari:

1. Open `http://<your-laptop>:8000`
2. Tap **Share** button (box with arrow)
3. Scroll down → **Add to Home Screen**
4. Name it "Astra"
5. Tap **Add**

**Result:** Astra appears as an app icon!

---

## 🎯 Next Steps After Setup

1. ✅ Test basic chat: "What do you see?"
2. ✅ Test voice: Click mic, speak
3. ✅ Test screen sharing: Go to Screen tab
4. ✅ Test OCR: Point camera at text, click "Read text"
5. ✅ Test from iPhone: Access via Tailscale
6. ✅ Add to iPhone home screen as PWA

---

## 💡 Usage Tips

### Voice Commands That Work Well:
- "What do you see?"
- "Are there any people?"
- "Count the laptops"
- "Read the text on the screen"
- "What's on my monitor?"
- "Where did I put my keys?" (searches memory)

### Pro Tips:
1. Keep laptop camera unobstructed
2. Use good lighting for better detection
3. Speak clearly for voice recognition
4. Use "Screen" tab for coding help
5. Check "Memory" tab for past observations

---

## 🆘 Getting Help

If something breaks:

1. Check backend terminal for errors
2. Check frontend console (F12)
3. Verify Ollama is running: `ollama list`
4. Restart both backend and frontend
5. Check `.env` settings match your setup

---

Made with ❤️ for your Dell Latitude!
