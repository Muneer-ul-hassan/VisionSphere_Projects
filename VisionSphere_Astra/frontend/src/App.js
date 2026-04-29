import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  // State
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [detections, setDetections] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [memories, setMemories] = useState([]);
  const [screenImage, setScreenImage] = useState(null);
  const [ocrText, setOcrText] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);

  // Refs
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check health on mount
  useEffect(() => {
    checkHealth();
    connectDetections();
    connectEvents();
    loadMemories();
  }, []);

  // Health check
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      console.error('Health check failed:', err);
      setHealth({ status: 'disconnected' });
    }
  };

  // Connect to detections WebSocket
  const connectDetections = () => {
    const ws = new WebSocket('ws://localhost:8000/api/ws/detections');
    ws.onopen = () => console.log('Connected to detections stream');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'detection') {
        setDetections(data);
      }
    };
    ws.onerror = (err) => console.error('Detections WebSocket error:', err);
    return () => ws.close();
  };

  // Connect to events WebSocket (Alerts, greetings, etc.)
  const connectEvents = () => {
    const ws = new WebSocket('ws://localhost:8000/api/ws/events');
    ws.onopen = () => console.log('Connected to events stream');
    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'proactive_alert') {
        // Add to chat
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
          isProactive: true
        }]);

        // Play audio if provided
        if (data.audio) {
          const audioBlob = b64ToBlob(data.audio, 'audio/mpeg');
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          setIsSpeaking(true);
          audio.onend = () => {
            setIsSpeaking(false);
            URL.revokeObjectURL(audioUrl);
          };
          audio.play();
        }
      }
    };
    ws.onerror = (err) => console.error('Events WebSocket error:', err);
    return () => ws.close();
  };

  // Utility to convert base64 to blob
  const b64ToBlob = (b64Data, contentType = '', sliceSize = 512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays, { type: contentType });
  };
  const loadMemories = async () => {
    try {
      const res = await fetch(`${API_BASE}/memory/visual?limit=10`);
      const data = await res.json();
      setMemories(data.memories || []);
    } catch (err) {
      console.error('Failed to load memories:', err);
    }
  };

  // Send message
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMsg = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    setMessages(prev => [...prev, { role: 'user', content: userMsg, timestamp: new Date() }]);

    try {
      const sceneQuestions = ['what do you see', 'what can you see', 'describe', 'whats there', "what's there", 'people', 'person'];
      const screenQuestions = ['screen', 'display', 'monitor', 'what\'s on my', 'whats on my'];
      const isSceneQuestion = sceneQuestions.some(q => userMsg.toLowerCase().includes(q));
      const isScreenQuestion = screenQuestions.some(q => userMsg.toLowerCase().includes(q));

      let response;
      if (isScreenQuestion) {
        const res = await fetch(`${API_BASE}/chat/screen?question=${encodeURIComponent(userMsg)}`, {
          method: 'POST'
        });
        response = await res.json();
        if (response.response) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: response.response,
            timestamp: new Date(),
            detections: response.objects
          }]);
          if (response.screen_image) {
            setScreenImage(response.screen_image);
          }
          speakResponse(response.response);
        }
      } else if (isSceneQuestion) {
        const res = await fetch(`${API_BASE}/chat/scene?question=${encodeURIComponent(userMsg)}`, {
          method: 'POST'
        });
        response = await res.json();
        if (response.response) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: response.response,
            timestamp: new Date(),
            detections: response.detections
          }]);
          speakResponse(response.response);
        }
      } else {
        const res = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg })
        });
        response = await res.json();
        if (response.response) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: response.response,
            timestamp: new Date()
          }]);
          speakResponse(response.response);
        }
      }
    } catch (err) {
      console.error('Chat error:', err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Make sure the backend is running.',
        timestamp: new Date(),
        error: true
      }]);
    }

    setIsLoading(false);
  };

  // Voice recording
  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await transcribeAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsListening(true);
    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Microphone access needed for voice input');
    }
  };

  const stopListening = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');

      const res = await fetch(`${API_BASE}/voice/transcribe`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.text) {
        setInputMessage(data.text);
        // Auto-send after transcription
        setTimeout(() => {
          setInputMessage(data.text);
        }, 100);
      }
    } catch (err) {
      console.error('Transcription error:', err);
    }
  };

  // Text to speech
  const speakResponse = async (text) => {
    try {
      const res = await fetch(`${API_BASE}/voice/synthesize?text=${encodeURIComponent(text)}`, {
        method: 'GET'
      });
      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      setIsSpeaking(true);
      audio.onend = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };
      audio.play();
    } catch (err) {
      console.error('TTS error:', err);
    }
  };

  // Screen capture
  const captureScreen = async () => {
    try {
      const res = await fetch(`${API_BASE}/screen/capture`);
      const data = await res.json();
      setScreenImage(data.image);
    } catch (err) {
      console.error('Screen capture error:', err);
    }
  };

  // OCR from camera
  const readText = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/ocr/read`, { method: 'POST' });
      const data = await res.json();
      setOcrText(data.text);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Found text: ${data.text}`,
        timestamp: new Date()
      }]);
    } catch (err) {
      console.error('OCR error:', err);
    }
    setIsLoading(false);
  };

  // Search memory for objects
  const searchObject = async (objectName) => {
    try {
      const res = await fetch(`${API_BASE}/memory/find-object?object_name=${encodeURIComponent(objectName)}`, {
        method: 'POST'
      });
      const data = await res.json();
      setSearchResults(data);
    } catch (err) {
      console.error('Search error:', err);
    }
  };

  // Search memories by text
  const searchMemories = async () => {
    if (!searchQuery.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/memory/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, n_results: 10 })
      });
      const data = await res.json();
      setSearchResults({ memories: data.results });
    } catch (err) {
      console.error('Search error:', err);
    }
  };

  // Quick actions
  const quickActions = [
    { label: '👁️ What do you see?', query: 'What do you see?' },
    { label: '👥 Any people?', query: 'Are there any people?' },
    { label: '📺 Screen check', query: 'What\'s on my screen?', type: 'screen' },
    { label: '📝 Read text', query: 'Read text', action: readText },
  ];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>🔮 Astra</h1>
          <div className="status">
            <span className={`status-dot ${health?.llm_connected ? 'connected' : 'disconnected'}`}></span>
            <span className="status-text">{health?.llm_connected ? 'Online' : 'Offline'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Left Panel - Video & Detections */}
        <div className="left-panel">
          <div className="video-section">
            <h2>📹 {activeTab === 'screen' ? 'Screen Capture' : 'Live View'}</h2>
            <div className="video-container">
              {screenImage ? (
                <img src={`data:image/jpeg;base64,${screenImage}`} alt="Screen" className="video-feed" />
              ) : detections?.image ? (
                <img src={`data:image/jpeg;base64,${detections.image}`} alt="Camera feed" className="video-feed" />
              ) : (
                <div className="video-placeholder">
                  <p>Waiting for camera...</p>
                  <p className="hint">Make sure backend is running</p>
                </div>
              )}
            </div>

            {/* Detection Stats */}
            {detections && (
              <div className="detection-stats">
                <div className="stat">
                  <span className="stat-value">{detections.total || 0}</span>
                  <span className="stat-label">Objects</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{Object.keys(detections.counts || {}).length}</span>
                  <span className="stat-label">Types</span>
                </div>
              </div>
            )}

            {/* Detected Objects */}
            {detections && detections.counts && Object.keys(detections.counts).length > 0 && (
              <div className="detected-objects">
                <h3>Detected:</h3>
                <div className="objects-list">
                  {Object.entries(detections.counts).map(([label, count]) => (
                    <span key={label} className="object-tag">{label} ×{count}</span>
                  ))}
                </div>
              </div>
            )}

            {/* OCR Text */}
            {ocrText && (
              <div className="ocr-section">
                <h3>📝 Text Found:</h3>
                <p className="ocr-text">{ocrText}</p>
                <button className="clear-btn" onClick={() => setOcrText(null)}>Clear</button>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Chat & Tabs */}
        <div className="right-panel">
          {/* Tabs */}
          <div className="tabs">
            <button className={`tab ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>💬 Chat</button>
            <button className={`tab ${activeTab === 'memory' ? 'active' : ''}`} onClick={() => setActiveTab('memory')}>🧠 Memory</button>
            <button className={`tab ${activeTab === 'screen' ? 'active' : ''}`} onClick={() => { setActiveTab('screen'); captureScreen(); }}>📺 Screen</button>
            <button className={`tab ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>⚙️</button>
          </div>

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="chat-container">
              <div className="messages">
                {messages.length === 0 && (
                  <div className="welcome-message">
                    <p>👋 Hi! I'm Astra</p>
                    <p>Ask me about what I see or your screen</p>
                  </div>
                )}

                {messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-content">
                      {msg.content}
                      {msg.detections && (
                        <div className="message-detections">
                          <small>Seen: {msg.detections.map(d => d.label).join(', ')}</small>
                        </div>
                      )}
                      {msg.error && <span className="error-hint">Check backend</span>}
                    </div>
                    <div className="message-actions">
                      <span className="message-time">{msg.timestamp?.toLocaleTimeString()}</span>
                      {!msg.error && (
                        <button className="speak-btn" onClick={() => speakResponse(msg.content)}>🔊</button>
                      )}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="message assistant loading">
                    <div className="message-content">
                      <span className="typing-indicator">Astra is thinking...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Quick Actions */}
              <div className="quick-actions">
                {quickActions.map((action, idx) => (
                  <button
                    key={idx}
                    className="quick-btn"
                    onClick={() => action.action ? action.action() : setInputMessage(action.query)}
                    disabled={isLoading}
                  >
                    {action.label}
                  </button>
                ))}
              </div>

              {/* Voice + Input */}
              <form className="chat-input" onSubmit={sendMessage}>
                <button
                  type="button"
                  className={`voice-btn ${isListening ? 'listening' : ''}`}
                  onClick={isListening ? stopListening : startListening}
                  disabled={isSpeaking}
                >
                  {isListening ? '🔴' : '🎤'}
                </button>
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask me anything..."
                  disabled={isLoading || isListening}
                />
                <button type="submit" disabled={isLoading || !inputMessage.trim()}>Send</button>
              </form>
            </div>
          )}

          {/* Memory Tab */}
          {activeTab === 'memory' && (
            <div className="memory-container">
              <h2>🧠 Memory</h2>
              <div className="memory-search">
                <input
                  type="text"
                  placeholder="Search memories..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && searchMemories()}
                />
                <button onClick={searchMemories}>Search</button>
              </div>

              {searchResults && (
                <div className="search-results">
                  <h3>Results:</h3>
                  {searchResults.memories?.map((mem, idx) => (
                    <div key={idx} className="memory-item">
                      <p>{mem.content}</p>
                      <small>{new Date(mem.metadata?.timestamp).toLocaleString()}</small>
                    </div>
                  ))}
                </div>
              )}

              <h3>Recent Visual Memories</h3>
              {memories.length === 0 ? (
                <p className="no-memories">No memories yet</p>
              ) : (
                <div className="memories-list">
                  {memories.map((memory, idx) => (
                    <div key={idx} className="memory-item">
                      <p className="memory-content">{memory.description}</p>
                      <small className="memory-time">
                        {new Date(memory.metadata?.timestamp).toLocaleString()}
                      </small>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Screen Tab */}
          {activeTab === 'screen' && (
            <div className="screen-container">
              <h2>📺 Screen Analysis</h2>
              <button className="action-btn" onClick={captureScreen}>📸 Capture Screen</button>
              <button className="action-btn" onClick={() => sendMessage({ preventDefault: () => {} })}>
                💬 Ask about screen
              </button>
              {screenImage && (
                <div className="screen-image-container">
                  <img src={`data:image/jpeg;base64,${screenImage}`} alt="Screen capture" className="screen-capture" />
                </div>
              )}
            </div>
          )}

          {/* Settings Tab */}
          {activeTab === 'settings' && (
            <div className="settings-container">
              <h2>⚙️ Settings</h2>
              <div className="setting-group">
                <h3>Status</h3>
                <div className="status-item">
                  <span>Backend:</span>
                  <span className={health?.status === 'healthy' ? 'status-ok' : 'status-error'}>
                    {health?.status || 'Unknown'}
                  </span>
                </div>
                <div className="status-item">
                  <span>LLM:</span>
                  <span className={health?.llm_connected ? 'status-ok' : 'status-error'}>
                    {health?.llm_connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="status-item">
                  <span>Model:</span>
                  <span>{health?.model || 'N/A'}</span>
                </div>
                <button className="refresh-btn" onClick={checkHealth}>🔄 Refresh</button>
              </div>

              <div className="setting-group">
                <h3>Actions</h3>
                <button className="action-btn" onClick={() => setMessages([])}>🗑️ Clear Chat</button>
                <button className="action-btn" onClick={async () => {
                  await fetch(`${API_BASE}/memory`, { method: 'DELETE' });
                  loadMemories();
                }}>🗑️ Clear Memory</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
