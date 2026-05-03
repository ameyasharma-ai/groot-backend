# ⚙️ GROOT Backend Core

> **Neural Processing & Audio Synthesis Server**

The GROOT Backend is a high-performance Python server built with FastAPI. It handles the "brain" of the system, including real-time speech-to-text transcription, multi-model LLM orchestration via OpenRouter, semantic long-term memory (RAG), and neural text-to-speech synthesis.

---

## 🧠 Core Intelligence

- **🗣️ Real-time STT**: Utilizes the **Groq Whisper (whisper-large-v3-turbo)** API for near-instant speech-to-text transcription.
- **🤖 Multi-Model Orchestration**: Connects to **OpenRouter** to leverage a wide array of LLMs (GPT-4o, Llama-3, Mistral, etc.). It features a **dynamic rerouting system** that automatically falls back to alternative free models if the primary model hits rate limits.
- **💾 RAG Memory System**: 
  - **Local Knowledge Base**: Uses TF-IDF vectorization and Cosine Similarity to store and retrieve specific facts from `knowledge_base.txt`.
  - **Wikipedia Integration**: Can perform live online searches to augment its knowledge.
- **🔊 Neural TTS**: Powered by **Edge-TTS** (Microsoft Edge Neural Voices) for high-quality, expressive speech synthesis across multiple characters (Lisa, Atlas, Nova).

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Server**: [Uvicorn](https://www.uvicorn.org/)
- **LLM SDK**: [OpenAI Python SDK](https://github.com/openai/openai-python) (OpenRouter compatible)
- **Audio Processing**: Groq API (STT), Edge-TTS (TTS)
- **Data Science**: Scikit-learn (TF-IDF Vectorization)
- **Networking**: WebSockets for low-latency duplex communication

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API Keys for **OpenRouter** and **Groq**

### Installation

1. Navigate to the `backend` directory:
   ```bash
   cd Backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements-cloud.txt
   ```
4. Create a `.env` file and add your keys:
   ```env
   OPENROUTER_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```
5. Launch the server:
   ```bash
   python server.py
   ```

## 🔌 API & WebSocket Protocol

The server exposes a main WebSocket endpoint at `/ws/chat`.

### Incoming Messages (to Server)
- **Audio Blob (bytes)**: RAW `.webm` audio data for transcription and processing.
- **JSON Payloads**:
  - `{"type": "interrupt"}`: Immediately halts current speech synthesis.
  - `{"type": "set_voice", "voice_id": "...", "system_prompt": "..."}`: Switches the AI personality and directive.

### Outgoing Messages (to Client)
- `{"type": "user_text", "text": "..."}`: Confirmed transcription of user input.
- `{"type": "bot_audio", "text": "...", "audio": "base64"}`: Synthesized speech chunk with corresponding text.
- `{"type": "action_status", "message": "..."}`: Current internal state (TRANSCRIBING, GENERATING, etc.).
- `{"type": "log", "message": "..."}`: Technical diagnostic logs for the terminal console.

---

*Powering the neural core of GROOT OS.*
