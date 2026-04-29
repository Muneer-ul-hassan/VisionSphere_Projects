import ollama
from typing import Optional, AsyncGenerator, List, Dict
from app.config import settings


class OllamaClient:
    """Client for interacting with local Ollama LLM"""

    def __init__(self):
        self.model = settings.llm_model
        self.base_url = settings.ollama_base_url
        self.temperature = settings.llm_temperature
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """You are Astra, a helpful AI assistant with visual capabilities.
You can see what the user's camera sees and help them understand their environment.

When describing scenes:
- Be EXTREMELY concise. Reply in 1-2 brief sentences max.
- Identify primary objects and their state.
- Do NOT be verbose. Talk fast.

When answering questions:
- Be direct and to the point.
- Admit uncertainty if detection is unclear."""

    async def check_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = ollama.list()
            models = [m['name'] for m in response['models']]
            return self.model in models or any(self.model.split(':')[0] in m for m in models)
        except Exception as e:
            print(f"Ollama connection check failed: {e}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = ollama.list()
            return [m['name'] for m in response['models']]
        except Exception:
            return []

    def chat(self, user_message: str, context: Optional[List[Dict]] = None) -> str:
        """Send a chat message and get response (sync)"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *(context or []),
            {"role": "user", "content": user_message}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature}
        )
        return response['message']['content']

    async def chat_stream(self, user_message: str, context: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
        """Stream chat response token by token"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *(context or []),
            {"role": "user", "content": user_message}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": self.temperature}
        )

        for part in response:
            if 'message' in part and 'content' in part['message']:
                yield part['message']['content']


# Global instance
llm_client = OllamaClient()
