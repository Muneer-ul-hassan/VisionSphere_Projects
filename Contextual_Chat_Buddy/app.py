from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

# Set your OpenAI API key as environment variable for safety
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

# Simple memory (list of past messages)
chat_history = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_msg = request.message
    chat_history.append({"role": "user", "content": user_msg})

    # Build prompt from chat history for context (basic)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=chat_history
    )

    bot_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": bot_reply})

    return {"reply": bot_reply}
