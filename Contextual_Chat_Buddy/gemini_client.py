from google import genai

# Initialize the client with your API key
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def ask_gemini(question: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=question,
    )
    return response.text
