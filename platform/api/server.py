from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ReasoningProvider:
    """Abstract base class for reasoning providers."""
    async def reason(self, prompt: str) -> str:
        raise NotImplementedError

class MockReasoningProvider(ReasoningProvider):
    """A mock provider for testing without API keys."""
    async def reason(self, prompt: str) -> str:
        return f"Mock AI Response to: {prompt} (System is in MVP Mode)"

class OpenAIReasoningProvider(ReasoningProvider):
    """OpenAI Provider."""
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    async def reason(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # Use a cheaper model for MVP
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {str(e)}"

# Select provider based on configuration
api_key = os.getenv("OPENAI_API_KEY")
if api_key and not api_key.startswith("sk-placeholder"):
    reasoning_provider = OpenAIReasoningProvider(api_key)
else:
    print("Warning: OPENAI_API_KEY not found or is placeholder. Using Mock Provider.")
    reasoning_provider = MockReasoningProvider()

@app.post("/chat")
async def chat(req: ChatRequest):
    answer = await reasoning_provider.reason(req.message)
    return {"reply": answer}

@app.get("/")
async def root():
    return {"status": "ok", "message": "Orolar AI Platform is running"}
