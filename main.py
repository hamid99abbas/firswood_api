# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os
import asyncio
from functools import partial

from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="Firswood Intelligence Chat API")

# ------------------------------------------------------------------------------
# CORS (browser-safe)
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

# ------------------------------------------------------------------------------
# Knowledge + Instructions
# ------------------------------------------------------------------------------
COMPANY_KNOWLEDGE = """
Firswood Intelligence builds production-ready AI systems.
We focus on reliable, integrated AI that delivers real business value.
"""

CORE_OPERATING_GUIDELINES = """
You are the AI assistant for Firswood Intelligence.

Rules:
- Calm, professional, grounded tone
- Plain English only
- Short responses (2–3 paragraphs max)
- No pricing, timelines, or guarantees
- Suggest discovery call for deeper questions
"""

def system_instruction() -> str:
    return f"""
{CORE_OPERATING_GUIDELINES}

Company Knowledge:
{COMPANY_KNOWLEDGE}

Date: {datetime.utcnow().strftime('%B %d, %Y')}
"""

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Message] = Field(default_factory=list)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str


# ------------------------------------------------------------------------------
# Gemini client
# ------------------------------------------------------------------------------
def get_gemini_client():
    return genai.Client(api_key=GOOGLE_API_KEY)


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        client = get_gemini_client()

        contents = []

        for msg in request.conversation_history[-10:]:
            contents.append(
                types.Content(
                    role="user" if msg.role == "user" else "model",
                    parts=[types.Part(text=msg.content)]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=request.message)]
            )
        )

        loop = asyncio.get_running_loop()

        generate = partial(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction(),
                temperature=0.7,
            ),
        )

        response = await loop.run_in_executor(None, generate)

        return ChatResponse(
            response=response.text.strip(),
            conversation_id=request.conversation_id or f"conv_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
