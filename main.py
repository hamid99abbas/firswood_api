# main.py - FINAL FIXED VERSION v3.1
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
import os
import json
from datetime import datetime
import requests
import traceback

app = FastAPI(title="Firswood Intelligence Chat API - AI Extraction v3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")

COMPANY_KNOWLEDGE = """
# Firswood Intelligence - Company Knowledge Base
Specializes in production-ready AI systems that deliver measurable business value.
Location: Manchester, UK | Website: www.firswoodintelligence.com
"""

CORE_OPERATING_GUIDELINES = """
# Firswood Intelligence AI Chatbot Operating Guidelines

## Your Role
You are a conversational AI assistant for Firswood Intelligence. Have natural conversations to understand user projects.

## CRITICAL RULES TO PREVENT REPETITION:
1. **NEVER say "Nice to meet you" more than ONCE per conversation**
2. **NEVER repeat greetings** - after first greeting, move on to questions
3. **KEEP RESPONSES ULTRA SHORT**: 1-2 sentences max, under 30 words
4. **ONE QUESTION per response**
5. **NO ECHOING**: Don't repeat what user just said
6. **MOVE FORWARD**: After getting info, ask about NEXT thing, don't confirm again

## Information to Gather:
- Name (ask once)
- Email (ask once)
- Company (ask once)
- Project details
- Timeline (ask once)

## Conversation Flow Example:
User: "I want a chatbot"
You: "What problem will it solve?"

User: "Customer support"
You: "What's your name?"

User: "John"
You: "What's your email?" (NOT "Nice to meet you John!")

User: "john@example.com"
You: "What company do you work for?" (NOT "Nice to meet you!")

## STRICT RULES:
- After name: Ask about email or company, NOT greet again
- After email: Ask about company or project, NOT greet again  
- After company: Ask about project details, NOT greet again
- Never use "Nice to meet you" twice
- Max 30 words per response
- Stay on topic
- Be efficient
"""

# IMPROVED: More explicit extraction prompt
DATA_EXTRACTION_PROMPT = """You are a data extraction AI. Analyze this conversation and extract information into JSON format.

EXTRACTION RULES:

1. **fullName**: Look for:
   - "my name is X"
   - "I'm X"
   - "this is X"
   - User says name after AI asks "what's your name?"
   - Capitalize properly (e.g., "hamid abbas" → "Hamid Abbas")

2. **workEmail**: Extract any email address (name@domain.com)

3. **company**: Look for:
   - "company is X"
   - "work at X"
   - "company name X"
   - User says company name after AI asks about company
   - Single-word responses to "what company?" (e.g., "emeron" → "Emeron")
   - Capitalize first letter

4. **phone**: Extract any phone number format

5. **projectType**: Categorize based on description:
   - If about answering questions → "Product Support Chatbot" or "Customer Support Chatbot"
   - If about tracking orders → "Order Tracking System"
   - If about documents → "Document Q&A Chatbot"
   - If about analytics → "Analytics Dashboard"
   - If general chatbot → "Chatbot"
   - Default: "AI Solution"

6. **timeline**: Standardize to ONE of these:
   - "ASAP" (if urgent, immediately, right away)
   - "1 month" (if 1 month, 30 days, next month)
   - "1-3 months" (if 2-3 months, couple months, quarter)
   - "3-6 months" (if 3-6 months, half year)
   - "6+ months" (if 6+ months, next year)
   - Look for: "3 months" → "1-3 months", "1 month" → "1 month"

7. **goal**: Summarize main problem/goal in 1-2 clear sentences from user's messages

IMPORTANT CONTEXT RULES:
- If AI asks "what's your name?" and user says "hamid abbas", extract as fullName
- If AI asks "what company?" and user says "emeron", extract as company
- If user says "3 months", timeline is "1-3 months"
- If user says "1 month", timeline is "1 month"
- Ignore filler words: "yes", "no", "okay", "sure"

Return ONLY valid JSON (no markdown, no explanation):
{
  "fullName": "string or null",
  "workEmail": "string or null", 
  "company": "string or null",
  "phone": "string or null",
  "projectType": "string or null",
  "timeline": "string or null",
  "goal": "string or null"
}

Conversation to analyze:
"""


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Message]] = []
    conversation_id: Optional[str] = None
    system_context: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    extracted_data: Optional[Dict[str, Any]] = None


class BriefSubmission(BaseModel):
    brief_data: dict
    conversation_id: str
    timestamp: str
    url: Optional[str] = None


def get_gemini_client():
    api_key = GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    return genai.Client(api_key=api_key)


def get_system_instruction(additional_context=""):
    base = f"""You are the AI assistant for Firswood Intelligence.

{CORE_OPERATING_GUIDELINES}

{COMPANY_KNOWLEDGE}

CRITICAL REMINDERS:
- NEVER repeat "Nice to meet you" - say it ONCE maximum
- After getting name, move to next question immediately
- After getting email, move to next question immediately
- Keep responses under 30 words
- Ask ONE question per response
- Be efficient and move forward

Current date: {datetime.now().strftime('%B %d, %Y')}
"""
    if additional_context:
        base += f"\n\nContext: {additional_context}"
    return base


async def extract_data_with_ai(conversation_history: List[Message]) -> Dict[str, Any]:
    """Use AI to extract structured data from conversation"""
    try:
        print("[EXTRACT] Starting AI extraction...")

        # Format conversation
        conversation_text = ""
        for i, msg in enumerate(conversation_history):
            role = "User" if msg.role == "user" else "AI"
            conversation_text += f"[Message {i + 1}] {role}: {msg.content}\n"

        print(f"[EXTRACT] Analyzing {len(conversation_history)} messages")

        # Create extraction prompt
        full_prompt = DATA_EXTRACTION_PROMPT + "\n" + conversation_text

        # Call Gemini
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=full_prompt)]
            )],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        # Parse response
        extracted_text = response.text.strip()

        # Clean markdown
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]

        extracted_data = json.loads(extracted_text.strip())

        print(f"[EXTRACT] ✅ Extracted: {json.dumps(extracted_data, indent=2)}")
        return extracted_data

    except Exception as e:
        print(f"[ERROR] Extraction failed: {str(e)}")
        traceback.print_exc()
        return {
            "fullName": None,
            "workEmail": None,
            "company": None,
            "phone": None,
            "projectType": None,
            "timeline": None,
            "goal": None
        }


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "version": "3.1.0",
        "features": ["AI extraction", "No regex", "Anti-repetition"],
        "endpoints": {
            "chat": "/api/chat",
            "submit_brief": "/api/submit-brief",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "google_api_configured": bool(GOOGLE_API_KEY),
        "slack_configured": bool(SLACK_WEBHOOK_URL)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat with AI extraction"""
    try:
        print(f"[CHAT] Message #{len(request.conversation_history) + 1}")

        client = get_gemini_client()

        # Build contents - ONLY user messages
        contents = []
        for msg in request.conversation_history:
            if msg.role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.content)]
                ))

        # Add current message
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        ))

        print(f"[CHAT] Sending {len(contents)} user messages to AI")

        # Generate response
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=get_system_instruction(request.system_context or ""),
                temperature=0.7,
            )
        )

        conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

        # Extract data using AI
        all_messages = request.conversation_history + [
            Message(role="user", content=request.message, timestamp=datetime.now().isoformat())
        ]
        extracted_data = await extract_data_with_ai(all_messages)

        print(f"[CHAT] ✅ Response generated with extraction")

        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            extracted_data=extracted_data
        )

    except Exception as e:
        print(f"[ERROR] Chat error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """Submit to Slack"""
    print(f"[BRIEF] Submitting...")

    if not SLACK_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Slack webhook not configured")

    try:
        brief = request.brief_data

        def clean(text, max_len=500):
            if not text or str(text).lower() in ['n/a', 'null', 'none']:
                return 'N/A'
            text = str(text).strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if len(text) > max_len:
                text = text[:max_len] + '...'
            return text

        full_name = clean(brief.get('fullName', 'N/A'), 100)
        work_email = clean(brief.get('workEmail', 'N/A'), 100)
        company = clean(brief.get('company', 'N/A'), 100)
        phone = clean(brief.get('phone', 'N/A'), 50)
        project_type = clean(brief.get('projectType', 'N/A'), 100)
        timeline = clean(brief.get('timeline', 'N/A'), 50)
        goal = clean(brief.get('goal', 'N/A'), 400)

        try:
            formatted_time = datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        slack_message = {
            "text": (
                f"🎉 *NEW LEAD - AI EXTRACTED!*\n\n"
                f"👤 *Name:* {full_name}\n"
                f"📧 *Email:* {work_email}\n"
                f"🏢 *Company:* {company}\n"
                f"📞 *Phone:* {phone}\n"
                f"💼 *Project:* {project_type}\n"
                f"📅 *Timeline:* {timeline}\n\n"
                f"🎯 *Goal:*\n{goal}\n\n"
                f"⏰ *Time:* {formatted_time}\n"
                f"🆔 *ID:* {request.conversation_id}\n"
                f"🔗 *Page:* {request.url or 'N/A'}"
            )
        }

        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ERROR] Slack error: {response.status_code}")
            raise HTTPException(status_code=500, detail=f"Slack error: {response.status_code}")

        print(f"[BRIEF] ✅ Submitted to Slack")

        return {
            "success": True,
            "message": "Brief submitted",
            "conversation_id": request.conversation_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Submit error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)