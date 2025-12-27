# main.py - FIXED v3.2 - No Repetition Loops
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

app = FastAPI(title="Firswood Intelligence Chat API v3.2")

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
# Firswood Intelligence
AI systems design and delivery practice.
Location: Manchester, UK
Website: www.firswoodintelligence.com
"""

CORE_OPERATING_GUIDELINES = """
# Firswood Intelligence AI Assistant - Operating Rules

## Your Mission
Have a natural conversation to understand the user's AI project. Gather information smoothly without feeling like an interrogation.

## CRITICAL CONVERSATION RULES:

### 1. NEVER REPEAT YOURSELF
- If you ask "What problem will the chatbot solve?" and they answer, DON'T ask it again
- Move to a NEW question after they answer
- Keep track of what you already know

### 2. CONVERSATION FLOW (Follow this order)
Ask about these topics IN ORDER (one at a time):
1. Project type/goal (what they want to build)
2. Problem it solves (why they need it)
3. Their name
4. Their email
5. Their company
6. Timeline
7. Additional details (budget, team size, etc.)

### 3. HOW TO ASK QUESTIONS
✅ GOOD:
- "What problem will it solve?" (first time)
- "What's your name?" (after project discussion)
- "What company do you work for?" (after name)
- "What's your timeline?" (after company)

❌ BAD:
- Asking "What problem?" twice
- Asking "What's your name?" twice
- Repeating ANY question you already asked

### 4. RESPONSE STYLE
- Keep responses SHORT: 1-2 sentences, max 30 words
- Ask ONE question per response
- Don't repeat information back to them
- Sound natural and conversational
- Move forward, never backward

### 5. RECOGNITION RULES
If user says:
- "I want a chatbot" → You know: project type is chatbot
- "For customer support" → You know: it's for support
- "My name is John" → You know: name is John
- "john@email.com" → You know: email is john@email.com

After you know something, NEVER ask about it again.

## Examples of Good Conversation:

User: "I want a chatbot"
You: "What problem will it solve?"

User: "Customer support"
You: "What's your name?" ← NEW question, don't repeat

User: "John"
You: "What's your email?" ← NEW question

User: "john@email.com"
You: "What company?" ← NEW question

## KEY RULE
Each response must ask something NEW. Never repeat a question.
"""

DATA_EXTRACTION_PROMPT = """Extract information from this conversation into JSON.

RULES:
1. **fullName**: Extract from "my name is X", "I'm X", or when user gives name after being asked
   - Capitalize: "hamid abbas" → "Hamid Abbas"

2. **workEmail**: Any email address (name@domain.com)

3. **company**: Extract from "company is X", "work at X", or single-word response to company question
   - Capitalize: "emeron" → "Emeron"

4. **phone**: Any phone number

5. **projectType**: Categorize based on what they want:
   - Chatbot for support → "Customer Support Chatbot"
   - Chatbot for products → "Product Support Chatbot"  
   - Order tracking → "Order Tracking System"
   - Document questions → "Document Q&A Chatbot"
   - Analytics → "Analytics Dashboard"
   - General chatbot → "Chatbot"

6. **timeline**: Standardize to one of:
   - "ASAP", "1 month", "1-3 months", "3-6 months", "6+ months"
   - "3 months" → "1-3 months"
   - "1 month" → "1 month"

7. **goal**: Main problem/goal in 1-2 sentences from user's description

CONTEXT AWARENESS:
- If AI asks "what company?" and user says "emeron" → company is "Emeron"
- If user says "answer customer questions" → goal includes that
- Ignore filler: "yes", "no", "okay"

Return ONLY valid JSON:
{
  "fullName": "string or null",
  "workEmail": "string or null",
  "company": "string or null",
  "phone": "string or null",
  "projectType": "string or null",
  "timeline": "string or null",
  "goal": "string or null"
}

Conversation:
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


def get_system_instruction(additional_context="", conversation_summary=""):
    base = f"""You are the AI assistant for Firswood Intelligence.

{CORE_OPERATING_GUIDELINES}

{COMPANY_KNOWLEDGE}

CRITICAL REMINDERS:
- NEVER ask the same question twice
- Each response must ask something NEW
- Keep responses under 30 words
- Move forward through the conversation
- Don't repeat yourself

{conversation_summary}

Current date: {datetime.now().strftime('%B %d, %Y')}
"""
    if additional_context:
        base += f"\n\nContext: {additional_context}"
    return base


async def extract_data_with_ai(conversation_history: List[Message]) -> Dict[str, Any]:
    """AI-powered extraction"""
    try:
        print(f"[EXTRACT] Analyzing {len(conversation_history)} messages...")

        conversation_text = ""
        for i, msg in enumerate(conversation_history):
            role = "User" if msg.role == "user" else "AI"
            conversation_text += f"[{i + 1}] {role}: {msg.content}\n"

        full_prompt = DATA_EXTRACTION_PROMPT + "\n" + conversation_text

        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=full_prompt)]
            )],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        extracted_text = response.text.strip()

        # Clean markdown
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]

        extracted_data = json.loads(extracted_text.strip())

        print(f"[EXTRACT] ✅ Success: {json.dumps(extracted_data, indent=2)}")
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


def generate_conversation_summary(extracted_data: Dict[str, Any]) -> str:
    """Generate summary of what we already know"""
    known_info = []

    if extracted_data.get('projectType'):
        known_info.append(f"Project type: {extracted_data['projectType']}")
    if extracted_data.get('goal'):
        known_info.append(f"Goal: {extracted_data['goal']}")
    if extracted_data.get('fullName'):
        known_info.append(f"Name: {extracted_data['fullName']}")
    if extracted_data.get('workEmail'):
        known_info.append(f"Email: {extracted_data['workEmail']}")
    if extracted_data.get('company'):
        known_info.append(f"Company: {extracted_data['company']}")
    if extracted_data.get('timeline'):
        known_info.append(f"Timeline: {extracted_data['timeline']}")

    if known_info:
        return f"\nINFO ALREADY COLLECTED:\n" + "\n".join(
            known_info) + "\n\nNEVER ask about these again. Ask about something NEW."
    return ""


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "version": "3.2.0",
        "features": ["AI extraction", "Anti-loop protection"],
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
    """Process chat with AI extraction and loop prevention"""
    try:
        print(f"\n[CHAT] Message #{len(request.conversation_history) + 1}")
        print(f"[CHAT] User says: {request.message[:50]}...")

        client = get_gemini_client()

        # First, extract current data to know what we have
        temp_history = request.conversation_history + [
            Message(role="user", content=request.message, timestamp=datetime.now().isoformat())
        ]
        extracted_data = await extract_data_with_ai(temp_history)

        # Generate summary of known info
        conv_summary = generate_conversation_summary(extracted_data)
        print(f"[CHAT] Known info: {conv_summary[:100]}...")

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

        print(f"[CHAT] Sending {len(contents)} user messages")

        # Generate response with conversation summary
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=get_system_instruction(request.system_context or "", conv_summary),
                temperature=0.7,
            )
        )

        conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

        print(f"[CHAT] ✅ Response: {response.text[:50]}...")

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
    print(f"[BRIEF] Submitting to Slack...")

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
            print(f"[ERROR] Slack: {response.status_code}")
            raise HTTPException(status_code=500, detail=f"Slack error: {response.status_code}")

        print(f"[BRIEF] ✅ Submitted successfully")

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