# main.py - FastAPI Backend with AI-Powered Data Extraction
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

# Initialize FastAPI
app = FastAPI(title="Firswood Intelligence Chat API - AI Extraction")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")

# Company knowledge base
COMPANY_KNOWLEDGE = """
# Firswood Intelligence - Company Knowledge Base

## Company Overview
Firswood Intelligence specializes in production-ready AI systems that deliver measurable business value.

**Company Details:**
- Company: Firswood Digital Services Limited
- Location: Manchester, UK
- Website: www.firswoodintelligence.com

## What We Build
1. Autonomous AI Agents
2. RAG & Enterprise Search
3. Conversational AI
4. Forecasting & Decision Intelligence
5. Real-Time Dashboards & Reporting
6. Computer Vision & On-Device AI
7. Full-Stack AI Product Development

## Engagement Models
1. Fixed-Scope Builds
2. Development Partnerships
3. High-Impact MVPs
"""

CORE_OPERATING_GUIDELINES = """
# Firswood Intelligence AI Chatbot Operating Guidelines

## Identity & Role
You are a conversational AI assistant for Firswood Intelligence. Your goal is to understand the user's project through natural conversation.

## Conversation Style:
1. **KEEP RESPONSES ULTRA SHORT**: Maximum 2 sentences, under 40 words
2. **ONE QUESTION MAXIMUM** per response
3. **NEVER REPEAT USER INPUT**: When user says "Hamid", respond "Nice to meet you, Hamid!" NOT "Hamidhamid"
4. **BE NATURAL**: Have a genuine conversation, not an interrogation
5. **NO VERBOSE EXPLANATIONS**: Get to the point fast

## Information to Gather Naturally:
- Their name
- Work email
- Company name
- Phone number (optional)
- Project type and goals
- Timeline

## Key Rules:
- Maximum 40 words per response
- ONE question per response
- Never echo user input
- Keep it conversational and warm
"""

# NEW: Data extraction prompt
DATA_EXTRACTION_PROMPT = """You are a data extraction AI. Analyze the conversation and extract key information.

Return a JSON object with these fields (use null if not found):
{
  "fullName": "string or null",
  "workEmail": "string or null",
  "company": "string or null",
  "phone": "string or null",
  "projectType": "string or null",
  "timeline": "string or null",
  "goal": "string or null"
}

EXTRACTION RULES:
1. **fullName**: Extract from phrases like "my name is X", "I'm X", "this is X", or standalone name responses. Capitalize properly.
2. **workEmail**: Extract any valid email address (name@domain.com)
3. **company**: Extract company name from "at X", "work at X", "company is X", or standalone company responses
4. **phone**: Extract phone numbers in any format
5. **projectType**: Categorize into: "Customer Support Chatbot", "Product Support Chatbot", "Order Tracking", "Document Q&A", "Analytics Dashboard", "Automation", "AI Platform", "MVP Development", or "Other"
6. **timeline**: Standardize to: "ASAP", "1 month", "1-3 months", "3-6 months", "6+ months", or null
7. **goal**: Extract the main problem/goal in 1-2 sentences (not questions, meaningful statements only)

IMPORTANT:
- Be smart about context: if AI asks "what's your name?" and user says "john", extract "John" as fullName
- Handle typos: "1 moth" = "1 month"
- Single word responses after questions are likely the answer to that question
- Ignore filler words like "yes", "no", "okay", "sure"
- Return ONLY valid JSON, no markdown, no explanation

Conversation to analyze:
"""


# Pydantic models
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


class ExtractionRequest(BaseModel):
    conversation_history: List[Message]
    conversation_id: str


class BriefSubmission(BaseModel):
    brief_data: dict
    conversation_id: str
    timestamp: str
    url: Optional[str] = None


# Initialize Gemini client
def get_gemini_client():
    api_key = GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)


def get_system_instruction(additional_context=""):
    base_instruction = f"""You are the AI assistant for Firswood Intelligence.

{CORE_OPERATING_GUIDELINES}

Company Knowledge Base:
{COMPANY_KNOWLEDGE}

Remember:
- Keep responses VERY SHORT (2-3 sentences max, under 60 words)
- Be conversational and warm
- ONE question per message maximum
- NEVER repeat or echo back user input
- Build trust through natural conversation

Current date: {datetime.now().strftime('%B %d, %Y')}
"""
    if additional_context:
        base_instruction += f"\n\nAdditional Context: {additional_context}"
    return base_instruction


# NEW: AI-powered data extraction function
async def extract_data_with_ai(conversation_history: List[Message]) -> Dict[str, Any]:
    """Use AI to extract structured data from conversation"""
    try:
        print("[EXTRACT] Starting AI-powered extraction...")

        # Format conversation for extraction
        conversation_text = ""
        for msg in conversation_history:
            role = "User" if msg.role == "user" else "AI Assistant"
            conversation_text += f"{role}: {msg.content}\n"

        # Create extraction prompt
        full_prompt = DATA_EXTRACTION_PROMPT + "\n" + conversation_text

        # Call Gemini for extraction
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=full_prompt)]
            )],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent extraction
                response_mime_type="application/json"  # Force JSON output
            )
        )

        # Parse JSON response
        extracted_text = response.text.strip()

        # Remove markdown code blocks if present
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]

        extracted_data = json.loads(extracted_text.strip())

        print(f"[EXTRACT] Successfully extracted: {json.dumps(extracted_data, indent=2)}")
        return extracted_data

    except Exception as e:
        print(f"[ERROR] Extraction failed: {str(e)}")
        traceback.print_exc()
        # Return empty data on failure
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
        "service": "Firswood Intelligence Chat API - AI Extraction",
        "status": "running",
        "version": "3.0.0",
        "features": ["AI-powered data extraction", "Natural conversation", "Smart lead capture"],
        "endpoints": {
            "chat": "/api/chat",
            "extract": "/api/extract-data",
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
        "slack_webhook_configured": bool(SLACK_WEBHOOK_URL)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return AI response with extracted data"""
    try:
        print(f"[CHAT] Processing message from conversation: {request.conversation_id}")

        client = get_gemini_client()

        # Only send USER messages to avoid echoing
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

        # NEW: Extract data using AI after every message
        all_messages = request.conversation_history + [
            Message(role="user", content=request.message, timestamp=datetime.now().isoformat())
        ]
        extracted_data = await extract_data_with_ai(all_messages)

        print(f"[CHAT] Response generated with extracted data")

        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            extracted_data=extracted_data
        )

    except Exception as e:
        print(f"[ERROR] Chat error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/api/extract-data")
async def extract_data(request: ExtractionRequest):
    """Extract structured data from conversation using AI"""
    try:
        print(f"[EXTRACT_API] Processing extraction for: {request.conversation_id}")

        extracted_data = await extract_data_with_ai(request.conversation_history)

        return {
            "success": True,
            "conversation_id": request.conversation_id,
            "extracted_data": extracted_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"[ERROR] Extract API error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting data: {str(e)}"
        )


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """Submit project brief to Slack"""
    print(f"[BRIEF_SUBMIT] Processing brief submission")

    if not SLACK_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Slack webhook not configured")

    try:
        brief = request.brief_data

        # Clean function
        def clean(text, max_length=500):
            if not text or text == 'N/A' or text == 'null':
                return 'N/A'
            text = str(text).strip()
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
            if len(text) > max_length:
                text = text[:max_length] + '...'
            return text

        # Extract and clean data
        full_name = clean(brief.get('fullName', 'N/A'), 100)
        work_email = clean(brief.get('workEmail', 'N/A'), 100)
        company = clean(brief.get('company', 'N/A'), 100)
        phone = clean(brief.get('phone', 'N/A'), 50)
        project_type = clean(brief.get('projectType', 'N/A'), 100)
        timeline = clean(brief.get('timeline', 'N/A'), 50)
        goal = clean(brief.get('goal', 'N/A'), 400)

        # Timestamp
        try:
            formatted_time = datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create Slack message
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
                f"🆔 *Conversation ID:* {request.conversation_id}\n"
                f"🔗 *Page:* {request.url or 'N/A'}"
            )
        }

        # Send to Slack
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ERROR] Slack API returned {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send to Slack: {response.status_code}"
            )

        print(f"[BRIEF_SUBMIT] Brief submitted successfully to Slack")

        return {
            "success": True,
            "message": "Brief submitted successfully",
            "conversation_id": request.conversation_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Brief submission error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting brief: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)