# main.py - FastAPI Backend with AI-Powered Natural Brief Collection
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
import os
import json
import re
from datetime import datetime
import requests

# Initialize FastAPI
app = FastAPI(title="Firswood Intelligence Chat API")

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

## What We Build
1. Autonomous AI Agents
2. RAG & Enterprise Search
3. Conversational AI (chatbots, voice assistants)
4. Forecasting & Decision Intelligence
5. Real-Time Dashboards & Reporting
6. Computer Vision & On-Device AI
7. Full-Stack AI Product Development

## Engagement Models
- Fixed-Scope Builds
- Development Partnerships  
- High-Impact MVPs

## Contact
- Website: www.firswoodintelligence.com
- Book a call: calendar.app.google/kVahCoFGsHhgiSE76
"""

STANDARD_SYSTEM_PROMPT = """You are the AI assistant for Firswood Intelligence, a specialized AI systems design and delivery practice.

## Your Role
Have genuine, helpful conversations about AI projects. Be conversational, curious, and knowledgeable.

## Conversation Style
- Keep responses SHORT (2-3 sentences, max 60 words)
- Ask ONE question at a time
- Be warm and conversational, like a knowledgeable colleague
- Show expertise through insights, not corporate speak
- Plain English only

## Guidelines
- Focus on understanding their project first
- Ask clarifying questions about technical requirements
- Share relevant insights when appropriate
- Guide towards discovery calls naturally (but not too early)
- Be honest about what's feasible

Company Knowledge:
{COMPANY_KNOWLEDGE}

Current date: {datetime.now().strftime('%B %d, %Y')}
""".format(COMPANY_KNOWLEDGE=COMPANY_KNOWLEDGE, datetime=datetime)

BRIEF_MODE_SYSTEM_PROMPT = """You are having a natural conversation to learn about a user's AI project for Firswood Intelligence.

## Your Mission
Through friendly conversation, naturally gather:
1. Their name
2. Email address
3. Company name
4. What they want to build (project description)
5. Timeline (when they need it)
6. Any technical details they share

## How to Gather Info Naturally

**CRITICAL RULES:**
- Keep responses VERY SHORT (2-3 sentences, max 50 words)
- Ask ONE question at a time only
- Be conversational, not interrogative
- Don't rush - build trust first
- Show genuine interest in their project

**Information Gathering Strategy:**
1. First 2-3 messages: Understand their PROJECT deeply
   - "What kind of chatbot?" 
   - "What problems will it solve?"
   - "How many users?"

2. Messages 3-4: Offer value as reason for email
   - "We have a case study on this - what's your email?"
   - "I can send you some examples - where should I send them?"

3. Messages 4-5: Get name & company naturally
   - "Great! What's your name?"
   - "And what company are you with?"

4. Message 5+: Timeline and next steps
   - "What's your timeline for this?"
   - "Would a quick call help discuss the details?"

**IMPORTANT:**
- NEVER list multiple questions at once
- Don't mention "brief" or "form"
- Don't ask for phone number unless they offer it
- Focus on their project, not data collection
- When you have name + email + project description, you can suggest a call

## Response Format
Always respond with valid JSON:
```json
{
  "response": "Your 2-3 sentence conversational message here",
  "extracted_info": {
    "name": "value or null",
    "email": "value or null", 
    "company": "value or null",
    "phone": "value or null",
    "project_description": "detailed description or null",
    "timeline": "value or null",
    "technical_details": "any tech details mentioned or null"
  },
  "brief_ready": false,
  "conversation_stage": "exploring_project|offering_value|gathering_contact|closing"
}
```

Set `brief_ready: true` ONLY when you have: name, email, and clear project description.

Current date: {datetime.now().strftime('%B %d, %Y')}
""".format(datetime=datetime)


# Pydantic models
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Message]] = []
    conversation_id: Optional[str] = None
    brief_mode: Optional[bool] = False
    brief_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    brief_context: Optional[Dict[str, Any]] = None
    brief_complete: Optional[bool] = False


class BriefSubmission(BaseModel):
    brief_data: dict
    conversation_id: str
    timestamp: str
    url: Optional[str] = None
    conversation_history: Optional[List[Message]] = []


def get_gemini_client():
    """Initialize Gemini client with API key"""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from AI response, handling various formats"""
    try:
        # Try direct JSON parse first
        if text.strip().startswith("{"):
            return json.loads(text)

        # Look for JSON code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        # Look for JSON anywhere in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # Fallback: return text as response
        return {
            "response": text,
            "extracted_info": {},
            "brief_ready": False,
            "conversation_stage": "exploring_project"
        }
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {
            "response": text,
            "extracted_info": {},
            "brief_ready": False,
            "conversation_stage": "exploring_project"
        }


def merge_extracted_info(current: dict, new: dict) -> dict:
    """Merge new extracted info with existing, keeping non-null values"""
    merged = current.copy()
    for key, value in new.items():
        if value and value != "null" and value != "None":
            merged[key] = value
    return merged


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "version": "2.0.0",
        "features": ["ai_chat", "natural_brief_collection", "slack_integration"],
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
        "slack_webhook_configured": bool(SLACK_WEBHOOK_URL)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message - handles both regular chat and natural brief collection
    """
    try:
        client = get_gemini_client()

        # Choose system prompt based on mode
        system_instruction = BRIEF_MODE_SYSTEM_PROMPT if request.brief_mode else STANDARD_SYSTEM_PROMPT

        # Build conversation history for context
        contents = []
        for msg in request.conversation_history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))

        # Prepare current message
        current_message = request.message

        # Add context if in brief mode
        if request.brief_mode and request.brief_context:
            collected = request.brief_context.get("collectedInfo", {})
            current_message = f"""User message: {request.message}

Already collected:
{json.dumps(collected, indent=2)}

Extract any NEW information from this message and respond naturally."""

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=current_message)]
        ))

        # Generate response
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )

        response_text = response.text
        print(f"🤖 AI Response: {response_text[:200]}...")

        # Process brief mode response
        brief_context = None
        brief_complete = False

        if request.brief_mode:
            # Parse JSON response
            parsed = extract_json_from_response(response_text)

            # Extract actual response text
            actual_response = parsed.get("response", response_text)

            # Get extracted info
            new_info = parsed.get("extracted_info", {})

            # Merge with existing collected info
            current_info = request.brief_context.get("collectedInfo", {}) if request.brief_context else {}
            merged_info = merge_extracted_info(current_info, new_info)

            # Check if brief is ready
            brief_complete = parsed.get("brief_ready", False)

            # Update context
            brief_context = {
                "stage": parsed.get("conversation_stage", "exploring_project"),
                "collectedInfo": merged_info
            }

            response_text = actual_response

            print(f"📊 Collected Info: {json.dumps(merged_info, indent=2)}")
            print(f"✅ Brief Complete: {brief_complete}")

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id or f"conv_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat(),
            brief_context=brief_context,
            brief_complete=brief_complete
        )

    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """
    Submit project brief and send to Slack with conversation context
    """
    if not SLACK_WEBHOOK_URL:
        print("❌ SLACK_WEBHOOK_URL not configured")
        raise HTTPException(
            status_code=500,
            detail="Slack webhook not configured"
        )

    try:
        brief = request.brief_data
        print(f"📋 Submitting brief for: {brief.get('name', 'N/A')}")

        # Format conversation history
        conversation_summary = ""
        if request.conversation_history:
            recent = request.conversation_history[-10:]  # Last 10 messages
            for msg in recent:
                role = "👤" if msg.role == "user" else "🤖"
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                conversation_summary += f"{role} {content}\n\n"

        # Create Slack message
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎯 New Lead - AI Collected",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Name:*\n{brief.get('name', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Email:*\n{brief.get('email', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Company:*\n{brief.get('company', 'Not provided')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Phone:*\n{brief.get('phone', 'Not provided')}"
                        }
                    ]
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Project:*\n{brief.get('project_description', 'N/A')}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Timeline:*\n{brief.get('timeline', 'Not specified')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Tech Details:*\n{brief.get('technical_details', 'Not specified')}"
                        }
                    ]
                }
            ]
        }

        # Add conversation summary
        if conversation_summary:
            slack_message["blocks"].extend([
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Conversation:*\n```{conversation_summary[:2000]}```"
                    }
                }
            ])

        # Add metadata and actions
        slack_message["blocks"].extend([
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Submitted:*\n{datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Page:*\n{request.url or 'N/A'}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📧 Reply",
                            "emoji": True
                        },
                        "url": f"mailto:{brief.get('email', '')}?subject=Re: Your Firswood Project&body=Hi {brief.get('name', '')},%0D%0A%0D%0AThanks for discussing your project with us.%0D%0A%0D%0A",
                        "style": "primary"
                    }
                ]
            }
        ])

        # Send to Slack
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ Slack error: {response.status_code}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send to Slack: {response.status_code}"
            )

        print(f"✅ Brief submitted successfully")

        return {
            "success": True,
            "message": "Brief submitted successfully",
            "conversation_id": request.conversation_id
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting brief: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)