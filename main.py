# main.py - FastAPI Backend for Firswood Chat Widget
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import os
import sys
from datetime import datetime
import requests
import json
import traceback

# Initialize FastAPI
app = FastAPI(title="Firswood Intelligence Chat API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
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
Firswood Intelligence specializes in production-ready AI systems that deliver measurable business value. We bridge the gap between AI experiments and operational systems.

**Company Details:**
- Company: Firswood Digital Services Limited
- Company No: 11608317
- Location: 651a Mauldeth Road West, Chorlton Cum Hardy, Manchester, M21 7SA
- Website: www.firswoodintelligence.com
- LinkedIn: linkedin.com/in/haseebkhanproduct

## Core Philosophy
- Builder-led: Work directly with the person designing and building
- Commercially grounded: Every decision considers cost, scale, and ROI
- Production-first: Systems meant for daily use, not demos
- Strategic partner mindset: Think like owners, not vendors

## What We Build

### 1. Autonomous AI Agents
Multi-agent systems that plan, delegate, verify, and refine outputs across research, analysis, proposal generation, and internal assistants.

### 2. RAG & Enterprise Search
Secure, source-grounded chat over documents, databases, and APIs with semantic search, traceability, and natural-language querying.

### 3. Conversational AI
Production chatbots that connect to live systems, fetch real data, and act across web, Slack, Telegram, or internal tools.

### 4. Forecasting & Decision Intelligence
Sales, demand, and inventory forecasts with automated data prep, time-series models, and executive-ready insight dashboards.

### 5. Real-Time Dashboards & Reporting
Natural-language access to metrics, AI-generated reports, and live integrations (Stripe, Notion, Sheets, internal APIs) with agent delegation.

### 6. Computer Vision & On-Device AI
Privacy-aware, real-time vision and mobile edge models for posture, movement, and activity detection with low latency and offline capability.

### 7. Full-Stack AI Product Development
End-to-end builds: backend AI services, secure APIs, cloud deploys (AWS, Vercel), and mobile/web apps (Flutter, Firebase) from MVP to production.

## Engagement Models

### 1. Fixed-Scope Builds
Perfect for defined problems. We design, build, and deploy a specific system for a fixed price and timeline.

### 2. Development Partnerships
Ongoing AI development. We act as your specialized engineering arm for continuous iteration.

### 3. High-Impact MVPs
Rapid builds for validation or fundraising. Get a functional core system to prove the value proposition.

## Ideal Clients
- Founders building AI-first products
- Businesses ready to automate at scale
- Teams with manual process bottlenecks
- Organizations with valuable unused data

## The Firswood Approach
1. Problem Definition: Understanding core operational bottlenecks before coding
2. System Architecture: Designing robust, scalable data pipelines
3. Production Deploy: Shipping live systems integrated into workflows
4. Iteration: Ongoing optimization based on real-world usage

## Contact
- Book a call: calendar.app.google/kVahCoFGsHhgiSE76
- Website form for inquiries
- LinkedIn: linkedin.com/in/haseebkhanproduct
"""

CORE_OPERATING_GUIDELINES = """
# Firswood Intelligence AI Chatbot Operating Guidelines

## Identity & Role
You are a conversational AI assistant for Firswood Intelligence. Your goal is to understand the user's project through natural conversation while subtly gathering key information.

## PRIMARY OBJECTIVE: Natural, Engaging Conversation
Have a genuine, helpful conversation about their project. Through natural discussion, learn:
- Their name
- Work email
- Company name
- Phone number (optional)
- Project details

## Conversation Style - CRITICAL RULES:
1. **KEEP RESPONSES ULTRA SHORT**: Maximum 2 sentences, ideally 1 sentence. Under 40 words.
2. **ONE QUESTION MAXIMUM** per response
3. **NO REPETITION**: If user says "8", write "8" not "88". Listen carefully.
4. **NO VERBOSE EXPLANATIONS**: Get to the point fast
5. **NATURAL FLOW**: Don't jump around topics randomly

## How to Gather Information:

**DO:**
- Ask ONE simple question at a time
- Listen to EXACT user input (if they say "8", acknowledge "8" not "88")
- Keep responses conversational and brief
- Show you're listening by referencing what they just said

**DON'T:**
- Repeat or rephrase what user just said unnecessarily
- Ask multiple questions in one response
- Write long explanations
- Mishear numbers or names
- Be overly formal

## Example Good Responses:
User: "I need a chatbot"
You: "Great! What problem will it solve?"

User: "Customer support"
You: "Got it. What kind of support queries?"

User: "My company is TechCorp"
You: "Thanks! What's your role at TechCorp?"

User: "We have 8 people"
You: "Perfect. What's your current process?"

## Key Rules:
- Maximum 40 words per response
- ONE question per response
- Listen carefully to exact user input
- No repetition or verbose explanations
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


class SlackNotificationRequest(BaseModel):
    conversation_id: str
    user_email: str
    user_name: str
    conversation_history: List[Message]
    timestamp: str
    url: Optional[str] = None


class BriefSubmission(BaseModel):
    brief_data: dict
    conversation_id: str
    timestamp: str
    url: Optional[str] = None


# Initialize Gemini client
def get_gemini_client():
    api_key = GOOGLE_API_KEY
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)
    return client


def get_system_instruction(additional_context=""):
    base_instruction = f"""You are the AI assistant for Firswood Intelligence, a specialized AI systems design and delivery practice.

{CORE_OPERATING_GUIDELINES}

Company Knowledge Base:
{COMPANY_KNOWLEDGE}

Remember:
- Keep responses VERY SHORT (2-3 sentences max, under 60 words)
- Be conversational and warm
- ONE question per message maximum
- Never mention "discovery calls" early in conversation
- Gather information naturally through genuine conversation
- Show expertise through insights, not corporate speak
- Build trust before asking for contact info

Current date: {datetime.now().strftime('%B %d, %Y')}
"""

    if additional_context:
        base_instruction += f"\n\nAdditional Context: {additional_context}"

    return base_instruction


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "slack_notify": "/api/notify-slack",
            "submit_brief": "/api/submit-brief",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "google_api_configured": bool(GOOGLE_API_KEY),
        "slack_webhook_configured": bool(SLACK_WEBHOOK_URL)
    }
    return health_status


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return AI response"""
    try:
        print(f"[CHAT] Processing message from conversation: {request.conversation_id}")

        # Initialize client
        client = get_gemini_client()

        # Convert conversation history to Gemini format
        contents = []
        for msg in request.conversation_history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))

        # Add current message
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        ))

        # Generate response
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=get_system_instruction(),
                temperature=0.7,
            )
        )

        conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

        print(f"[CHAT] Response generated successfully for: {conversation_id}")

        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"[ERROR] Chat error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/api/notify-slack")
async def notify_slack(request: SlackNotificationRequest):
    """Send notification to Slack when user requests human support"""

    print(f"[SLACK_NOTIFY] Processing notification for: {request.user_email}")

    if not SLACK_WEBHOOK_URL:
        print("[ERROR] Slack webhook not configured")
        raise HTTPException(
            status_code=500,
            detail="Slack webhook not configured"
        )

    try:
        # Format conversation history
        history_text = ""
        for msg in request.conversation_history[-6:]:  # Last 6 messages
            role = "👤 Visitor" if msg.role == "user" else "🤖 AI"
            content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            history_text += f"{role}: {content}\n\n"

        # Safe timestamp handling
        try:
            formatted_time = datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"[WARN] Timestamp parsing error: {e}")
            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create Slack message
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🆘 Human Support Requested",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Name:*\n{request.user_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Email:*\n{request.user_email}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{formatted_time}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Page:*\n{request.url or 'N/A'}"
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
                        "text": f"*Recent Conversation:*\n```{history_text}```"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Conversation ID:* `{request.conversation_id}`"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "📧 Reply via Email",
                                "emoji": True
                            },
                            "url": f"mailto:{request.user_email}?subject=Re: Firswood Chat Support&body=Hi {request.user_name},%0D%0A%0D%0AThanks for reaching out via our website chat.%0D%0A%0D%0A",
                            "style": "primary"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "📅 Book Call",
                                "emoji": True
                            },
                            "url": "https://calendar.app.google/kVahCoFGsHhgiSE76"
                        }
                    ]
                }
            ]
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
                detail=f"Failed to send Slack notification: {response.status_code}"
            )

        print(f"[SLACK_NOTIFY] Notification sent successfully")

        return {
            "success": True,
            "message": "Team notified via Slack",
            "conversation_id": request.conversation_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Slack notification error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error sending Slack notification: {str(e)}"
        )


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """Submit project brief and send to Slack"""

    print(f"[BRIEF_SUBMIT] Processing brief submission")
    print(f"[BRIEF_SUBMIT] Conversation ID: {request.conversation_id}")
    print(f"[BRIEF_SUBMIT] Brief data keys: {list(request.brief_data.keys())}")

    if not SLACK_WEBHOOK_URL:
        print("[ERROR] Slack webhook not configured")
        raise HTTPException(
            status_code=500,
            detail="Slack webhook not configured"
        )

    try:
        brief = request.brief_data

        # Helper function to clean text for Slack
        def clean_for_slack(text, max_length=500):
            """Clean and truncate text for Slack Block Kit"""
            if not text or text == 'N/A':
                return 'N/A'

            # Convert to string and strip
            text = str(text).strip()

            # Escape special characters for Slack
            text = text.replace('&', '&amp;')
            text = text.replace('<', '&lt;')
            text = text.replace('>', '&gt;')

            # Remove any control characters
            text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')

            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + '...'

            return text

        # Safe value extraction with defaults and cleaning
        full_name = clean_for_slack(brief.get('fullName', 'N/A'), 100)
        work_email = clean_for_slack(brief.get('workEmail', 'N/A'), 100)
        company = clean_for_slack(brief.get('company', 'N/A'), 100)
        phone = clean_for_slack(brief.get('phone', 'N/A'), 50)
        project_type = clean_for_slack(brief.get('projectType', 'N/A'), 100)
        timeline = clean_for_slack(brief.get('timeline', 'N/A'), 50)
        goal = clean_for_slack(brief.get('goal', 'N/A'), 400)

        print(f"[BRIEF_SUBMIT] Name: {full_name}, Email: {work_email}, Company: {company}")
        print(f"[BRIEF_SUBMIT] Goal length: {len(goal)}")

        # Safe timestamp handling
        try:
            formatted_time = datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"[WARN] Timestamp parsing error: {e}")
            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create SIMPLE Slack message - just text, no fancy blocks
        slack_message = {
            "text": (
                f"🎉 *NEW LEAD!*\n\n"
                f"👤 *Name:* {full_name}\n"
                f"📧 *Email:* {work_email}\n"
                f"🏢 *Company:* {company}\n"
                f"📞 *Phone:* {phone}\n"
                f"💼 *Project:* {project_type}\n"
                f"📅 *Timeline:* {timeline}\n\n"
                f"🎯 *Goal:*\n{goal}\n\n"
                f"🆔 *Conversation ID:* {request.conversation_id}\n"
                f"🔗 *Page:* {request.url or 'N/A'}"
            )
        }

        print(f"[BRIEF_SUBMIT] Sending to Slack...")

        # Log the payload for debugging
        print(f"[BRIEF_SUBMIT] Slack payload size: {len(json.dumps(slack_message))} bytes")

        # Send to Slack
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ERROR] Slack API returned {response.status_code}: {response.text}")
            # Log the full payload that failed
            print(f"[ERROR] Failed payload: {json.dumps(slack_message, indent=2)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send brief to Slack: {response.status_code} - {response.text}"
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