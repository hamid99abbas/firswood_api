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

# Environment variables - Get directly from os.environ
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")

# Company knowledge base (shortened)
COMPANY_KNOWLEDGE = """
Firswood Intelligence builds production-ready AI systems:
- AI Agents & Automation
- RAG & Enterprise Search  
- Conversational AI (chatbots)
- Forecasting & Analytics
- Computer Vision
- Full-stack AI development

We focus on systems that work in production, not demos.
Contact: hello@firswood.com
"""

CORE_OPERATING_GUIDELINES = """
You are a friendly AI assistant for Firswood Intelligence. Your job is to have a casual, helpful chat about their project.

## Your Goal:
Learn about their project naturally. Through conversation, find out:
- What they want to build
- Their name
- Email (to send resources)
- Company
- Timeline

## Rules:
1. Keep responses SHORT - max 3 sentences
2. Ask ONE question at a time
3. Be curious, not sales-y
4. NO "discovery calls" until message 5+
5. Build trust BEFORE asking for contact info

## Conversation Strategy:
Messages 1-2: Understand their project
Messages 3-4: Show expertise, ask for email to send resources
Messages 5-6: Get name/company naturally
Messages 7+: Discuss timeline, then mention discovery call

## Example Good Responses:
"Customer support chatbots are great for handling repetitive questions. What kind of issues do your customers usually ask about?"

"Nice! We've built similar systems. I can send you a relevant case study - what's your work email?"

"Got it. What's your name, by the way?"

## Bad Responses (DON'T DO):
❌ Long paragraphs
❌ Multiple questions at once
❌ Mentioning "discovery calls" early
❌ Being too formal or corporate
❌ Listing capabilities

Talk like a helpful colleague, not a salesperson.
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
        # Try one more time directly from environment
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)
    return client


def get_system_instruction(additional_context=""):
    base_instruction = f"""{CORE_OPERATING_GUIDELINES}

What Firswood does:
{COMPANY_KNOWLEDGE}

CRITICAL RULES:
- Max 3 sentences per response
- Ask ONE question only
- Be conversational, not corporate
- NO "discovery calls" in first 4 messages
- Get email by offering case studies/resources

Today: {datetime.now().strftime('%B %d, %Y')}
"""

    if additional_context:
        base_instruction += f"\n\n{additional_context}"

    return base_instruction


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "slack": "/api/notify-slack",
            "health": "/health",
            "debug": "/debug/env"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables (REMOVE IN PRODUCTION)"""
    return {
        "google_api_key_exists": bool(GOOGLE_API_KEY),
        "google_api_key_length": len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0,
        "google_api_key_prefix": GOOGLE_API_KEY[:10] + "..." if GOOGLE_API_KEY else "None",
        "slack_webhook_exists": bool(SLACK_WEBHOOK_URL),
        "env_keys_count": len(os.environ.keys()),
        "has_google_key_in_environ": "GOOGLE_API_KEY" in os.environ,
        "python_version": sys.version
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message and return AI response
    """
    try:
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

        return ChatResponse(
            response=response.text,
            conversation_id=request.conversation_id or f"conv_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/api/notify-slack")
async def notify_slack(request: SlackNotificationRequest):
    """
    Send notification to Slack when user requests human support
    """
    if not SLACK_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Slack webhook not configured")

    try:
        # Format conversation history
        history_text = ""
        for msg in request.conversation_history[-6:]:  # Last 6 messages
            role = "👤 Visitor" if msg.role == "user" else "🤖 AI"
            content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            history_text += f"{role}: {content}\n\n"

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
                            "text": f"*Time:*\n{datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
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
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to send Slack notification")

        return {
            "success": True,
            "message": "Team notified via Slack",
            "conversation_id": request.conversation_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending Slack notification: {str(e)}")


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """
    Submit project brief and send to Slack
    """
    if not SLACK_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Slack webhook not configured")

    try:
        brief = request.brief_data

        # Create Slack message with brief details
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "📋 New Project Brief Submitted",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Name:*\n{brief.get('fullName', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Email:*\n{brief.get('workEmail', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Company:*\n{brief.get('company', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Phone:*\n{brief.get('phone', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Project Type:*\n{brief.get('projectType', 'N/A')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Timeline:*\n{brief.get('timeline', 'N/A')}"
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
                        "text": f"*What they want to achieve:*\n{brief.get('goal', 'N/A')}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Submitted:*\n{datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Page:*\n{request.url or 'N/A'}"
                        }
                    ]
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
                            "url": f"mailto:{brief.get('workEmail', '')}?subject=Re: Your Firswood Project Brief&body=Hi {brief.get('fullName', '')},%0D%0A%0D%0AThanks for submitting your project brief.%0D%0A%0D%0A",
                            "style": "primary"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "📞 Call",
                                "emoji": True
                            },
                            "url": f"tel:{brief.get('phone', '')}"
                        }
                    ]
                }
            ]
        }

        # Send to Slack
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to send brief to Slack")

        return {
            "success": True,
            "message": "Brief submitted successfully",
            "conversation_id": request.conversation_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting brief: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)