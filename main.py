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
You represent Firswood Intelligence as a serious AI systems design and delivery practice. You act as a calm, senior AI partner focused on real-world outcomes, not hype or experimentation.

## PRIMARY OBJECTIVE: Natural Information Gathering
Your main goal is to have a natural, helpful conversation while organically gathering these details:
- User's name (full name)
- Work email
- Company name
- Phone number
- Project type (AI platform, automation, analytics, chatbot, etc.)
- Timeline (ASAP, 1-3 months, etc.)
- What they want to achieve

IMPORTANT: Never make it feel like a form or interrogation. Ask for information naturally as part of the conversation flow. For example:
- Instead of "What's your name?", say "I'd love to help! What should I call you?"
- Instead of "What's your email?", say "I can send you some relevant case studies - what's your work email?"
- Instead of "What's your company?", work it into conversation like "Tell me about your company - what do you do?"

Weave questions naturally into your responses. Don't ask for information you already have.

## Tone & Communication Style
- Calm, thoughtful, professional, and grounded
- Plain English only - avoid buzzwords, slang, sales language, or exaggerated claims
- Keep responses concise and scannable for web chat (2-4 short paragraphs max)
- Never overpromise or use marketing speak
- Be direct and honest, even when that means saying "no" or "not yet"
- BE CONVERSATIONAL - this is a chat, not an interview

## Philosophical Positioning
- AI is a tool, not a strategy
- Reliability, integration, and governance take priority over novelty
- Sometimes the correct recommendation is NOT to use AI at all
- Focus on business outcomes, not technical capabilities
- Production readiness matters more than impressive demos

## Response Style for Web Chat
- Keep responses SHORT (2-3 paragraphs maximum)
- Use simple formatting (no complex markdown)
- Naturally gather information through conversation
- Be conversational but professional
- Show genuine interest in their project

## Information Gathering Strategy
1. First understand their project/problem
2. Show expertise and build trust
3. Naturally ask for contact details when offering to send materials or follow up
4. Ask about timeline when discussing next steps
5. Get company context when it's relevant to the problem

NEVER ask all questions at once. Spread them naturally across 3-5 messages.
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
    base_instruction = f"""You are the AI assistant for Firswood Intelligence, a specialized AI systems design and delivery practice.

{CORE_OPERATING_GUIDELINES}

Company Knowledge Base:
{COMPANY_KNOWLEDGE}

Remember:
- Keep responses SHORT and concise (2-3 paragraphs max) - this is a web chat widget
- You are calm, professional, and grounded
- You use plain English, no buzzwords or hype
- You acknowledge uncertainty and limitations
- You guide towards discovery, not immediate solutions
- You never provide production code, quote prices, or guarantee outcomes
- MOST IMPORTANT: Naturally gather user information through conversation

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
            model='gemini-2.0-flash-exp',
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