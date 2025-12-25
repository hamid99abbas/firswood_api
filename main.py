# main.py - FastAPI Backend with AI-Powered Brief Collection
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
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

# Company knowledge base
COMPANY_KNOWLEDGE = """
# Firswood Intelligence - Company Knowledge Base

## Company Overview
Firswood Intelligence specializes in production-ready AI systems that deliver measurable business value.

## What We Build
1. Autonomous AI Agents
2. RAG & Enterprise Search
3. Conversational AI
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

BRIEF_COLLECTION_SYSTEM_PROMPT = """
You are collecting project brief information through natural conversation for Firswood Intelligence.

Your goal is to collect these key details through friendly, conversational questions:
1. Contact Information:
   - Full name
   - Email address
   - Company name
   - Phone number (optional)

2. Project Details:
   - What they want to build/achieve
   - Project type (AI platform, automation, analytics, MVP, etc.)
   - Timeline expectations
   - Budget range (if they mention it)
   - Any specific requirements or constraints

IMPORTANT GUIDELINES:
- Be conversational and natural - don't make it feel like a form
- Ask ONE question at a time
- Build on their previous answers
- Show genuine interest in their project
- If they provide multiple pieces of information at once, acknowledge everything
- Don't ask for information they've already provided
- If they seem hesitant about contact info, explain it's to send them a proposal
- Keep responses SHORT (2-3 sentences max)

BRIEF COMPLETION:
When you have collected AT MINIMUM:
- Full name
- Email address  
- Clear description of what they want to build

Then you can consider the brief complete. Extract ALL collected information into a structured format.

RESPONSE FORMAT:
Always respond with a JSON object:
{
  "response": "Your conversational message to the user",
  "collected_info": {
    "fullName": "extracted name or null",
    "email": "extracted email or null",
    "company": "extracted company or null",
    "phone": "extracted phone or null",
    "projectDescription": "what they want to build",
    "projectType": "AI platform/automation/etc or null",
    "timeline": "their timeline or null",
    "budget": "budget info or null"
  },
  "brief_complete": false,
  "next_question": "What information you still need"
}

Set brief_complete to true only when you have name, email, and project description.

Current conversation stage: The user just expressed interest in sharing their project details.
"""

STANDARD_SYSTEM_PROMPT = """
You are the AI assistant for Firswood Intelligence, a specialized AI systems design and delivery practice.

Keep responses:
- SHORT (2-3 paragraphs maximum)
- Professional but conversational
- Focused on understanding their needs
- Grounded in reality, not hype

When users ask questions:
- Provide helpful, accurate information
- Guide them towards discovery calls for detailed discussions
- Never quote prices or make commitments
- Acknowledge when something needs human expertise

If someone seems ready to discuss a project, suggest they click "Tell us about your project" or book a discovery call.

Company Knowledge:
{COMPANY_KNOWLEDGE}

Current date: {datetime.now().strftime('%B %d, %Y')}
""".format(COMPANY_KNOWLEDGE=COMPANY_KNOWLEDGE, datetime=datetime)


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


@app.get("/")
async def root():
    """Root endpoint with API information"""
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "google_api_configured": bool(GOOGLE_API_KEY),
        "slack_webhook_configured": bool(SLACK_WEBHOOK_URL)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message - handles both regular chat and brief collection
    """
    try:
        client = get_gemini_client()

        # Determine system prompt based on mode
        if request.brief_mode:
            system_instruction = BRIEF_COLLECTION_SYSTEM_PROMPT
        else:
            system_instruction = STANDARD_SYSTEM_PROMPT

        # Build conversation history
        contents = []
        for msg in request.conversation_history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))

        # Add current message with context if in brief mode
        current_message = request.message
        if request.brief_mode and request.brief_context:
            current_message = f"""
User message: {request.message}

Current brief context:
{json.dumps(request.brief_context, indent=2)}

Respond conversationally while extracting any new information.
"""

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=current_message)]
        ))

        # Generate response
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )

        response_text = response.text

        # Parse brief mode response
        brief_context = None
        brief_complete = False

        if request.brief_mode:
            try:
                # Try to extract JSON from response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                    parsed = json.loads(json_str)
                elif response_text.strip().startswith("{"):
                    parsed = json.loads(response_text)
                else:
                    # AI didn't return JSON, create structure
                    parsed = {
                        "response": response_text,
                        "collected_info": request.brief_context.get("collectedInfo", {}),
                        "brief_complete": False
                    }

                # Extract values
                actual_response = parsed.get("response", response_text)
                collected_info = parsed.get("collected_info", {})
                brief_complete = parsed.get("brief_complete", False)

                # Merge with existing context
                current_info = request.brief_context.get("collectedInfo", {})
                current_info.update({k: v for k, v in collected_info.items() if v})

                brief_context = {
                    "stage": "complete" if brief_complete else "collecting",
                    "collectedInfo": current_info
                }

                response_text = actual_response

            except Exception as e:
                print(f"Error parsing brief response: {e}")
                # Continue with regular response
                brief_context = request.brief_context

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id or f"conv_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat(),
            brief_context=brief_context,
            brief_complete=brief_complete
        )

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """
    Submit project brief and send to Slack
    """
    if not SLACK_WEBHOOK_URL:
        print("ERROR: SLACK_WEBHOOK_URL not configured")
        raise HTTPException(
            status_code=500,
            detail="Slack webhook not configured"
        )

    try:
        brief = request.brief_data
        print(f"📋 Submitting brief for: {brief.get('fullName', 'N/A')}")

        # Format conversation history
        conversation_summary = ""
        if request.conversation_history:
            recent_messages = request.conversation_history[-8:]  # Last 8 messages
            for msg in recent_messages:
                role = "👤 User" if msg.role == "user" else "🤖 AI"
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                conversation_summary += f"{role}: {content}\n\n"

        # Create Slack message
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎯 New Project Brief - AI Collected",
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
                        "text": f"*Project Description:*\n{brief.get('projectDescription', 'N/A')}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Project Type:*\n{brief.get('projectType', 'Not specified')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Timeline:*\n{brief.get('timeline', 'Not specified')}"
                        }
                    ]
                }
            ]
        }

        # Add conversation summary if available
        if conversation_summary:
            slack_message["blocks"].append({
                "type": "divider"
            })
            slack_message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Conversation Summary:*\n```{conversation_summary}```"
                }
            })

        # Add metadata
        slack_message["blocks"].extend([
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
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📧 Reply via Email",
                            "emoji": True
                        },
                        "url": f"mailto:{brief.get('email', '')}?subject=Re: Your Firswood Project&body=Hi {brief.get('fullName', '')},%0D%0A%0D%0AThanks for sharing your project details with us.%0D%0A%0D%0A",
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