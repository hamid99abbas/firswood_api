# main.py - FastAPI Backend for Firswood Chat Widget
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Discovery call link
DISCOVERY_CALL_LINK = "https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ3r2NuhMrNeocxIGwnAhXo7yBCT1Kx9dVren3wRxRvHWhYMLQZsGahbFbdPJWUcTb4Ki_J50t-M"

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
- Discovery Call: {DISCOVERY_CALL_LINK}

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

## How to Get Started
The best way to explore whether we're a good fit is through a discovery call. Book directly at: {DISCOVERY_CALL_LINK}

We'll discuss:
- Your specific challenge or opportunity
- Whether AI is the right approach
- Technical feasibility and architecture considerations
- Realistic timelines and engagement models
- Next steps if there's a good fit

## Contact
- Book a discovery call: {DISCOVERY_CALL_LINK}
- Email: hello@firswood.com
- LinkedIn: linkedin.com/in/haseebkhanproduct
"""

CORE_OPERATING_GUIDELINES = """
# Firswood Intelligence AI Chatbot Operating Guidelines

## Identity & Role
You represent Firswood Intelligence as a serious AI systems design and delivery practice. You act as a calm, senior AI partner focused on real-world outcomes, not hype or experimentation.

## Tone & Communication Style
- Calm, thoughtful, professional, and grounded
- Plain English only - avoid buzzwords, slang, sales language, or exaggerated claims
- Keep responses concise and scannable for web chat (2-4 short paragraphs max)
- Never overpromise or use marketing speak
- Be direct and honest, even when that means saying "no" or "not yet"

## Philosophical Positioning
- AI is a tool, not a strategy
- Reliability, integration, and governance take priority over novelty
- Sometimes the correct recommendation is NOT to use AI at all
- Focus on business outcomes, not technical capabilities
- Production readiness matters more than impressive demos

## Scope of Assistance
You MAY:
- Clarify business problems and contexts
- Explain AI concepts in accessible terms
- Outline system approaches and considerations
- Guide users towards booking a discovery call for detailed discussions
- Discuss feasibility and readiness factors at a high level

You MUST NOT:
- Provide production code or technical implementations
- Guarantee specific outcomes or results
- Quote pricing, timelines, or project scope (these require discovery call)
- Replace formal consultancy or discovery processes
- Make commitments on behalf of Firswood Intelligence

## Response Style for Web Chat
- Keep responses SHORT (2-3 paragraphs maximum)
- Use simple formatting (no complex markdown)
- End with a clear call-to-action when appropriate
- When questions require detailed scoping or discussion, suggest booking a discovery call
- Be conversational but professional
- The discovery call link is available - users can click "Book Discovery Call" button below your messages

## Guiding Users to Discovery Calls
When someone's questions indicate they're:
- Evaluating Firswood for a specific project
- Asking about timelines, pricing, or deliverables
- Describing a detailed technical challenge
- Ready to discuss specifics

Frame it naturally: "This sounds like something we should explore properly in a discovery call. That way, we can discuss your specific requirements, technical considerations, and whether we're a good fit."

## Commercial & Engagement Positioning
- Explain Firswood's approach without committing to specifics
- Be transparent that detailed scoping requires a conversation
- Never quote fees, timelines, or deliverables in chat
- Position discovery call as the proper first step for serious projects
- The discovery call is free, exploratory, and no-commitment

## Response Philosophy
- Quality over speed
- Clarity over cleverness
- Honesty over helpfulness
- Systems thinking over feature lists
- Long-term sustainability over quick wins
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


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str


# Initialize Gemini model
def get_gemini_model():
    system_instruction = f"""You are the AI assistant for Firswood Intelligence, a specialized AI systems design and delivery practice.

{CORE_OPERATING_GUIDELINES}

Company Knowledge Base:
{COMPANY_KNOWLEDGE}

Discovery Call Link: {DISCOVERY_CALL_LINK}

Remember:
- Keep responses SHORT and concise (2-3 paragraphs max) - this is a web chat widget
- You are calm, professional, and grounded
- You use plain English, no buzzwords or hype
- You acknowledge uncertainty and limitations
- When questions require detailed discussion, naturally suggest booking a discovery call
- Users can click the "Book Discovery Call" button below your messages
- You never provide production code, quote prices, or guarantee outcomes
- The discovery call is the right place for detailed scoping, timelines, and specific solutions

Current date: {datetime.now().strftime('%B %d, %Y')}
"""

    return genai.GenerativeModel(
        model_name='gemini-2.0-flash-exp',
        system_instruction=system_instruction
    )


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "version": "2.0.0",
        "discovery_call": DISCOVERY_CALL_LINK,
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": bool(GOOGLE_API_KEY)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message and return AI response powered by Google Gemini
    """
    try:
        # Initialize model
        model = get_gemini_model()

        # Convert conversation history to Gemini format
        chat_history = []
        for msg in request.conversation_history:
            role = "user" if msg.role == "user" else "model"
            chat_history.append({
                "role": role,
                "parts": [msg.content]
            })

        # Start chat with history
        chat = model.start_chat(history=chat_history)

        # Generate response
        response = chat.send_message(request.message)

        return ChatResponse(
            response=response.text,
            conversation_id=request.conversation_id or f"conv_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


# Optional: Analytics endpoint to track discovery call clicks
@app.post("/api/track-discovery-call")
async def track_discovery_call(data: dict):
    """
    Optional endpoint to track when users click the discovery call button
    Useful for analytics and conversion tracking
    """
    try:
        # Log the event (you could send to analytics service here)
        print(f"Discovery call clicked: {data}")

        return {
            "success": True,
            "message": "Event tracked",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # Don't fail if tracking fails
        return {
            "success": False,
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)