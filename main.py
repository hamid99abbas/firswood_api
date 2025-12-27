# main.py - 3 Phase Conversation Flow
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

app = FastAPI(title="Firswood Intelligence Chat API v4.0")

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
AI systems design and delivery practice specializing in production-ready AI.
Location: Manchester, UK
Website: www.firswoodintelligence.com
LinkedIn: linkedin.com/in/haseebkhanproduct
Booking: calendar.app.google/kVahCoFGsHhgiSE76

## What We Build
1. Autonomous AI Agents
2. RAG & Enterprise Search
3. Conversational AI
4. Forecasting & Decision Intelligence
5. Real-Time Dashboards
6. Computer Vision & On-Device AI
7. Full-Stack AI Product Development

## Engagement Models
1. Fixed-Scope Builds
2. Development Partnerships
3. High-Impact MVPs

## Philosophy
- Builder-led: Work directly with the person designing and building
- Production-first: Systems meant for daily use, not demos
- Commercially grounded: Every decision considers cost, scale, and ROI
"""

FAQ_KNOWLEDGE = """
# Common Questions & Answers

## Business Discovery & Feasibility

**Too much manual processing - can AI help?**
Yes, AI excels at automating repetitive tasks like processing enquiries, data entry, and routing. We'd assess your current workflow to identify where automation adds the most value.

**What problems are suitable for AI?**
Best fits: Pattern recognition, data processing, customer support, forecasting, content generation. Less suitable: Tasks requiring deep human judgment, physical manipulation, or highly regulated decisions without human oversight.

**Is my business ready for AI?**
Key indicators: You have defined processes, accessible data (even if messy), team buy-in, and clear success metrics. We can assess readiness in a discovery session.

**Can AI work with our CRM?**
Absolutely. We integrate AI on top of existing systems (Salesforce, HubSpot, custom databases) rather than replacing them. AI pulls data, generates insights, and writes back actions.

**Should we fix processes first?**
Sometimes yes. If processes are fundamentally broken, AI will automate the problem. We help identify whether you need process optimization, AI, or both.

## AI System Design & Architecture

**What's a production-ready chatbot?**
It handles errors gracefully, integrates with your systems, has monitoring/logging, maintains conversation context, escalates to humans when stuck, and runs reliably 24/7.

**Integration with existing tools?**
We connect via APIs to your CRM, databases, Slack, email, or custom systems. AI becomes another "team member" with system access.

**Demo vs Production AI?**
Demo: Works in controlled conditions, impressive but fragile. Production: Handles edge cases, has error handling, monitoring, security, and runs reliably under load.

**Handling failures?**
We build fallbacks: if AI is unsure, it escalates to humans. Systems include confidence scoring, logging, and monitoring to catch issues early.

**Human-in-the-loop?**
AI handles routine tasks autonomously, flags uncertain cases for human review. Humans approve high-stakes decisions. Think "AI proposes, human disposes."

## Data & Readiness

**What data do we need?**
Depends on use case. Chatbots need FAQs/docs. Forecasting needs historical data. We can work with limited data and improve as you collect more.

**Our data is messy?**
That's normal. We clean and structure data as part of the project. Messy data slows things down but doesn't rule out AI.

**Can AI use internal documents?**
Yes, through RAG (Retrieval-Augmented Generation). AI searches your docs, finds relevant info, and answers questions with citations.

**Data security?**
We implement access controls, encryption, and can deploy on-premise or in your cloud environment. Data never leaves your control.

**Do we need to train models?**
Usually no. We use pre-trained models (like GPT, Claude) and fine-tune with your data. Custom training only for specialized needs.

## Automation & Operations

**Can AI automate internal workflows?**
Yes - document processing, report generation, email routing, data entry, scheduling. We identify high-impact internal use cases.

**What's safe to automate?**
Low-risk, high-volume tasks with clear success criteria. Examples: categorizing tickets, extracting invoice data, scheduling meetings.

**Where to avoid automation?**
High-stakes decisions (hiring, medical, legal), creative strategy, relationship-building, or tasks requiring empathy and nuance.

**AI for decision support?**
AI provides analysis, recommendations, and drafts. Humans make final decisions. This works well for complex scenarios.

**AI agents vs traditional automation?**
Traditional automation follows fixed rules. AI agents adapt, handle exceptions, and work with unstructured data.

## Commercial & Strategic

**How to measure AI value?**
Time saved, cost reduction, error rate, customer satisfaction, response time. We define metrics upfront and track them.

**What's a typical engagement?**
Discovery (1-2 weeks) → Design (2-3 weeks) → Build (4-8 weeks) → Deploy (1-2 weeks) → Iterate. 3-6 months total for most projects.

**Biggest risks businesses underestimate?**
1) Change management 2) Data quality 3) Scope creep 4) Over-promising 5) Lack of clear success metrics

**Timeline from idea to production?**
MVP: 6-12 weeks. Full system: 3-6 months. Depends on complexity and integrations.

**Realistic first project scope?**
Focus on ONE high-impact use case. Example: FAQ chatbot, document Q&A, or specific workflow automation. Prove value, then expand.

## Governance, Risk & Ethics

**Preventing unsafe decisions?**
Confidence thresholds, human approval for high-stakes actions, comprehensive testing, monitoring, and clear escalation paths.

**When AI gets it wrong?**
Systems log all interactions. We investigate, retrain, add safeguards. Humans are always in the loop for important decisions.

**Compliance and accountability?**
We document decision logic, maintain audit trails, implement access controls. For regulated industries, we involve legal/compliance teams early.

**AI in regulated industries?**
Yes, with extra care. Healthcare, finance, legal all use AI successfully with proper governance, human oversight, and compliance frameworks.

**How much human oversight?**
Varies by risk. Low-risk tasks: minimal. High-stakes: continuous. We design appropriate oversight levels.

## Boundaries & Expectations

**Can AI run the business fully autonomous?**
No. AI handles specific tasks but needs human oversight, strategy, and judgment. Think "AI assistant" not "AI CEO."

**Can AI replace our team?**
AI augments teams, not replaces them. It handles repetitive work so humans focus on creative, strategic, relationship-building tasks.

**Can you guarantee results?**
We guarantee professional delivery and best practices. Business outcomes depend on many factors. We set realistic expectations and measure progress.

**Quick implementation without change?**
No. Effective AI requires some process change, training, and adoption. We minimize disruption but change is necessary for impact.

**Just plug into ChatGPT?**
ChatGPT is a tool, not a solution. Production systems need custom prompts, integrations, security, monitoring - that's what we build.
"""

PHASE_1_SYSTEM = f"""You are the AI assistant for Firswood Intelligence.

{COMPANY_KNOWLEDGE}

## YOUR ROLE - PHASE 1: ANSWER QUESTIONS

The user is asking questions about AI. Your job is to:
1. Answer their questions thoroughly using the FAQ knowledge below
2. If question is NOT in FAQ, answer based on general AI knowledge and Firswood's expertise
3. Be helpful, knowledgeable, and conversational
4. Keep responses under 100 words
5. After answering 2-3 questions, ask: "Do you have a project in mind where you'd like to use AI?"

{FAQ_KNOWLEDGE}

IMPORTANT RULES:
- If question is in FAQ, use that answer
- If question is NOT in FAQ, provide a thoughtful answer based on your AI knowledge
- Never say "I don't know" - always try to help
- If truly outside expertise, say: "That's outside my area, but I can connect you with someone who can help. Do you have a specific project in mind?"
- Reference Firswood Intelligence capabilities when relevant
- Be warm and approachable
- After 2-3 exchanges, gently transition: "By the way, do you have a specific AI project you're thinking about?"
"""

PHASE_2_SYSTEM = f"""You are the AI assistant for Firswood Intelligence.

{COMPANY_KNOWLEDGE}

## YOUR ROLE - PHASE 2: PROJECT DISCOVERY

The user has a project. Your job is to:
1. Understand their project through natural conversation
2. Gather: name, email, company, project details, timeline
3. Keep responses SHORT (1-2 sentences, max 40 words)
4. Ask ONE question at a time
5. After gathering info, suggest a discovery call

CONVERSATION FLOW:
1. Ask about their project: "What kind of AI project?"
2. Ask about the problem: "What problem does it solve?"
3. Ask for their name: "What's your name?"
4. Once you have the name, acknowledge it ONCE: "Thanks, [Name]."
5. Ask for email: "What's your email?"
6. Ask for company: "What company are you with?"
7. Ask for timeline: "What's your timeline?"
8. Offer call: "Would you like to schedule a discovery call?"

CRITICAL RULES:
- NEVER repeat the person's name more than once per response
- After acknowledging their name, just ask the next question
- Keep responses under 40 words
- Move forward through the flow
- Don't be repetitive

GOOD EXAMPLES:
User: "Hamid"
You: "Thanks, Hamid. What's your email?"

User: "hamid@email.com"  
You: "Got it. What company are you with?"

BAD EXAMPLES:
User: "Hamid"
You: "Thanks, Hamid! Nice to meet you, Hamid. What's your email, Hamid?" ❌ (too many name repetitions)

User: "hamid@email.com"
You: "Thanks for sharing your email, hamid@email.com. I have your email saved." ❌ (too wordy)
"""

PHASE_3_SYSTEM = """You are the AI assistant for Firswood Intelligence.

## YOUR ROLE - PHASE 3: BOOK CALL

The user has been asked about a discovery call. Provide a response based on their answer.

**CRITICAL: Use EXACTLY this format for the booking link - DO NOT change it:**

If user says YES (or variations like "sure", "ok", "let's do it"):
```
Perfect! You can book a time that works for you using the link below:

[Book Your Discovery Call](https://calendar.app.google/kVahCoFGsHhgiSE76)

Looking forward to discussing your project in detail!
```

If user says NO (or variations like "not now", "maybe later", "not interested"):
```
No problem at all! If you change your mind or have more questions, I'm here anytime. Feel free to reach out whenever you're ready.
```

RULES:
- Use the EXACT markdown format: [Book Your Discovery Call](https://calendar.app.google/kVahCoFGsHhgiSE76)
- Don't mention the user's name unnecessarily
- Keep it short and friendly
- Don't add extra explanations
"""

DATA_EXTRACTION_PROMPT = """Extract information from this conversation into JSON.

EXTRACTION RULES:

1. **fullName**: Look for explicit name statements
   - "my name is Hamid" → "Hamid"
   - "I'm John Smith" → "John Smith"
   - Capitalize properly

2. **workEmail**: Any email address format

3. **company**: IMPORTANT - Look carefully for company name
   - If AI asks "what company?" and user responds with a single word → that's the company
   - "company is Emebron" → "Emebron"
   - "we are Acme Corp" → "Acme Corp"
   - User response after "what company are you with?" → that's the company
   - ALWAYS capitalize first letter

4. **phone**: Any phone number

5. **projectType**: Categorize based on description
   - Customer support → "Customer Support Chatbot"
   - Order tracking → "Order Tracking System"
   - Document questions → "Document Q&A Chatbot"
   - General chatbot → "Chatbot"

6. **timeline**: Extract and standardize
   - "3 months", "3-4 months" → "1-3 months"
   - "1 month" → "1 month"
   - "6 months" → "3-6 months"
   - "ASAP", "urgent" → "ASAP"

7. **goal**: Summarize main problem/objective in 1-2 sentences

CONTEXT AWARENESS:
- Pay special attention to single-word responses after questions
- If AI asks "What company?" and user says "emebron" → company is "Emebron"
- If AI asks "timeline?" and user says "3 4 months" → timeline is "1-3 months"

Return ONLY valid JSON (no markdown, no explanations):
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
    conversation_phase: Optional[str] = "phase1"  # phase1, phase2, phase3


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    conversation_phase: str
    extracted_data: Optional[Dict[str, Any]] = None
    should_submit_brief: bool = False


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


def detect_phase_transition(message: str, conversation_history: List[Message], current_phase: str) -> str:
    """Detect if we should move to next phase"""
    msg_lower = message.lower()

    # Phase 1 → Phase 2: User mentions having a project
    if current_phase == "phase1":
        project_keywords = ['i want', 'i need', 'we need', 'build', 'create', 'develop', 'project', 'yes i have',
                            'yes we have']
        if any(keyword in msg_lower for keyword in project_keywords):
            print(f"[PHASE] Transition 1→2: User has a project")
            return "phase2"

    # Phase 2 → Phase 3: User wants to book call
    if current_phase == "phase2":
        call_keywords = ['schedule', 'book', 'call', 'meeting', 'talk', 'discuss', 'discovery', 'yes']
        # Also check if AI asked about booking and user said yes
        if len(conversation_history) > 0:
            last_ai_msg = next((m.content for m in reversed(conversation_history) if m.role == "assistant"), "")
            if any(word in last_ai_msg.lower() for word in ['discovery call', 'schedule', 'book']):
                if any(keyword in msg_lower for keyword in call_keywords):
                    print(f"[PHASE] Transition 2→3: User wants to book call")
                    return "phase3"

    return current_phase


async def extract_data_with_ai(conversation_history: List[Message]) -> Dict[str, Any]:
    """AI-powered extraction"""
    try:
        conversation_text = ""
        for i, msg in enumerate(conversation_history):
            role = "User" if msg.role == "user" else "AI"
            conversation_text += f"{role}: {msg.content}\n"

        full_prompt = DATA_EXTRACTION_PROMPT + "\n" + conversation_text

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

        extracted_text = response.text.strip()
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]

        extracted_data = json.loads(extracted_text.strip())
        print(f"[EXTRACT] ✅ {extracted_data}")
        return extracted_data

    except Exception as e:
        print(f"[ERROR] Extraction: {str(e)}")
        return {
            "fullName": None,
            "workEmail": None,
            "company": None,
            "phone": None,
            "projectType": None,
            "timeline": None,
            "goal": None
        }


def should_submit_brief(extracted_data: Dict[str, Any], old_phase: str, new_phase: str, user_message: str) -> bool:
    """Check if we should submit brief - when user responds to discovery call question"""
    # Submit when:
    # 1. Moving from phase 2 to phase 3 (user said YES)
    # 2. User said NO to discovery call (we're still in phase 2 but they declined)

    has_email = bool(extracted_data.get('workEmail'))
    has_project = bool(extracted_data.get('projectType') or extracted_data.get('goal'))

    # Check if transitioning to phase 3 (YES to call)
    if old_phase == "phase2" and new_phase == "phase3":
        result = has_email and has_project
        print(f"[BRIEF_CHECK] Phase 2→3 (YES): Email: {has_email}, Project: {has_project} → Submit: {result}")
        return result

    # Check if user declined call (stayed in phase 2 but said no)
    if old_phase == "phase2" and new_phase == "phase2":
        decline_keywords = ['no', 'not now', 'maybe later', 'not ready', 'not yet', 'later']
        msg_lower = user_message.lower().strip()
        if any(keyword in msg_lower for keyword in decline_keywords):
            result = has_email and has_project
            print(
                f"[BRIEF_CHECK] User declined call (NO): Email: {has_email}, Project: {has_project} → Submit: {result}")
            return result

    return False


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "version": "4.0.0",
        "features": ["3-phase conversation", "FAQ answering", "Project discovery"],
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
        "google_api": bool(GOOGLE_API_KEY),
        "slack": bool(SLACK_WEBHOOK_URL)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """3-phase conversation handler"""
    try:
        current_phase = request.conversation_phase or "phase1"
        message_count = len(request.conversation_history) + 1

        print(f"\n[CHAT] Phase: {current_phase}, Message #{message_count}")
        print(f"[CHAT] User: {request.message[:60]}...")

        # Detect phase transition BEFORE generating response
        old_phase = current_phase
        new_phase = detect_phase_transition(
            request.message,
            request.conversation_history,
            current_phase
        )

        if new_phase != old_phase:
            print(f"[PHASE] Switching from {old_phase} to {new_phase}")
            current_phase = new_phase

        # Select system prompt based on phase
        if current_phase == "phase1":
            system_prompt = PHASE_1_SYSTEM
        elif current_phase == "phase2":
            system_prompt = PHASE_2_SYSTEM
        else:  # phase3
            system_prompt = PHASE_3_SYSTEM

        # Build conversation contents
        contents = []
        for msg in request.conversation_history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        ))

        # Generate response
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            )
        )

        # Extract data and check if should submit
        extracted_data = None
        should_submit = False

        if current_phase == "phase2" or new_phase == "phase3":
            # Extract data when in phase 2 or moving to phase 3
            temp_history = request.conversation_history + [
                Message(role="user", content=request.message),
                Message(role="assistant", content=response.text)
            ]
            extracted_data = await extract_data_with_ai(temp_history)

            # Check if should submit (on phase 2→3 transition OR if user declined call)
            should_submit = should_submit_brief(extracted_data, old_phase, new_phase, request.message)

        conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

        print(f"[CHAT] ✅ Response sent (phase: {current_phase})")

        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            conversation_phase=current_phase,
            extracted_data=extracted_data,
            should_submit_brief=should_submit
        )

    except Exception as e:
        print(f"[ERROR] Chat: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/submit-brief")
async def submit_brief(request: BriefSubmission):
    """Submit to Slack"""
    print(f"[BRIEF] Submitting...")

    if not SLACK_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Slack not configured")

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
            formatted_time = datetime.fromisoformat(request.timestamp).strftime('%Y-%m-%d %H:%M')
        except:
            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M')

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
                f"⏰ {formatted_time}\n"
                f"🆔 {request.conversation_id}"
            )
        }

        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Slack: {response.status_code}")

        print(f"[BRIEF] ✅ Submitted")

        return {
            "success": True,
            "message": "Brief submitted",
            "conversation_id": request.conversation_id
        }

    except Exception as e:
        print(f"[ERROR] Submit: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)