# main.py - v4.0 - Natural Conversation + Discovery Call Flow
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
AI systems design and delivery practice specializing in production-ready solutions.
Location: Manchester, UK
Website: www.firswoodintelligence.com
Discovery Call: calendar.app.google/kVahCoFGsHhgiSE76
"""

CORE_OPERATING_GUIDELINES = """
# Firswood Intelligence AI Assistant - Natural Conversation Guide

## Your Personality
You're a friendly, knowledgeable AI consultant. You're genuinely interested in their project and want to help. 
You're conversational, not robotic. Think of yourself as a helpful colleague, not a form.

## Conversation Style

### Be Natural and Warm:
✅ "That sounds really useful! What kind of products do you sell?"
✅ "Nice to meet you, Hamid! What company are you with?"
✅ "I love that idea. How soon are you looking to launch?"
✅ "Makes sense! Where's the product data currently stored?"

❌ "What is your company name?"
❌ "Please provide timeline."
❌ "What problem will this solve?"

### Show Interest:
- React to what they say with brief acknowledgments
- Use phrases like: "That makes sense", "I see", "Great!", "Interesting", "Love it"
- Ask follow-up questions that show you're listening
- Build on their previous answers

### Keep It Flowing:
- One question at a time
- Short responses (20-40 words)
- Natural transitions between topics
- Don't make it feel like an interrogation

## Information Gathering Flow

**Phase 1: Understanding the Project (2-3 messages)**
- What they want to build
- Why they need it / what problem it solves
- Key details about their use case

**Phase 2: Getting to Know Them (2-3 messages)**
- Their name (casual: "By the way, what's your name?")
- Their company (natural: "What company are you with?")
- Their email (friendly: "What's your work email? I'll loop you in.")

**Phase 3: Logistics (1-2 messages)**
- Timeline ("How soon are you looking to launch?")
- Budget/team size if relevant

**Phase 4: Next Steps (1 message)**
- Suggest discovery call
- Share calendar link
- End warmly

## Conversation Examples

### Example 1: Product Chatbot
User: "I need a chatbot for my website"
You: "Sounds great! What would the chatbot help with?"

User: "Answering product questions"
You: "Nice! What kind of products do you sell?"

User: "Electronics - drills and switches"
You: "Got it. By the way, what's your name?"

User: "Sarah"
You: "Nice to meet you, Sarah! What company are you with?"

User: "BuildPro"
You: "Perfect. What's your work email?"

User: "sarah@buildpro.com"
You: "Thanks! How soon are you looking to launch this?"

User: "2-3 months"
You: "Makes sense. I'd love to discuss this in detail. Want to book a quick discovery call? Here's my calendar: [link]"

### Example 2: Customer Support Bot
User: "I want an AI for customer support"
You: "That's a great use case! What kind of support queries would it handle?"

User: "Order status, returns, FAQs"
You: "Perfect. Where's your order data currently stored?"

User: "Shopify and our internal system"
You: "Got it. What's your name, by the way?"

## Key Rules

1. **Be Conversational**: Sound human, not like a chatbot
2. **Show Interest**: React to what they say
3. **Ask Naturally**: Work questions into the flow
4. **Never Repeat**: If you know something, don't ask again
5. **Build Trust**: Be helpful and knowledgeable
6. **End with Action**: Always suggest the discovery call

## Response Length
- 15-40 words per response
- One main point or question
- Optional brief reaction/acknowledgment first

## Tone
Friendly professional. Like talking to a knowledgeable colleague over coffee, not filling out a form.
"""

DATA_EXTRACTION_PROMPT = """Extract information from this conversation into JSON.

RULES:
1. **fullName**: Extract from "my name is X", "I'm X", or name given after being asked
   - Capitalize: "sarah" → "Sarah", "hamid abbas" → "Hamid Abbas"

2. **workEmail**: Any email address

3. **company**: Extract from "company is X", "work at X", "with X", or single-word response
   - Capitalize: "buildpro" → "BuildPro"

4. **phone**: Any phone number

5. **projectType**: Categorize intelligently:
   - "Customer Support Chatbot" - for support queries, tickets, help desk
   - "Product Support Chatbot" - for product questions, comparisons, recommendations
   - "Order Tracking System" - for order status, tracking, delivery
   - "Document Q&A Chatbot" - for answering questions from documents
   - "Analytics Dashboard" - for reporting, metrics, insights
   - "AI Agent" - for complex multi-step tasks
   - "Chatbot" - general conversational AI

6. **timeline**: Standardize to:
   - "ASAP" (urgent, immediately)
   - "1 month" (1 month, 4 weeks, 30 days)
   - "1-3 months" (2-3 months, quarter)
   - "3-6 months" (3-6 months, half year)
   - "6+ months" (long term, next year)

7. **goal**: Summarize the main problem/goal in 1-2 clear sentences

CONTEXT RULES:
- Single-word responses to questions are likely the answer
- "3 months" → "1-3 months"
- Ignore filler: "yes", "no", "okay"
- Be smart about context

Return ONLY valid JSON (no markdown):
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
    ready_for_call: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    extracted_data: Optional[Dict[str, Any]] = None
    ready_for_call: bool = False


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


def get_system_instruction(additional_context="", conversation_summary="", ready_for_call=False):
    if ready_for_call:
        return f"""You are the AI assistant for Firswood Intelligence.

The user has provided all the key information. Now it's time to wrap up and suggest a discovery call.

{COMPANY_KNOWLEDGE}

IMPORTANT: 
- Thank them for sharing details
- Briefly summarize what you understand (1 sentence)
- Suggest booking a discovery call to discuss in detail
- Share the calendar link: calendar.app.google/kVahCoFGsHhgiSE76
- Keep it warm and friendly
- Max 50 words

Example:
"Thanks for all the details, Sarah! So you're looking to build a product support chatbot for BuildPro's electronics. I'd love to discuss the technical approach and timeline in detail. Want to book a quick 20-minute discovery call? Here's my calendar: calendar.app.google/kVahCoFGsHhgiSE76"

Current date: {datetime.now().strftime('%B %d, %Y')}
"""

    base = f"""You are the AI assistant for Firswood Intelligence.

{CORE_OPERATING_GUIDELINES}

{COMPANY_KNOWLEDGE}

CRITICAL REMINDERS:
- Be warm and conversational
- Show genuine interest in their project
- Never repeat questions you already asked
- Keep responses 15-40 words
- One question per response
- Build natural rapport

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

        # Clean markdown
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]

        extracted_data = json.loads(extracted_text.strip())

        print(f"[EXTRACT] ✅ Extracted data")
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
    """Generate summary of what we know"""
    known_info = []

    if extracted_data.get('projectType'):
        known_info.append(f"- Project type: {extracted_data['projectType']}")
    if extracted_data.get('goal'):
        known_info.append(f"- Goal: {extracted_data['goal'][:80]}")
    if extracted_data.get('fullName'):
        known_info.append(f"- Name: {extracted_data['fullName']}")
    if extracted_data.get('workEmail'):
        known_info.append(f"- Email: {extracted_data['workEmail']}")
    if extracted_data.get('company'):
        known_info.append(f"- Company: {extracted_data['company']}")
    if extracted_data.get('timeline'):
        known_info.append(f"- Timeline: {extracted_data['timeline']}")

    if known_info:
        return f"\n=== INFO ALREADY COLLECTED ===\n" + "\n".join(
            known_info) + "\n\nNEVER ask about these again. Ask something NEW or move to discovery call.\n"
    return ""


def check_ready_for_call(extracted_data: Dict[str, Any], message_count: int) -> bool:
    """Check if we have enough info to suggest discovery call"""
    has_email = extracted_data.get('workEmail') and extracted_data['workEmail'] not in [None, 'N/A', 'null']
    has_name = extracted_data.get('fullName') and extracted_data['fullName'] not in [None, 'N/A', 'null']
    has_company = extracted_data.get('company') and extracted_data['company'] not in [None, 'N/A', 'null']
    has_project = extracted_data.get('projectType') and extracted_data['projectType'] not in [None, 'N/A', 'null']
    has_goal = extracted_data.get('goal') and extracted_data['goal'] not in [None, 'N/A', 'null']

    # Count filled fields
    filled = sum([has_email, has_name, has_company, has_project, has_goal])

    # Ready if: has email + at least 3 other fields + at least 5 messages
    ready = has_email and filled >= 4 and message_count >= 5

    print(f"[READY_CHECK] Email: {has_email}, Filled: {filled}/5, Messages: {message_count}, Ready: {ready}")

    return ready


@app.get("/")
async def root():
    return {
        "service": "Firswood Intelligence Chat API",
        "status": "running",
        "version": "4.0.0",
        "features": ["Natural conversation", "AI extraction", "Discovery call flow"],
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Natural conversation with discovery call flow"""
    try:
        message_num = len(request.conversation_history) + 1
        print(f"\n[CHAT] Message #{message_num}: {request.message[:50]}...")

        client = get_gemini_client()

        # Extract current data
        temp_history = request.conversation_history + [
            Message(role="user", content=request.message, timestamp=datetime.now().isoformat())
        ]
        extracted_data = await extract_data_with_ai(temp_history)

        # Check if ready for discovery call
        ready_for_call = check_ready_for_call(extracted_data, message_num)

        # Generate conversation summary
        conv_summary = generate_conversation_summary(extracted_data)

        # Build contents - ONLY user messages
        contents = []
        for msg in request.conversation_history:
            if msg.role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.content)]
                ))

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        ))

        # Generate response
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=get_system_instruction(
                    request.system_context or "",
                    conv_summary,
                    ready_for_call
                ),
                temperature=0.8,  # Higher for more natural conversation
            )
        )

        conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

        print(f"[CHAT] ✅ Response generated (ready_for_call: {ready_for_call})")

        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            extracted_data=extracted_data,
            ready_for_call=ready_for_call
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
                f"🎉 *NEW LEAD - READY FOR DISCOVERY CALL!*\n\n"
                f"👤 *Name:* {full_name}\n"
                f"📧 *Email:* {work_email}\n"
                f"🏢 *Company:* {company}\n"
                f"📞 *Phone:* {phone}\n"
                f"💼 *Project:* {project_type}\n"
                f"📅 *Timeline:* {timeline}\n\n"
                f"🎯 *Goal:*\n{goal}\n\n"
                f"⏰ *Submitted:* {formatted_time}\n"
                f"🆔 *ID:* {request.conversation_id}\n"
                f"🔗 *Page:* {request.url or 'N/A'}\n\n"
                f"📅 *Next Step:* User invited to book discovery call"
            )
        }

        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=slack_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Slack error")

        print(f"[BRIEF] ✅ Submitted to Slack")

        return {
            "success": True,
            "message": "Brief submitted - ready for discovery call",
            "conversation_id": request.conversation_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)