from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException

from ai_agent import get_response_from_ai_agent


# --------------------------------
# REQUEST SCHEMA
# --------------------------------
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


# --------------------------------
# ALLOWED FREE GEMINI MODELS
# --------------------------------
ALLOWED_MODEL_NAMES = [
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash",
]


# --------------------------------
# FASTAPI APP
# --------------------------------
app = FastAPI(
    title="LangGraph AI Agent",
    description="AI Chatbot Architect API",
    version="1.0.0"
)


# --------------------------------
# CHAT ENDPOINT
# --------------------------------
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    AI Chatbot Architect API

    Accepts chatbot requirements and returns
    a complete system design recommendation.
    """

    # -----------------------------
    # VALIDATIONS
    # -----------------------------
    if request.model_provider.lower() == "gemini":
        if request.model_name not in ALLOWED_MODEL_NAMES:
            raise HTTPException(
                status_code=400,
                detail="Invalid Gemini model selected"
            )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages list cannot be empty"
        )

    # -----------------------------
    # USER QUERY (PLAIN STRING)
    # -----------------------------
    user_query = request.messages[-1]

    # -----------------------------
    # ARCHITECT PROMPT
    # -----------------------------
    architect_prompt = f"""
You are a senior AI Architect.

Your job is to help users DESIGN chatbots end-to-end.

For every request:
1. Understand the chatbot goal
2. Recommend:
   - Backend framework
   - Database
   - AI model
   - Tools (search, memory, RAG)
   - APIs & integrations
   - Deployment strategy
3. Explain WHY each choice is made
4. Provide a simple architecture overview

Be:
- Practical
- Beginner-friendly
- Production-ready

User-defined behavior:
{request.system_prompt}
"""

    # -----------------------------
    # CALL AI AGENT (FIXED)
    # -----------------------------
    try:
        ai_response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=user_query,               # ✅ plain string
            allow_search=request.allow_search,
            system_prompt=architect_prompt, # ✅ injected properly
            provider=request.model_provider
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {
        "status": "success",
        "role": "AI Chatbot Architect",
        "response": ai_response
    }


# --------------------------------
# RUN SERVER
# --------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",   # 🔥 IMPORTANT FIX
        host="127.0.0.1",
        port=9999,
        reload=True
    )
