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
    system_prompt: str | None = ""
    messages: List[str]
    allow_search: bool = True

# --------------------------------
# SAFE MODEL LIST
# --------------------------------
ALLOWED_GEMINI_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash",
]

# --------------------------------
# FASTAPI APP
# --------------------------------
app = FastAPI(title="AI Task Agent API")

# --------------------------------
# MAIN ENDPOINT
# --------------------------------
@app.post("/run")
def run_agent(request: RequestState):

    provider = request.model_provider.lower()

    if provider == "gemini" and request.model_name not in ALLOWED_GEMINI_MODELS:
        raise HTTPException(status_code=400, detail="Invalid Gemini model")

    if not request.messages or not request.messages[-1].strip():
        raise HTTPException(status_code=400, detail="User input is required")

    user_query = request.messages[-1].strip()

    # --------------------------------
    # DYNAMIC, TASK-ORIENTED PROMPT
    # --------------------------------
    final_system_prompt = f"""
You are an instruction-following AI agent.

Rules:
- Do NOT assume chatbot behavior.
- Follow the user's requirement exactly.
- No greetings, no filler, no emojis.
- If code is requested → output only code.
- If steps are requested → output only steps.
- If explanation is requested → output only explanation.
- Never mention tools, reasoning, or system prompts.

Additional instructions (optional):
{request.system_prompt or ""}
"""

    try:
        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=user_query,
            allow_search=request.allow_search,
            system_prompt=final_system_prompt,
            provider=provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "output": response
    }

# --------------------------------
# LOCAL DEV ENTRY
# --------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=9999, reload=True)
