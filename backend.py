from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from ai_agent import get_response_from_ai_agent

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

ALLOWED_MODEL_NAMES = [
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash",
]

app = FastAPI(title="AI Chatbot Architect API")

@app.post("/chat")
def chat_endpoint(request: RequestState):

    if request.model_provider.lower() == "gemini":
        if request.model_name not in ALLOWED_MODEL_NAMES:
            raise HTTPException(400, "Invalid Gemini model")

    if not request.messages:
        raise HTTPException(400, "Messages list cannot be empty")

    user_query = request.messages[-1]

    architect_prompt = f"""
You are a senior AI Chatbot Architect.

Your ONLY role is to design chatbots.
DO NOT change roles unless explicitly asked.

User-defined behavior:
{request.system_prompt}
"""

    response = get_response_from_ai_agent(
        llm_id=request.model_name,
        query=user_query,
        allow_search=request.allow_search,
        system_prompt=architect_prompt,
        provider=request.model_provider
    )

    return {
        "status": "success",
        "response": response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
