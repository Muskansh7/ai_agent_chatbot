from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

from ai_agent import get_response_from_ai_agent
from advisor_prompts import ADVISOR_PROMPTS


# --------------------------------
# REQUEST SCHEMA
# --------------------------------
class RequestState(BaseModel):
    model_name: str
    model_provider: str

    # single | council
    mode: str = "single"

    # Used when mode == single
    advisor: str = ""

    # Used when mode == council
    advisors: List[str] = Field(default_factory=list)

    # Used when advisor == Create Custom Advisor
    custom_prompt: str = ""

    messages: List[str]

    allow_search: bool = True


# --------------------------------
# ALLOWED GEMINI MODELS
# --------------------------------
GEMINI_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
]


# --------------------------------
# FASTAPI APP
# --------------------------------
app = FastAPI(
    title="AI Advisor Platform",
    description="Single Advisor + AI Council",
    version="2.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://aiplacementadvisor.vercel.app",
        "https://aiplacementadvisor-git-main-muskansh7s-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------
# CHAT ENDPOINT
# --------------------------------
@app.post("/chat")
def chat_endpoint(request: RequestState):

    print("\n========== REQUEST ==========")
    print(request.model_dump())
    print("=============================\n")

    provider = request.model_provider.lower()

    # -----------------------------
    # MODEL VALIDATION
    # -----------------------------
    if provider == "gemini":

        if request.model_name not in GEMINI_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Gemini model: {request.model_name}"
            )

    elif provider == "openai":

        if request.model_name not in OPENAI_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid OpenAI model: {request.model_name}"
            )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}"
        )

    # -----------------------------
    # MESSAGE VALIDATION
    # -----------------------------
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty."
        )

    if request.mode not in ["single", "council"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}"
        )

    user_query = request.messages[-1]

    # =====================================================
    # SINGLE ADVISOR
    # =====================================================
    if request.mode == "single":

        if request.advisor == "Create Custom Advisor":

            if not request.custom_prompt.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Custom advisor prompt is required."
                )

            system_prompt = request.custom_prompt
            role = "Custom Advisor"

        else:

            print("\n========== ADVISOR DEBUG ==========")
            print("Advisor Received:", repr(request.advisor))
            print("Available Advisors:", list(ADVISOR_PROMPTS.keys()))
            print("===================================\n")

            if request.advisor not in ADVISOR_PROMPTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid advisor selected: {request.advisor}"
                )

            system_prompt = ADVISOR_PROMPTS[request.advisor]
            role = request.advisor

    # =====================================================
    # COUNCIL MODE
    # =====================================================
    else:

        if len(request.advisors) == 0:
            raise HTTPException(
                status_code=400,
                detail="Select at least one advisor."
            )

        advisor_sections = []

        for advisor in request.advisors:

            if advisor not in ADVISOR_PROMPTS:
                continue

            advisor_sections.append(
                f"""
Advisor:
{advisor}

Instructions:
{ADVISOR_PROMPTS[advisor]}
"""
            )

        system_prompt = f"""
You are an AI Council Moderator.

The following placement advisors are participating.

{chr(10).join(advisor_sections)}

Your job is to moderate the discussion.

Instructions:

1. Every advisor answers ONLY from their expertise.
2. Avoid repeating information.
3. Mention disagreements only if meaningful.
4. Finish with one practical Council Recommendation.
"""

        role = "AI Council"
    # =====================================================
    # CALL AI AGENT
    # =====================================================

    # =====================================================
    # CALL AI AGENT
    # =====================================================

    import traceback

    try:

        ai_response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=user_query,
            allow_search=request.allow_search,
            system_prompt=system_prompt,
            provider=request.model_provider,
        )

    except Exception as e:

        print("\n========== AI ERROR ==========")
        traceback.print_exc()
        print("==============================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    # =====================================================
    # RESPONSE
    # =====================================================

    return {
        "status": "success",
        "mode": request.mode,
        "advisor": role,
        "provider": request.model_provider,
        "model": request.model_name,
        "search_enabled": request.allow_search,
        "response": ai_response,
    }