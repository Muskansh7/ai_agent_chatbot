from dotenv import load_dotenv
import os

load_dotenv()

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def get_response_from_ai_agent(
    llm_id: str,
    query: str,
    allow_search: bool,
    system_prompt: str,
    provider: str
) -> str:

    # ----------------------------
    # PROVIDER
    # ----------------------------
    provider = provider.lower()

    if provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found")

        llm = ChatGoogleGenerativeAI(
            model=llm_id,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found")

        llm = ChatOpenAI(
            model=llm_id,
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )

    else:
        raise ValueError("Provider must be 'gemini' or 'openai'")

    # ----------------------------
    # TOOLS (OPTIONAL SEARCH)
    # ----------------------------
    tools = []

    if allow_search and TAVILY_API_KEY:
        tools.append(
            TavilySearch(
                api_key=TAVILY_API_KEY,
                max_results=3
            )
        )

    # ----------------------------
    # STRONG SYSTEM PROMPT (KEY FIX)
    # ----------------------------
    final_system_prompt = f"""
You are a task-oriented AI agent.

User requirement:
{query}

Rules:
- Do NOT behave like a chatbot unless explicitly asked.
- Do NOT add greetings, emojis, or filler.
- If code is requested → output ONLY code.
- If steps are requested → output ONLY steps.
- If explanation is requested → output ONLY explanation.
- Never mention tools, thoughts, or reasoning.
- Output must be clean and final.
"""

    # ----------------------------
    # CREATE AGENT PER REQUEST
    # ----------------------------
    agent = create_react_agent(
        model=llm,
        tools=tools,
        messages_modifier=lambda msgs: [
            SystemMessage(content=final_system_prompt)
        ] + msgs
    )

    # ----------------------------
    # INVOKE
    # ----------------------------
    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # ----------------------------
    # RETURN ONLY FINAL AI MESSAGE
    # ----------------------------
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content

            if isinstance(content, list):
                return "\n".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and "text" in part
                ).strip()

            return str(content).strip()

    return "No response generated."
