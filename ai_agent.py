from dotenv import load_dotenv
import os

# --------------------------------
# LOAD ENV VARIABLES
# --------------------------------
load_dotenv()

# --------------------------------
# LANGGRAPH & LANGCHAIN IMPORTS
# --------------------------------
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# --------------------------------
# API KEYS
# --------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --------------------------------
# SEARCH TOOL (TAVILY)
# --------------------------------
search_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=2
)

# --------------------------------
# MAIN AGENT FUNCTION
# --------------------------------
def get_response_from_ai_agent(
    llm_id: str,
    query: str,
    allow_search: bool,
    system_prompt: str,
    provider: str
) -> str:
    """
    Executes LangGraph ReAct agent and returns
    CLEAN, HUMAN-READABLE AI output.
    """

    # --------------------------------
    # PROVIDER SELECTION
    # --------------------------------
    if provider.lower() == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=llm_id,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

    elif provider.lower() == "openai":
        llm = ChatOpenAI(
            model=llm_id,
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )

    else:
        raise ValueError("Unsupported provider. Use 'gemini' or 'openai'.")

    # --------------------------------
    # TOOLS (SEARCH ENABLE / DISABLE)
    # --------------------------------
    tools = [search_tool] if allow_search else []

    # --------------------------------
    # CREATE LANGGRAPH AGENT
    # ❌ NO state_modifier
    # --------------------------------
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # --------------------------------
    # BUILD MESSAGES (CORRECT FORMAT)
    # --------------------------------
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    # --------------------------------
    # INVOKE AGENT
    # --------------------------------
    result = agent.invoke({"messages": messages})

    # --------------------------------
    # CLEAN AI OUTPUT (VERY IMPORTANT)
    # --------------------------------
    ai_text_parts = []

    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            content = msg.content

            # Gemini sometimes returns list[dict]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        ai_text_parts.append(part["text"])
            else:
                ai_text_parts.append(str(content))

    # --------------------------------
    # FINAL CLEAN RESPONSE
    # --------------------------------
    return "\n\n".join(ai_text_parts).strip()
