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

    provider = provider.lower()

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=llm_id,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            model=llm_id,
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )
    else:
        raise ValueError("Invalid provider")

    tools = []
    if allow_search and TAVILY_API_KEY:
        tools.append(TavilySearch(api_key=TAVILY_API_KEY, max_results=3))

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=query.strip())
    ]

    result = agent.invoke({"messages": messages})

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, list):
                return "\n".join(
                    p["text"] for p in msg.content
                    if isinstance(p, dict) and "text" in p
                ).strip()
            return str(msg.content).strip()

    return "No response generated."
