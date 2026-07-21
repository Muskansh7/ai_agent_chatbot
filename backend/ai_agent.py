from dotenv import load_dotenv
import os
from response_format import RESPONSE_FORMAT

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
    max_results=5
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
    a clean, structured AI response.
    """

    # --------------------------------
    # PROVIDER SELECTION
    # --------------------------------
    if provider.lower() == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=llm_id,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

    elif provider.lower() == "openai":
        llm = ChatOpenAI(
            model=llm_id,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3
        )

    else:
        raise ValueError("Unsupported provider. Use 'gemini' or 'openai'.")

    # --------------------------------
    # TOOLS
    # --------------------------------
    tools = [search_tool] if allow_search else []

    # --------------------------------
    # CREATE LANGGRAPH REACT AGENT
    # --------------------------------
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # --------------------------------
    # COMBINE ADVISOR PROMPT + RESPONSE FORMAT
    # --------------------------------
    full_prompt = f"""
{system_prompt}

{RESPONSE_FORMAT}

Additional Instructions:

- Maintain the personality of the selected advisor.
- Follow the response format strictly.
- Use Markdown headings.
- Use bullet points whenever possible.
- If a section is not applicable, omit it.
- Never respond as a plain chatbot.
- If web search is enabled and the question requires recent information,
  use the search tool before answering.
"""

    # --------------------------------
    # BUILD MESSAGES
    # --------------------------------
    messages = [
        SystemMessage(content=full_prompt),
        HumanMessage(content=query)
    ]

    # --------------------------------
    # INVOKE AGENT
    # --------------------------------
    result = agent.invoke({"messages": messages})

    # --------------------------------
    # CLEAN AI OUTPUT
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
    response = "\n\n".join(ai_text_parts).strip()

    # Remove markdown wrappers if present
    response = response.replace("```markdown", "")
    response = response.replace("```md", "")
    response = response.strip()

    return response