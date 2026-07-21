import streamlit as st
from ai_agent import get_response_from_ai_agent

st.set_page_config(page_title="AI Task Agent", layout="wide")

MODEL_CATALOG = {
    "Gemini": [
        "models/gemini-flash-latest",
        "models/gemini-flash-lite-latest",
        "models/gemini-2.0-flash",
    ],
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ],
}

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("⚙️ Agent Settings")

    system_prompt = st.text_area(
        "Agent Instructions (Optional)",
        placeholder="Example: Respond only with code. No explanation.",
        height=120
    )

    provider = st.radio("Model Provider", list(MODEL_CATALOG.keys()))
    model_name = st.selectbox("Model", MODEL_CATALOG[provider])
    allow_search = st.checkbox("Enable Web Search (Tavily)", value=True)

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🧠 AI Task Agent")

user_query = st.text_area(
    "Enter your requirement",
    placeholder="Example: Write a FastAPI JWT auth middleware",
    height=120
)

# -----------------------------
# RUN AGENT
# -----------------------------
if st.button("🚀 Run Agent", use_container_width=True):

    if not user_query.strip():
        st.warning("Please enter a requirement.")
    else:
        st.session_state.chat_history.append(("user", user_query))

        final_prompt = f"""
You are an instruction-following AI agent.

Rules:
- Do NOT assume chatbot behavior.
- Follow the user's requirement exactly.
- No greetings or filler.
- Output must be clean and final.

Additional instructions:
{system_prompt}
"""

        with st.spinner("Processing..."):
            try:
                response = get_response_from_ai_agent(
                    llm_id=model_name,
                    query=user_query,
                    allow_search=allow_search,
                    system_prompt=final_prompt,
                    provider=provider.lower()
                )
                st.session_state.chat_history.append(("agent", response))
            except Exception as e:
                st.error(str(e))

# -----------------------------
# CONVERSATION DISPLAY
# -----------------------------
st.divider()
st.subheader("💬 Conversation")

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
