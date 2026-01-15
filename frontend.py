# -------------------------------
# AI Agent Chatbot UI (Streamlit)
# -------------------------------

import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:9999/chat"

st.set_page_config(
    page_title="AI Agent Chatbot",
    layout="centered"
)

# -------------------------------
# SIDEBAR CONFIG
# -------------------------------
with st.sidebar:
    st.title("🤖 Agent Settings")

    system_prompt = st.text_area(
        "Agent Behavior",
        value="Act as an expert AI chatbot architect and assistant",
        height=100
    )

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

    provider = st.radio(
        "Provider",
        options=list(MODEL_CATALOG.keys())
    )

    model_name = st.selectbox(
        "Model",
        MODEL_CATALOG[provider]
    )

    allow_search = st.checkbox("Enable Web Search", value=True)

# -------------------------------
# MAIN UI
# -------------------------------
st.title("🧠 AI Agent Chatbot")
st.caption("Design and build intelligent AI chatbots")

user_query = st.text_area(
    "Describe your chatbot requirement",
    placeholder="Example: Design a chatbot for a college website",
    height=120
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🚀 Ask Agent", use_container_width=True):

    if not user_query.strip():
        st.warning("Please enter a message")
    else:
        st.session_state.chat_history.append(("user", user_query))

        payload = {
            "model_name": model_name,
            "model_provider": provider.lower(),
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_search,
        }

        with st.spinner("Designing chatbot architecture..."):
            try:
                res = requests.post(BACKEND_URL, json=payload, timeout=120)

                if res.status_code == 200:
                    data = res.json()
                    st.session_state.chat_history.append(
                        ("agent", data["response"])
                    )
                else:
                    st.error(res.json())

            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running. Start FastAPI.")
            except Exception as e:
                st.error(str(e))

# -------------------------------
# CHAT DISPLAY (INLINE)
# -------------------------------
st.divider()
st.subheader("💬 Conversation")

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"""
            <div style="background:#f1f3f6;padding:12px;border-radius:8px;margin-bottom:8px">
            <b>You</b><br>{msg}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background:#e8f5e9;padding:14px;border-radius:8px;margin-bottom:12px">
            <b>AI Architect</b><br>{msg}
            </div>
            """,
            unsafe_allow_html=True
        )
