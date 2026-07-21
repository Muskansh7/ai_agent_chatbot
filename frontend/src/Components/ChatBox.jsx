import { useLocation, useNavigate } from "react-router-dom";
import { useState } from "react";
import api from "../api/api";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const ChatBox = () => {
  const location = useLocation();
  const navigate = useNavigate();

  console.log("Location State:", location.state);

  const councilMode = location.state?.mode === "council";

  const advisor =
    location.state?.advisor || {
      icon: "🤖",
      title: "AI Placement Advisor",
      description: "Your intelligent placement companion.",
      skills: [],
    };

  const selectedAdvisors = location.state?.advisors || [];

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    const body = councilMode
      ? {
          model_provider: "openai",
          model_name: "gpt-4.1-mini",
          mode: "council",
          advisor: "",
          advisors: selectedAdvisors,
          custom_prompt: "",
          messages: [input],
          allow_search: true,
        }
      : {
          model_provider: "openai", 
          model_name: "gpt-4.1-mini",
          mode: "single",
          advisor: advisor.title,
          advisors: [],
          custom_prompt: "",
          messages: [input],
          allow_search: true,
        };

    console.log("Sending Payload:", body);

    try {
      const res = await api.post("/chat", body);

      console.log("Backend Response:", res.data);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.data.response,
        },
      ]);
    } catch (error) {
      console.error("Full Error:", error);

      if (error.response) {
        console.log("Status:", error.response.status);
        console.log("Backend Response:", error.response.data);

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              error.response.data.detail ||
              "Backend returned an error.",
          },
        ]);
      } else if (error.request) {
        console.log("No response received from backend.");

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "❌ Backend didn't respond. Make sure FastAPI is running.",
          },
        ]);
      } else {
        console.log(error.message);

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: error.message,
          },
        ]);
      }
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  return (
    <div className="chat-page">

      <div className="chat-top">
        <button
          className="back-btn"
          onClick={() => navigate("/")}
        >
          ← Back
        </button>
      </div>

      <div className="advisor-banner">
        <div className="advisor-icon">
          {councilMode ? "⚡" : advisor.icon}
        </div>

        <div className="advisor-info">
          <h2>
            {councilMode ? "AI Council" : advisor.title}
          </h2>

          <p>
            {councilMode
              ? "Multiple AI advisors collaborate to provide a balanced answer."
              : advisor.description}
          </p>

          <div className="advisor-skills">
            {councilMode ? (
              selectedAdvisors.map((item, index) => (
                <span key={index}>{item}</span>
              ))
            ) : advisor.skills.length > 0 ? (
              advisor.skills.map((skill, index) => (
                <span key={index}>{skill}</span>
              ))
            ) : (
              <span>General Guidance</span>
            )}
          </div>
        </div>
      </div>

      <div className="welcome-card">
        <h3>Welcome 👋</h3>

        <p>
          You're chatting with{" "}
          <strong>
            {councilMode ? "AI Council" : advisor.title}
          </strong>
        </p>

        <p>
          Ask anything related to coding, interviews, placements,
          resumes or career guidance.
        </p>
      </div>

      <div className="chat-history">
  {messages.map((msg, index) => (
    <div
      key={index}
      className={
        msg.role === "user"
          ? "user-message"
          : "ai-message"
      }
    >
      <strong>
        {msg.role === "user" ? "You" : "AI"}
      </strong>

      <br />

      {msg.role === "assistant" ? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {msg.content}
        </ReactMarkdown>
      ) : (
        msg.content
      )}
    </div>
  ))}

  {loading && (
    <div className="ai-message">
      Thinking...
    </div>
  )}
</div>

      <div className="chat-input">
        <textarea
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            councilMode
              ? "Ask the AI Council anything..."
              : `Ask ${advisor.title} anything...`
          }
        />

        <button
          onClick={handleSend}
          disabled={loading}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </div>

    </div>
  );
};

export default ChatBox;