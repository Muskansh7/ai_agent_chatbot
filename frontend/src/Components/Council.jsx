import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const advisors = [
  {
    icon: "💻",
    title: "Coding Mentor",
  },
  {
    icon: "💼",
    title: "Career Coach",
  },
  {
    icon: "🎤",
    title: "Interview Coach",
  },
  {
    icon: "📚",
    title: "Research Assistant",
  },
  {
    icon: "🤝",
    title: "Friend",
  },
  {
    icon: "💰",
    title: "FINANCIAL ADVISOR", 
  },
];

const Council = () => {
  const navigate = useNavigate();

  const [selected, setSelected] = useState([]);

  const toggleAdvisor = (advisor) => {
    if (selected.find((a) => a.title === advisor.title)) {
      setSelected(selected.filter((a) => a.title !== advisor.title));
    } else {
      setSelected([...selected, advisor]);
    }
  };

  const startCouncil = () => {
    if (selected.length === 0) {
      alert("Please select at least one advisor.");
      return;
    }

    navigate("/chat", {
      state: {
        mode: "council",

        advisor: {
          icon: "⚡",
          title: "AI Council",
          description:
            "Multiple AI experts will collaborate to provide a balanced answer.",

          skills: selected.map((a) => a.title),
        },

        advisors: selected,
      },
    });
  };

  return (
    <div className="council-page">

      <div className="council-header">

        <h1>AI Council</h1>

        <p>
          Select multiple advisors and receive a collaborative,
          balanced response.
        </p>

      </div>

      <h3>Select Advisors</h3>

      <div className="council-grid">

        {advisors.map((advisor) => (

          <div
            key={advisor.title}
            className={`council-card ${
              selected.find((a) => a.title === advisor.title)
                ? "selected"
                : ""
            }`}
            onClick={() => toggleAdvisor(advisor)}
          >

            <input
              type="checkbox"
              checked={
                selected.find((a) => a.title === advisor.title)
                  ? true
                  : false
              }
              readOnly
            />

            <span>

              {advisor.icon} {advisor.title}

            </span>

          </div>

        ))}

      </div>

      <button
        className="primary-btn council-btn"
        onClick={startCouncil}
      >
        Start Council
      </button>

    </div>
  );
};

export default Council;