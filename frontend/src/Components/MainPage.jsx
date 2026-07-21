import React from "react";
import AdvisorCard from "./AdvisorCard";

const advisors = [
  {
    icon: "💻",
    title: "Coding Mentor",
    description:
      "Master DSA, algorithms, debugging and system design interviews.",
    skills: ["DSA", "Algorithms", "Debugging", "System Design"],
  },
  {
    icon: "💼",
    title: "Career Coach",
    description:
      "Improve your resume, LinkedIn profile and placement strategy.",
    skills: ["Resume", "ATS", "LinkedIn", "Career"],
  },
  {
    icon: "🎤",
    title: "Interview Coach",
    description:
      "Practice technical and HR interviews with detailed feedback.",
    skills: ["Technical", "HR", "Mock", "Feedback"],
  },
  {
    icon: "📚",
    title: "Research Assistant",
    description:
      "Learn AI, ML, Web Development and emerging technologies.",
    skills: ["AI", "ML", "Cloud", "Research"],
  },
  {
    icon: "🤝",
    title: "Friend",
    description:
      "Stay motivated, overcome placement stress and stay consistent.",
    skills: ["Support", "Motivation", "Balance"],
  },
  {
    icon: "⚡",
    title: "AI Council",
    description:
      "Combine multiple advisors for well-rounded guidance.",
    skills: ["Multiple Experts", "Balanced Advice"],
  },
];

const MainPage = () => {
  const handleSelect = (advisor) => {
    console.log("Selected:", advisor);
    // Later we'll navigate to Chat Page
  };

  return (
    <main className="main-page">

      <section className="hero">

        <h1>AI Placement Advisor</h1>

        <p>
          Get expert guidance for coding interviews, placements,
          resumes, career growth and interview preparation —
          all from specialized AI advisors.
        </p>

      </section>

      <section
  id="advisors"
  className="advisor-section"
>

        <h2>Choose Your Advisor</h2>

        <div className="advisor-grid">

          {advisors.map((advisor, index) => (
            <AdvisorCard
              key={index}
              icon={advisor.icon}
              title={advisor.title}
              description={advisor.description}
              skills={advisor.skills}
              onClick={() => handleSelect(advisor.title)}
            />
          ))}

        </div>

      </section>

    </main>
  );
};

export default MainPage;