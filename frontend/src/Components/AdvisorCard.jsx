import React from "react";
import { useNavigate } from "react-router-dom";

const AdvisorCard = ({
  icon,
  title,
  description,
  skills = [],
}) => {

  const navigate = useNavigate();

  const handleClick = () => {
    navigate("/chat", {
      state: {
        advisor: {
          icon,
          title,
          description,
          skills,
        },
      },
    });
  };

  return (
    <div
      className="advisor-card"
      onClick={handleClick}
    >
      <div className="advisor-icon">
        {icon}
      </div>

      <h3>{title}</h3>

      <p>{description}</p>

      <div className="advisor-skills">
        {skills.map((skill, index) => (
          <span key={index}>{skill}</span>
        ))}
      </div>

      <button
        className="advisor-btn"
        onClick={(e) => {
          e.stopPropagation();
          handleClick();
        }}
      >
        Start Chat →
      </button>
    </div>
  );
};

export default AdvisorCard;