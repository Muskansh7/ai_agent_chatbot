import React from "react";
import { NavLink, useNavigate } from "react-router-dom";

const Header = () => {

  const navigate = useNavigate();

  const scrollToAdvisors = () => {

    if (window.location.pathname !== "/") {
      navigate("/");

      setTimeout(() => {
        document
          .getElementById("advisors")
          ?.scrollIntoView({
            behavior: "smooth",
          });
      }, 100);

    } else {

      document
        .getElementById("advisors")
        ?.scrollIntoView({
          behavior: "smooth",
        });

    }

  };

  return (
    <header className="header">

      <div className="logo-section">

        <h2>Placement Advisor</h2>

      </div>

      <nav className="nav-links">

        <NavLink to="/">
          Home
        </NavLink>

        <button
          className="nav-btn"
          onClick={scrollToAdvisors}
        >
          Advisors
        </button>

        <NavLink to="/council">
          AI Council
        </NavLink>

        <NavLink to="/settings">
          Settings
        </NavLink>

      </nav>

    </header>
  );
};

export default Header;