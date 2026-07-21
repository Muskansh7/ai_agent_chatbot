import React from "react";

const Footer = () => {
  return (
    <footer className="footer">

      <p>
        © {new Date().getFullYear()} AI Placement Advisor
      </p>

      <p>
        Built for smarter placement preparation.
      </p>

    </footer>
  );
};

export default Footer;