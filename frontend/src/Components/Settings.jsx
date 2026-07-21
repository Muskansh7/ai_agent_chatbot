import React, { useState } from "react";

const Settings = () => {
  const [provider, setProvider] = useState("gemini");
  const [model, setModel] = useState("models/gemini-2.0-flash");
  const [search, setSearch] = useState(true);

  const geminiModels = [
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
  ];

  const openaiModels = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
  ];

  const handleProvider = (e) => {
    const value = e.target.value;

    setProvider(value);

    if (value === "gemini") {
      setModel(geminiModels[0]);
    } else {
      setModel(openaiModels[0]);
    }
  };

  return (
    <div className="settings-page">

      <h1>Settings</h1>

      <p>
        Configure your AI advisor preferences.
      </p>

      <div className="setting-card">

        <label>AI Provider</label>

        <select
          value={provider}
          onChange={handleProvider}
        >
          <option value="gemini">Google Gemini</option>
          <option value="openai">OpenAI</option>
        </select>

      </div>

      <div className="setting-card">

        <label>Model</label>

        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {(provider === "gemini"
            ? geminiModels
            : openaiModels
          ).map((item) => (
            <option key={item}>
              {item}
            </option>
          ))}
        </select>

      </div>

      <div className="setting-card toggle">

        <label>Enable Web Search</label>

        <input
          type="checkbox"
          checked={search}
          onChange={() => setSearch(!search)}
        />

      </div>

      <button className="save-btn">
        Save Settings
      </button>

    </div>
  );
};

export default Settings;