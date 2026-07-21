import "./style.css";

import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";

import Header from "./Components/Header";
import Footer from "./Components/Footer";

import MainPage from "./Components/MainPage";
import ChatBox from "./Components/ChatBox";
import Settings from "./Components/Settings";
import Council from "./Components/Council";

function App() {
  return (
    <BrowserRouter>

      <Header />

      <main className="app-content">

        <Routes>

          {/* Landing Page */}
          <Route
            path="/"
            element={<MainPage />}
          />

          {/* Individual Advisor Chat */}
          <Route
            path="/chat"
            element={<ChatBox />}
          />

          {/* AI Council */}
          <Route
            path="/council"
            element={<Council />}
          />

          {/* Settings */}
          <Route
            path="/settings"
            element={<Settings />}
          />

        </Routes>

      </main>

      <Footer />

    </BrowserRouter>
  );
}

export default App;