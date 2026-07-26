// src/main.jsx
//
// This is the entry point — the very first file that runs.
// It takes the App component and puts it into the HTML page.
// You should not need to edit this file.

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
