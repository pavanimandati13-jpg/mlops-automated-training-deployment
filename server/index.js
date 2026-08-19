const path = require("path");
const express = require("express");
const apiRoutes = require("./routes");

const app = express();
const PORT = process.env.PORT || 3000;

// Minimal CORS middleware (no external dependency needed since the
// frontend is served from the same origin as the API; kept permissive
// here in case you split the frontend onto a different host later).
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

app.use(express.json());

// API
app.use("/api", apiRoutes);

// Frontend (static)
app.use(express.static(path.join(__dirname, "..", "public")));

app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "..", "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`MLOps pipeline dashboard running at http://localhost:${PORT}`);
});
