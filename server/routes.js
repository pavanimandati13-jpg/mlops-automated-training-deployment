const express = require("express");
const pipeline = require("./pipeline");

const router = express.Router();

let pipelineRunning = false;

// Kick off a new pipeline run (data -> train -> evaluate -> gate -> deploy)
router.post("/pipeline/run", async (req, res) => {
  if (pipelineRunning) {
    return res.status(409).json({ error: "A pipeline run is already in progress." });
  }
  pipelineRunning = true;
  // Fire and forget - client polls /pipeline/status for progress
  pipeline.runPipeline().finally(() => {
    pipelineRunning = false;
  });
  res.status(202).json({ message: "Pipeline run started.", status: pipeline.getStatus() });
});

// Poll current pipeline status/progress
router.get("/pipeline/status", (req, res) => {
  res.json(pipeline.getStatus());
});

// Run history (past training + evaluation results)
router.get("/pipeline/history", (req, res) => {
  res.json(pipeline.getHistory());
});

// Currently deployed model info
router.get("/model", (req, res) => {
  const model = pipeline.getRegisteredModel();
  if (!model) {
    return res.status(404).json({ error: "No model deployed yet." });
  }
  res.json({
    run_id: model.runId,
    trained_at: model.trainedAt,
    metrics: model.metrics,
    feature_names: pipeline.FEATURE_NAMES,
  });
});

// Inference endpoint
router.post("/predict", (req, res) => {
  try {
    const { features } = req.body;
    const result = pipeline.predict(features);
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

router.get("/health", (req, res) => {
  res.json({ status: "ok", uptime_seconds: process.uptime() });
});

module.exports = router;
