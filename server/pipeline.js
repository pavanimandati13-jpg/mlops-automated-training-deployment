/**
 * pipeline.js
 * A self-contained (dependency-free) ML pipeline engine:
 *   generate data -> train (gradient descent linear regression)
 *   -> evaluate -> quality gate -> "deploy" (register in-memory model)
 *
 * This mirrors the stages of a real MLOps pipeline (data prep, training,
 * evaluation, model registry, serving) without requiring a Python ML
 * runtime, so the whole stack runs on Node.js alone.
 */

const FEATURE_NAMES = ["sqft", "bedrooms", "age_years"];
const MIN_R2_THRESHOLD = 0.6;

let runHistory = [];      // log of every pipeline run
let registeredModel = null; // the currently "deployed" model
let currentStatus = {
  stage: "idle",          // idle | data | train | evaluate | gate | deploy | done | failed
  progress: 0,
  message: "Waiting to start.",
  runId: null,
};

function randomNormal(mean = 0, std = 1) {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

/** Step 1: Data generation / prep (stand-in for loading + cleaning a real dataset). */
function generateDataset(nSamples = 300) {
  const trueWeights = [180, 12000, -900]; // price ~ sqft*180 + bedrooms*12000 - age*900
  const trueBias = 15000;

  const X = [];
  const y = [];

  for (let i = 0; i < nSamples; i++) {
    const sqft = Math.max(400, randomNormal(1500, 450));
    const bedrooms = Math.max(1, Math.round(randomNormal(3, 1)));
    const age = Math.max(0, randomNormal(15, 10));
    const noise = randomNormal(0, 15000);

    const price =
      trueBias +
      sqft * trueWeights[0] +
      bedrooms * trueWeights[1] +
      age * trueWeights[2] +
      noise;

    X.push([sqft, bedrooms, age]);
    y.push(price);
  }
  return { X, y };
}

function trainTestSplit(X, y, testRatio = 0.2, seed = 42) {
  const n = X.length;
  const indices = [...Array(n).keys()];
  // simple deterministic shuffle
  let s = seed;
  for (let i = n - 1; i > 0; i--) {
    s = (s * 9301 + 49297) % 233280;
    const j = Math.floor((s / 233280) * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const testSize = Math.floor(n * testRatio);
  const testIdx = indices.slice(0, testSize);
  const trainIdx = indices.slice(testSize);

  return {
    XTrain: trainIdx.map((i) => X[i]),
    yTrain: trainIdx.map((i) => y[i]),
    XTest: testIdx.map((i) => X[i]),
    yTest: testIdx.map((i) => y[i]),
  };
}

function standardize(X) {
  const nFeatures = X[0].length;
  const means = new Array(nFeatures).fill(0);
  const stds = new Array(nFeatures).fill(0);

  for (let j = 0; j < nFeatures; j++) {
    means[j] = X.reduce((sum, row) => sum + row[j], 0) / X.length;
  }
  for (let j = 0; j < nFeatures; j++) {
    const variance =
      X.reduce((sum, row) => sum + (row[j] - means[j]) ** 2, 0) / X.length;
    stds[j] = Math.sqrt(variance) || 1;
  }

  const Xscaled = X.map((row) =>
    row.map((val, j) => (val - means[j]) / stds[j])
  );

  return { Xscaled, means, stds };
}

/** Step 2: Training - gradient descent linear regression. */
function trainModel(XTrain, yTrain, { epochs = 300, lr = 0.1 } = {}) {
  const { Xscaled, means, stds } = standardize(XTrain);
  const nFeatures = Xscaled[0].length;
  let weights = new Array(nFeatures).fill(0);
  let bias = 0;
  const n = Xscaled.length;
  const lossHistory = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW = new Array(nFeatures).fill(0);
    let gradB = 0;
    let loss = 0;

    for (let i = 0; i < n; i++) {
      const pred =
        Xscaled[i].reduce((sum, x, j) => sum + x * weights[j], 0) + bias;
      const error = pred - yTrain[i];
      loss += error * error;
      for (let j = 0; j < nFeatures; j++) {
        gradW[j] += error * Xscaled[i][j];
      }
      gradB += error;
    }

    for (let j = 0; j < nFeatures; j++) {
      weights[j] -= (lr * gradW[j]) / n;
    }
    bias -= (lr * gradB) / n;

    if (epoch % 20 === 0 || epoch === epochs - 1) {
      lossHistory.push({ epoch, loss: loss / n });
    }
  }

  return { weights, bias, means, stds, lossHistory };
}

function predictWithModel(model, X) {
  const { weights, bias, means, stds } = model;
  return X.map((row) => {
    const scaled = row.map((val, j) => (val - means[j]) / stds[j]);
    return scaled.reduce((sum, x, j) => sum + x * weights[j], bias);
  });
}

/** Step 3: Evaluation - R^2, MAE, RMSE. */
function evaluateModel(model, XTest, yTest) {
  const preds = predictWithModel(model, XTest);
  const n = yTest.length;
  const meanY = yTest.reduce((a, b) => a + b, 0) / n;

  let ssRes = 0;
  let ssTot = 0;
  let mae = 0;

  for (let i = 0; i < n; i++) {
    ssRes += (yTest[i] - preds[i]) ** 2;
    ssTot += (yTest[i] - meanY) ** 2;
    mae += Math.abs(yTest[i] - preds[i]);
  }

  const r2 = 1 - ssRes / ssTot;
  const rmse = Math.sqrt(ssRes / n);
  mae = mae / n;

  return { r2, rmse, mae };
}

/** Step 4: Quality gate. */
function gateModel(metrics, threshold = MIN_R2_THRESHOLD) {
  return metrics.r2 >= threshold;
}

/** Orchestrates the full run, updating currentStatus as it goes (for polling/SSE). */
async function runPipeline() {
  const runId = `run_${Date.now()}`;
  const startedAt = new Date().toISOString();

  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

  try {
    currentStatus = { stage: "data", progress: 10, message: "Generating & preparing dataset...", runId };
    await sleep(500);
    const { X, y } = generateDataset(300);
    const { XTrain, yTrain, XTest, yTest } = trainTestSplit(X, y);

    currentStatus = { stage: "train", progress: 40, message: "Training model (gradient descent)...", runId };
    await sleep(700);
    const model = trainModel(XTrain, yTrain);

    currentStatus = { stage: "evaluate", progress: 70, message: "Evaluating on held-out test set...", runId };
    await sleep(500);
    const metrics = evaluateModel(model, XTest, yTest);

    currentStatus = { stage: "gate", progress: 85, message: "Running quality gate check...", runId };
    await sleep(400);
    const passed = gateModel(metrics);

    if (!passed) {
      currentStatus = {
        stage: "failed",
        progress: 100,
        message: `Quality gate FAILED (R2=${metrics.r2.toFixed(3)} < ${MIN_R2_THRESHOLD}).`,
        runId,
      };
      runHistory.unshift({ runId, startedAt, metrics, passed, deployed: false, featureNames: FEATURE_NAMES });
      return currentStatus;
    }

    currentStatus = { stage: "deploy", progress: 95, message: "Registering & deploying model...", runId };
    await sleep(400);
    registeredModel = { ...model, runId, trainedAt: new Date().toISOString(), metrics };

    currentStatus = {
      stage: "done",
      progress: 100,
      message: `Model deployed successfully (R2=${metrics.r2.toFixed(3)}).`,
      runId,
    };

    runHistory.unshift({ runId, startedAt, metrics, passed, deployed: true, featureNames: FEATURE_NAMES });
    if (runHistory.length > 20) runHistory = runHistory.slice(0, 20);

    return currentStatus;
  } catch (err) {
    currentStatus = { stage: "failed", progress: 100, message: `Pipeline error: ${err.message}`, runId };
    return currentStatus;
  }
}

function getStatus() {
  return currentStatus;
}

function getHistory() {
  return runHistory;
}

function getRegisteredModel() {
  return registeredModel;
}

function predict(features) {
  if (!registeredModel) {
    throw new Error("No model is currently deployed. Run the pipeline first.");
  }
  if (!Array.isArray(features) || features.length !== FEATURE_NAMES.length) {
    throw new Error(`Expected ${FEATURE_NAMES.length} features: ${FEATURE_NAMES.join(", ")}`);
  }
  const [prediction] = predictWithModel(registeredModel, [features]);
  return {
    prediction: Math.round(prediction),
    model_run_id: registeredModel.runId,
    trained_at: registeredModel.trainedAt,
  };
}

module.exports = {
  FEATURE_NAMES,
  runPipeline,
  getStatus,
  getHistory,
  getRegisteredModel,
  predict,
};
