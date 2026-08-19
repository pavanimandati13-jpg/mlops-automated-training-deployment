const STAGES = ["data", "train", "evaluate", "gate", "deploy"];
const STAGE_PROGRESS = { data: 10, train: 40, evaluate: 70, gate: 85, deploy: 95, done: 100, failed: 100 };

const runBtn = document.getElementById("run-btn");
const statusMessage = document.getElementById("status-message");
const conveyorFill = document.getElementById("conveyor-fill");
const stageEls = Array.from(document.querySelectorAll(".stage"));
const consoleLog = document.getElementById("console-log");
const deployedBadge = document.getElementById("deployed-badge");

const metricR2 = document.getElementById("metric-r2");
const metricRmse = document.getElementById("metric-rmse");
const metricMae = document.getElementById("metric-mae");

const predictForm = document.getElementById("predict-form");
const predictResult = document.getElementById("predict-result");

const historyBody = document.getElementById("history-body");

let pollTimer = null;
let lastLoggedStage = null;

function logLine(text, tone = "") {
  const p = document.createElement("p");
  p.className = `console-line ${tone ? "console-line--" + tone : ""}`;
  const time = new Date().toLocaleTimeString();
  p.textContent = `[${time}] ${text}`;
  consoleLog.appendChild(p);
  consoleLog.scrollTop = consoleLog.scrollHeight;
}

function setStageVisual(status) {
  const { stage, progress } = status;
  conveyorFill.style.width = `${STAGE_PROGRESS[stage] ?? progress ?? 0}%`;

  const activeIndex = STAGES.indexOf(stage);

  stageEls.forEach((el, i) => {
    el.classList.remove("is-active", "is-done", "is-failed");
    if (stage === "failed") {
      // mark stages up to where it likely failed based on progress
      if (i <= activeIndex || activeIndex === -1) el.classList.add("is-failed");
      return;
    }
    if (stage === "done") {
      el.classList.add("is-done");
      return;
    }
    if (i < activeIndex) el.classList.add("is-done");
    else if (i === activeIndex) el.classList.add("is-active");
  });
}

function fmtMoney(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return "$" + Math.round(n).toLocaleString();
}

async function fetchStatus() {
  const res = await fetch("/api/pipeline/status");
  return res.json();
}

async function fetchHistory() {
  const res = await fetch("/api/pipeline/history");
  return res.json();
}

async function fetchModel() {
  const res = await fetch("/api/model");
  if (!res.ok) return null;
  return res.json();
}

function renderHistory(rows) {
  if (!rows || rows.length === 0) {
    historyBody.innerHTML = `<tr><td colspan="5" class="history-empty">No runs yet.</td></tr>`;
    return;
  }
  historyBody.innerHTML = rows
    .map((r) => {
      const gateClass = r.passed ? "tag-pass" : "tag-fail";
      const gateText = r.passed ? "PASS" : "FAIL";
      const deployText = r.deployed ? "yes" : "no";
      return `<tr>
        <td>${r.runId.replace("run_", "")}</td>
        <td>${r.metrics.r2.toFixed(3)}</td>
        <td>${fmtMoney(r.metrics.rmse)}</td>
        <td class="${gateClass}">${gateText}</td>
        <td>${deployText}</td>
      </tr>`;
    })
    .join("");
}

async function updateDeployedBadge() {
  const model = await fetchModel();
  if (model) {
    deployedBadge.textContent = model.run_id.replace("run_", "");
    deployedBadge.className = "deployed-badge deployed-badge--live";
    metricR2.textContent = model.metrics.r2.toFixed(3);
    metricRmse.textContent = fmtMoney(model.metrics.rmse);
    metricMae.textContent = fmtMoney(model.metrics.mae);
  }
}

async function poll() {
  const status = await fetchStatus();
  setStageVisual(status);
  statusMessage.textContent = status.message;

  if (status.stage !== lastLoggedStage) {
    lastLoggedStage = status.stage;
    const tone =
      status.stage === "done" ? "green" : status.stage === "failed" ? "brick" : "amber";
    logLine(status.message, tone);
  }

  if (status.stage === "done" || status.stage === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    runBtn.disabled = false;
    const history = await fetchHistory();
    renderHistory(history);
    await updateDeployedBadge();
  }
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  lastLoggedStage = null;
  logLine("Triggering new pipeline run…", "");

  const res = await fetch("/api/pipeline/run", { method: "POST" });
  if (res.status === 409) {
    logLine("A run is already in progress.", "brick");
    runBtn.disabled = false;
    return;
  }

  pollTimer = setInterval(poll, 500);
  poll();
});

predictForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const sqft = parseFloat(document.getElementById("in-sqft").value);
  const bedrooms = parseFloat(document.getElementById("in-bedrooms").value);
  const age = parseFloat(document.getElementById("in-age").value);

  predictResult.classList.remove("is-error");
  predictResult.textContent = "Requesting prediction…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: [sqft, bedrooms, age] }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed.");

    predictResult.innerHTML = `Estimated price: <strong>${fmtMoney(data.prediction)}</strong> &nbsp;<span style="color:var(--text-faint)">(model ${data.model_run_id.replace("run_","")})</span>`;
  } catch (err) {
    predictResult.classList.add("is-error");
    predictResult.textContent = err.message;
  }
});

// Initial load
(async function init() {
  const status = await fetchStatus();
  setStageVisual(status);
  statusMessage.textContent = status.message;

  const history = await fetchHistory();
  renderHistory(history);

  await updateDeployedBadge();
})();
