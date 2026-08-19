# MLOps Pipeline Control — Full-Stack Web App

**Cloud Computing Project — Automated Machine Learning Model Training and
Deployment Pipeline (MLOps)**

A full-stack demo of an MLOps pipeline dashboard: trigger a training run,
watch it move through Ingest → Train → Evaluate → Gate → Deploy in real
time, inspect evaluation metrics, query the deployed model, and review run
history — all from the browser.

## Stack

- **Backend:** Node.js + Express (REST API, in-memory pipeline engine)
- **Frontend:** Static HTML, CSS, and vanilla JavaScript (no framework/build
  step required)
- **ML engine:** A dependency-free JavaScript linear regression (gradient
  descent) trained on a generated synthetic housing-price dataset — stands
  in for a real model/dataset so the whole project runs with just `node`,
  no Python or GPU required.

## Folder Structure

```
mlops-webapp/
├── server/
│   ├── index.js       # Express app entry point
│   ├── routes.js       # REST API routes
│   └── pipeline.js     # Data gen, training, evaluation, gate, deploy, predict
├── public/
│   ├── index.html      # Dashboard UI
│   ├── css/style.css   # Design system (control-room theme)
│   └── js/app.js       # Frontend logic (polling, charts, forms)
├── package.json
└── README.md
```

## Run it

```bash
npm install
npm start
```

Then open **http://localhost:3000** in your browser.

- Click **RUN PIPELINE** to kick off a new training run. The stage line at
  the top lights up live as the run progresses through each phase.
- Once a run passes its quality gate (R² ≥ 0.60) the model is "deployed" and
  becomes queryable.
- Use the **Inference request** panel to send a prediction (square footage,
  bedrooms, age → estimated price) to the deployed model.
- The **Run history** table keeps the last 20 runs with their metrics and
  gate result.

## REST API

| Method | Endpoint                 | Description                              |
|--------|---------------------------|-------------------------------------------|
| POST   | `/api/pipeline/run`       | Start a new pipeline run                  |
| GET    | `/api/pipeline/status`    | Poll current run stage/progress           |
| GET    | `/api/pipeline/history`   | Last 20 runs with metrics                 |
| GET    | `/api/model`               | Currently deployed model + metrics        |
| POST   | `/api/predict`             | `{ "features": [sqft, bedrooms, age] }`   |
| GET    | `/api/health`               | Health check                              |

## Deploying to the cloud

This is a stateless Node.js app (state resets on restart, held in memory),
so it can be deployed as-is to any Node-friendly host:

- **Render / Railway / Fly.io** — connect the repo, set start command
  `npm start`.
- **AWS** — Elastic Beanstalk, or containerize with a simple
  `node:20-alpine` Dockerfile and run on ECS/Fargate.
- **Azure** — App Service (Node runtime) or Azure Container Apps.
- **GCP** — Cloud Run (containerize) or App Engine (Node standard
  environment).

For a persistent model registry in production, swap the in-memory
`registeredModel` / `runHistory` in `server/pipeline.js` for a real store
(e.g. S3/Blob/GCS for model artifacts, a database for run history).

## Notes

This app is a self-contained **teaching/demo pipeline** — it generates its
own synthetic data and trains a simple linear regression in pure
JavaScript so the entire project runs with just Node.js installed. For a
production MLOps setup you would typically swap in a real dataset, a
proper ML framework (scikit-learn/PyTorch/TensorFlow) run as a separate
training service, and a model registry — the API and dashboard here are
built so those pieces can be swapped in behind the same endpoints.
