# Architecture

## Overview

An end-to-end ML pipeline for predictive maintenance built on Google Cloud
Platform. Trains an XGBoost classifier to predict equipment failure from
sensor readings. Covers the full ML lifecycle — data ingestion, exploratory
analysis, feature engineering, training, experiment tracking, model registry,
and deployment.

## Data Flow
1. **Download** — `data/download_dataset.py` downloads UCI dataset, uploads to GCS, loads into BigQuery
2. **Explore** — `notebooks/01_exploratory_analysis.ipynb` performs EDA, validates feature engineering ideas
3. **Train** — `training/train.py` engineers features, trains XGBoost, evaluates model
4. **Store** — model artifacts saved to GCS, metrics logged to Vertex AI Experiments, model registered in Vertex AI Model Registry
5. **Deploy** — `serving/deploy.py` deploys registered model to Vertex AI Endpoint, serves predictions via REST API

## Services Used

- **BigQuery** — stores raw training data and is queried directly by the
  training script. No intermediate files needed.

- **Cloud Storage** — stores versioned model artifacts under
  `models/equipment-failure-classifier/TIMESTAMP/`. Every training run
  creates a new timestamped directory.

- **Vertex AI Experiments** — tracks every training run with its
  hyperparameters and evaluation metrics. Enables comparison across
  multiple runs to find the best model.

- **Vertex AI Model Registry** — stores registered model versions with
  lineage, metrics, and deployment history. The deploy script always
  picks the latest registered version automatically.

- **Vertex AI Endpoints** — serves online predictions via REST API.
  Auto-scales between 1-2 replicas based on traffic. Uses Google's
  pre-built XGBoost serving container — no custom serving code needed.

- **Vertex AI Workbench** — managed JupyterLab environment for EDA.
  Auto-authenticated to GCP, direct BigQuery access, idle shutdown
  after 60 minutes.

- **Artifact Registry** — stores Docker images for serving containers.

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset

| Property | Value |
|---|---|
| Rows | 10,000 |
| Features | 14 raw + 4 engineered |
| Target | machine_failure (binary) |
| Failure rate | 3.4% (339 failures) |
| Failure types | Tool wear, heat dissipation, power, overstrain, random |

## Feature Engineering

Four derived features identified during EDA and added at training time:

| Feature | Formula | Rationale |
|---|---|---|
| power_proxy | torque × rotational_speed | Captures physics relationship |
| torque_speed_ratio | torque / rotational_speed | Best single predictor (corr=0.206) |
| temp_difference | process_temp - air_temp | Negative correlation with failure |
| tool_wear_log | log(1 + tool_wear_min) | Compresses right-skewed distribution |

Product type (L/M/H) encoded as 0/1/2 for XGBoost compatibility.

## Model

XGBoost binary classifier with class imbalance handling:

| Hyperparameter | Value | Rationale |
|---|---|---|
| n_estimators | 300 | Sufficient trees for convergence |
| max_depth | 6 | Captures feature interactions |
| learning_rate | 0.1 | Conservative, pairs with 300 trees |
| subsample | 0.8 | Reduces overfitting |
| colsample_bytree | 0.8 | Reduces overfitting |
| scale_pos_weight | 28.5 | Handles 97/3 class imbalance |

## Model Performance

Evaluated on 20% held-out test set (2,000 rows, stratified split):

| Metric | Value |
|---|---|
| AUC-ROC | 0.9805 |
| F1 Score | 0.8235 |
| Precision | 0.8235 |
| Recall | 0.8235 |

Confusion matrix:
TN=1920  FP=12
FN=12    TP=56

Recall is the most important metric for predictive maintenance — a missed
failure (FN) is more costly than a false alarm (FP).

## Key Findings from EDA

- **Class imbalance** — 96.6% non-failure, 3.4% failure. Handled with
  scale_pos_weight in XGBoost.
- **Torque-speed relationship** — strong inverse correlation (-0.88).
  Failures cluster at high torque + low RPM (binding) and low torque +
  high RPM (component failure).
- **Tool wear** — left-skewed distribution for failures. Most failures
  occur at high tool wear but a tail of early failures exists (random
  failure mode).
- **Low linear correlation** — no single feature correlates strongly with
  failure (max 0.19). XGBoost captures the non-linear interactions that
  matter.

## Deployment

The model is served via a Vertex AI Endpoint using Google's pre-built
XGBoost serving container. Input must be a list of feature values in
the exact order defined in get_feature_columns():

```python
[
  air_temperature_k,
  process_temperature_k,
  rotational_speed_rpm,
  torque_nm,
  tool_wear_min,
  tool_wear_log,
  type_encoded,
  power_proxy,
  torque_speed_ratio,
  temp_difference
]
```

Returns a probability between 0.0 and 1.0. Values above 0.5 indicate
predicted failure.

## Cost

| Phase | Service | Estimated Cost |
|---|---|---|
| Data download | BigQuery + GCS | ~$0.00 |
| EDA | Workbench (e2-standard-2) | ~$0.10/hour |
| Training | Local Mac | $0.00 |
| Deployment | Vertex AI Endpoint | ~$0.10-0.30/hour |
| Storage | GCS + BigQuery | ~$0.01/month |

Always undeploy the endpoint when done testing. Run teardown.sh to
delete all resources when the project is complete.
