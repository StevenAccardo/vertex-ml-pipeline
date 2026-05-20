import os
import json
import joblib
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from google.cloud import bigquery, storage
import google.cloud.aiplatform as aiplatform
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  roc_auc_score,
  f1_score,
  precision_score,
  recall_score,
  classification_report,
  confusion_matrix
)
import xgboost as xgb

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

PROJECT_ID         = os.getenv("GCP_PROJECT_ID")
REGION             = os.getenv("GCP_REGION")
BQ_DATASET         = os.getenv("BQ_DATASET")
GCS_BUCKET         = os.getenv("GCS_BUCKET")
VERTEX_EXPERIMENT  = os.getenv("VERTEX_EXPERIMENT")
VERTEX_MODEL_NAME  = os.getenv("VERTEX_MODEL_NAME")

BQ_TABLE      = "raw_equipment_data"
MODEL_DIR     = "/tmp/model"
MODEL_FILE    = "model.joblib"
METADATA_FILE = "metadata.json"

def load_data():
  print("Loading data from BigQuery...")
  client = bigquery.Client(project=PROJECT_ID)
  query  = f"""
    SELECT *
    FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
  """
  df = client.query(query).to_dataframe()
  print(f"Loaded {len(df)} rows")
  return df

def engineer_features(df):
  print("Engineering features...")

  # Derived features from EDA
  df['power_proxy']        = df['torque_nm'] * df['rotational_speed_rpm']
  df['torque_speed_ratio'] = df['torque_nm'] / df['rotational_speed_rpm']
  df['temp_difference']    = df['process_temperature_k'] - df['air_temperature_k']
  df['tool_wear_log']      = np.log1p(df['tool_wear_min'])

  # Encode product type as numeric
  type_map = {"L": 0, "M": 1, "H": 2}
  df['type_encoded'] = df['type'].map(type_map)

  return df

def get_feature_columns():
  return [
    'air_temperature_k',
    'process_temperature_k',
    'rotational_speed_rpm',
    'torque_nm',
    'tool_wear_min',
    'tool_wear_log',
    'type_encoded',
    'power_proxy',
    'torque_speed_ratio',
    'temp_difference',
  ]

def prepare_data(df):
  print("Preparing train/test split...")
  feature_cols = get_feature_columns()
  target_col   = 'machine_failure'

  X = df[feature_cols]
  y = df[target_col]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserve class ratio in both splits
  )

  print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
  print(f"Train failure rate: {y_train.mean()*100:.1f}%")
  print(f"Test failure rate:  {y_test.mean()*100:.1f}%")

  return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params=None):
  print("Training XGBoost model...")

  scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
  print(f"scale_pos_weight: {scale_pos_weight:.1f}")

  default_params = {
    'n_estimators':      300,
    'max_depth':         6,
    'learning_rate':     0.1,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'scale_pos_weight':  scale_pos_weight,
    'eval_metric':       'auc',
    'random_state':      42,
    'n_jobs':            -1,
  }

  if params:
    default_params.update(params)

  model = xgb.XGBClassifier(**default_params)
  model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    verbose=50
  )

  return model, default_params

def evaluate_model(model, X_test, y_test):
  print("\nEvaluating model...")

  y_pred_proba = model.predict_proba(X_test)[:, 1]
  y_pred       = model.predict(X_test)

  metrics = {
    'auc_roc':   round(float(roc_auc_score(y_test, y_pred_proba)), 4),
    'f1_score':  round(float(f1_score(y_test, y_pred)), 4),
    'precision': round(float(precision_score(y_test, y_pred)), 4),
    'recall':    round(float(recall_score(y_test, y_pred)), 4),
  }

  print(f"\nMetrics:")
  for k, v in metrics.items():
    print(f"  {k}: {v}")

  print(f"\nClassification Report:")
  print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))

  print(f"\nConfusion Matrix:")
  cm = confusion_matrix(y_test, y_pred)
  print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
  print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

  return metrics

def save_model(model, metrics, params):
  print("\nSaving model...")
  os.makedirs(MODEL_DIR, exist_ok=True)

  # Save model
  model_path = os.path.join(MODEL_DIR, MODEL_FILE)
  joblib.dump(model, model_path)
  print(f"Model saved to {model_path}")

  # Save metadata
  metadata = {
    'model_name':    VERTEX_MODEL_NAME,
    'trained_at':    datetime.utcnow().isoformat(),
    'features':      get_feature_columns(),
    'metrics':       metrics,
    'params':        {k: v for k, v in params.items() if k != 'scale_pos_weight'},
    'framework':     'xgboost',
    'framework_version': xgb.__version__,
  }

  metadata_path = os.path.join(MODEL_DIR, METADATA_FILE)
  with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
  print(f"Metadata saved to {metadata_path}")

  return model_path, metadata_path

def upload_to_gcs(model_path, metadata_path):
  print("\nUploading to GCS...")
  client    = storage.Client(project=PROJECT_ID)
  bucket    = client.bucket(GCS_BUCKET)
  timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
  gcs_prefix = f"models/{VERTEX_MODEL_NAME}/{timestamp}"

  for local_path in [model_path, metadata_path]:
    filename = os.path.basename(local_path)
    gcs_path = f"{gcs_prefix}/{filename}"
    bucket.blob(gcs_path).upload_from_filename(local_path)
    print(f"Uploaded gs://{GCS_BUCKET}/{gcs_path}")

  return f"gs://{GCS_BUCKET}/{gcs_prefix}"

def log_experiment(metrics, params, gcs_model_uri):
  print("\nLogging to Vertex AI Experiments...")

  aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    experiment=VERTEX_EXPERIMENT
  )

  run_name = f"xgboost-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

  with aiplatform.start_run(run_name) as run:
    run.log_params({k: str(v) for k, v in params.items()})
    run.log_metrics(metrics)

  print(f"Experiment run logged: {run_name}")
  return run_name

def register_model(gcs_model_uri, metrics):
  print("\nRegistering model in Vertex AI Model Registry...")

  aiplatform.init(project=PROJECT_ID, location=REGION)

  model = aiplatform.Model.upload(
    display_name=VERTEX_MODEL_NAME,
    artifact_uri=gcs_model_uri,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest",
    description=f"XGBoost equipment failure classifier. AUC-ROC: {metrics['auc_roc']}",
  )

  print(f"Model registered: {model.resource_name}")
  return model

def main():
  print("=" * 50)
  print("VERTEX AI TRAINING PIPELINE")
  print("=" * 50)

  df                              = load_data()
  df                              = engineer_features(df)
  X_train, X_test, y_train, y_test = prepare_data(df)
  model, params                   = train_model(X_train, y_train)
  metrics                         = evaluate_model(model, X_test, y_test)
  model_path, metadata_path       = save_model(model, metrics, params)
  gcs_model_uri                   = upload_to_gcs(model_path, metadata_path)
  log_experiment(metrics, params, gcs_model_uri)
  register_model(gcs_model_uri, metrics)

  print("\n" + "=" * 50)
  print("TRAINING COMPLETE")
  print(f"Model URI: {gcs_model_uri}")
  print("=" * 50)

if __name__ == "__main__":
  main()
