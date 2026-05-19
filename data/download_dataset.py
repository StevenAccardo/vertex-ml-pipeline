import os
import sys
import urllib.request
import pandas as pd
from google.cloud import storage, bigquery
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
REGION      = os.getenv("GCP_REGION")
BQ_DATASET  = os.getenv("BQ_DATASET")
GCS_BUCKET  = os.getenv("GCS_BUCKET")
BQ_LOCATION = os.getenv("BQ_LOCATION")

DATASET_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
LOCAL_PATH   = "/tmp/ai4i2020.csv"
GCS_PATH     = "data/ai4i2020.csv"
BQ_TABLE     = "raw_equipment_data"
SCHEMA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'schemas', 'equipment_features.json')

COLUMN_RENAME = {
  "UDI":                    "udi",
  "Product ID":             "product_id",
  "Type":                   "type",
  "Air temperature [K]":    "air_temperature_k",
  "Process temperature [K]":"process_temperature_k",
  "Rotational speed [rpm]": "rotational_speed_rpm",
  "Torque [Nm]":            "torque_nm",
  "Tool wear [min]":        "tool_wear_min",
  "Machine failure":        "machine_failure",
  "TWF":                    "twf",
  "HDF":                    "hdf",
  "PWF":                    "pwf",
  "OSF":                    "osf",
  "RNF":                    "rnf",
}

def download_dataset():
  print(f"Downloading dataset from UCI...")
  urllib.request.urlretrieve(DATASET_URL, LOCAL_PATH)
  print(f"Downloaded to {LOCAL_PATH}")

def upload_to_gcs():
  print(f"Uploading to GCS: gs://{GCS_BUCKET}/{GCS_PATH}")
  client = storage.Client(project=PROJECT_ID)
  bucket = client.bucket(GCS_BUCKET)
  blob   = bucket.blob(GCS_PATH)
  blob.upload_from_filename(LOCAL_PATH)
  print(f"Uploaded to GCS")

def load_to_bigquery():
  print(f"Loading into BigQuery: {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}")

  df = pd.read_csv(LOCAL_PATH)
  df = df.rename(columns=COLUMN_RENAME)

  print(f"Dataset shape: {df.shape}")
  print(f"Columns: {list(df.columns)}")
  print(f"Sample:\n{df.head(3)}")

  client    = bigquery.Client(project=PROJECT_ID)
  table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

  job_config = bigquery.LoadJobConfig(
    schema=client.schema_from_json(SCHEMA_PATH),
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
  )

  job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
  job.result()

  table = client.get_table(table_ref)
  print(f"Loaded {table.num_rows} rows into {table_ref}")

def main():
  download_dataset()
  upload_to_gcs()
  load_to_bigquery()
  print("\nDataset ready. Next step: open Vertex AI Workbench for EDA.")

if __name__ == "__main__":
  main()
