#!/bin/bash

source .env

echo "Setting up GCP project: $GCP_PROJECT_ID"

echo "Enabling APIs..."
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable notebooks.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable dataproc.googleapis.com

echo "Creating BigQuery dataset..."
bq mk --dataset --location="$BQ_LOCATION" "$GCP_PROJECT_ID:$BQ_DATASET"

echo "Creating GCS bucket..."
gcloud storage buckets create "gs://$GCS_BUCKET" --location="$GCP_REGION"

echo "Creating GCS folders..."
gcloud storage cp /dev/null "gs://$GCS_BUCKET/data/.keep"
gcloud storage cp /dev/null "gs://$GCS_BUCKET/models/.keep"
gcloud storage cp /dev/null "gs://$GCS_BUCKET/pipelines/.keep"

echo "Creating Artifact Registry repository for Docker images..."
gcloud artifacts repositories create vertex-ml-pipeline \
  --repository-format=docker \
  --location="$GCP_REGION" \
  --description="Docker images for vertex ml pipeline"

echo "Setup complete."
echo ""
echo "Next steps:"
echo "1. Run data/download_dataset.py to load data into BigQuery"
echo "2. Create a Vertex AI Workbench notebook for EDA"
echo "3. Run the training pipeline"
