#!/bin/bash

source .env

echo "WARNING: This will delete all resources in $GCP_PROJECT_ID"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Teardown cancelled."
  exit 0
fi

echo "Undeploying models from endpoints..."
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region="$GCP_REGION" \
  --project="$GCP_PROJECT_ID" \
  --filter="displayName=$ENDPOINT_NAME" \
  --format="value(name)" | head -1)

if [ -n "$ENDPOINT_ID" ]; then
  echo "Deleting endpoint: $ENDPOINT_ID"
  gcloud ai endpoints delete "$ENDPOINT_ID" \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --quiet
fi

echo "Deleting Cloud Run service..."
gcloud run services delete "$ENDPOINT_NAME" \
  --region="$GCP_REGION" \
  --project="$GCP_PROJECT_ID" \
  --quiet 2>/dev/null || echo "No Cloud Run service found."

echo "Deleting Artifact Registry repository..."
gcloud artifacts repositories delete vertex-ml-pipeline \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT_ID" \
  --quiet 2>/dev/null || echo "No Artifact Registry repository found."

echo "Deleting BigQuery dataset..."
bq rm -r -f "$GCP_PROJECT_ID:$BQ_DATASET"

echo "Deleting GCS bucket..."
gcloud storage rm --recursive "gs://$GCS_BUCKET"

echo "Teardown complete."
echo "To fully delete all resources run:"
echo "gcloud projects delete $GCP_PROJECT_ID"
