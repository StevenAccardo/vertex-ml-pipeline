import os
from dotenv import load_dotenv
import google.cloud.aiplatform as aiplatform

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

PROJECT_ID        = os.getenv("GCP_PROJECT_ID")
REGION            = os.getenv("GCP_REGION")
VERTEX_MODEL_NAME = os.getenv("VERTEX_MODEL_NAME")
ENDPOINT_NAME     = os.getenv("ENDPOINT_NAME")

def get_latest_model():
  print("Finding latest registered model...")
  aiplatform.init(project=PROJECT_ID, location=REGION)

  models = aiplatform.Model.list(
    filter=f"display_name={VERTEX_MODEL_NAME}",
    order_by="create_time desc",
  )

  if not models:
    raise ValueError(f"No model found with name: {VERTEX_MODEL_NAME}")

  latest = models[0]
  print(f"Found model: {latest.resource_name}")
  return latest

def create_or_get_endpoint():
  print("Creating endpoint...")
  aiplatform.init(project=PROJECT_ID, location=REGION)

  endpoints = aiplatform.Endpoint.list(
    filter=f"display_name={ENDPOINT_NAME}",
    order_by="create_time desc",
  )

  if endpoints:
    print(f"Using existing endpoint: {endpoints[0].resource_name}")
    return endpoints[0]

  endpoint = aiplatform.Endpoint.create(
    display_name=ENDPOINT_NAME,
    project=PROJECT_ID,
    location=REGION,
  )
  print(f"Created endpoint: {endpoint.resource_name}")
  return endpoint

def deploy_model(model, endpoint):
  print("Deploying model to endpoint...")
  print("This takes 5-10 minutes...")

  endpoint.deploy(
    model=model,
    deployed_model_display_name=VERTEX_MODEL_NAME,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=2,
    traffic_percentage=100,
  )

  print(f"Model deployed successfully")
  print(f"Endpoint resource name: {endpoint.resource_name}")
  return endpoint

def test_prediction(endpoint):
  print("\nTesting prediction...")

  # XGBoost container expects a list of lists, not a list of dicts
  # Order must match get_feature_columns() exactly
  test_instance = [
    298.1,   # air_temperature_k
    308.6,   # process_temperature_k
    1251,    # rotational_speed_rpm
    69.8,    # torque_nm
    205,     # tool_wear_min
    5.326,   # tool_wear_log
    1,       # type_encoded
    87289.8, # power_proxy
    0.0558,  # torque_speed_ratio
    10.5,    # temp_difference
  ]

  prediction = endpoint.predict(instances=[test_instance])
  print(f"Test prediction: {prediction.predictions}")
  print(f"Expected: failure (1)")
  
def main():
  model    = get_latest_model()
  endpoint = create_or_get_endpoint()
  deploy_model(model, endpoint)
  test_prediction(endpoint)
  print("\nDeployment complete.")
  print("Remember to undeploy when done to avoid billing.")

if __name__ == "__main__":
  main()
