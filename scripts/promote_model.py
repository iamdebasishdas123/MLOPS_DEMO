# promote model

import os
import mlflow
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "iamdebasishdas123"
    repo_name = "MLOPS_DEMO"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "model"
    
    try:
        # Get the latest version in staging
        latest_staging = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_staging:
            logger.error("No model found in Staging stage")
            return
        
        latest_version_staging = latest_staging[0].version
        logger.info(f"Found model version {latest_version_staging} in Staging")

        # Archive the current production model(s)
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                logger.info(f"Archived model version {version.version}")
        else:
            logger.info("No model found in Production stage")

        # Promote the new model to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )
        logger.info(f"Model version {latest_version_staging} promoted to Production")
        print(f"✅ Model version {latest_version_staging} successfully promoted to Production")
        
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    promote_model()