# register model

import json
import mlflow
import mlflow.sklearn
import logging
import os
import pickle
import dagshub

# Set up DagsHub credentials for MLflow tracking
dagshub_token = "159ec2545ad2e371c4774da563d6ef820e7c0ef9"
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "iamdebasishdas123"
repo_name = "MLOPS_DEMO"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Load the model directly from the local pickle file
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.debug('Model loaded from models/model.pkl')
        
        # Start a new MLflow run to log and register the model
        with mlflow.start_run():
            # Log the model in this run
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            logger.debug(f'Model logged to MLflow run {run_id}')
            
            # Register the model from this run
            registered_model = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            logger.debug(f'Model {model_name} registered with version {registered_model.version}')
            
            # Transition the model to "Staging" stage
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Staging"
            )
            
            logger.debug(f'Model {model_name} version {registered_model.version} transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "local_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()