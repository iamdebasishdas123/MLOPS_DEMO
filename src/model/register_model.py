# register model

import json
import mlflow
import mlflow.sklearn
import logging
import os
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


def get_valid_model_path(run_id: str, expected_path: str) -> str:
    """
    Verify the artifact path exists under the given run.
    If not, list all artifacts and attempt to find a valid model path.
    """
    client = mlflow.tracking.MlflowClient()

    def list_artifacts_recursive(run_id, path=""):
        """Recursively list all artifacts under a run."""
        artifacts = client.list_artifacts(run_id, path)
        paths = []
        for artifact in artifacts:
            if artifact.is_dir:
                paths.extend(list_artifacts_recursive(run_id, artifact.path))
            else:
                paths.append(artifact.path)
        return paths

    # Check if the expected path exists directly
    artifacts_at_path = client.list_artifacts(run_id, expected_path)
    if artifacts_at_path:
        logger.debug("Artifact path '%s' found under run %s", expected_path, run_id)
        return expected_path

    # Expected path not found — list everything and search for MLmodel file
    logger.warning(
        "Artifact path '%s' not found under run %s. Scanning all artifacts...",
        expected_path, run_id
    )
    all_artifacts = list_artifacts_recursive(run_id)
    logger.debug("All artifacts found: %s", all_artifacts)

    # MLflow models always contain an 'MLmodel' file — find its parent directory
    mlmodel_files = [p for p in all_artifacts if p.endswith("MLmodel")]
    if mlmodel_files:
        # Use the parent directory of the first MLmodel file found
        valid_path = mlmodel_files[0].replace("/MLmodel", "").replace("\\MLmodel", "")
        logger.debug("Found valid model path via MLmodel file: '%s'", valid_path)
        return valid_path

    raise FileNotFoundError(
        f"No valid model artifact found under run {run_id}. "
        f"Available artifacts: {all_artifacts}"
    )


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    client = mlflow.tracking.MlflowClient()

    # Load the model from the local pickle file
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.debug('Model loaded from models/model.pkl')

    # Set the correct experiment (avoid defaulting to experiment 0)
    mlflow.set_experiment("my_experiment")  # 🔁 replace with your actual experiment name

    # Log AND register inside the same run context
    with mlflow.start_run(run_name="model_registration") as run:
        model_info_mlflow = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",        
            registered_model_name=model_name  # ✅ register directly here
        )
        new_run_id = run.info.run_id
        logger.debug('Model logged and registered in run %s', new_run_id)

    # Transition to Staging
    # Get the latest version just registered
    versions = client.get_latest_versions(model_name, stages=["None"])
    latest_version = versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging"
    )
    logger.debug(
        'Model %s version %s transitioned to Staging.',
        model_name, latest_version
    )


def main():
    model_info_path = 'reports/experiment_info.json'
    model_info = load_model_info(model_info_path)

    model_name = "model"
    register_model(model_name, model_info)


if __name__ == '__main__':
    main()