import mlflow
import dagshub
dagshub.init(repo_owner='iamdebasishdas123', repo_name='MLOPS_DEMO', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)