import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


