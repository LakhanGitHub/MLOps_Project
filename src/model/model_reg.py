import json
import mlflow
import dagshub
from mlflow import MlflowClient


# Initialize dagshub and set mlflow for experiment tracking
dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)
mlflow.set_experiment('Final_Model')
mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

#load the run id and model name from saved json file

report_path = 'reports/run_info.json'
with open(report_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id']
model_name = run_info['model_name']

#create mlflow client
client = MlflowClient()

#create the model uri
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

#register model
reg = mlflow.register_model(model_uri,model_name)

#get the model version
model_version = reg.version

#transition the model into staging
new_stage = 'Staging'

client.transition_model_version_stage(
    name = model_name,
    version=model_version,
    stage = new_stage,
    archive_existing_versions=True

)

print(f"model {model_name} versioni {model_version} transitioned to {new_stage} stage.")
