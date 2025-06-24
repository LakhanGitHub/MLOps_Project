import json
import mlflow
import dagshub
from mlflow import MlflowClient
import pandas as pd


# Initialize dagshub and set mlflow for experiment tracking
dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

#load the run id and model name from saved json file
model_name = 'Best_Model' #registered model

try:
    client = mlflow.tracking.MlflowClient()

    versions = client.get_latest_versions(model_name, stages=['Staging'])

    if versions:
        latest_version = versions[0].version
        run_id = versions[0].run_id #fatching the run id and latest version
        print(f'latest verion in staging :{latest_version}, run id {run_id}')

        logged_model = f'runs:/{run_id}/Best Model'  # artifact name "Best Model" 4f9f80232/artifacts/Best Model/model.pkl
        print(f'logged model :{logged_model}')

        #load the model using the loggged_model variable
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print(f'model loaded from {logged_model}')

        #input data for prediction
        data  = pd.DataFrame({
            'ph':[3.71608],
            'Hardness':[204.89045],
            'Solids':[20791.31890],
            'Chloramines':[7.3000],
            'Sulfate':[381.5164],
            'Conductivity':[564.3086],
            'Organic_carbon':[10.3700],
            'Trihalomethanes':[86.99],
            'Turbidity':[2.9634]

        })

        prediction = loaded_model.predict(data)
        print(f'prediction {prediction}')
    else:
        print('no model exist in production stage')
except Exception as e:
    print(f'error fetching model :{e}')




