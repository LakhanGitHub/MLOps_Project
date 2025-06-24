import pandas as pd
import numpy as np
import pickle
import json
from mlflow.models.signature import infer_signature

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow.sklearn
import mlflow
import dagshub
import os

# Initialize dagshub and set mlflow for experiment tracking
dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)
mlflow.set_experiment('Best_model')
mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

model = pickle.load(open('model.pkl','rb'))

y_pred= model.predict(X_test)

with mlflow.start_run(run_name="DVC Model") as run:


    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params({
    'n_estimators': 400,
    'max_depth': 5
    })

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", pre)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

         
    mlflow.log_artifact(__file__)
   # Log the trained model
    signature_ = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, 'Best Model', signature= signature_)

    #save run id and model info into JSON file:

    run_info = {'run_id': run.info.run_id, 'model_name':'Best_Model'}
    report_path = 'reports/run_info.json'
    with open(report_path, 'w') as file:
        json.dump(run_info,file,indent=4)


    metrics_dict = {
        'acc':acc,
        'precision':pre,
        'recall':recall,
        'f1 score': f1
    }

    with open('metrics.json','w') as file:
        json.dump(metrics_dict, file, indent=4)