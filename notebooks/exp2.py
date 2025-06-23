from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import pickle
import mlflow.sklearn
import mlflow
import dagshub

#initilize dagshub and set mlflow for experiment tracking
dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)
#set Experiment
mlflow.set_experiment('Experiment2')
#set uri tracking
mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

#load data
data = pd.read_csv(r'E:\Data_Science\MLOps-Exp\water_potability.csv')

#tain test split
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

#impute missing values with median
def fill_missing_value(df):
    for col in df.columns:
        if df[col].isnull().any():
            mean_value = df[col].median()
            df[col].fillna(mean_value, inplace=True)

    return df
train_processed_data = fill_missing_value(train_data)
test_processed_data = fill_missing_value(test_data)

#seprate train(X) and test(y)
X_train = train_processed_data.drop(columns=['Potability'], axis=1)
y_train = train_processed_data['Potability']

X_test = test_processed_data.iloc[:,0:-1]
y_test = test_processed_data.iloc[:,-1]

#define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest':RandomForestClassifier(),
    'SVM': SVC(),
    'Decision Tree':DecisionTreeClassifier(),
    'K-Nearst Neighbors':KNeighborsClassifier(),
    'XGboost':XGBClassifier()   
}

with mlflow.start_run(run_name='water_potability-experiment'):
    #iterate over model
    for model_name, model in models.items():
        #start child run
        with mlflow.start_run(run_name=model_name, nested=True):
            model.fit(X_train, y_train)
            
            #save model with pickle
            model_filename = f'{model_name.replace(" ","_")}.pkl'
            pickle.dump(model, open(model_filename,'wb'))

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            #log the metircs and parameters
            mlflow.log_metric('accuracy',acc)
            mlflow.log_metric('precision',pre)
            mlflow.log_metric('recall',recall)
            mlflow.log_metric('f1_score',f1)

            #save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confution Metrics for {model_name}')

            plt.savefig(f'confution_matrix{model_name.replace(" ","_")}.png')
            #log artifact
            
            mlflow.log_artifact(f'confution_matrix{model_name.replace(" ","_")}.png')

            #log source code file
            mlflow.log_artifact(__file__)
            
            #log mode
            mlflow.sklearn.log_model(model,model_name.replace(" ","_"))

            #set tags addition metadata
            mlflow.set_tag('Author', 'Lakhan')
            mlflow.set_tag('model', f'{model_name}')

    #print accuracy
    print('Trainand Log all the model successfully')
            


