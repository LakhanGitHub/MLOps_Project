from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow
import dagshub

#initilize dagshub and set mlflow for experiment tracking
dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)
mlflow.set_experiment('Experiment1')
mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

#load data
data = pd.read_csv(r'E:\Data_Science\MLOps-Exp\water_potability.csv')

#tain test split
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

#impute missing values
def fill_missing_value(df):
    for col in df.columns:
        if df[col].isnull().any():
            mean_value = df[col].median()
            df[col].fillna(mean_value, inplace=True)

    return df
train_processed_data = fill_missing_value(train_data)
test_processed_data = fill_missing_value(test_data)

#randomfortestclassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

#seprate train(X) and test(y)
X_train = train_processed_data.drop(columns=['Potability'], axis=1)
y_train = train_processed_data['Potability']

X_test = test_processed_data.iloc[:,0:-1].values
y_test = test_processed_data.iloc[:,-1]



n_enstimators = 100

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n_enstimators)
    clf.fit(X_train,y_train)

    #save file into pikle
    pickle.dump(clf, open('model.pkl','wb'))

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    #log the metircs and parameters
    mlflow.log_metric('accurace',acc)
    mlflow.log_metric('precision',pre)
    mlflow.log_metric('recall',recall)
    mlflow.log_metric('f1_score',f1_score)

    mlflow.log_param('n_estimators', n_enstimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confution Metrics')

    plt.savefig('confution_matrix.png')
    #log artifact
    
    mlflow.log_artifact('confution_matrix.png')
    mlflow.log_artifact(__file__)
    
    #log mode
    mlflow.sklearn.log_model(clf,'RandomForestClassifier')

    #set tags addition metadata
    mlflow.set_tag('Author', 'Lakhan')
    mlflow.set_tag('model', 'Random Forest')

    #print accuracy
    print('Accuracy', acc)
    print('Precision', pre)
    print('Recall', recall)
    print('f1_score', f1_score)


