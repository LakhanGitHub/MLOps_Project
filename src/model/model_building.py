import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

train_data = pd.read_csv('./data/processed/train_processed.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

clf = RandomForestClassifier(n_estimators=400, max_depth=10)
clf.fit(X_train, y_train)

#save model into pickle file

pickle.dump(clf, open('model.pkl', 'wb'))