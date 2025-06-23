import pandas as pd
import numpy as np
import os

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

def fill_missing_value(df):
    for col in df.columns:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)

    return df

train_processed_data = fill_missing_value(train_data)
test_processed_data = fill_missing_value(test_data)

#create a path to save processed data
data_path = os.path.join('data','processed')
os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'), index=False)
test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)