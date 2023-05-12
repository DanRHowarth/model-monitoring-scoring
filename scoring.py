from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

# dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

#################Function for model scoring
def score_model(test_data_path:str, model_path:str):
    # this function should take a trained model, load test data, and calculate an F1 score
    # for the model relative to the test data
    #it should write the result to the latestscore.txt file

    data = pd.read_csv(f'{test_data_path}/testdata.csv')
    X = data.drop(columns=['exited', 'corporation'])
    y = data['exited']

    model = pickle.load(open(f'{model_path}/trainedmodel.pkl', 'rb'))
    preds = model.predict(X)
    f1 = metrics.f1_score(y, preds)

    with open(f'{model_path}/latestscore.txt', 'w') as file:
        file.write(str(f1))

    return f1

if __name__ == '__main__':
    score_model(test_data_path, model_path)