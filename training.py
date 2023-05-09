from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])



def train_model(dataset_csv_path: str, model_path: str):
    """
    function to ingest data, train a model and save trained model as a pickle file
    :param dataset_csv_path: path to load in data from
    :param model_path: destination to save model to
    """
    data = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')
    X = data.drop(columns=['exited', 'corporation'])
    y = data['exited']

    # use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)

    # fit the logistic regression to your data
    lr.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(lr, open(f'{model_path}/trainedmodel.pkl', 'wb'))

if __name__ == '__main__':
    train_model(dataset_csv_path, model_path)
