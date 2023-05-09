from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle(model_path:str, prod_deployment_path:str, dataset_csv_path:str):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
        
    # load pickle file and save to prod
    model = pickle.load(open(f'{model_path}/trainedmodel.pkl', 'rb'))
    pickle.dump(model, open(f'{prod_deployment_path}/trainedmodel.pkl', 'wb'))

    # the latestscore.txt value
    shutil.copyfile(f'{model_path}/latestscore.txt', f'{prod_deployment_path}/latestscore.txt')

    # ingestfiles.txt
    shutil.copyfile(f'{dataset_csv_path}/ingestedfiles.txt', f'{prod_deployment_path}/ingestedfiles.txt')

if __name__ == '__main__':
    store_model_into_pickle(model_path, prod_deployment_path, dataset_csv_path)
        

