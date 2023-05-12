from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, check_missing_data
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction/", methods=['POST', 'OPTIONS'])
def predict():
    filepath = request.args.get('filepath')
    predictions = model_predictions(model_path, filepath)
    return str(predictions)


# #######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():
    score = score_model(test_data_path, model_path)
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():
    stats = dataframe_summary(dataset_csv_path)
    return str(stats)
#
#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    execution_time()
    outdated_packages_list()
    check_missing_data(dataset_csv_path)

    return 'diagnostics run and saved in repo'

if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    app.run(host='0.0.0.0', port=5000)
