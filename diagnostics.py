import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])


##################Function to get model predictions
def model_predictions(model_path, test_data_path):
    #read the deployed model and a test dataset, calculate predictions

    data = pd.read_csv(f'{test_data_path}/testdata.csv').drop(columns=['exited', 'corporation'])
    model = pickle.load(open(f'{model_path}/trainedmodel.pkl', 'rb'))
    predictions = model.predict(data)

    return list(predictions)

##################Function to get summary statistics
def dataframe_summary(dataset_csv_path):

    data = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')

    stats = []
    for col in data.select_dtypes(include=np.number).columns:
        stats.append(data[col].agg(['mean', 'median', 'std']))

    return stats

##################Function to get timings
def execution_time():

    timing = []

    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ing_time = timeit.default_timer() - starttime
    timing.append(ing_time)


    starttime = timeit.default_timer()
    os.system('python3 training.py')
    train_time = timeit.default_timer() - starttime
    timing.append(train_time)

    print(timing)

    return timing

##################Function to check dependencies
def outdated_packages_list():
    requirements = subprocess.check_output(['pip', 'list', '--outdated'])
    with open('outdated.txt', 'wb') as f:
        f.write(requirements)

### check for missing data

def check_missing_data(dataset_csv_path):
    data = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')
    return data.isna().sum()/data.shape[0]


if __name__ == '__main__':
    model_predictions(model_path, test_data_path)
    dataframe_summary(dataset_csv_path)
    execution_time()
    outdated_packages_list()
    check_missing_data(dataset_csv_path)





    
