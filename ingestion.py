import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path


# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

def merge_multiple_dataframe(input_folder_path, output_folder_path):
    """
    function to load in data from a load path and save concatenated output to
    target path, along with an ingestion record.
    :param input_folder_path: path to load in data from
    :param output_folder_path: path to save data to
    :return:
    """
    # load in data and save to csv using config params
    datasets = Path(input_folder_path).glob('*.csv')

    data = []
    filenames = []

    for filename in datasets:
        df = pd.read_csv(filename)
        data.append(df)
        filenames.append(filename)

    concat_data = pd.concat(data).drop_duplicates().reset_index(drop=True)
    concat_data.to_csv(f'{output_folder_path}/finaldata.csv', index=False)

    # save a data ingestion record

    # In order to record the time of the latest ingestion process, you need to get a current timestamp:
    dateTimeObj = datetime.now()
    date = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)

    with open(F'{output_folder_path}/ingestedfiles.txt', 'w') as writer:
        writer.write('Date of Ingestion: ' + date + '\n')
        for i, file in enumerate(filenames):
            writer.write(f'File {i}: ' + str(file) + '\n')


if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path)