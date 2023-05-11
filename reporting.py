import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import matplotlib.pyplot as plt



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])


def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    predictions = model_predictions(model_path, test_data_path)
    actual = pd.read_csv(f'{test_data_path}/testdata.csv')['exited']

    cm = metrics.confusion_matrix(actual, predictions)
    display = metrics.ConfusionMatrixDisplay(cm).plot()
    plt.savefig(f'{output_model_path}/confusionmatrix.png')



if __name__ == '__main__':
    score_model()
