import json
import os
import re
from pathlib import Path
from scoring import score_model


# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

prod_deploy_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']



##################Check and read new data
# first, read ingestedfiles.txt

with open(f'{prod_deploy_path}/ingestedfiles.txt', 'r') as base_file:
    file_contents = base_file.read()
    base_datasets = re.findall(r'File \d+: .*/(.+\.csv)', file_contents)

directory = Path(input_folder_path)  # replace this with the actual path to your directory
new_files = directory.glob('*.csv')
check_files = []
for file in new_files:
    check_files.append(file.name)

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
# print(check_files, base_datasets)
# check_files = ['dataset1.csv', 'dataset2.csv']
set1 = set(check_files)
set2 = set(base_datasets)
common_elements = set1.intersection(set2)

# if there has been change, then we need to ingest the new data and check for model drift
if len(common_elements) != len(base_datasets):

    print('running ingestion.py')
    # script needs to run the code in ingestion.py to ingest all the new data.
    os.system('python3 ingestion.py')

    print('getting old and new scores')
    ## check for model drift
    # Read the score from the latest model, recorded in latestscore.txt from the deployment directory,
    # specified in the prod_deployment_path entry of your config.json file.
    with open(f'{prod_deploy_path}/latestscore.txt', 'r') as score_file:
        last_score = float(score_file.read())


    # Make predictions using the trainedmodel.pkl model in the /production_deployment directory
    # and the most recent data you obtained from the previous "Checking and Reading New Data" step.
    # Get a score for the new predictions from step 2 by running the scoring.py.
    new_score = score_model(output_folder_path, output_model_path, 'finaldata.csv')
    ## issue here - prod path is wrong - think i just need to change to output_mdoel_path but just need to dbc

    # new_score = subprocess.check_output(['python3', 'scoring.py'])
    # new_score = float(new_score.decode('utf-8').strip())

    print('new: ', new_score)
    print('old: ', last_score)

    # Check whether the new score from step 3 is higher or lower than the score recorded in latestscore.txt in step 1
    # using the raw comparison test. If the score from step 3 is lower, then model drift has occurred. Otherwise, it has not.
    if new_score < last_score:
        print('model drift has occured')

        print('running training.py')
        # script needs to run the code in ingestion.py to ingest all the new data.
        os.system('python3 training.py')

        print('running deployment.py')
        # script needs to run the code in ingestion.py to ingest all the new data.
        os.system('python3 deployment.py')

    else:
        print('no model drift')
        exit()

    # The last part of your script should run the code from apicalls.py and reporting.py on the most recently
    # deployed model
    print('running app.py and apicalls.py')
    # script needs to run the code in ingestion.py to ingest all the new data.
    os.system('python3 app.py')
    # do I need to wait here
    os.system('python3 apicalls.py')

    print('running reporting.py')
    # script needs to run the code in ingestion.py to ingest all the new data.
    os.system('python3 reporting.py')

else:

    print('data not changed so no need to retain')

    exit()












##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here




##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







