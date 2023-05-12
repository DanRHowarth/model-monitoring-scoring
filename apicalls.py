import os
import requests
import json


with open('config.json', 'r') as f:
    config = json.load(f)
model_path = os.path.join(config['output_model_path'])

#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:5000"

#Call each API endpoint and store the responses
response1 = requests.post('http://127.0.0.1:5000/prediction?filepath=testdata').content
response2 = requests.get('http://127.0.0.1:5000/scoring').content
response3 = requests.get('http://127.0.0.1:5000/summarystats').content
response4 = requests.get('http://127.0.0.1:5000/diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace

with open(f'{model_path}/apireturns.txt','w') as file:
    for response in responses:
        file.write(str(response))


