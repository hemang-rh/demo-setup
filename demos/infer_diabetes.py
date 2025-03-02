import requests
import dotenv
import os

dotenv.load_dotenv()
MODEL_URL = os.getenv('MODEL_URL')
TOKEN = os.getenv('TOKEN')

print(f"Model url: {MODEL_URL}")

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

payload = {
    "inputs": [
        {
            "name": "dense_input",
            "data": [[6, 148, 72, 35, 0, 33.6, 0.627, 50],
                     [1, 85, 66, 29, 0, 26.6, 0.351, 31]],
            "datatype": "FP32",
            "shape": [2, 8]
        }
    ]
}

response = requests.post(
    MODEL_URL, json=payload, headers=headers, verify=False
)

if (response.status_code == 200):
    outputs = response.json()
    response_dict = response.json()
    print('\n***Output from the model***')
    print(response_dict['outputs'][0]['data'])
else:
    print("Error making request:", response.status_code, response.status_code)
