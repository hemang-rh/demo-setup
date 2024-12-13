# MODEL_PATH = "https://iris-model-demo-multimodel.apps.cluster-9csp7.9csp7.sandbox3279.opentlc.com/v2/models/iris-model/infer"
# TOKEN = "Token"

import requests

MODEL_PATH = "Serving URL"
TOKEN = "Token"

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
    MODEL_PATH, json=payload, headers=headers, verify=False
)

# either pretty-print results or print error
if (response.status_code == 200):
    outputs = response.json()
    response_dict = response.json()
    print(response_dict['outputs'][0]['data'])
else:
    print("Error making request:", response.status_code, response.content)
