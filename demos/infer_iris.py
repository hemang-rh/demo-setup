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
            "name": "X",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [3, 4, 3, 2]
        }
    ]
}

response = requests.post(
    MODEL_PATH, json=payload, headers=headers, verify=False
)

# either pretty-print results or print error
if (response.status_code == 200):
    outputs = response.json()
    # print(json.dumps(outputs, indent=4))
    response_dict = response.json()
    print(response_dict['outputs'][0]['data'])
else:
    print("Error making request:", response.status_code, response.content)
