import requests
import pickle

MODEL_PATH = "Serving URL"
TOKEN = "Token"

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

def rest_request(data):
    json_data = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": [1, 5],
                "datatype": "FP32",
                "data": data
            }
        ]
    }

    response = requests.post(MODEL_PATH, json=json_data)
    response_dict = response.json()
    return response_dict['outputs'][0]['data']