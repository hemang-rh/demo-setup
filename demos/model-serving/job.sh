#!/bin/bash

# shellcheck disable=SC1091

# MODLEL_URL="https://ai-on-openshift.io/odh-rhoai/img-triton/card.fraud.detection.onnx"
# MODEL_NAME="card.fraud.detection.onnx"

# MODEL_URL_1="https://github.com/RedHatTraining/AI26X-apps/raw/main/deploying/rhoaiserving-using/rf_iris.onnx"
# MODEL_NAME_1="rf_iris.onnx"

MODEL_URL_1="https://github.com/hemang-rh/demo-setup/raw/main/demos/models/rf_iris.onnx"
MODEL_NAME_1="rf_iris.onnx"

MODEL_URL_2="https://github.com/hemang-rh/demo-setup/raw/main/demos/models/diabetes-from-tensorflow-keras.onnx"
MODEL_NAME_2="diabetes-from-tensorflow-keras.onnx"


MODEL_URL_3="https://github.com/hemang-rh/demo-setup/raw/main/demos/models/card.fraud.detection-1.onnx"
MODEL_NAME_3="card.fraud.detection-1.onnx"

echo "Downloading model from ${MODEL_NAME_1}"
curl -sL ${MODEL_URL_1} -o ${MODEL_NAME_1}

echo "Downloading model from ${MODEL_NAME_2}"
curl -sL ${MODEL_URL_2} -o ${MODEL_NAME_2}

echo "Downloading model from ${MODEL_NAME_3}"
curl -sL ${MODEL_URL_3} -o ${MODEL_NAME_3}

pip install minio requests boto3 --quiet

python3 /app/minio-tools.py

ls -lh *.onnx