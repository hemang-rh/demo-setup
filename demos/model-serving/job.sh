#!/bin/bash

# shellcheck disable=SC1091

MODLEL_URL="https://ai-on-openshift.io/odh-rhoai/img-triton/card.fraud.detection.onnx"
MODEL_NAME="card.fraud.detection.onnx"

MODEL_URL_1="https://github.com/RedHatTraining/AI26X-apps/raw/main/deploying/rhoaiserving-using/rf_iris.onnx"
MODEL_NAME_1="rf_iris.onnx"

echo "Downloading model from ${MODEL_NAME}"
curl -L ${MODLEL_URL} -o ${MODEL_NAME}

echo "Downloading model from ${MODEL_NAME_1}"
curl -L ${MODEL_URL_1} -o ${MODEL_NAME_1}


pip install minio
pip install requests
pip install boto3

python3 /app/minio-tools.py

ls -lh /app
ls -lh *.onnx