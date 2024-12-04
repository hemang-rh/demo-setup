import logging
import os
# from minio import Minio
import requests
import boto3

logging.basicConfig(level=logging.INFO)

# model_url = "https://raw.githubusercontent.com/opendatahub-io-contrib/ai-on-openshift/main/docs/odh-rhoai/img-triton/card.fraud.detection.onnx"
# model_name = "card.fraud.detection.onnx"

# model_url_1 = "https://github.com/RedHatTraining/AI26X-apps/raw/main/deploying/rhoaiserving-using/rf_iris.onnx"
# model_name_1 = "rf_iris.onnx"


MODEL_NAME_1="rf_iris.onnx"
MODEL_NAME_2="diabetes-from-tensorflow-keras.onnx"
MODEL_NAME_3="card.fraud.detection-1.onnx"

aws_access_key_id = os.getenv("MINIO_ROOT_USER")
aws_secret_access_key = os.getenv("MINIO_ROOT_PASSWORD")
aws_s3_endpoint = os.getenv("MINIO_ENDPOINT")
bucket_name = os.getenv("MINIO_BUCKET_NAME")

logging.info(f"AWS Access key: {aws_access_key_id}")
logging.info(f"AWS Secret key: {aws_secret_access_key}")
logging.info(f"AWS S3 endpoint: {aws_s3_endpoint}")
logging.info(f"Bucket name: {bucket_name}")


def download_file_from_github(github_url, file_name):
    logging.info(f"Downloading file {file_name} from {github_url}")
    response = requests.get(github_url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        logging.info(f"File {file_name} downloaded successfully "
                     f"from {github_url}")
    else:
        logging.error(f"Failed to download file from {github_url}, "
                      f"status code: {response.status_code}")


s3 = boto3.client(
    "s3",
    endpoint_url=aws_s3_endpoint,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)


def create_bucket(bucket_name):
    logging.info(f"Creating bucket {bucket_name}")
    try:
        s3.list_buckets()["Buckets"]
        s3.head_bucket(Bucket=bucket_name)
        logging.info(f'Bucket {bucket_name} already exists')
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=bucket_name)
        logging.info(f'Bucket {bucket_name} created')


def upload_file_to_bucket(file_name, bucket_name):
    logging.info(f"Uploading file {file_name} to bucket {bucket_name}")
    try:
        s3.upload_file(file_name, bucket_name, file_name)
        logging.info(f"File {file_name} uploaded to bucket "
                     f"{bucket_name} successfully")
    except Exception as e:
        logging.error(f"Failed to upload file {file_name} to bucket "
                      f"{bucket_name}: {e}")


if __name__ == '__main__':
    # download_file_from_github(model_url, model_name)
    # download_file_from_github(model_url_1, model_name_1)
    create_bucket(bucket_name)
    upload_file_to_bucket(MODEL_NAME_1, bucket_name)
    upload_file_to_bucket(MODEL_NAME_2, bucket_name)
    upload_file_to_bucket(MODEL_NAME_3, bucket_name)
