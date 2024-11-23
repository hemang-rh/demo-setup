curl -L -o rf_iris.onnx https://github.com/RedHatTraining/AI26X-apps/raw/main/deploying/rhoaiserving-using/rf_iris.onnx
curl -L -o minio-data-connection.yaml https://github.com/RedHatTraining/AI26X-apps/raw/main/deploying/rhoaiserving-using/minio-data-connection.yaml

AWS_ACCESS_KEY_ID: $MINIO_ROOT_USER
AWS_SECRET_ACCESS_KEY: $MINIO_ROOT_PASSWORD
AWS_DEFAULT_REGION: us
AWS_S3_ENDPOINT: http://minio.ic-shared-minio.svc:9000
AWS_S3_BUCKET: user0

AWS_S3_BUCKET=models\n",
"AWS_S3_ENDPOINT='http://minio.minio.svc:9000'\n",
