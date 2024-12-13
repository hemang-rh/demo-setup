## Model Serving Demo

#### Test deployed model
```
> export MODEL_PATH=http://modelmesh-serving.rhoai-multimodel:8008/v2/models/iris-model
> curl -s $MODEL_PATH | jq
```