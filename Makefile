BASE:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/sh

.PHONY: setup-cluster

setup-cluster:
	@echo "Configuring the cluster..."
	until oc apply -k components/00-branding; do : ; done
	until oc apply -k components/01-console-link; do : ; done
	until oc apply -k components/02-argocd; do : ; done
	until oc apply -k components/03-configure-gpu; do : ; done
	until oc apply -k components/04-authorino-operator; do : ; done
	until oc apply -k components/05-serverless-operator; do : ; done
	until oc apply -k components/06-servicemesh-operator; do : ; done
	until oc apply -k components/07-rhoai-operator; do : ; done
	until oc apply -k components/09-serving-runtime; do : ; done

.PHONY: model-serving-demo-setup model-serving-demo-teardown

model-serving-demo-setup:
	@echo "Configuring the model serving demo..."
	until oc apply -k demos/model-serving; do : ; done

model-serving-demo-teardown:
	@echo "Tearing down the model serving demo..."
	until oc delete -k demos/model-serving; do : ; done