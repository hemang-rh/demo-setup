BASE:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/sh

.PHONY: prereqs gpu rhoai fullsetup

prereqs:
	@echo "Configuring cluster prereqs..."
	./scripts/setup.sh -p

gpu:
	@echo "Configuring the GPU..."
	./scripts/setup.sh -g

rhoai:
	@echo "Configuring RHOAI..."
	./scripts/setup.sh -r

fullsetup:
	@echo "Setting up the cluster..."
	./scripts/setup.sh -a

.PHONY: model-serving-demo

model-serving-demo:
	@echo "Configuring the model serving demo..."
	# until oc apply -k demos/model-serving; do : ; done
	./scripts/setup.sh -d model-serving


model-serving-demo-teardown:
	@echo "Tearing down the model serving demo..."
	oc delete -k demos/model-serving

