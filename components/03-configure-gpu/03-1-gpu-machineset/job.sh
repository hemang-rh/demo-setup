#!/bin/bash

# shellcheck disable=SC1091
. /scripts/ocp.sh

INSTANCE_TYPE=${INSTANCE_TYPE:-g4dn.4xlarge}

ocp_aws_cluster || exit 0
echo "Creating GPU machineset for ${INSTANCE_TYPE}"
ocp_aws_create_gpu_machineset "${INSTANCE_TYPE}"
echo "Tainting GPU machineset for ${INSTANCE_TYPE}"
ocp_aws_taint_gpu_machineset "${INSTANCE_TYPE}"
# ocp_scale_machineset
# ocp_machineset_create_autoscale
# ocp_aws_machineset_fix_storage
# ocp_machineset_taint_gpu