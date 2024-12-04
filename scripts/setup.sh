#!/bin/bash

# shellcheck disable=SC1091

################# standard init #################

# 8 seconds is usually enough time for the average user to realize they foobar
export SLEEP_SECONDS=8

check_shell(){
  [ -n "$BASH_VERSION" ] && return
  echo -e "${ORANGE}WARNING: These scripts are ONLY tested in a bash shell${NC}"
  sleep "${SLEEP_SECONDS:-8}"
}

check_git_root(){
  if [ -d .git ] && [ -d scripts ]; then
    GIT_ROOT=$(pwd)
    export GIT_ROOT
    echo "GIT_ROOT:   ${GIT_ROOT}"
  else
    echo "Please run this script from the root of the git repo"
    exit
  fi
}

get_script_path(){
  SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
  echo "SCRIPT_DIR: ${SCRIPT_DIR}"
}

check_shell
check_git_root
get_script_path

# shellcheck source=/dev/null
. "${SCRIPT_DIR}/functions.sh"

################# standard init #################

DEFAULT_HTPASSWD=scratch/htpasswd-local
DEFAULT_ADMIN_PASS=scratch/password.txt

check_cluster_version(){
  OCP_VERSION=$(oc version | sed -n '/Server Version: / s/Server Version: //p')
  AVOID_VERSIONS=()
  TESTED_VERSIONS=("4.14.37" "4.16.14")

  echo "Current OCP version: ${OCP_VERSION}"
  echo "Tested OCP version(s): ${TESTED_VERSIONS[*]}"
  echo ""

  # shellcheck disable=SC2076
  if [[ " ${AVOID_VERSIONS[*]} " =~ " ${OCP_VERSION} " ]]; then
    echo "OCP version ${OCP_VERSION} is known to have issues with this demo"
    echo ""
    echo 'Recommend: "oc adm upgrade --to-latest=true"'
    echo ""
  fi
}

validate_cli(){
  echo ""
  echo "Validating command requirements..."
  bin_check oc
  bin_check helm
  bin_check jq
  echo ""
}

validate_cluster(){
  ocp_check_login
  check_cluster_version
}

add_admin_user(){
  DEFAULT_USER="admin"
  DEFAULT_PASS=$(genpass)

  HT_USERNAME=${1:-${DEFAULT_USER}}
  HT_PASSWORD=${2:-${DEFAULT_PASS}}

  echo "${HT_PASSWORD}" > "${DEFAULT_ADMIN_PASS}"

  htpasswd_ocp_get_file
  htpasswd_add_user "${HT_USERNAME}" "${HT_PASSWORD}"
  htpasswd_ocp_set_file
  htpasswd_validate_user "${HT_USERNAME}" "${HT_PASSWORD}"
}

help(){
  loginfo "This script installs RHOAI and other dependencies"
  loginfo "Usage: $(basename "$0")"
  loginfo "Options:"
  loginfo " -h, --help      Usage"
  loginfo " -p, --prereqs   Install prerequisites"
  loginfo " -g, --gpu       Configure gpu support"
  loginfo " -r, --rhoai     Configure RHOAI"
  loginfo " -d, --demo      Run a specific demo"
  return 0
}

while getopts "h:agprd:" flag; do
  case $flag in
    h) help ;;
    a) all=1 ;;
    g) gpu=1 ;;
    p) prereqs=1 ;;
    r) rhoai=1 ;;
    d) demo=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done


install_prereqs(){
  logbanner "Install prerequisites"

  validate_cluster || return 1

  retry oc apply -k "${GIT_ROOT}"/components/00-prereqs

  validate_cli || echo "!!!NOTICE: you are missing cli tools needed!!!"

  logbanner "Add administrative user"
  loginfo "Creating user 'admin'"

  if [ -f "${DEFAULT_ADMIN_PASS}" ]; then
    HT_PASSWORD=$(cat "${DEFAULT_ADMIN_PASS}")

    loginfo "Delete ${DEFAULT_ADMIN_PASS} to create a NEW password
    "

    htpasswd_validate_user "${HT_USERNAME}" "${HT_PASSWORD}"

    return
  else
    retry oc apply -k "${GIT_ROOT}"/components/01-admin-user
    add_admin_user admin
  fi
}

configure-gpu() {
  logbanner "Add GPU node"
  retry oc apply -k "${GIT_ROOT}"/components/02-gpu-node
  sleep 60
  ocp_scale_machineset

  logbanner "Install GPU operators"
  retry oc apply -k "${GIT_ROOT}"/components/03-gpu-operators

  logbanner "Install GPU dashboard"
  retry oc apply -k "${GIT_ROOT}
"/components/05-gpu-timeslicing

  logbanner "Run sample gpu application"
  retry oc apply -k "${GIT_ROOT}"/components/06-gpu-app
}

configure-rhoai() {
  logbanner "Install RHOAI and its dependencies"
  logbanner "Install Authorino operator"
  retry oc apply -k "${GIT_ROOT}"/components/07-authorino-operator

  logbanner "Install serverless operator"
  retry oc apply -k "${GIT_ROOT}"/components/08-serverless-operator

  logbanner "Install service mesh operator"
  retry oc apply -k "${GIT_ROOT}"/components/09-servicemesh-operator

  logbanner "Install RHOAI operator"
  retry oc apply -k "${GIT_ROOT}"/components/10-rhoai-operator

  logbanner "Configure serving runtime"
  retry oc apply -k "${GIT_ROOT}"/components/11-serving-runtime
}

workshop_uninstall(){
  logbanner "Uninstall Workshop"

  oc -n kube-system get secret/kubeadmin || return 1

  rm "${DEFAULT_HTPASSWD}"{,.txt} "${DEFAULT_ADMIN_PASS}"

  oc -n pipeline-test delete --all datasciencepipelinesapplications

  oc delete datasciencecluster default-dsc
  oc delete dscinitialization default-dsci

  sleep 8

  oc -n istio-system delete --all servicemeshcontrolplanes.maistra.io
  oc -n istio-system delete --all servicemeshmemberrolls.maistra.io
  oc delete --all -A servicemeshmembers.maistra.io

  oc -n knative-serving delete knativeservings.operator.knative.dev knative-serving
  oc delete consoleplugin console-plugin-nvidia-gpu

  oc delete csv -A -l operators.coreos.com/authorino-operator.openshift-operators
  oc delete csv -A -l operators.coreos.com/devworkspace-operator.openshift-operators
  oc delete csv -A -l operators.coreos.com/servicemeshoperator.openshift-operators
  oc delete csv -A -l operators.coreos.com/web-terminal.openshift-operators

  oc delete -n openshift-operators deploy devworkspace-webhook-server

  GPU_MACHINE_SET=$(oc -n openshift-machine-api get machinesets -o name | grep -E 'gpu|g4dn' | head -n1)
  for set in ${GPU_MACHINE_SET}
  do
    oc -n openshift-machine-api delete "$set"
  done

  oc delete \
    -f "${GIT_ROOT}"/configs/00 \
    -f "${GIT_ROOT}"/configs/01 \
    -f "${GIT_ROOT}"/configs/02 \
    -f "${GIT_ROOT}"/configs/03 \
    -f "${GIT_ROOT}"/configs/04 \
    -f "${GIT_ROOT}"/configs/05 \
    -f "${GIT_ROOT}"/configs/06 \
    -f "${GIT_ROOT}"/configs/07 \
    -f "${GIT_ROOT}"/configs/08 \
    -f "${GIT_ROOT}"/configs/08/minio \
    -f "${GIT_ROOT}"/configs/09 \
    -f "${GIT_ROOT}"/configs/uninstall

  oc apply \
    -f "${GIT_ROOT}"/configs/restore

}

setup(){

  if [ "$prereqs" ]; then
    loginfo "Installing prerequisites"
    install_prereqs
  elif [ "$gpu" ]; then
    loginfo "Configuring GPU support"
    configure-gpu
  elif [ "$rhoai" ]; then
    loginfo "Configuring RHOAI"
    configure-rhoai
  elif [ "$all" ]; then
    loginfo "Setup cluster"
    install_prereqs
    configure-gpu
    configure-rhoai
  elif [ "$demo" = "model-serving" ]; then
    loginfo "Configuring model serving demo"
    retry oc apply -k "${GIT_ROOT}"/demos/model-serving
  else
    loginfo "No valid option passed"
    help
  fi
}

is_sourced || setup