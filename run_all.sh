#!/bin/bash

IMG_DIR="/home/pwagstro/Documents/workspace/gpt-flux-ouroboros/images"
OLD_PWD=$(pwd)
ITERATIONS=60
WORKFLOW=$OLD_PWD/workflow_api_new.json

for x in $(ls $IMG_DIR | tail +7); do
    ls -al $IMG_DIR/$x
    WORK_DIR="${x%.*}"
    mkdir -p $OLD_PWD/$WORK_DIR
    pushd $OLD_PWD/$WORK_DIR
    python $OLD_PWD/run_workflow.py --workflow $WORKFLOW --image $IMG_DIR/$x --output $WORK_DIR.jsonl --iterations $ITERATIONS
    popd
done
