#!/bin/bash
#####################################
# Script to initialize the server
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2hyperparameter"}
CONTAINER=${CONTAINER:-"wav2vec2_3"}
export NV_GPU="3"

################################
################################

# Ensure that the server is closed when the script exits
function cleanup_server {
    echo "Shutting down ${CONTAINER} container"
    docker stop ${CONTAINER} > /dev/null 2>&1
}

if [ "$(docker inspect -f "{{.State.Running}}" ${CONTAINER})" = "true" ]; then
    if [ "$1" == "norestart" ]; then
       echo "${CONTAINER} is already running ..."
       exit 0
    fi
    cleanup_server || true
fi

# Run the container
set -x
nvidia-docker run -it -d --rm --name ${CONTAINER} --runtime=nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 -v /data/:/data/:ro -v /home:/home -v $PWD:/workspace/wav2vec2/ ${NAME}
set +x

# Execute
nvidia-docker exec -it ${CONTAINER} bash Experiments/English_Experiment_1_lineal_5.sh
