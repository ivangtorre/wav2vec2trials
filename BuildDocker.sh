#!/bin/bash
#####################################
# Build Docker Image ################
#####################################

# VARIABLES ####################
NAME=${NAME:-"wav2vec2"}
DOCKERFILEPATH=${DOCKERFILEPATH:-"Docker"}

#docker build --no-cache -t ${NAME} Docker2.8/.
docker build -t ${NAME} ${DOCKERFILEPATH}/.
