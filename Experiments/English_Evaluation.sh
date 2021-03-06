#!/bin/bash


##### PARAMETERS ##################################
MODEL_PATH=${1:-${MODEL_PATH:-"results/english.-WITH-CONTROLS.-EPOCH-10.LR_TYPE-linear.-21-06-25.10.45.40"}}
DATA_DIR=${2:-${DATA_DIR:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/data/"}}
TEST_MILD=${3:-${TEST_MILD:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/df_test_mild.csv"}}
TEST_MODERATE=${4:-${TEST_MODERATE:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/df_test_moderate.csv"}}
TEST_SEVERE=${5:-${TEST_SEVERE:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/df_test_severe.csv"}}
TEST_VSEVERE=${6:-${TEST_VSEVERE:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/df_test_very_severe.csv"}}
######################################################


CMD="python3 utils/evaluation.py"
CMD+=" --model_path=$MODEL_PATH"
CMD+=" --cache_dir=$DATA_DIR"
CMD+=" --test_mild=$TEST_MILD"
CMD+=" --test_moderate=$TEST_MODERATE"
CMD+=" --test_severe=$TEST_SEVERE"
CMD+=" --test_vsevere=$TEST_VSEVERE"
CMD+=" --num_proc=10"



set -x
$CMD
set +x