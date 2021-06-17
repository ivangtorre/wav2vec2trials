#!/bin/bash


##### PARAMETERS ##################################
DATA_DIR=${1:-${DATA_DIR:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/data/"}}  # The folder where audios are stored
LANGUAGE=${2:-${LANGUAGE:-"english"}} # Language
DATASET=${2:-${DATASET:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/df_train.csv"}}  # Train Dataset Location
DATASET_EVAL=${2:-${DATASET_EVAL:-"/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/english/2021_06_17/df_eval.csv"}}  # Eval Dataset Location
RESULT_DIR=${3:-${RESULT_DIR:-"results"}}
MODELXLSR=${4:-${MODELXLSR:-"facebook/wav2vec2-large-xlsr-53"}}
MODEL_DIR=${6:-${MODEL_DIR:-"/datasets/modelxlsr"}}
NUM_GPUS=${7:-${NUM_GPUS:-1}}
EPOCHS=${9:-${EPOCHS:-10}}
SEED=${10:-${SEED:-6}}
BATCH_SIZE=${11:-${BATCH_SIZE:-1}} # original 16
LEARNING_RATE=${12:-${LEARNING_RATE:-"0.0002"}}
WARMUP_RATIO=${13:-${WARMUP_RATIO:-"0.1"}}
SAVE_STATES=${14:-${SAVE_STATES:-5000}}
EVAL_STATES=${15:-${EVAL_STATES:-5000}}
LOG_STATES=${16:-${LOG_STATES:-100}}
SAVE_LIMIT=${17:-${SAVE_LIMIT:-2}}
FEAT_PROJ_DROPOUT=${18:-${FEAT_PROJ_DROPOUT:-0.05}}
LAYER_DROPOUT=${19:-${LAYER_DROPOUT:-0.02}}
ACCUM_STEPS=${20:-${ACCUM_STEPS:-3}}
MASK_TIME=${21:-${MASK_TIME:-0.057}}
HIDDEN_DROPOUT=${22:-${HIDDEN_DROPOUT:-0.025}}
ACTIVATION_DROPOUT=${23:-${ACTIVATION_DROPOUT:-0.026}}
ATTENTION_DROPOUT=${24:-${ATTENTION_DROPOUT:-0.036}}
LR_TYPE=${25:-${LR_TYPE:-"linear"}}
MAX_TRAIN_SAMPLES=${27:-${MAX_TRAIN_SAMPLES:-3172}}  # Set to 0 for using all dataset
MAX_VAL_SAMPLES=${28:-${MAX_VAL_SAMPLES:-100}}  # Set to 0 for using all dataset
######################################################


DATESTAMP=`date +'%y-%m-%d.%H.%M.%S'`
RESULT_DIR=$RESULT_DIR/$LANGUAGE.-EPOCH-$EPOCHS.LR_TYPE-$LR_TYPE.-$DATESTAMP
mkdir -p "$RESULT_DIR"


CMD="python3 utils/train_wav2vec2.py"
CMD+=" --model_name_or_path=$MODELXLSR"
CMD+=" --dataset_config_name=$DATASET"
CMD+=" --dataset_eval=$DATASET_EVAL"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --overwrite_output_dir"
CMD+=" --num_train_epochs=$EPOCHS"
CMD+=" --per_device_train_batch_size=$BATCH_SIZE"
CMD+=" --learning_rate=$LEARNING_RATE"
CMD+=" --warmup_ratio=$WARMUP_RATIO"
CMD+=" --lr_scheduler_type=$LR_TYPE"
CMD+=" --evaluation_strategy=steps"
CMD+=" --save_steps=$SAVE_STATES"
CMD+=" --eval_steps=$EVAL_STATES"
CMD+=" --logging_steps=$LOG_STATES"
CMD+=" --save_total_limit=$SAVE_LIMIT"
CMD+=" --freeze_feature_extractor"
CMD+=" --feat_proj_dropout=$FEAT_PROJ_DROPOUT"
CMD+=" --layerdrop=$LAYER_DROPOUT"
CMD+=" --gradient_checkpointing"
CMD+=" --fp16"
CMD+=" --do_train --do_eval"
CMD+=" --cache_dir=$DATA_DIR"
CMD+=" --model_cache_dir=$MODEL_DIR"
CMD+=" --logging_dir=$RESULT_DIR"
CMD+=" --preprocessing_num_workers=10"
CMD+=" --gradient_accumulation_steps=$ACCUM_STEPS"
CMD+=" --mask_time_prob=$MASK_TIME"
CMD+=" --hidden_dropout=$HIDDEN_DROPOUT"
CMD+=" --activation_dropout=$ACTIVATION_DROPOUT"
CMD+=" --attention_dropout=$ATTENTION_DROPOUT"
CMD+=" --max_train_samples=$MAX_TRAIN_SAMPLES"
CMD+=" --max_val_samples=$MAX_VAL_SAMPLES"


export GBS=$(expr $BATCH_SIZE \* $NUM_GPUS)
printf -v TAG "wav2vec2_train_benchmark_amp-%s_gbs%d" "$AMP" $GBS
LOGFILE=$RESULT_DIR/$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"


set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x
