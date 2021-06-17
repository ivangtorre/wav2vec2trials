import wandb
import os
import yaml
from wandb.sweeps import GridSearch, RandomSearch, BayesianSearch
import torchaudio
import random
import pandas as pd
from datasets import Dataset
import json
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, set_seed,)
set_seed(3)
random.seed(3)

# Get the datasets:
df = pd.read_csv("df_final.csv", delimiter=',')
df = df[~df["transcription"].isnull()]
#df = df[0:500]

# FILTER DURATION
df["duration"] = (df["mark_end"] - df["mark_start"])
df = df[df["duration"] < 25000]
df = df.reset_index(drop=True)

# REFORMAT AND SAVE
df = df[["transcription", "file_cut"]]
df.columns = ["sentence", "path"]
df["path"] = "/data/CORPORA/ACOUSTIC_CORPUS/APHASIA/audiosV3/" + df["path"]
all_dataset = Dataset.from_pandas(df)
#all_dataset.to_csv("dummy_dataset.csv")

# SPLIT TRAIN-TEST
all_dataset = all_dataset.train_test_split(test_size=0.4)
train_dataset = all_dataset["train"]
dummy_dataset = all_dataset["test"]
dummy_dataset = dummy_dataset.train_test_split(test_size=0.375)
eval_dataset = dummy_dataset["test"]
test_dataset = dummy_dataset["train"]


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1,
                                keep_in_memory=True, remove_columns=train_dataset.column_names)
vocab_test = eval_dataset.map(extract_all_chars, batched=True, batch_size=-1,
                              keep_in_memory=True, remove_columns=eval_dataset.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_list.insert(0, "[UNK]")
vocab_list.insert(0, "[PAD]")
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch


train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(speech_file_to_array_fn, remove_columns=eval_dataset.column_names)

tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0,
                                             do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Prepare data for training ##################
def prepare_dataset(batch):
    """
    check that all files have the correct sampling rate
    :param batch:
    :return:
    """
    assert (len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batch_size=2, batched=True)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names, batch_size=2, batched=True)

#train_dataset.save_to_disk("tmp_train")
#eval_dataset.save_to_disk("tmp_test")


#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
os.environ["WANDB_API_KEY"] = "258f2d07e6f93bd28c84ea84742840a7467b9e50"
os.environ["WANDB_PROJECT"] = "wav2vec2aphasia"
num_times = 50

sweep = wandb.controller()
sweep.configure_search(GridSearch)
sweep.configure_program("train_wav2vec2_sweep.py")

with open("configurations/sweep.yaml", 'r') as stream:
    sweepconfig = yaml.safe_load(stream)

sweep_id = wandb.sweep(sweepconfig)
wandb.agent(sweep_id, count=num_times)
