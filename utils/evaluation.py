import torch
import torchaudio
from typing import Optional
import pandas as pd
import os
from transformers import HfArgumentParser, Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, load_metric
import argparse
import numpy as np


def load_test(path, args):
    df = pd.read_csv(path, delimiter=',')
    df = df[~df["transcription"].isnull()]

    eliminar = pd.read_csv("eliminar.csv", ",", header=None)[0].values.tolist()
    for item in eliminar:
        df = df[~df["file_cut"].str.contains(item)]

    df = df.reset_index(drop=True)

    df = df[["transcription", "file_cut"]]
    df.columns = ["sentence", "path"]
    df["path"] = args.cache_dir + df["path"]
    return Dataset.from_pandas(df)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--model_path", type=str, required=True, help='Path to pretrained model')
    parser.add_argument("--cache_dir", type=str, required=True, help='where audios are stored')
    parser.add_argument("--test_mild", type=str, required=True, help='test file csv')
    parser.add_argument("--test_moderate", type=str, required=True, help='test file csv')
    parser.add_argument("--test_severe", type=str, required=True, help='test file csv')
    parser.add_argument("--test_vsevere", type=str, required=True, help='test file csv')
    parser.add_argument("--num_proc", default=1, type=int, required=True, help='Number of processors')
    return parser.parse_args()


def main(args):
    # LOAD DATA
    #mild_dataset = load_test(args.test_mild, args)
    #moderate_dataset = load_test(args.test_moderate, args)
    #severe_dataset = load_test(args.test_severe, args)
    vsevere_dataset = load_test(args.test_vsevere, args)

    # LOAD MODEL
    wer = load_metric("wer")
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
    model.to("cuda")

    #mild_dataset = mild_dataset.map(speech_file_to_array_fn, remove_columns=mild_dataset.column_names, num_proc=args.num_proc)
    #moderate_dataset = moderate_dataset.map(speech_file_to_array_fn, remove_columns=moderate_dataset.column_names, num_proc=args.num_proc)
    #severe_dataset = severe_dataset.map(speech_file_to_array_fn, remove_columns=severe_dataset.column_names, num_proc=args.num_proc)
    vsevere_dataset = vsevere_dataset.map(speech_file_to_array_fn, remove_columns=vsevere_dataset.column_names, num_proc=args.num_proc)

    # EVALUATE
    def evaluate(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
            #pred_ids = torch.argmax(logits, dim=-1)  # GREEDY
            pred_ids = np.argmax(logits, axis=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)



        return batch

    result = test_dataset.map(evaluate, batched=True, batch_size=8)

    #result_mild = mild_dataset.map(evaluate, batched=True, batch_size=8)
    #result_moderate = moderate_dataset.map(evaluate, batched=True, batch_size=8)
    #result_severe = severe_dataset.map(evaluate, batched=True, batch_size=8)
    result_vsevere = vsevere_dataset.map(evaluate, batched=True, batch_size=8)

    print("************************")
    print("MILD TEST:")
    #print("WER: {:2f}".format(100 * wer.compute(predictions=result_mild["pred_strings"], references=result_mild["target_text"])))
    print("")
    print("MODERATE TEST:")
    #print("WER: {:2f}".format(100 * wer.compute(predictions=result_moderate["pred_strings"], references=result_moderate["target_text"])))
    print("")
    print("SEVERE TEST:")
    #print("WER: {:2f}".format(100 * wer.compute(predictions=result_severe["pred_strings"], references=result_severe["target_text"])))
    print("")
    print("VERY SEVERE TEST:")
    print("WER: {:2f}".format(100 * wer.compute(predictions=result_vsevere["pred_strings"], references=result_vsevere["target_text"])))
    print("************************")

    print(result_vsevere["pred_strings"])
    print(result_vsevere["target_text"])


if __name__ == "__main__":
    args = parse_args()
    main(args)