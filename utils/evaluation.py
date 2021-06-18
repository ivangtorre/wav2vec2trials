import torch
import torchaudio
from typing import Optional
import pandas as pd
from dataclasses import dataclass, field
from transformers import HfArgumentParser, Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, load_metric

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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: str = field(metadata={"help": "Path to pretrained model"})
    cache_dir: str = field(metadata={"help": "where audios are stored"})
    test_mild: str = field(metadata={"help": "test file csv"})
    test_moderate: str = field(metadata={"help": "test file csv"})
    test_severe: str = field(metadata={"help": "test file csv"})
    test_vsevere: str = field(metadata={"help": "test file csv"})


def main():
    parser = HfArgumentParser(ModelArguments)
    args = parser.parse_args_into_dataclasses()

    # LOAD DATA
    mild_dataset = load_test(args.test_mild, args)
    moderate_dataset = load_test(args.test_mild, args)
    severe_dataset = load_test(args.test_mild, args)
    vsevere_dataset = load_test(args.test_mild, args)

    # LOAD MODEL
    wer = load_metric("wer")
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
    model.to("cuda")

    mild_dataset = mild_dataset.map(speech_file_to_array_fn, remove_columns=mild_dataset.column_names)
    moderate_dataset = moderate_dataset.map(speech_file_to_array_fn, remove_columns=moderate_dataset.column_names)
    severe_dataset = severe_dataset.map(speech_file_to_array_fn, remove_columns=severe_dataset.column_names)
    vsevere_dataset = vsevere_dataset.map(speech_file_to_array_fn, remove_columns=vsevere_dataset.column_names)

    # EVALUATE
    def evaluate(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
            pred_ids = torch.argmax(logits, dim=-1)  # GREEDY
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

    result = mild_dataset.map(evaluate, batched=True, batch_size=8)

    print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["target_text"])))
