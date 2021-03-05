# DOCKER BUILD
First build docker
```bash
sudo docker build -t  wav2vec2 .
```

Then create the following emtpy folders using any route you want:
* /data/TMP_IVAN/wav2vec2/datasets
* /data/TMP_IVAN/wav2vec2/models
* /data/TMP_IVAN/wav2vec2/results

Create container:
```bash
sudo NV_GPU='5' nvidia-docker run -it -d --rm --name "wav2vec2_5" --runtime=nvidia --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/wav2vec2/ -v /data/TMP_IVAN/wav2vec2/datasets:/datasets/  -v /data/TMP_IVAN/wav2vec2/results:/results/   -v /data/TMP_IVAN/wav2vec2/models:/models/ -v /data/:/data/:ro wav2vec2
```

# INFERENCE:
Execute the container:
```bash
sudo nvidia-docker exec -it wav2vec2_5 bash
```

## First download pretrained model
If you do not have any checkpoint, dowload it to the folder:
```bash
cd /models && \
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt && \
cd /workspace/fairseq/fairseq/
```

## Execute inference
```bash
python3 examples/speech_recognition/infer.py /workspace/wav2vec2/examples/inference \
        --task audio_pretraining \
        --nbest 1 \
        --path /models/wav2vec_small_960h.pt \
        --gen-subset dev_clean \
        --results-path /results \
        --w2l-decoder viterbi \
        --word-score -1 \
        --sil-weight 0 \
        --criterion ctc \
        --labels ltr \
        --max-tokens 4000000 \
        --post-process letter
```

Results are stored in /results


# PREPARE DATA:
```bash
python3 examples/wav2vec/wav2vec_manifest.py /datasets/LibriSpeech/dev-clean/1272/128104 \
        --dest /datasets/ \
        --ext flac \
        --valid-percent 0.2
```