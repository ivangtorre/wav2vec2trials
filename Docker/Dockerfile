FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   libsndfile1 \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN apt update && \
    apt install -y language-pack-en && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.utf8
ENV LC_ALL='en_US.utf8'

#RUN pip3 install torchaudio torchvision==0.9.1+cu111
#RUN pip3 install torch==1.8.1+cu112 torchaudio==0.8.1 torchtext==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir jupyter

RUN git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    pip3 install -e .

RUN pip3 install datasets && \
    pip3 install soundfile && \
    pip3 install jiwer && \
    pip3 install lang-trans && \
    pip3 install librosa && \
#    pip3 install wandb && \
    pip3 install unidecode && \
    pip3 install soundfile

WORKDIR /workspace/wav2vec2

CMD ["/bin/bash"]


#
#
##################################################3
##FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
##LABEL maintainer="Hugging Face"
##LABEL repository="transformers"
#
#RUN apt update && \
#    apt install -y bash \
#                   build-essential \
#                   git \
#                   curl \
#                   ca-certificates \
#                   python3 \
#                   python3-pip && \
#    rm -rf /var/lib/apt/lists
#
#RUN python3 -m pip install --no-cache-dir --upgrade pip && \
#    python3 -m pip install --no-cache-dir \
#    mkl \
#    torch
#
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && \
#    python3 setup.py install && \
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#
#WORKDIR /workspace
#COPY . transformers/
#RUN cd transformers/ && \
#    python3 -m pip install --no-cache-dir .
#
#CMD ["/bin/bash"]
#
#
#
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#
#RUN apt update && \
#    apt install -y bash \
#                   build-essential \
#                   git \
#                   curl \
#                   ca-certificates \
#                   libsndfile1 \
#                   python3 \
#                   python3-pip && \
#    rm -rf /var/lib/apt/lists
#
#RUN apt update && \
#    apt install -y language-pack-en && \
#    locale-gen en_US.UTF-8
#
#ENV LANG=en_US.utf8
#ENV LC_ALL='en_US.utf8'
#
#RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#
#WORKDIR /workspace
#RUN python3 -m pip install --no-cache-dir --upgrade pip && \
#    python3 -m pip install --no-cache-dir jupyter
#
#RUN git clone https://github.com/huggingface/transformers.git && \
#    cd transformers && \
#    pip3 install -e .
#
#RUN pip3 install datasets && \
#    pip3 install soundfile && \
#    pip3 install jiwer==2.2.0 && \
#    pip3 install lang-trans==0.6.0 && \
#    pip3 install librosa==0.8.0
#
#WORKDIR /workspace/wav2vec2
#
#CMD ["/bin/bash"]

