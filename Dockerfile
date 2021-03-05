FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
#FROM wav2letter/wav2letter:cuda-e7c4d17
RUN apt-get update && apt-get -y install \
     apt-utils \
    build-essential \
    cmake \
    gcc \
    git \
    libbz2-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libboost-thread-dev \
    libeigen3-dev \
    libgflags-dev \
    libfftw3-dev \
    libgoogle-glog-dev \
    liblzma-dev \
    libpq-dev \
    libopenblas-dev \
    libsndfile-dev \
    libsndfile1-dev \
    nano \
    python3-pip \
    swig \
    sox \
    wget \
    zlib1g-dev

RUN apt-get install -y language-pack-en && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.utf8
ENV LC_ALL='en_US.utf8'


RUN pip3 install --upgrade pip
RUN pip3 install soundfile editdistance packaging
#RUN pip3 install https://github.com/kpu/kenlm/archive/master.zip



###############################
###   TORCH, APEX TENSORRT  ###
###############################
RUN pip3 install \
    install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


# De momento prescindimos de APEX
#WORKDIR /tmp/apex
#RUN git clone https://github.com/NVIDIA/apex.git && \
#    cd apex && \
#    pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#    --global-option="--fast_multihead_attn" .


################################
# INSTALL FAIRSEQ ##############
################################
WORKDIR /workspace/fairseq
RUN git clone https://github.com/pytorch/fairseq && \
    cd fairseq && \
    pip3 install --editable ./
    #&&\
    #cp /home/VICOMTECH/igonzalez/WAV2VEC2/wav2vec2/recognize.py examples/wav2vec/


##################################
## KENLM #########################
#################################
WORKDIR /workspace/external_lib

RUN git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j 16

ENV KENLM_ROOT_DIR=/workspace/external_lib/kenlm/

# Install wav2letter
WORKDIR /workspace/wav2letter
RUN git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git && \
    cd wav2letter/bindings/python && \
    pip3 install -e .

WORKDIR /workspace/fairseq/fairseq
COPY . .