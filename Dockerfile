FROM ubuntu:20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y python3-dev wget libgl1-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda update -n base -c defaults conda

RUN conda create -y -n "vsm" python=3.8.11

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate vsm \
    && pip install -r requirements.txt"