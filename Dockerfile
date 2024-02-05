FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
    && apt-get install -y build-essential curl \
    && apt-get install -y git-all \
    && apt-get install -y ffmpeg \
    && apt-get install -y sox libsox-dev python3-dev python3-pip python3-distutils \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY /cache /cache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --upgrade pip --no-cache-dir && python3 -m pip install -r requirements.txt