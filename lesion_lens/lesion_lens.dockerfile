#FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
#FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
#    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# try : replace opencv-python by opencv-python-headless (and remove ffmpeg/libsm6/libxext6 ?)
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
      python3.11 \
      python3-pip \
      python3.11-distutils \
      # libgl1 \
      ffmpeg \
      libsm6 \
      libxext6 \
#      nvidia-driver-555 \
      && \
    ln -s python3.11 /usr/bin/python && \
    python -m pip install --upgrade \
        poetry==1.6.1 && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /src && \
    mkdir /model && \
    mkdir /input && \
    mkdir /output && \
    mkdir /opt/ml && \
    mkdir /opt/ml/model && \
    export PYTHONPATH=$PYTHONPATH:/code/deepdat_ssrl

#COPY model /model

COPY pyproject.toml poetry.lock /src/
#COPY pyproject.toml /src/

WORKDIR /src
RUN poetry lock --no-update && \
    poetry install --without dev --no-root && \
    rm -rf $POETRY_CACHE_DIR
#RUN poetry lock && \
#    poetry install --without dev --no-root && \
#    rm -rf $POETRY_CACHE_DIR

RUN groupadd -r grandchallenge &&  \
    useradd --no-log-init -r -g grandchallenge grandchallenge
USER grandchallenge

COPY src .

# For I/O with docker run, use:
# -v host/path:container/path
ENTRYPOINT ["poetry", "run", "python", "/src/subream_inference.py"]
