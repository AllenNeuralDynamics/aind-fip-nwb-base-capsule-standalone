# hash:sha256:4868ca6b3e3c836c474f47ff9948de19c9185e34245f60dfb174ace8267db8bc
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.2 \
    aind-data-schema==1.2.0 \
    harp-python==0.4.1 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.2.1 \
    pynwb \
    scipy \
    aind-log-utils \
    hdmf-zarr==0.11.3 \
    python-dateutil==2.9.0.post0 \
    regex==2024.11.6 \
    scipy==1.13.1 \
    uuid==1.30 \
    aind-nwb-utils
