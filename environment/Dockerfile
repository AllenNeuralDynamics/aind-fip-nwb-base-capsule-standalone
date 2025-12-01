# hash:sha256:c706f1b99750957e9a95b8e1083b6488537af707c0db7e7ac2483d26a90e1491
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.2 \
    aind-data-schema==2.0.1 \
    aind-log-utils==0.2.4 \
    aind-nwb-utils \
    aind-qcportal-schema==0.4.0 \
    harp-python==0.4.1 \
    hdmf-zarr==0.11.1 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.3.1 \
    pynwb==3.0.0 \
    python-dateutil==2.9.0.post0 \
    regex==2024.11.6 \
    scipy==1.13.1 \
    uuid==1.30

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
