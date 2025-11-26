# hash:sha256:39556274b2d1a9880a80653664f41f991fab12a55bf9b589fb27181b9aefd82a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.2 \
    aind-data-schema==2.0.1 \
    aind-log-utils==0.2.4 \
    aind-nwb-utils==0.1.2 \
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
