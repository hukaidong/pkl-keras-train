# syntax=docker/dockerfile:1.4

FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<-EOT
    pip install --upgrade --user pip
EOT

COPY requirements /pkl-keras-train/requirements
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<-EOT
    pip install -r /pkl-keras-train/requirements/requirements.txt
    pip install -r /pkl-keras-train/requirements/development.txt
EOT
