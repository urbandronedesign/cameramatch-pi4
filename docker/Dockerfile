# Pi4 inference server for CameraMatch (Blender add-on backend).
# Upstream pins torch 2.12.1+cu126, which has no kernels for Blackwell (RTX 50xx, sm_120),
# so we use the cu130 build of the same torch version.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 (the version upstream validated) via uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv venv /opt/venv --python 3.11
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN uv pip install "torch==2.12.1" --index-url ${TORCH_INDEX} \
    && uv pip install setuptools wheel

# Pi4 pinned to a known commit (repo is very young; avoid surprise breakage).
ARG PI4_COMMIT=32dfbea9ce6268ea17446208a804710df9475c32
RUN git clone https://github.com/yyfz/Pi4 /opt/Pi4 \
    && git -C /opt/Pi4 checkout ${PI4_COMMIT}
RUN cd /opt/Pi4 && uv pip install --no-build-isolation -e ".[example]"
# Phase 2 refinement (bundle adjustment).
RUN uv pip install "scipy>=1.14"
# OneFormer (ADE20K) semantic masks: trees / water / sky excluded from camera tracking (server/semantic.py)
RUN uv pip install "transformers==5.23.0"

COPY server /app
WORKDIR /app
EXPOSE 8765
CMD ["python", "/app/server.py"]
