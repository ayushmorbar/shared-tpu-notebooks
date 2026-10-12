# Image every student TPU job runs. Built and pushed by scripts/01_build_image.sh.
#
# The base is Google's JAX TPU image so libtpu, jax, and flax match what the TPU nodes
# expect. Course libraries go on top without letting pip move those versions.
ARG BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest
FROM ${BASE_IMAGE}

USER root
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /opt/tpu/requirements.txt
RUN python3 -m pip freeze | grep -iE '^(jax|jaxlib|libtpu|flax|optax|orbax-checkpoint|numpy|ml-dtypes)==' \
        > /opt/tpu/constraints.txt \
    && python3 -m pip install -c /opt/tpu/constraints.txt \
        --index-url https://download.pytorch.org/whl/cpu torch \
    && python3 -m pip install -c /opt/tpu/constraints.txt -r /opt/tpu/requirements.txt \
    && python3 -c "import jax, flax, torch, tokenizers, Levenshtein, jiwer, wandb, google.cloud.storage; print('jax', jax.__version__, 'flax', flax.__version__, 'torch', torch.__version__)"

COPY tpu_entrypoint.py /opt/tpu/tpu_entrypoint.py
RUN (id -u 1000 >/dev/null 2>&1 || useradd --uid 1000 --create-home --shell /bin/bash student) \
    && chmod 0755 /opt/tpu/tpu_entrypoint.py

USER 1000
WORKDIR /work
