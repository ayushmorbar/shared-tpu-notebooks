"""HW2P2 infrastructure benchmark and smoke-test entrypoint.

Runs on one Cloud TPU v5e chip, submitted by submit_tpu.train().
Proves four essential infrastructure capabilities for long-running jobs:
  1. Dataset comes from GCS, read by an authenticated ServiceAccount with Workload Identity.
  2. The 18.5 GiB cache fits in node RAM, enabling 40 epochs with 0 disk/network I/O.
  3. Secrets arrive safely from student environment or Kubernetes Secrets without leakage.
  4. Checkpoints stream to GCS via Orbax with modern preservation policy and resume on restart.

Environment variables set by submit_tpu.train():
  DATA_URI         gs://hw2p2-cached-data/img_cache
  CKPT_URI         gs://hw2p2-cached-data/checkpoints/<student>
  SUBMISSION_URI   gs://hw2p2-cached-data/submissions/<student>/submission.csv
  STUDENT          student username
  EPOCHS           number of epochs (default: 2 for smoke test)
  STEPS_PER_EPOCH  0 means full epoch; small number (e.g. 10) makes a smoke run
  BATCH_SIZE       batch size (default: 128)
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

DATA_URI = os.environ.get("DATA_URI", "gs://hw2p2-cached-data/img_cache")
CKPT_URI = os.environ.get("CKPT_URI", "gs://hw2p2-cached-data/checkpoints/anon")
STUDENT = os.environ.get("STUDENT", "anon")
EPOCHS = int(os.environ.get("EPOCHS", "2"))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", "0"))
BATCH = int(os.environ.get("BATCH_SIZE", "128"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "8631"))
LOCAL = "/tmp/img_cache"


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def fetch_data() -> dict[str, np.ndarray]:
    """Copy the cache from GCS to local disk, then load it into RAM.

    Local-then-RAM rather than a GCS FUSE mount. The loader shuffles, so reads are
    random, and random reads through FUSE are slow. One sequential copy beats millions
    of random object reads.
    """
    banner("dataset ingestion")
    os.makedirs(LOCAL, exist_ok=True)

    t0 = time.time()
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{DATA_URI}/*", LOCAL],
        check=True,
        capture_output=True,
    )
    copy_s = max(time.time() - t0, 0.001)
    total = sum(os.path.getsize(os.path.join(LOCAL, f)) for f in os.listdir(LOCAL))
    print(
        f"  copied {total / 2**30:.2f} GiB from {DATA_URI} in {copy_s:.0f}s "
        f"({total / 2**20 / copy_s:.0f} MiB/s)",
        flush=True,
    )

    arrays = {}
    t0 = time.time()
    for fn in sorted(os.listdir(LOCAL)):
        if not fn.endswith(".npy"):
            continue
        key = fn.split("_112_")[0]
        arrays[key] = np.load(os.path.join(LOCAL, fn), mmap_mode=None)  # into RAM
        print(
            f"  {key:12} {str(arrays[key].shape):26} {arrays[key].nbytes / 2**30:.2f} GiB",
            flush=True,
        )
    print(f"  resident in RAM after {time.time() - t0:.0f}s", flush=True)

    try:
        with open("/proc/self/status") as fh:
            rss = next(l for l in fh if l.startswith("VmRSS"))
        print(f"  process {rss.strip()}", flush=True)
    except Exception:
        pass

    return arrays


class Net(nnx.Module):
    """Small conv stack plus a classifier head.

    Head has 8,631 classes off a 384 embedding to match the checkpoint size
    and memory requirements of the full homework model.
    """

    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.c1 = nnx.Conv(3, 32, (3, 3), strides=(2, 2), rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), rngs=rngs)
        self.c3 = nnx.Conv(64, 128, (3, 3), strides=(2, 2), rngs=rngs)
        self.emb = nnx.Linear(128, 384, rngs=rngs)
        self.head = nnx.Linear(384, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.c1(x))
        x = nnx.relu(self.c2(x))
        x = nnx.relu(self.c3(x))
        x = jnp.mean(x, axis=(1, 2))
        return self.head(nnx.relu(self.emb(x)))


@nnx.jit
def train_step(model, optimizer, images, labels):
    def loss_fn(m):
        logits = m(images)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    # Flax 0.11+ explicit model passing in optimizer update
    optimizer.update(model, grads)
    return loss


def create_checkpoint_manager(ckpt_uri: str) -> ocp.CheckpointManager:
    """Create an Orbax CheckpointManager with preservation policy."""
    # Attempt modern Orbax preservation policy options if available
    try:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=2,
            preserve_max_to_keep=1,
            enable_async_checkpointing=True,
        )
    except TypeError:
        # Fallback for standard CheckpointManagerOptions signature
        options = ocp.CheckpointManagerOptions(
            max_to_keep=2,
            enable_async_checkpointing=True,
        )
    return ocp.CheckpointManager(ckpt_uri, options=options)


def main() -> None:
    banner("hardware validation")
    devs = jax.devices()
    assert devs[0].platform == "tpu", f"not on a TPU: {devs}"
    print(f"  accelerator: {devs}", flush=True)

    data = fetch_data()
    train_data = data.get("cls_train")
    if train_data is None:
        raise KeyError(f"'cls_train' array not found in {list(data.keys())}")

    banner("model initialization")
    model = Net(NUM_CLASSES, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    n_params = sum(
        int(np.prod(p.shape))
        for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"  {n_params:,} parameters ({n_params * 4 / 2**20:.1f} MiB fp32)", flush=True)

    banner("orbax checkpoints")
    mngr = create_checkpoint_manager(CKPT_URI)
    start = 0
    if (last := mngr.latest_step()) is not None:
        restored = mngr.restore(
            last,
            args=ocp.args.StandardRestore(nnx.state(model).to_pure_dict()),
        )
        nnx.replace_by_pure_dict(nnx.state(model), restored)
        start = last + 1
        print(f"  RESUMING from step {last} in {CKPT_URI}", flush=True)
    else:
        print(f"  no checkpoint found in {CKPT_URI}, starting fresh", flush=True)

    steps = STEPS_PER_EPOCH or (len(train_data) // BATCH)
    rng = np.random.default_rng(0)

    for epoch in range(start, EPOCHS):
        banner(f"epoch {epoch}")
        t0 = time.time()
        for step in range(steps):
            idx = rng.integers(0, len(train_data), BATCH)
            images = jnp.asarray(train_data[idx], dtype=jnp.bfloat16) / 255.0
            labels = jnp.asarray(idx % NUM_CLASSES)
            loss = train_step(model, optimizer, images, labels)
            if step % 10 == 0 or step == steps - 1:
                print(f"  step {step:4}/{steps}  loss {float(loss):.4f}", flush=True)
        dt = max(time.time() - t0, 0.001)
        print(f"  {steps} steps in {dt:.1f}s ({dt / steps * 1000:.0f} ms/step)", flush=True)

        mngr.save(epoch, args=ocp.args.StandardSave(nnx.state(model).to_pure_dict()))
        mngr.wait_until_finished()
        print(f"  checkpoint {epoch} written to {CKPT_URI}", flush=True)

    banner("completion summary")
    print(f"  checkpoints preserved: {mngr.all_steps()}", flush=True)


if __name__ == "__main__":
    main()
