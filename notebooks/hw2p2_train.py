#!/usr/bin/env python3
"""
hw2p2_train.py: Modular ResNet-50 + ElasticFace Training Engine for CMU 11-785 HW2P2.

Can be run:
  1. Directly in Jupyter Notebook:
     import hw2p2_train
     hw2p2_train.main(epochs=10, batch_size=128)
     
  2. Via Cloud TPU Kueue runner:
     import submit_tpu
     submit_tpu.run('''
     import hw2p2_train
     hw2p2_train.main(epochs=10, batch_size=256)
     ''')
     
  3. From shell / CLI:
     python hw2p2_train.py --epochs 10 --batch-size 128
"""

import argparse
import collections
import hashlib
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tqdm.auto import tqdm


# ==============================================================================
# Model Architecture (Flax NNX)
# ==============================================================================

class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_planes, planes, kernel_size=(1, 1), strides=(1, 1),
                              padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(planes, rngs=rngs)

        self.conv2 = nnx.Conv(planes, planes, kernel_size=(3, 3), strides=(stride, stride),
                              padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(planes, rngs=rngs)

        self.conv3 = nnx.Conv(planes, planes * self.expansion, kernel_size=(1, 1), strides=(1, 1),
                              padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(planes * self.expansion, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        identity = x
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = nnx.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return nnx.relu(out + identity)


class Downsample(nnx.Module):
    def __init__(self, in_planes, out_planes, stride, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_planes, out_planes, kernel_size=(1, 1), strides=(stride, stride),
                             padding=((0, 0), (0, 0)), use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(out_planes, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))


class ResNet50Backbone(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.inplanes = 64
        self.conv1 = nnx.Conv(3, 64, kernel_size=(7, 7), strides=(2, 2),
                              padding=((3, 3), (3, 3)), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)

        self.layer1 = self._make_layer(64,  3, stride=1, rngs=rngs)
        self.layer2 = self._make_layer(128, 4, stride=2, rngs=rngs)
        self.layer3 = self._make_layer(256, 6, stride=2, rngs=rngs)
        self.layer4 = self._make_layer(512, 3, stride=2, rngs=rngs)

    def _make_layer(self, planes, blocks, stride, *, rngs):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = Downsample(self.inplanes, planes * Bottleneck.expansion, stride, rngs=rngs)
        block_list = [Bottleneck(self.inplanes, planes, stride, downsample, rngs=rngs)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            block_list.append(Bottleneck(self.inplanes, planes, rngs=rngs))
        return nnx.List(block_list)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for blk in self.layer1: x = blk(x)
        for blk in self.layer2: x = blk(x)
        for blk in self.layer3: x = blk(x)
        for blk in self.layer4: x = blk(x)
        return jnp.mean(x, axis=(1, 2))


class ElasticFace(nnx.Module):
    def __init__(self, in_features, out_features, s=40.0, m=0.3, std=0.05, *, rngs: nnx.Rngs):
        limit = float(np.sqrt(6.0 / (in_features + out_features)))
        key = rngs.params()
        w = jax.random.uniform(key, (out_features, in_features), minval=-limit, maxval=limit)
        self.weight = nnx.Param(w)
        self.s = s
        self.m = m
        self.std = std

    def __call__(self, embeddings, labels, margin_key):
        emb_n = embeddings / (jnp.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        w = self.weight.value
        w_n = w / (jnp.linalg.norm(w, axis=1, keepdims=True) + 1e-12)

        cosine = jnp.matmul(emb_n, w_n.T)
        cosine = jnp.clip(cosine, -1.0 + 1e-5, 1.0 - 1e-5)

        margin = self.m + jax.random.normal(margin_key, labels.shape) * self.std
        theta = jnp.arccos(cosine)
        one_hot = jax.nn.one_hot(labels, w.shape[0])
        target_theta = theta + margin[:, None]
        target_logits = jnp.cos(target_theta)
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine
        return logits * self.s


class Network(nnx.Module):
    def __init__(self, num_classes, embedding_dim=384, s=40.0, m=0.3, std=0.05, *, rngs: nnx.Rngs):
        self.backbone = ResNet50Backbone(rngs=rngs)
        self.embedding = nnx.Linear(2048, embedding_dim, rngs=rngs)
        self.bn_emb = nnx.BatchNorm(embedding_dim, rngs=rngs)
        self.cls_layer = ElasticFace(embedding_dim, num_classes, s=s, m=m, std=std, rngs=rngs)

    def __call__(self, x, labels=None, margin_key=None):
        x = self.backbone(x)
        feats = self.embedding(x)
        feats = self.bn_emb(feats)
        feats = feats / (jnp.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        if labels is None:
            return {"feats": feats, "out": None}
        logits = self.cls_layer(feats, labels, margin_key)
        return {"feats": feats, "out": logits}


# ==============================================================================
# Data Loading & Memory Mapping
# ==============================================================================

def _load_image(path, size):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

def _fill(out, paths, size, threads, desc, chunk=4096):
    with ThreadPoolExecutor(threads) as ex:
        for s in range(0, len(paths), chunk):
            part = paths[s:s + chunk]
            for j, arr in enumerate(ex.map(lambda p: _load_image(p, size), part)):
                out[s + j] = arr

class ImageDataset:
    def __init__(self, root, num_classes=None, image_size=112,
                 cache_dir=None, cache_name=None, build_threads=None):
        self.root = root
        self.image_paths = []
        self.labels = None
        self.classes = None
        self._cache = None

        labels_file = os.path.join(root, "labels.txt")
        images_dir = os.path.join(root, "images")

        if os.path.exists(labels_file):
            entries = []
            with open(labels_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        entries.append((parts[0], int(parts[1])))
            entries.sort(key=lambda e: (e[1], e[0]))
            all_labels = sorted({lbl for _, lbl in entries})
            selected = set(all_labels[:num_classes]) if num_classes else set(all_labels)
            label_map = {lbl: i for i, lbl in enumerate(sorted(selected))}
            labels = []
            for name, lbl in entries:
                if lbl in selected:
                    self.image_paths.append(os.path.join(images_dir, name))
                    labels.append(label_map[lbl])
            self.labels = np.asarray(labels, dtype=np.int32)
            self.classes = sorted(set(labels))
            self.num_classes = len(selected)
        else:
            self.image_paths = [os.path.join(images_dir, f)
                                for f in sorted(os.listdir(images_dir))]
            self.num_classes = None

        os.makedirs(cache_dir, exist_ok=True)
        h = hashlib.md5("\n".join(os.path.basename(p) for p in self.image_paths).encode())
        if self.labels is not None:
            h.update(self.labels.tobytes())
        n = len(self.image_paths)
        shape = (n, image_size, image_size, 3)
        self._cache_path = os.path.join(
            cache_dir, f"{cache_name or 'data'}_{image_size}_{n}_{h.hexdigest()[:8]}.npy")

        if not os.path.exists(self._cache_path):
            tmp = self._cache_path + ".tmp"
            mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=shape)
            threads = build_threads or min(16, (os.cpu_count() or 4))
            _fill(mm, self.image_paths, image_size, threads, f"Caching {cache_name}")
            mm.flush()
            del mm
            os.replace(tmp, self._cache_path)

    @property
    def cache(self):
        if self._cache is None:
            self._cache = np.load(self._cache_path, mmap_mode='r')
        return self._cache

    def __len__(self):
        return len(self.image_paths)

    def batch_dict(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        imgs = np.asarray(self.cache[idx])
        out = {'image': imgs}
        if self.labels is not None:
            out['label'] = self.labels[idx]
        return out


class JaxLoader:
    def __init__(self, dataset, batch_size, sharding, shuffle=True, seed=0):
        self.ds = dataset
        self.sharding = sharding
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_batches = len(dataset) // batch_size

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        n = len(self.ds)
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        self.epoch += 1

        for b in range(self.num_batches):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            b_dict = self.ds.batch_dict(idx)
            yield jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(self.sharding, x), b_dict)


MEAN = jnp.array([0.5, 0.5, 0.5], jnp.float32)
STD  = jnp.array([0.5, 0.5, 0.5], jnp.float32)

def normalize(x, dtype=jnp.bfloat16):
    return ((x.astype(jnp.float32) / 255.0 - MEAN) / STD).astype(dtype)

def augment(key, x, pad=8):
    B, H, W, C = x.shape
    k_flip, k_y, k_x = jax.random.split(key, 3)
    flip = jax.random.bernoulli(k_flip, 0.5, (B, 1, 1, 1))
    x = jnp.where(flip, x[:, :, ::-1, :], x)
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    oy = jax.random.randint(k_y, (B,), 0, 2 * pad + 1)
    ox = jax.random.randint(k_x, (B,), 0, 2 * pad + 1)
    crop = lambda im, y, xx: jax.lax.dynamic_slice(im, (y, xx, 0), (H, W, C))
    return jax.vmap(crop)(xp, oy, ox)


# ==============================================================================
# Training Engine
# ==============================================================================

def main(data_root="./dataset/hw2p2_data", cache_dir="./img_cache",
         checkpoint_dir="./model_checkpoints", epochs=5, batch_size=128, lr=0.03):
    
    devices = jax.devices()
    platform = devices[0].platform.lower()
    compute_dtype = jnp.bfloat16 if platform == 'tpu' else jnp.float32
    mesh = Mesh(np.asarray(devices), ('data',))
    sharding = NamedSharding(mesh, P('data'))

    print(f"=== Starting HW2P2 Training ===")
    print(f"Backend      : {platform.upper()} x{len(devices)} ({devices[0].device_kind})")
    print(f"Precision    : {compute_dtype}")
    print(f"Epochs       : {epochs}")
    print(f"Batch Size   : {batch_size}")
    print(f"Learning Rate: {lr}")

    train_ds = ImageDataset(os.path.join(data_root, "cls_data", "train"),
                            cache_dir=cache_dir, cache_name="cls_train")
    val_ds   = ImageDataset(os.path.join(data_root, "cls_data", "dev"),
                            cache_dir=cache_dir, cache_name="cls_val")

    train_loader = JaxLoader(train_ds, batch_size, sharding, shuffle=True)
    val_loader   = JaxLoader(val_ds, batch_size, sharding, shuffle=False)

    rngs = nnx.Rngs(0)
    model = Network(num_classes=train_ds.num_classes, embedding_dim=384, rngs=rngs)
    nnx.update(model, jax.device_put(nnx.state(model), NamedSharding(mesh, P())))

    total_steps = len(train_loader) * epochs
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps)
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=1e-4))

    train_key = jax.random.key(11785)

    @nnx.jit
    def train_step(model, optimizer, batch):
        nonlocal train_key
        train_key, subkey = jax.random.split(train_key)
        aug_k, margin_k = jax.random.split(subkey)
        images = normalize(augment(aug_k, batch['image']), dtype=compute_dtype)
        labels = batch['label']

        def loss_fn(model):
            outputs = model(images, labels, margin_k)
            logits = outputs["out"]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), labels)
            return loss.mean(), logits

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model)
        optimizer.update(grads)
        acc = (jnp.argmax(logits, axis=-1) == labels).mean() * 100.0
        return loss, acc

    for epoch in range(epochs):
        train_loader.set_epoch(epoch)
        t0 = time.time()
        running_loss, running_acc = 0.0, 0.0
        
        for batch in train_loader:
            loss, acc = train_step(model, optimizer, batch)
            running_loss += float(loss)
            running_acc += float(acc)

        elapsed = time.time() - t0
        avg_loss = running_loss / len(train_loader)
        avg_acc  = running_acc / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s) - Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%", flush=True)

        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({'epoch': epoch+1, 'model': nnx.state(model).to_pure_dict()}, f)

    print("=== Training Complete! ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./dataset/hw2p2_data")
    parser.add_argument("--cache-dir", default="./img_cache")
    parser.add_argument("--checkpoint-dir", default="./model_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()

    main(data_root=args.data_root, cache_dir=args.cache_dir,
         checkpoint_dir=args.checkpoint_dir, epochs=args.epochs,
         batch_size=args.batch_size, lr=args.lr)
