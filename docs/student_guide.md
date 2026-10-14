# 🎓 Student Guide: Training on the Cloud TPU Hub

Welcome to the **CMU 11-785 Cloud TPU Hub**! This guide is designed to help you navigate the environment, set up your private credentials, run heavy training jobs on Cloud TPUs, and submit your work to Kaggle and Gradescope.

---

## 1. How Your Environment Works

To keep the classroom affordable and fast for hundreds of students, we separate your notebook editor from the Cloud TPU hardware:

```
+------------------------------------+          +------------------------------------+
|  Your JupyterLab Notebook Pod      |          |  Dedicated Cloud TPU v5e Pod       |
|  - CPU only (2 vCPU, 16 GiB RAM)   |          |  - Physical Cloud TPU v5e (197 TF) |
|  - 32 GiB Persistent Home Volume   |          |  - 47 GiB Physical RAM             |
|  - Location: /home/jovyan/         |          |  - High-Speed NVMe Local Scratch   |
|                                    |          |                                    |
|  * Edit code & visualize data      | dispatch |  * Copies dataset in ~90s to RAM   |
|  * Configure private credentials   | -------->|  * Trains with 0 disk/network I/O  |
|  * Package Gradescope submissions  |          |  * Streams checkpoints to GCS      |
+------------------------------------+          +------------------------------------+
                  ▲                                                │
                  │              Job Completed                     │
                  └────────────────────────────────────────────────┘
                     Automatically pulls submission.csv & checkpoints
                     back to your private /home/jovyan/ directory!
```

1. **Your Notebook Pod (`/home/jovyan`)**: You have a 32 GiB private persistent disk that stays with you all semester. Your code, personal keys (`.env`), and evaluated checkpoints live here.
2. **Central Tartan Course Store (`/tartan/11785/` and `/ocean/`)**: Read-only shared storage hosting course datasets (`/tartan/11785/datasets/`), starter notebooks (`/tartan/11785/assignments/`), and the shared live Python environment.
3. **Live Shared Environment**: Select the **`Python 3.11 (CMU 11-785 TPU Shared Env)`** kernel from the top-right kernel dropdown. All required packages (`flax`, `jax`, `torch`, `augmax`, `timm`, `orbax`) are pre-installed and updated live by course staff.
4. **The Cloud TPU Pod**: When you call `submit_tpu.train("hw2p2_train.py", ...)`, your script is submitted to a fair-share queue managed by **Kueue**. As soon as a Cloud TPU v5e chip is available, your pod runs, streams its logs directly to your notebook output, and shuts down immediately when finished so other students can train.

---

## 2. Setting Up Your Personal Credentials (Zero Sharing)

Your credentials (WandB, Kaggle, Hugging Face) are **strictly personal** and are never shared with classmates.

### Step 1: Copy the Template
In your notebook or terminal, copy the template into your persistent home directory:
```bash
cp notebooks/student.env.template /home/jovyan/.env
```

### Step 2: Fill In Your Keys
Open `/home/jovyan/.env` in the JupyterLab file browser and fill in your keys:
```env
# Weights & Biases (https://wandb.ai/authorize)
WANDB_API_KEY="your_personal_wandb_key"

# Kaggle API (https://www.kaggle.com/settings -> Create New Token)
KAGGLE_USERNAME="your_kaggle_username"
KAGGLE_API_TOKEN="your_kaggle_api_token"

# Hugging Face (Optional: https://huggingface.co/settings/tokens)
HF_TOKEN="your_hf_token"

# You can also add ANY custom secret you need!
# COMET_API_KEY="your_key"
# NEPTUNE_API_TOKEN="your_token"
```

### Step 3: Privacy Guarantee
- `/home/jovyan/.env` is stored **only on your private disk**.
- It is **never committed to Git**.
- When `submit_tpu.train()` runs, it injects these keys into your TPU pod.
- The logs print only masked verification:
  ```
  [Secret] WANDB_API_KEY           : present, len=40, sha256[:8]=9f8a2b1c
  [Secret] KAGGLE_USERNAME         : present, len=12, sha256[:8]=3a7b11ef
  ```
  Your cleartext keys are never visible in logs or to other students.

---

## 3. Training on the Cloud TPU v5e

To launch training, run this cell in your notebook:

```python
import submit_tpu

# Submits your script to the Cloud TPU queue and streams output live
logs = submit_tpu.train(
    "hw2p2_train.py",
    epochs=40,
    download_submission=True,           # auto-downloads submission.csv
    local_submission_path="/home/jovyan/submission.csv",
)
```

### What Happens Under the Hood:
1. **Queueing**: Your job enters the Kueue queue `tpu`. If a warm chip is ready, it starts in seconds. If the pool is spinning up a new node, you will see 30-second live status heartbeats (`[Queue Update]`, `[Node Build]`).
2. **Fast Cache Copy (78–117s)**: The TPU pod copies the pre-processed 18.5 GiB `.npy` dataset from Google Cloud Storage to local scratch at ~200 MiB/s.
3. **In-RAM Training**: The dataset is loaded into the TPU node's 47 GiB RAM (`VmRSS ~24 GB`), allowing 40 epochs to run with **zero disk or network I/O**!
4. **Orbax Checkpoints**: Checkpoints stream to GCS after each epoch. If your job is ever interrupted, re-running it automatically **resumes from the latest epoch**.
5. **Userspace Delivery**: When training finishes, `submission.csv` is automatically copied into `/home/jovyan/submission.csv` on your notebook disk!

---

## 4. Memory Management: CPU Pod vs. TPU Node

> ⚠️ **CRITICAL RULE:**
> - **On your CPU Notebook Pod**: You have a 16 GiB memory limit. If you load an 18.5 GiB dataset into RAM, the Linux kernel will kill your notebook pod (`OOMKilled`). **Always use `mmap_mode='r'` when exploring data in notebook cells.**
> - **On the Cloud TPU Pod**: The node has 47 GiB physical RAM. `hw2p2_train.py` safely loads data directly into RAM (`mmap_mode=None`) because it runs on the TPU node, not your notebook pod!

---

## 5. Submitting Your Homework

### 1. Kaggle Leaderboard Submission
Once `submission.csv` is in `/home/jovyan/`:
- **Via Kaggle CLI in notebook:**
  ```python
  !kaggle competitions submit -c hw-2-p-2-spring-2026-sandbox-testing -f /home/jovyan/submission.csv -m "ResNet-34 ElasticFace"
  ```
- **Or Manual Download:** Right-click `/home/jovyan/submission.csv` in the JupyterLab file browser on the left, click **Download**, and upload to the Kaggle competition page.

### 2. Gradescope Submission Packaging
Run the final cell in `F26_HW2P2_JAX_TPU_Hub.ipynb`. It will automatically:
1. Verify that your model parameters are strictly **$< 30,000,000$**.
2. Package `HW2P2_final_submission.zip` containing:
   - `submission.csv`
   - Your exported notebook
   - Architecture metadata
3. Right-click `HW2P2_final_submission.zip` → **Download**, and upload to **Gradescope**!

---

## 6. Frequently Asked Questions & Troubleshooting

### Q: "My job is waiting in queue. Did it freeze?"
**A:** No! A cold Cloud TPU node takes 2–4 minutes (and up to 16 minutes in peak times) for Google Cloud to provision. `submit_tpu` prints periodic heartbeats:
```
[Queue Update] Waiting in queue 'tpu' (60s elapsed)...
[Node Build] TPU node provisioning in progress (120s elapsed)...
```
As soon as the chip is online, your training logs will stream live.

### Q: "Can I add custom secrets for other libraries?"
**A:** Yes! Any key you add to `/home/jovyan/.env` (e.g., `COMET_API_KEY="xyz"`) or pass via `extra_env={"COMET_API_KEY": "xyz"}` in `submit_tpu.train()` is automatically forwarded into your TPU pod.

### Q: "Can I run a quick 2-minute smoke test before a 40-epoch run?"
**A:** Yes! Run:
```python
submit_tpu.train("train_hw2p2.py", epochs=2, steps_per_epoch=10)
```
This tests the full hardware pipeline, secret read, and checkpoint save in under 2 minutes.
