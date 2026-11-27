# Magnolia HPC Cluster: Working Guide

A complete, project-independent reference for running GPU/CPU work on **Magnolia**, the SLURM cluster at the
University of Southern Mississippi (USM). It is written for two readers:

- **Me**: which command to use for what, and why.
- **A future Claude session in a different project**: everything needed to work on this cluster without being
  re-briefed. *(Claude: read this whole file before writing job scripts or setup commands. Facts marked
  **verified** were tested on the cluster; anything marked **untested** is standard SLURM/Linux that hasn't been
  tried here yet.)*

Learned while setting up and training a YOLO (PyTorch/Ultralytics) project, November 2026. The raw recovered wiki
pages and the original survey outputs live in `Magnolia_HPC.md` (the pneumonia-detection repo); this file distills them.

> **Copying this to a new project:** drop this file into the new repo (or point Claude at it) and add one line
> to that project's `CLAUDE.md`: *"Cluster work: see MAGNOLIA_CLUSTER_GUIDE.md."* Replace `<username>` with the
> real username only locally, never in a public repo.

---

## 1. Quick facts (TL;DR)

| Thing | Value | Status |
|---|---|---|
| Login | `ssh <username>@magnolia.usm.edu` (password auth) | verified |
| Scheduler | SLURM (`sbatch`, `srun`, `squeue`, `scancel`, `sinfo`, `scontrol`) | verified |
| **Usable GPUs** | **2× Tesla P100-PCIE-12GB** on node `gpu001`, partition `gpu` | verified |
| GPU request syntax | **`--partition=gpu --gres=gpu:p100:1`**: always say `p100` | verified |
| Unusable GPUs | 4× Tesla K80 on `gpu002` (too old for modern PyTorch; driver 470) | verified |
| A100s | 6× A100 80GB, partition `suliaoma`, **owner-only** (QoS `bxmarg`) | verified (no access) |
| OS on compute/login nodes | **CentOS 7, glibc 2.17** | verified |
| GPU driver (gpu001) | 550.54.14, CUDA 12.4 | verified |
| Newest PyTorch that installs | **torch 2.5.1 + torchvision 0.20.1 (cu124)** (torch ≥ 2.6 needs glibc 2.28) | verified |
| Python | `module load python/3.11.5` (default). No conda module. | verified |
| Internet on compute nodes | **Yes** (pip, Kaggle, W&B online, GitHub all work from jobs) | verified |
| Home | `/homes/01/<username>`, **filesystem 99% full** (shared, no per-user quota) | verified |
| Scratch | `/scratch` (Lustre, ~11 TB free), but **no personal folder** (admin must create one) | verified |
| Default job time limit | **5 h** on most partitions; always set `--time` | verified |
| Max job time (`gpu`) | unlimited (our scripts use `2-00:00:00`) | verified |
| `sacct` | **broken** ("Connection refused"; accounting DB down). Read job logs instead. | verified |
| Python HTTPS | **broken by default** (`CERTIFICATE_VERIFY_FAILED`); fix with `export SSL_CERT_FILE="$(python -m certifi)"` | verified |
| Admin / contact | Dr. Brian G. Olson, brian.olson@usm.edu | from wiki |

---

## 2. Hardware and partitions

### What you can actually use

| Partition | Nodes | What's there | Access | Default / max time | Default mem per CPU |
|---|---|---|---|---|---|
| **`gpu`** | `gpu001` | **2× P100 12 GB**, 20 CPU cores, ~128 GB RAM | everyone | 5 h / unlimited | 6000 MB |
| `gpu` | `gpu002` | 4× K80 (**don't use**) | everyone | 5 h / unlimited | 6000 MB |
| **`node`** *(default)* | `node[001-076]` | 20 cores, ~120–128 GB RAM each, no GPU | everyone | 5 h / unlimited | 6000 MB |
| `himem` | `himem[001-008]` | 512 GB – 2 TB RAM nodes, no GPU | everyone | 5 h / unlimited | 25000 MB |
| `lomem` | 11 nodes | 16 cores, 64 GB | everyone | 2 h / **1 day** | 1000 MB |
| `float` | mix | — | QoS `float` only | 1 h / 5 days | 1500 MB |

### What you can't use (owner partitions)

| Partition | Hardware | QoS required |
|---|---|---|
| `suliaoma` | `gpu[003-005]`: **6× A100 80GB PCIe**, 256 GB RAM, Rocky Linux 9 | `bxmarg` |
| `kisame` | 20 big CPU nodes (64 cores, 512 GB) | `msbroms` |
| `egg` | 11 CPU nodes | `bxmarg` |
| `systodo`, `syscheck` | test nodes | admins |

These become open to everyone only if the admin enables *preemption partitions* (on the 2024 roadmap; not done).
Access was requested from the admin 2026-11-14; no reply as of 2026-11-27.

### The GPU node in practice

- `gpu001` has **2 P100s and 20 cores shared by everyone** on the cluster. Two jobs can each take one P100.
- Our training jobs request **10 CPUs** (half the node), so two P100 jobs fit side by side.
- P100 = Pascal (compute capability **sm_60**), **no tensor cores**, so mixed precision (AMP) works but gives a
  smaller speedup than on modern GPUs. fp16 ran fine (Ultralytics AMP check passed).
- **Why `gpu:p100:1` matters:** a plain `--gres=gpu:1` was scheduled onto the K80 node (gpu002) in testing.
  K80 is compute capability 3.7, unsupported by current PyTorch builds, and its driver (470 / CUDA 11.4) is too
  old for cu124 wheels.

### Scale (for intuition)

~100+ nodes total, roughly **a few server racks in one room**, not a warehouse. 1,896 CPU cores at the 2017
launch, plus the 2024 additions. The GPU share available to normal users is small (2 usable cards), so it's a
shared, modest resource.

---

## 3. Connecting and moving files

### SSH (verified)

```bash
ssh <username>@magnolia.usm.edu
```

- Windows 10/11: works directly in PowerShell (built-in OpenSSH). No PuTTY needed.
- First connect: verify the host key fingerprint
  `SHA256:glQIZZKlsAwf3QJNNDQSxFpRzE2rKs4GfxByO6ZBzis` (ECDSA), then answer `yes`.
- You land on a **login node** (`magnolia01`). **Never run heavy work there** (training, big pip builds are borderline;
  setup scripts were OK). Heavy work goes through `sbatch`/`srun`.
- Change the initial password once with `passwd`.

### Passwordless login with an SSH key (untested; standard)

```powershell
# On the laptop (PowerShell). Press Enter to accept the defaults.
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <username>@magnolia.usm.edu "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Optional `~/.ssh/config` entry on the laptop so `ssh magnolia` works:

```
Host magnolia
    HostName magnolia.usm.edu
    User <username>
```

### Copying files: `scp` (verified, run on the **laptop**)

```powershell
# Laptop -> cluster (e.g. a secrets file that is gitignored)
scp .env <username>@magnolia.usm.edu:~/<repo>/

# Cluster -> laptop: a whole run directory
scp -r <username>@magnolia.usm.edu:~/<repo>/runs/detect/<run_name> runs/detect/

# Cluster -> laptop: only some files (glob is expanded on the cluster side)
scp -r <username>@magnolia.usm.edu:~/<repo>/runs/detect/<run_name>/eval_val* runs/detect/<run_name>/
```

- Run `scp` **from the laptop**, not from inside an SSH session.
- `rsync` is nicer for big/repeat copies but isn't built into Windows; use `scp`, or `rsync` from WSL/Git Bash if installed.

### Code: use git, not scp (verified)

The repo is public on GitHub, so the cluster clones and pulls over HTTPS without credentials.

```bash
# once, on the cluster
git clone https://github.com/<github-user>/<repo>.git ~/<repo>

# every time code changes: commit + push on the laptop, then on the cluster:
cd ~/<repo> && git pull
```

- Pushing *from* the cluster would need GitHub credentials (a token or SSH key); we haven't set that up. We only
  committed from the cluster once (a lock file), which is the exception, not the rule.
- Data, weights, runs, `.venv`, and `.env` are **gitignored** and must be created or copied separately.

---

## 4. Storage

| Location | Size / state | Use for |
|---|---|---|
| `$HOME` = `/homes/01/<username>` | NFS, **105 TB filesystem 99% full** (1.3 TB free, shared), no per-user quota (`quota -s` shows none) | code, venvs, small results |
| `/home` | separate 500 GB NFS (60% used) | not used |
| `/scratch` | **Lustre**, 104 TB, ~11 TB free, fast parallel I/O | big datasets, job I/O; **needs admin-created folder** |
| `/scratch/jobs/` | SLURM's temporary per-job space | not for you |
| `$SCRATCH` | **not set** | — |

**Facts (verified):**
- `mkdir /scratch/<username>` fails: **Permission denied** (root-owned). Personal folders live under
  `/scratch/users/` and are created by the admin. One was requested 2026-11-14 (no reply yet). Purge policy unknown.
- So datasets currently live in **home**: `~/pneumonia-data` (~15 GB: raw DICOMs + zip + generated PNGs). Reading
  training images from home over NFS was **not** a bottleneck at the speeds we measured.
- Keep home usage modest; the shared filesystem is nearly full.

**Pattern that works well: keep data outside the repo and symlink it in.**

```bash
mkdir -p ~/myproject-data
ln -sfn ~/myproject-data ~/<repo>/data     # repo code keeps using data/...
readlink ~/<repo>/data                    # check where it points
```

- **gitignore gotcha:** a symlink is a *file* to git, so `data/` in `.gitignore` does **not** match it.
  Use `/data` (no trailing slash). Otherwise `git status` shows `data` as new and you may commit a symlink.
- **Don't delete `~/myproject-data` thinking it's a duplicate** of `~/<repo>/data`: it's the same files via the link.
- If a scratch folder is ever granted: `mv ~/myproject-data /scratch/users/<username>/ && ln -sfn /scratch/users/<username>/myproject-data ~/<repo>/data`.
- Jobs run with the same `$HOME` mounted on every node, so paths are identical everywhere.

---

## 5. Software environment

### Modules (verified)

```bash
module avail                          # list (old layout; this is what works on CentOS 7 nodes)
module avail 2>&1 | grep -i python    # python/2.7.14 3.5.4 3.6.2 3.7.11 3.8.6 3.9.6 3.11.5(default)
module load python/3.11.5
module list
module purge                          # start clean (do this at the top of every job script)
```

- **No conda/anaconda module.** Use `python -m venv`.
- Other modules: gcc 6.4–13.2, `cuda-toolkit/8.0.61, 10.1.243 (default), 11.6.2`, intel, mkl, ffmpeg, gaussian,
  lammps, etc. **You do not need any CUDA module for PyTorch**: pip wheels bundle their own CUDA runtime; only the
  node's driver matters.
- A new module layout (`module load newsetup`, then `module load thisnode`) exists for the Rocky Linux 9
  nodes (the A100 partition). It isn't needed on the CentOS 7 nodes we use.

### Python venv recipe (verified)

```bash
cd ~/<repo>
module purge && module load python/3.11.5
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Every new shell / job** must re-run `module load python/3.11.5 && source .venv/bin/activate`, or `python`, `pip`,
`wandb`, `yolo` etc. won't be found (`wandb: command not found` was exactly this).

### The glibc 2.17 constraint (the big one)

CentOS 7 ships **glibc 2.17**. Many modern Python wheels need glibc ≥ 2.28 (`manylinux_2_28`), and building from
source on the cluster fails (missing build tools such as meson). Rules that worked:

1. **Install with `--only-binary=:all:`** so pip picks the newest version that has a pre-built wheel for this OS
   instead of trying (and failing) to compile. Example failure without it: numpy 2.4 tried to build from source:
   `meson-python ... not found`.
2. **Don't copy the laptop's exact pins** for numeric packages. Pin only what must match (e.g. `ultralytics`,
   augmentation libs); let pip resolve the rest, then freeze it to a lock file.
3. **PyTorch:** `torch==2.5.1 torchvision==0.20.1` from the `cu124` index (torch 2.6+ dropped glibc 2.17).
   CUDA 12.4 matches the P100 node's driver 550.
4. **One combined pip call** with the PyTorch index plus PyPI as an extra index, so torch's dependencies resolve:

```bash
pip install --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 torchvision==0.20.1 -r requirements-cluster.txt
pip check
pip freeze > requirements-cluster.lock.txt    # record what actually got installed
```

**Versions that resolved and work (Python 3.11, Nov 2026):** torch 2.5.1+cu124, torchvision 0.20.1+cu124,
numpy 2.2.6, pandas 2.3.2, scipy 1.16.3, scikit-learn 1.7.2, matplotlib 3.11.2, pillow 12.2.0, opencv 4.14.0.94
(headless, see below), ultralytics 8.4.14, polars 2.2.0, pydicom 3.0.2, wandb 0.23.0, kaggle 2.2.5, certifi.

### OpenCV: use the headless build (verified)

The normal `opencv-python` wheel bundles X11/xcb libraries that clash with the GPU node's system libs:

```
ImportError: .../libxcb-shm...: undefined symbol: xcb_send_request_with_fds
```

(It imports fine on the login and CPU nodes, so the failure only shows up inside a GPU job.) Fix: replace it with
the headless build (same `cv2` module and version, no GUI):

```bash
pip uninstall -y opencv-python
pip install --only-binary=:all: --no-deps "opencv-python-headless<5"
```

If a package (like ultralytics) depends on `opencv-python`, pip installs it again on every install, so redo the swap
after installs. `pip check` will then complain that opencv-python is missing; that's expected and harmless.
Pin `opencv<5`: the resolver otherwise picked a brand-new 5.x major version.

### HTTPS / SSL certificates (verified)

The module's Python can't find the system CA bundle, so any Python HTTPS download fails:

```
ssl.SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED
```

(`curl` works; it's only Python.) Fix, after activating the venv (certifi comes with `requests`):

```bash
export SSL_CERT_FILE="$(python -m certifi)"
```

Put this line in **every** job script and setup script, after `source .venv/bin/activate`. Without it,
downloads of model weights, datasets via Python, etc. fail.

### Undeclared dependency: `kaggle` needs `six` (verified)

`kaggle` 2.x imports `six` without declaring it → `ModuleNotFoundError: No module named 'six'`. Add `six` to the
requirements explicitly.

### API keys and secrets

- **W&B:** run `wandb login` **once** on the login node (with the venv active). The key is stored in `~/.netrc` and
  works in all jobs. Compute nodes have internet, so W&B logs online. No offline mode or `wandb sync` needed.
- **Kaggle / other keys in a `.env`:** `.env` is gitignored, so `scp` it from the laptop. Then
  `chmod 600 .env` (other users share the filesystem).
- **Windows line endings:** a `.env` written on Windows has CRLF. Sourcing it in bash fails with
  `$'\r': command not found` (often printed as just `: command not found`), and values keep a trailing `\r`.
  Fix once with `sed -i 's/\r$//' .env`, or load it robustly in scripts:

```bash
if [ -f .env ]; then set -a; source <(tr -d '\r' < .env); set +a; fi
```

- Also add a `.gitattributes` with `*.sh text eol=lf` and `*.sbatch text eol=lf`, so scripts edited on Windows
  still have LF endings on the cluster.
- **Don't print secrets** into logs (no `cat .env`). To check it exists without revealing it: `cut -c1-16 .env`.

---

## 6. SLURM essentials

### Commands you'll actually use

| Goal | Command | Notes |
|---|---|---|
| Submit a batch job | `sbatch script.sbatch [args]` | prints `Submitted batch job <id>` |
| My jobs | `squeue -u $USER` | `ST`: `PD` = pending, `R` = running, `CG` = completing |
| Who's on the GPUs | `squeue -p gpu` | if two jobs are `R` on gpu001, both P100s are busy |
| Wider view | `squeue -u $USER -o "%.10i %.20j %.8T %.10M %.12l %R"` | adds elapsed/limit time and the pending reason (untested format, standard) |
| Cancel a job | `scancel <jobid>` / `scancel -u $USER` (all mine) | |
| Job details | `scontrol show job <jobid>` | node, times, reasons, resources |
| Partitions | `sinfo` / `sinfo -o "%P %l %G %N"` | GRES column shows GPU types |
| Free cores/RAM per partition | `coresavail` | Magnolia-specific helper |
| Node state / GPU allocation | `scontrol show node gpu001` | `AllocTRES` shows GPUs in use (standard, untested here) |
| My QoS / account | `sacctmgr show assoc user=$USER format=user,account,qos%40` | (not yet run) |
| Job history | ~~`sacct`~~ | **broken**: "Connection refused". Use the log files. |

**Pending reasons** (in the `NODELIST(REASON)` column of `squeue`): `(Resources)` = waiting for a free GPU/CPU;
`(Priority)` = others are ahead of you; `(QOSNotAllowed)` / `(AssocGrp...)` = you're not allowed on that partition.

**Job IDs** are plain numbers (e.g. `582685`). Job log files include them, see the templates.

### `sbatch` vs `srun`

| | `sbatch` | `srun` (interactive) |
|---|---|---|
| Blocks your terminal? | No: queued, returns immediately | **Yes**: waits until resources free up, then runs attached |
| Survives logout / closed laptop? | **Yes** | **No**: dies if the SSH session closes |
| Output | log file | your terminal |
| Use for | training, data prep, anything > a few minutes | quick checks, short evals (minutes), debugging |

If `srun` prints `job <id> queued and waiting for resources`, the GPUs are busy. Ctrl+C cancels it. Prefer
`sbatch` then, or run the short thing on your own machine.

---

## 7. Templates (copy and adapt)

### 7.1 GPU training job (verified; this is the pneumonia project's `slurm/train.sbatch`)

```bash
#!/bin/bash -l
#SBATCH --job-name=myproj-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1          # must say p100: plain gpu:1 can land on the unusable K80 node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10         # half of gpu001's 20 cores, so both P100s can be used at once
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00          # default is only 5 h; days-hh:mm:ss
#SBATCH --output=slurm/logs/%x-%j.out   # %x = job name, %j = job id; the folder must already exist

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"             # = the directory you ran sbatch from (submit from the repo root)

module purge
module load python/3.11.5
source .venv/bin/activate
export SSL_CERT_FILE="$(python -m certifi)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # proves which GPU you got

python train.py --workers $((SLURM_CPUS_PER_TASK - 2)) "$@"   # "$@" passes extra sbatch args through
```

- `#!/bin/bash -l` (login shell) makes `module` available inside the job.
- `mkdir -p slurm/logs` once; SLURM won't create the folder, and the job fails silently without it. Keep
  `slurm/logs/*` gitignored, with a `.gitkeep`.
- Arguments after the script name reach it as `$1`, `$2`, … e.g. `sbatch slurm/train.sbatch configs/x.yaml --set batch=32`.
- `#SBATCH` options can be overridden at submit time: `sbatch --time=12:00:00 --job-name=n640 slurm/train.sbatch ...`.
- **DataLoader workers = CPUs − 2** worked well (8 workers with 10 CPUs).
- **Auto-resume pattern** (used in our script): if the run's `last.pt` checkpoint exists, call the resume command
  instead of starting fresh, so re-submitting the identical command continues after a time-limit kill or crash.

### 7.2 CPU-only job (verified; data download + preprocessing)

```bash
#!/bin/bash -l
#SBATCH --job-name=myproj-prep
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.11.5
source .venv/bin/activate
export SSL_CERT_FILE="$(python -m certifi)"
if [ -f .env ]; then set -a; source <(tr -d '\r' < .env); set +a; fi   # API keys, CRLF-safe

python prepare.py --workers "$SLURM_CPUS_PER_TASK"
```

- Kaggle downloads work from compute nodes: `kaggle competitions download -c <competition> -p data/`.
- Unzipping: `python -m zipfile -e file.zip outdir/` works without needing `unzip`.
- The full RSNA dataset (~4 GB zip, ~27k DICOMs → PNG) took well under the 4 h limit with 16 CPUs.

### 7.3 Short one-off GPU command with `srun` (verified)

```bash
srun -p gpu --gres=gpu:p100:1 --cpus-per-task=4 --mem=16G --time=00:30:00 bash -c \
  'module load python/3.11.5 && source .venv/bin/activate && export SSL_CERT_FILE="$(python -m certifi)" && python evaluate.py runs/x'
```

- Single quotes around the `bash -c` string, so `$(...)` runs on the compute node, not the login node.
- Runs in the current directory (the repo root).
- Evaluating a YOLO model on ~4,000 images took ~3–8 min on a P100.

### 7.4 Interactive shell on a GPU node (verified with `--pty`)

```bash
srun -p gpu --gres=gpu:p100:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash -l
# now you're on gpu001:
module load python/3.11.5 && source .venv/bin/activate && export SSL_CERT_FILE="$(python -m certifi)"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit    # releases the GPU. Don't leave idle interactive sessions holding one of the 2 P100s.
```

### 7.5 One-liner checks (verified)

```bash
# GPU, OS and glibc on the P100 node
srun -p gpu --gres=gpu:p100:1 --pty bash -c "nvidia-smi; head -2 /etc/os-release; ldd --version | head -1"

# Internet from a compute node (an HTTP 404 from the API root is fine: it means the server was reached)
srun -p gpu --gres=gpu:p100:1 --pty bash -c "curl -I --max-time 10 https://api.wandb.ai; echo exit=\$?"
```

Don't use `curl -s` for connectivity tests: it hides the errors.

---

## 8. Monitoring a running job

```bash
squeue -u $USER                                  # is it running? how long?
ls -t slurm/logs | head                          # newest logs first
tail -f slurm/logs/<name>-<jobid>.out            # live log (Ctrl+C stops watching, NOT the job)
tail -c 400 slurm/logs/<name>-<jobid>.out        # last bytes (progress bars make huge single "lines")
grep -a "it/s" slurm/logs/<name>-<jobid>.out | tail -2   # training speed; -a because progress bars contain control chars
grep -ai "error\|traceback\|nan" slurm/logs/<name>-<jobid>.out | head
```

- Ultralytics/tqdm progress bars write carriage returns, so logs look like one giant line. Use `tail -c`, `grep -a`.
- **Check the GPU of a running job** (untested; needs SLURM ≥ 20.11 for `--overlap`):
  `srun --jobid=<jobid> --overlap --pty nvidia-smi`.
- If the code logs to **W&B**, the web dashboard is the easiest monitor (works from a phone).
- A job that disappears from `squeue` has finished (or crashed): read the end of its log. `sacct` can't tell you
  (it's broken).
- Ctrl+C in a `tail -f` is safe. Ctrl+C in an `srun` **kills** that job.

---

## 9. Laptop ↔ cluster workflow (what worked)

1. **Edit code on the laptop** → commit → push.
2. **Cluster:** `cd ~/<repo> && git pull`.
3. **Submit:** `sbatch slurm/<script>.sbatch <args>` from the repo root.
4. **Check after ~10 minutes:** `squeue -u $USER`, then the log's speed line, to confirm it's running at the expected speed.
5. **When done:** evaluate on the cluster (`srun`, minutes) or copy the run to the laptop (`scp -r`) and evaluate there.
6. **Bring results home:** `scp -r` the run folder (weights are small for YOLO: 5–20 MB).

**One-time setup on a new project:**

```bash
ssh <username>@magnolia.usm.edu
git clone https://github.com/<github-user>/<repo>.git && cd <repo>
DATA_DIR=$HOME/<repo>-data bash slurm/setup_env.sh   # venv + packages + data symlink (project script)
module load python/3.11.5 && source .venv/bin/activate
wandb login                                          # once per account, not per project
# from the laptop: scp .env <username>@magnolia.usm.edu:~/<repo>/   then on the cluster: chmod 600 .env
mkdir -p slurm/logs
```

**Keep laptop and cluster consistent:**
- Two requirements files: `requirements.txt` (laptop, exact pins) and `requirements-cluster.txt` (loose, binary-only).
  Pin the libraries that change training behaviour (e.g. `ultralytics`, `albumentationsx`) **identically** in both.
- Commit the cluster's `pip freeze` (`requirements-cluster.lock.txt`) for reproducibility.
- Record provenance per run (git commit, config, data-split hash) so runs from both machines can be trusted and compared.
- Model files trained on the cluster (torch 2.5.1) load fine on the laptop (torch 2.10), verified.

---

## 10. Measured performance (P100 12 GB, verified Nov 2026)

YOLO26 (Ultralytics 8.4.14) on RSNA chest X-rays, 18,678 training images (PNG 1024², resized on the fly), batch 16,
AMP on, 8 DataLoader workers, data read from home (NFS):

| Model / image size | Speed | Per epoch | 30 epochs | GPU memory |
|---|---|---|---|---|
| yolo26n, 640 px | 3.2 batches/s | ~6.5 min | **3.3 h** | 2.6–2.8 GB |
| yolo26s, 640 px | 2.2 batches/s | ~9.5 min | **4.7 h** | 5.6 GB |
| yolo26n, 1024 px | — | ~14 min | **7.0 h** | ~7 GB (est.) |
| yolo26n, 640 px, 22,886 imgs/epoch | — | ~8 min (est.) | ~4 h (est.) | ~2.7 GB |

- **~3× faster than the laptop** (RTX 3050 4 GB, Windows, 2 workers) for the same config, mostly thanks to more
  DataLoader workers and 12 GB VRAM.
- A 4× bigger model (n→s) was only ~1.45× slower: small models are partly **data-loading bound** at this setup.
- Validation pass on 4,003 images: ~20–30 s. Inference: ~2.3–2.9 ms/image.
- Rule of thumb for time requests: estimate, then set `--time` to ~2× (we use 2 days; nothing is lost by asking
  more on the `gpu` partition, but longer requests can wait longer if the scheduler backfills).

---

## 11. Troubleshooting: every error we hit

| Symptom | Cause | Fix |
|---|---|---|
| `meson-python ... not found` / numpy building from source during `pip install` | No glibc-2.17 wheel for that version (numpy 2.4) | `--only-binary=:all:`; don't pin laptop versions of numpy/pandas |
| `torch` ≥ 2.6 won't install / "no matching distribution" | torch 2.6+ needs glibc 2.28 | `torch==2.5.1 torchvision==0.20.1` from the `cu124` index |
| Resolver picks `opencv-python` 5.x | new major version | pin `opencv-python<5` |
| `mkdir: cannot create directory '/scratch/<user>': Permission denied` | scratch folders are admin-created | use `$HOME/<project>-data` + symlink; ask the admin |
| `CERTIFICATE_VERIFY_FAILED` when Python downloads anything | module Python can't find CA certs | `export SSL_CERT_FILE="$(python -m certifi)"` after activating the venv |
| `wandb: command not found` (or `python` is the wrong one) | venv not active in this shell | `module load python/3.11.5 && source .venv/bin/activate` |
| `.env` missing on the cluster | it's gitignored (correctly) | `scp .env <username>@magnolia.usm.edu:~/<repo>/` from the laptop, then `chmod 600 .env` |
| Job log ends with `: command not found` right after sourcing `.env` | Windows CRLF line endings | `sed -i 's/\r$//' .env`, or source via `tr -d '\r'` |
| `ModuleNotFoundError: No module named 'six'` (from kaggle) | kaggle doesn't declare `six` | add `six` to requirements |
| `git status` shows `data` (symlink) as a new file | `data/` in `.gitignore` only matches directories | use `/data`; `git reset HEAD data` if staged |
| `ImportError: libxcb-shm...: undefined symbol: xcb_send_request_with_fds` on the GPU node | GUI OpenCV wheel's bundled X11 libs clash with gpu001's | switch to `opencv-python-headless` |
| Job submitted, then gone from `squeue` immediately, no output | crashed at startup | read `slurm/logs/<name>-<id>.out`; check that `slurm/logs/` exists |
| `sacct: error: ... Connection refused` | cluster accounting DB down | use the log files |
| `srun: job <id> queued and waiting for resources` (hangs) | both P100s busy | Ctrl+C; use `sbatch`, or run it locally |
| Job lands on a K80 (`nvidia-smi` in the log shows "Tesla K80"); PyTorch would warn the GPU is unsupported | requested `gpu:1` without a type (landing on the K80 is verified; we never trained there) | `--gres=gpu:p100:1` |
| W&B: "Tried to log to step 30 that is less than the current step 31" | Ultralytics logs the final val out of order | harmless |
| Run directory exists after a failed job; re-submit refuses to start | the script won't overwrite a run | `rm -rf runs/<name>` if it never trained, else resume |
| (Windows laptop) DataLoader workers crash, out of memory | each worker is a full process on Windows | fewer workers (2 on 16 GB RAM) |

---

## 12. Etiquette, admin and citation

- **It's shared, and the GPU part is tiny (2 usable P100s).** Use one P100 per job, don't hold idle interactive
  sessions, cancel jobs you don't need (`scancel`), and check `squeue -p gpu` before queuing several jobs.
- Don't run heavy compute on the login node.
- Keep home usage small (the filesystem is 99% full).
- Software install requests and access questions: **Dr. Brian G. Olson (brian.olson@usm.edu)**.
- **Open requests** (sent 2026-11-14, no reply as of 2026-11-27): a personal scratch directory (+ purge policy) and
  access to the A100 partition (`suliaoma`). If granted, update this file. A100 nodes run **Rocky Linux 9**
  (newer glibc → newer torch possible) and use the new module layout (`module load newsetup` + `module load thisnode`)
  and `--partition=suliaoma --qos=<granted qos> --gres=gpu:a100:1`.
- **Citation** for publications using the cluster:
  > The authors acknowledge HPC at The University of Southern Mississippi supported by the National Science
  > Foundation under the Major Research Instrumentation (MRI) program via Grant # ACI 1626217.

---

## 13. Unknown / not yet checked

| What | How to check |
|---|---|
| My QoS list (any route to A100s?) | `sacctmgr show assoc user=$USER format=user,account,qos%40` |
| Scratch purge policy | ask the admin |
| Container runtime (fallback for glibc limits) | `which apptainer singularity` |
| Does `srun --jobid=<id> --overlap` work (SLURM version)? | `sinfo --version` |
| Jupyter on a compute node (port forwarding) | untested; would need `srun --pty` + an SSH tunnel through the login node |
| When the CentOS 7 nodes move to Rocky 9 | admin/news. **After that, everything in §5 changes**: rebuild the venv, newer torch possible, modules via `thisnode` |

---

## 14. For Claude in a new project: checklist

When asked to run something on Magnolia:

1. Job scripts: `#!/bin/bash -l`, `module purge`, `module load python/3.11.5`, `source .venv/bin/activate`,
   `export SSL_CERT_FILE="$(python -m certifi)"`, `cd "$SLURM_SUBMIT_DIR"`, an explicit `--time`, and
   `--output=slurm/logs/%x-%j.out` (create the folder).
2. GPU: `--partition=gpu --gres=gpu:p100:1`, ≤ 10 CPUs, ≤ 60G memory, so the second P100 stays usable for others.
   Assume **12 GB VRAM, no tensor cores, fp16 OK**.
3. Packages: torch 2.5.1/cu124 max; `--only-binary=:all:`; headless OpenCV; pin only behaviour-critical libraries; freeze a lock file.
4. Data: outside the repo (`$HOME/<project>-data`), symlinked in; `.gitignore` uses `/data`.
5. Secrets: never in git; `scp` from the laptop; CRLF-safe loading; never print them.
6. Prefer `sbatch` over `srun` for anything longer than a few minutes. `sacct` is broken, so read the logs.
7. The user runs cluster commands themselves: give exact commands and say what output to expect.
