# Magnolia HPC Cluster (USM): Reference Notes

> Recovered from the old USM HPC wiki via the Wayback Machine (the live site is currently down).
> The content is from the archived pages, so some of it may be out of date. In particular, see the
> **May 28, 2024 hardware/software update**, which isn't captured here yet.

---

## Overview

Magnolia is the high-performance computing cluster at the **University of Southern Mississippi (USM)**. It has been fully operational since **September 5, 2017**.

- **New to HPC at USM?** Start with the FAQs on the wiki.
- **Latest update listed:** May 28, 2024: *Major Software and Hardware Updates* (page not yet recovered).
- **Contact for questions and software install requests:** Dr. Brian G. Olson (brian.olson@usm.edu)

Wiki sections: Connecting to Magnolia · Tutorials · Workshops · Capabilities · News · FAQs

---

## Capabilities

### Hardware (as of 2017)

| Resource | Spec |
|---|---|
| Compute cores | 1,896 |
| CUDA cores | 7,168 |
| High-memory nodes | 4 nodes, 512 GB RAM each |
| Normal compute nodes | 20 cores, 6 GB RAM per core (~120 GB/node) |
| Scratch filesystem | Rapid temporary filesystem (**Lustre**) for running jobs |
| Archive storage | Network-attached storage for data archiving |

### Software

- Intel and GNU compilers (Fortran, C, C++)
- Other software can be installed on request
- See the wiki's *Software* page for the full list (not yet recovered)

### DataSpace digital archive

- Archive for raw data, documents, theses and dissertations
- Front end for searching and processing data

### Funding

Supported by the National Science Foundation's Major Research Instrumentation (MRI) program, Grant **#ACI 1626217**.

### How to cite

Reference USM HPC in any research report, journal or publication. Suggested citation:

> The authors acknowledge HPC at The University of Southern Mississippi supported by the National
> Science Foundation under the Major Research Instrumentation (MRI) program via Grant # ACI 1626217.

---

## Connecting to Magnolia

> **Note:** Change your password after your first login by running `passwd`.

### Requesting an account

Requests are typically processed within **48 hours**. Credentials are sent by email once approved.

- **Individual:** use the request page on the wiki.
- **Class:** the instructor of record can request temporary student accounts by emailing Dr. Brian G. Olson. Class accounts are per semester and expire **2 weeks after the semester ends**.

### Required program: SSH

Connect with `ssh`. On first connection you'll be shown the host's fingerprint. Verify it against the wiki page `SSH_Fingerprints/magnolia.usm.edu`. If it doesn't match, contact Dr. Olson.

Known fingerprint (from the archived page):

```
ECDSA key fingerprint is SHA256:glQIZZKlsAwf3QJNNDQSxFpRzE2rKs4GfxByO6ZBzis.
```

### Linux / macOS

Replace `<username>` with your assigned username:

```bash
ssh <username>@magnolia.usm.edu
```

If the fingerprint matches, type `yes` and press Enter.

### Windows

Use an SSH client:

- **PuTTY** (putty.org): recommended for text-only use. Enter `magnolia.usm.edu` as the Host Name and click **Open**.
- **MobaXterm** (mobaxterm.mobatek.net): recommended if you need GUIs over X11 forwarding.
- Windows 10/11 also ships with OpenSSH, so `ssh <username>@magnolia.usm.edu` works in PowerShell.

### Full login example

```text
user $ ssh exampleuser@magnolia.usm.edu
The authenticity of host 'magnolia.usm.edu' can't be established.
ECDSA key fingerprint is SHA256:glQIZZKlsAwf3QJNNDQSxFpRzE2rKs4GfxByO6ZBzis.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'magnolia.usm.edu' (ECDSA) to the list of known hosts.
exampleuser@magnolia.usm.edu's password:

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Welcome to Magnolia, the high performance computational cluster
at the University of Southern Mississippi.
...
                          ** Messages **

September 14, 2017 - To get a copy of the newest version of job
submission examples, which work with new and updated modules, run
the following command at anytime:
                          update-examples
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

[exampleuser@magnolia01 ~]$
```

> **Tip:** Run `update-examples` after logging in to get the current job-submission example scripts.
> They should show the correct SLURM partitions, GPU syntax and modules for this cluster.
> (This tip and the Windows OpenSSH note above are additions, not from the original wiki.)

### What next?

- Initial setup of account
- Examples (job submission)
- List of tutorials

---

## Tutorials

Subcategories:

- Installing Programs
- LAMMPS Examples
- Quantum ESPRESSO Examples

Pages:

- Compiling LAMMPS
- Connect to Magnolia
- Initial Setup
- Initial Setup (flory)
- Schrödinger Command Line Job Submissions
- Selector-mpi (flory)
- Update Examples

## Workshops

- Introduction to GNU/Linux, Parts 1–3
- Introduction to the Magnolia Cluster Workshop
- Magnolia HPC Cluster Workshop (July 2023)
- Submitting jobs and compiling programs on Magnolia cluster

---

## May 2024 Update ("Update 202405")

> Recovered from the wiki. Recent node purchases and the end of CentOS prompted major updates.
> The cluster is moving from **CentOS 7 to Rocky Linux 9**. Custom packages (modules, and
> user-installed miniconda packages) must be recompiled for Rocky 9's newer libraries.

### Roadmap (status as of May 28, 2024)

| Task | Done? | Notes |
|---|---|---|
| Install new compute hardware, Part I | ✅ | Vendor installed (see below) |
| Install new compute hardware, Part II | ❌ | Not yet shipped by vendor |
| Install OS on new nodes | ✅ | Rocky Linux 9 (CentOS 7 EOL June 30, 2024) |
| Update backup/cloning system | ✅ | Works with both Rocky 9 and CentOS 7 |
| Update Slurm | ✅ | Version compatible with both OSes |
| Add new nodes to Slurm | ✅ | New partitions `kisame` and `suliaoma` (owners only until preemption is enabled); 4 nodes added to `himem` (open to all) |
| Update module setup | ❌ | WIP, preview with `module load newsetup` |
| Update modules for Rocky 9 | ❌ | WIP |
| Install CUDA drivers on new GPU nodes | ✅ | |
| Preemption partitions | ❌ | TODO (this is what would open `kisame`/`suliaoma` to everyone) |
| Upgrade CentOS 7 compute nodes to Rocky 9 | ❌ | TODO |
| Upgrade login nodes to Rocky 9 | ❌ | TODO |

**New hardware (Part I):**

| Nodes | CPUs | RAM | GPUs |
|---|---|---|---|
| 6 compute nodes | 2× Intel Xeon Gold 6448H | 512 GiB | none |
| 4 high-memory nodes | 2× Intel Xeon Gold 6426Y | 2 TiB | none |
| 3 GPU nodes | 2× Intel Xeon Gold 6426Y | 256 GiB | **2× NVIDIA A100 80GB PCIe** each |

**Contact:** if an upgrade breaks a package, contact Brian Olson with details.

### Checking free resources: `coresavail`

```text
$ coresavail
Number of nodes in partition with N available cores and RAM.

Nodes  Partition  Available Cores  Available RAM (MiB)
6      kisame     64               515134
3      suliaoma   32               257094
4      himem      32               2063430
24     node       20               128000
4      himem      20               512000
2      gpu        20               128000
17     lomem      16               64000
```

`kisame` and `suliaoma` are only accessible to the groups that purchased that hardware.

### Modules: the new layout

During the transition some nodes run CentOS 7 and others Rocky 9, so modules are split by OS. Preview the new layout with:

```bash
module load newsetup
module avail
```

- `/modules/node/common/...` holds modules that run on **both** OSes.
- `/modules/node/Types/` holds node-type modules named `<distro>-<version>-<cpu>`: `centos-7-63`, `centos-7-79`, `rocky-9.3-143`.
- **`thisnode`** detects the current node type and loads the matching module. After the upgrade this will likely be required in every job script. `module help thisnode` shows the node type, e.g. `Node Type: centos-7-79`.

**Common modules (both OSes), as of May 28, 2024:**

| Category | Modules |
|---|---|
| Languages | `cuda-toolkit/8.0.61`, `cuda-toolkit/10.1.243` (default), `cuda-toolkit/11.6.2`, `gcc/6.4.0` to `gcc/13.2.0`, `intel/2017.4.196` |
| Libraries | isl, lapack, libjpeg-turbo, libpng, mkl, mpfr, ncurses, openssl, trimmomatic, x264, zstd |
| Programs | bamtools, bedtools, bowtie, bowtie2, cmake, fastx-toolkit, ffmpeg, gaussian, nco, salmon |
| MPI | `impi/2017.4.196` |

**Extra on Rocky 9 nodes** (`module load rocky-9.3-143`): `openmpi-gcc/4.1.5`, `lammps/20220623`, `hdf5`, `netcdf`. These fail on CentOS 7 nodes.

> No `python`/`conda` module appears in the new layout yet. The usage stats below show `python` was
> used 433 times, so it likely exists in the **old** layout (plain `module avail` without `newsetup`).

### Installation priority (most-used modules, last 90 days)

The admin prioritizes porting modules used 10+ times in 90 days:

hdf5 (2089), netcdf (2079), intel (1381), mkl (1317), lammps (1294), mpfr (1225), gcc (1222), libpng (1109), isl (1057), x264 (804), ffmpeg (792), esmf (749), MCT (744), scrip-coawst (742), fftw (737), nco (683), ncview (676), gaussian (645), **python (433)**, **cuda-toolkit (317)**, bedtools (258), samtools (258), openmpi (221), bowtie (203), R (165), orca (150), fastx-toolkit (146), digimat (109), icu (96), readline (93), ncurses (88), matlab (69), impi (56), cmake (54), openmpi-gcc (52), lapack (45), salmon (34), git (27), trimmomatic (20), newsetup (18), gmake (14), bowtie2 (13), lammps-tools (13), mopac (13), gromacs (10), STAR (10)

If you need a module not on the list, request it from the contacts above.

### Sample job script for Rocky Linux nodes (from the wiki)

```bash
#!/bin/bash -l

#SBATCH --partition=suliaoma
#SBATCH --gres=gpu:1
#SBATCH --qos=bxmarg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000G
#SBATCH --time=00:10:00
#SBATCH --job-name=example
#SBATCH --output=myoutput-%j.txt

cd "$SLURM_SUBMIT_DIR"

module purge
module load newsetup
module load thisnode

srun hostname
```

> Note: `--mem=1000G` exceeds the 256 GiB on a suliaoma node, so this is likely a typo on the wiki.

---

## Cluster survey (run by me, 2026-11-13)

### Partitions: `sinfo -o "%P %l %G %N"`

```text
PARTITION TIMELIMIT  GRES                NODELIST
node*     infinite   (null)              node[001-076]
gpu       infinite   gpu:p100:2(S:0-1)   gpu001
gpu       infinite   gpu:k80:4(S:0-1)    gpu002
himem     infinite   (null)              himem[001-008]
lomem     1-00:00:00 (null)              lomem[009-010,012-020]
kisame    infinite   (null)              node[078,084,088-089,092,094-098,102-104,107-111,113,115]
egg       infinite   (null)              node[077,080-083,085-086,100,105,116-117]
systodo   infinite   (null)              node[087,090-091,093,099,106,112,114]
syscheck  infinite   (null)              node[079,101]
suliaoma  infinite   gpu:a100:2(S:0-1)   gpu[003-005]
float     5-00:00:00 (null)              himem[001-004],lomem[009-010,012-020],node[001-076]
```

### Partition details: `scontrol show partition` (summary)

| Partition | Nodes | GPUs | Who can use it (QoS) | Default time | Max time | Default mem/CPU |
|---|---|---|---|---|---|---|
| `node` *(default)* | node[001-076], 1520 CPUs | none | everyone (`normal`, `cores800`, `cores400`) | 5 h | unlimited | 6000 MB |
| **`gpu`** | gpu001 (**2× P100**), gpu002 (**4× K80**); 40 CPUs, 250 GB total | 6 | everyone (`normal`) | **5 h** | unlimited | 6000 MB |
| `himem` | himem[001-008], 208 CPUs | none | everyone | 5 h | unlimited | 25000 MB |
| `lomem` | 11 nodes, 176 CPUs | none | everyone, plus several QoS | 2 h | 1 day | 1000 MB |
| `float` | himem/lomem/node mix, 1776 CPUs | none | QoS `float` only | 1 h | 5 days | 1500 MB |
| **`suliaoma`** | gpu[003-005] (**6× A100 80GB**), 96 CPUs | 6 | **QoS `bxmarg` only** (hardware owners) | 5 h | unlimited | 8000 MB |
| `kisame` | 20 nodes, 1280 CPUs | none | QoS `msbroms` only | 5 h | unlimited | 8000 MB |
| `egg` | 11 nodes, 704 CPUs | none | QoS `bxmarg` only | 5 h | unlimited | 8000 MB |
| `systodo`, `syscheck` | test nodes | none | QoS `nodetester` (admins) | 5 h | unlimited | 8000 MB |

All partitions are `OverSubscribe=FORCE:1` and `PreemptMode=OFF`.

### K80 node: `srun -p gpu --gres=gpu:1 --pty nvidia-smi`

Both attempts landed on a **Tesla K80** (gpu002):

```text
NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.4
GPU  Name        ...  Memory-Usage        GPU-Util
  0  Tesla K80        0MiB / 11441MiB     99% (1st run) / 0% (2nd run)
No running processes found
```

### Internet check from a compute node

```bash
srun -p gpu --gres=gpu:1 --pty bash -c "nvidia-smi; curl -sI https://api.wandb.ai | head -1"
```

The `curl` line printed nothing here, but this first test landed on the K80 node with `-s` hiding errors. The retest on the P100 node (below) shows **compute nodes do have internet**.

---
### P100 node (gpu001): GPU, OS, glibc

```bash
srun -p gpu --gres=gpu:p100:1 --pty bash -c "nvidia-smi; head -2 /etc/os-release; ldd --version | head -1"
```

```text
NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4
GPU  Name                   Memory-Usage        GPU-Util
  0  Tesla P100-PCIE-12GB   0MiB / 12288MiB     1%
No running processes found

NAME="CentOS Linux"
VERSION="7 (Core)"
ldd (GNU libc) 2.17
```

The P100 node has a **much newer driver (550, CUDA 12.4)** than the K80 node (470, CUDA 11.4). The cards are the **12 GB** P100 variant, not 16 GB.

### Internet from a compute node (retest)

```bash
srun -p gpu --gres=gpu:p100:1 --pty bash -c "curl -I --max-time 10 https://api.wandb.ai; echo exit=\$?"
```

```text
HTTP/1.1 404 Not Found
x-wandb-trace-id: ...
via: 1.1 google
exit=0
```

A 404 from the API root is a normal response, meaning the server was reached. **Compute nodes have internet access.**

### Python modules

```bash
module avail 2>&1 | grep -iE "python|conda|anaconda"
```

```text
python/2.7.14   python/3.5.4   python/3.6.2   python/3.7.11
python/3.8.6    python/3.9.6   python/3.11.5 (default)
```

No conda/anaconda module.

### Storage

```bash
echo $HOME $SCRATCH; df -h | grep -iE "lustre|scratch|home"; quota -s
```

```text
$HOME    = /homes/01/<username>
$SCRATCH = (not set)

Filesystem                  Size  Used  Avail Use%  Mounted on
nfs01-ib:/homes/01          105T  104T  1.3T   99%  /homes/01
nfs01-ib:/home              500G  300G  201G   60%  /home
172.18.1.243@o2ib:/scratch  104T   94T   11T   90%  /scratch    (Lustre)

Disk quotas: none
```

---

## Implications for this project

1. **Train on `gpu001`: 2× Tesla P100 12 GB (Pascal, sm_60).** Always request `--gres=gpu:p100:1`. A plain `--gres=gpu:1` can land on the K80 node (gpu002), which current PyTorch doesn't support.
2. **The A100s (`suliaoma`, 6× A100 80GB) are owner-only** (QoS `bxmarg`) until preemption is enabled. Worth asking the admin about.
3. **Driver 550 / CUDA 12.4 on the P100 node** means a `cu124` PyTorch build works. The laptop's `cu130` build does not.
4. **CentOS 7 / glibc 2.17** is the binding constraint. PyTorch moved its Linux wheels to glibc 2.28 starting with **torch 2.6**, so the newest torch we can pip-install here is **2.5.1** (`torch==2.5.1 torchvision==0.20.1`, `cu124`). Some other pinned packages may also lack glibc-2.17 wheels; the setup script will reveal which.
5. **Python: `module load python/3.11.5`** (the laptop uses 3.13; all pinned packages support 3.11).
6. **Internet works on compute nodes**, so W&B can log **online** directly (no offline mode or `wandb sync` needed), and pip/Kaggle work from jobs too.
7. **Storage:** `/homes/01` is 99% full (shared), so keep only code and the venv in home. Put the dataset (~15 GB) on **`/scratch`** (Lustre, 11 TB free) and symlink `data/` to it. `$SCRATCH` isn't set, so we'll use `/scratch/<username>`.
8. **Default job time is 5 h**, so job scripts set `--time` explicitly.
9. **Expectations:** a P100 is roughly comparable to the laptop's RTX 3050 per step (no tensor cores). The wins are 12 GB VRAM, 20 CPU cores per node for data loading, ~128 GB RAM, and two GPUs for parallel runs.

## Requests to the admin

| Date | Request | Status |
|---|---|---|
| 2026-11-14 | Personal scratch directory (`/scratch/users/<username>`), scratch purge policy, A100 (`suliaoma`) access | No reply as of 2026-11-25. Meanwhile: data in `$HOME/pneumonia-data`, training on the `gpu` partition P100s. |

Note: you can't create your own folder under `/scratch` (it's root-owned; `/scratch/users/` folders are admin-created), and `/scratch/jobs/` is SLURM's temporary per-job space.

## Still to check (nice to have, not blocking)

| # | What | Command |
|---|---|---|
| 1 | Can you create a folder on scratch? Is scratch purged after N days? | `mkdir -p /scratch/$USER && ls -ld /scratch/$USER` (and check the wiki/admin for a purge policy) |
| 2 | Your QoS (any chance of A100 access?) | `sacctmgr show assoc user=$USER format=user,account,qos%40` |
| 3 | How busy the P100s are | `squeue -p gpu` and `coresavail` |
| 4 | Is a container runtime available (fallback if pip installs fail on CentOS 7) | `which apptainer singularity` |
