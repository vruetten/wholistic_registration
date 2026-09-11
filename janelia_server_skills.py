---
name: janelia-cluster
description: Running jobs on the Janelia cluster
allowed-tools: bjobs
---

# janelia-cluster

## Version History

* v0.1 first try!

## Quickstart commands

From outside the cluster (ssh wrapper)
`ssh login1 'bash -l -c "bjobs ..."'`

Running a job on a GPU node:
`bsub -n $SLOTS -gpu "num=1" -q $QUEUE -W $MINUTES -R "affinity[core(1)]" -J $JOBNAME -o $LOGFILE $COMMAND`****

Job status
- `bjobs`
- `bjobs -l $JOBID`
- `bjobs -o "job_name stat exec_host" -noheader`

Kill jobs
  - `bkill $JOBID` — kill one
  - `bkill 0` — kill all yours
  - `bkill -J "jobname_*" 0` — kill by name pattern

Launch interactive job: 
`ssh -Y login1 'bsub -XF -Is -n 8 -gpu "num=1" -q gpu_a100 -W 48:00 /bin/bash'`

## Choosing parameters for bsub

`$SLOTS`
  - Scale the number of slots so that the job will have enough memory
  - Consider parallelization in the code that could benefit from more slots

`$QUEUE`
  - Choose queue based on GPU type needed
  - GPU must have enough VRAM for the job
  - Consider price of the job: ($NSLOTS * .05 + price[$QUEUE]) * $MINUTES / 60

`$MINUTES`
  - Estimate how long the job will take to run
  - Consider adding some buffer time to avoid job termination if it runs slightly longer than expected

`$JOBNAME`
  - Choose a descriptive name for the job with most identifiable information at the end of the name for easier searching in `bjobs`

## Submitting jobs

### GPU queues

| Queue | GPU | VRAM | Price/GPU/hr | Slots/GPU | RAM/slot |
|---|---|---|---|---|---|
| gpu_a100 | A100 | 80GB | $0.20 | 12 | 40GB |
| gpu_l4 | L4 | 24GB | $0.10 | 8 | 15GB |
| gpu_l4_16 | L4 | 24GB | $0.10 | 16 | 15GB |
| gpu_l4_large | L4 | 24GB | $0.10 | 64 | 15GB |
| gpu_h100 | H100 | 80GB | $0.50 | 12 | 40GB |
| gpu_h200 | H200 | 141GB | $0.80 | 12 | 40GB |
| gpu_t4 | T4 | 16GB | $0.10 | 48 | 15GB |
| gpu_short | All | - | $0.10 | 8 | 15GB |

- CPU slots: $0.05/slot/hour
- Default wall time if no -W: 2 hours. Max: 14 days.
- Per-user limit: ~50% of GPUs in each queue

### CPU queues

| Queue | Runtime Limit | Description |
|---|---|---|
| interactive | Default 8h, max 48h | GUI/interactive apps. Limit: 128 slots or 4 jobs per user |
| local | 14 days | Default for jobs without runtime. CPU-optimized nodes. Limit: 5999 slots per user |
| short | 1 hour | Jobs < 1 hour. No slot limit per user. Gets priority scheduling |

### COST

- **$0.05/slot/hour** for CPU
- GPUs billed additionally per GPU type

### Runtime Limits

- `-W` sets hard runtime (minutes or HH:MM format); `-We` sets estimate
- **local** queue: 14 days max
- **short** queue: 1 hour
- **interactive** queue: default 8 hours, max 48 hours
- **gpu_short**: 1 hour
- **gpu_\<type\>** queues: 14 days max; default 2 hours if no `-W` specified
- Jobs exceeding runtime are sent SIGUSR2 then killed

### Common Job Submission Options

| Option | Description |
|---|---|
| `-J <name>` | Job name (avoid: usernames, spaces, "spark", "janelia", "master", "int") |
| `-n <slots>` | Number of slots (1-128). Env var: `LSB_DJOB_NUMPROC` |
| `-o <file>` | Stdout file (suppresses email notification) |
| `-e <file>` | Stderr file |
| `-W <min>` | Hard runtime limit (minutes or HH:MM) |
| `-We <min>` | Runtime estimate (helps scheduler, won't kill job) |

### Additional -gpu Options

| Setting | Description | Janelia Notes |
|---|---|---|
| `num=num_gpus` | Number of GPUs | Max = GPUs per host |
| `mode=shared\|exclusive_process` | GPU sharing mode | Default: exclusive_process |
| `mps=yes\|no` | Multi-Process Service | Default: no (bugs in the past) |
| `j_exclusive=yes\|no` | Exclusive GPU access | Do not change; always exclusive |
| `gmodel=full_model_name` | Request specific GPU model | Only needed for gpu_short; use full model name |
| `gmem=mem_value` | Minimum GPU memory | Use with gpu_short only; e.g. `gmem=16G` |
| `nvlink=yes` | Require NVLink | Not needed; A100/H100/H200 always have nvlink |

Default `-gpu` settings: `"num=1:mode=exclusive_process:mps=no:j_exclusive=yes"`

### Types of Jobs

| Type | Description |
|---|---|
| **Batch** | Single segment, executed once |
| **Array** | Parallel independent tasks with same workload |
| **Parallel** | Cooperating tasks (MPI), must run simultaneously |
| **Interactive** | User login to compute node |

### Batch Jobs

```bash
# Single-threaded
bsub -n 1 -J <name> -o /dev/null 'command > output'

# Multi-threaded
bsub -n <1-128> -J <name> -o /dev/null 'command > output'
```

### Array Jobs

```bash
bsub -n <slots> -J "jobname[1-n]" -o /dev/null 'command file.$LSB_JOBINDEX > output.$LSB_JOBINDEX'
```

Limit concurrent members with `%val`:

```bash
bsub -J "myArray[1-1000]%15" /path/to/mybinary input.$LSB_JOBINDEX
```

Max array size: 1 million elements.

### Environment Variables

By default the submitting environment is passed to the job.

### Job Environment Variables

| Variable | Description |
|---|---|
| `$LSB_JOBID` | Job ID number |
| `$LSB_JOBINDEX` | Array Task Index |
| `$LSB_JOBINDEX_STEP` | Array step value |
| `$LSB_BATCH_JID` | Combined JobID and Array Index |
| `$LSB_DJOB_NUMPROC` | Value of `-n` (slots) |
| `$LSB_JOBNAME` | Value of `-J` (job name) |

### XDG_RUNTIME_DIR Issue

If job errors about `/run/user/<userid>`, fix with `unset XDG_RUNTIME_DIR` before submitting or inside the job.

## Monitoring and Modifying Existing Jobs

- `bjobs` or `bjobs -u all` to see jobs
- Job states: `RUN`, `PEND`, `UNKNOWN`

### Job Management

| Task | Command |
|---|---|
| Delete all your jobs | `bkill 0` |
| Delete individual job | `bkill <job id>` |
| Delete array job | `bkill <job id>` |
| Delete single array task | `bkill "<job id>[<task#>]"` |
| Delete range of tasks | `bkill "12354[1-15, 321, 500-600]"` |
| Delete by job name | `bkill -J <jobname> 0` |
| Delete by queue | `bkill -q <queue> 0` |

### GPU Host Statistics

```bash
lsload -gpuload <hostname>
```

- `gpu_ut` = processing utilization
- `CUDA_VISIBLE_DEVICES_ORIG` gives gpuid inside job
- `bjobs -l <jobid>` shows GPU assignment in EXTERNAL MESSAGES

### Slots per GPU

Request slots matching the ratio in the GPU table. Over-requesting strands GPUs.

## Storage

| Path | Backed up | Notes |
|---|---|---|
| `/groups/` | Yes (nightly, 30-day offsite) | Primary storage for scientific data |
| `/nrs/` | No | Cheaper tier for computationally reproducible data |
| `/scratch/$USER/` | No | Node-local SSD, ~25GB/slot, clean up after job |
| `/nearline/` | Yes | Not visible from compute nodes |
| `/tmp/` | No | Do not use; use `/scratch/` instead |

- Data transfer node: `dtn.int.janelia.org` (for copying to/from nearline)
- Submit large file copies as cluster jobs, not on login nodes
- Check quota by running `df -h /nrs/branson` or `df -h /groups/branson` on `login1`

### Workstation vs cluster paths

- Only `/groups/`, `/nrs/`, and `/misc/` are mounted on **both** workstations and cluster nodes (login + compute), with the same paths and contents — use these for anything that needs to be visible from both.
- `/home/` and `/tmp/` paths (e.g. `/home/<user>@hhmi.org/`) exist **only on the workstation**. They do NOT resolve on cluster nodes.
- The user's `$HOME` resolves to different paths in different contexts:
  - On the workstation: `/home/<user>@hhmi.org/`
  - On `login1` and compute nodes: typically `/groups/<lab>/home/<user>/`
- When constructing commands that will run on the cluster, prefer `$HOME` (resolved at run time on the cluster) or an explicit `/groups/<lab>/home/<user>/...` path, over a workstation-local `/home/<user>@hhmi.org/...` path.

### Conda

- A user-managed conda is installed at `$HOME/miniforge3/` on each machine. Because `$HOME` differs between workstation and cluster, this resolves to a *different* miniforge3 install in each place — they are independent installs unless the user explicitly mirrors envs.
- To activate: `source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate <env>`. This works in any context (workstation, login1, compute node) because `$HOME` resolves locally.
- Envs need to be built separately on the cluster. Cluster jobs that activate `$HOME/miniforge3` will pick up the **cluster's** miniforge3 (under `/groups/<lab>/home/<user>/`), not the workstation's.

## Containers (Apptainer/Singularity)

- Compute nodes run Apptainer 1.4.x
- Can run containers from older versions; new containers won't run on older Apptainer
- Can build from Docker containers
- GPU containers use host nvidia driver libraries (mapped at runtime)
- Use `--nv` flag for GPU access: `singularity exec --nv -B /groups -B /nrs -B /scratch image.sif command`

## CUDA

- Installed at `/usr/local/` on all compute nodes
- Current default: CUDA 13.1 (`/usr/local/cuda`)
- Also `/usr/local/cuda-11` and `/usr/local/cuda-12`
- Load specific version: `module load cuda-<version>`
- Many conda apps bundle their own CUDA toolkit


Copied from Kristin Branson's repo. 