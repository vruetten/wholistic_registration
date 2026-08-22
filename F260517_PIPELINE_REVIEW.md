# F260517 High-Resolution Registration Pipeline — Code Review

**Subject:** `src/wholistic_registration/tests/test_F260517_v2.ipynb` and the
projection code it relies on (`src/wholistic_registration/utils/calFlowCrossResolution.py`).****
**Companion docs:** `src/wholistic_registration/pipeline/HighResolution.md` (the
design this notebook implements).
**Audience:** the project supervisor (for the *why*) and the student who will fix
it (for the *what to change*).
**Status of the code:** a working prototype that produces output, but with one
correctness bug, two questionable design choices, and several reproducibility
problems. None of these crash the run; they degrade output quality or make the
results hard to trust and impossible to reproduce on another machine.

---

## How to read this report

This pipeline registers a **sparse moving stack** (20 z-planes acquired over time)
to a **dense reference volume** (220 z-planes) and re-projects the moving signal
into the reference's coordinate frame so the result looks like a stable,
motion-corrected movie. If you have never seen the code, read the *Background*
section first — it defines the handful of variables the rest of the report keeps
referring to. Then each concern below follows the same four-step shape:

1. **Intention** — what this piece of code is *trying* to achieve.
2. **The code** — where it lives and what it actually does.
3. **Why it doesn't make sense** — the gap between the two.
4. **What to do instead** — a concrete change, plus how to confirm it worked.

### Severity summary


| #   | Concern                                                                                        | Type            | Severity  | Effort |
| --- | ---------------------------------------------------------------------------------------------- | --------------- | --------- | ------ |
| 1   | Periodic intensity recalibration feeds the model its own output instead of the new moving data | Correctness     | 🟥 High   | S      |
| 2   | Membrane channel is re-projected with max-splat scatter, which is lossy and hole-prone         | Design          | 🟧 Medium | S–M    |
| 3   | Forward-only with a fixed anchor → drift accumulates with nothing to bound it                  | Design          | 🟧 Medium | M      |
| 4   | `if i > 75:` hard-codes a per-dataset behaviour switch mid-run                                 | Hygiene         | 🟨 Low    | XS     |
| 5   | Hard-coded machine paths + a dead import make the notebook unrunnable elsewhere                | Reproducibility | 🟧 Medium | S      |


Effort: XS < 30 min · S ≈ 1 h · M = a few hours.

---

## Background — the data flow and the five variables that matter

The notebook processes one time frame at a time. For each frame `i` it calls the
registration engine and gets three arrays back:

```python
phase_new, motion_current, mem_mapped = calFlowCrossResolution.getMotion_v2(
    mov_mem_i, ref_mem_adj, option, verbose=False,
)
```

What each one means:

- `**mov_mem_i**` — the *raw moving* membrane stack for this frame, shape `(X, Y, K)`
with `K = 20` planes. This is the real, observed data.
- `**ref_mem_adj*`* — the *reference* membrane volume after its intensity
histogram has been warped to match the moving data (the "intensity adjustment"
step). Geometry unchanged, only brightness remapped.
- `**phase_new*`* — shape `(X, Y, K, 3)`. For every moving voxel it gives the
`(x, y, z)` coordinate **in the reference volume** that the voxel maps to. This
is the deformation field $\phi$.
- `**motion_current`** — the part of $\phi$ that was optimised this frame (the
displacement on top of the fixed initial map). It is carried to the next frame
as a warm start.
- `**mem_mapped**` — this is the one people misread. It is **the reference
sampled at `phase_new`**, i.e. `ref_mem_adj(φ(X))`. Internally the engine calls
it `data_ref_sampled` (`calFlowCrossResolution.py:2589`). It is the model's
*reconstruction of the moving image*, **not** the moving image itself.

You can prove `mem_mapped` is reference-derived from one line in the notebook:

```python
mem_err = float(np.mean(np.abs(mov_mem_i - mem_mapped)))
```

`mem_err` runs around **170** for every frame. If `mem_mapped` were the moving
image this difference would be ~0. It is large precisely because `mem_mapped` is
the *reference's best guess* at the moving image, and the gap between them is the
registration residual.

The frame is then re-projected to fixed reference planes and four volumes are
saved per frame: raw moving membrane, projected membrane, raw moving sparse-cell,
projected sparse-cell. The Fiji macro `Show4ChTiff.ijm` stacks those four for
viewing.

Two more facts the concerns below rely on:

- The reference intensity adjustment is **learned once at startup from the raw
moving frames 0–4**, then **periodically re-learned every 40 frames**.
- The run is **forward-only**: frame `i` is initialised from frame `i-1`'s motion
(scaled by 0.7), and the initial map $\phi_0$ never changes.

---

## Concern 1 — The periodic intensity recalibration uses the wrong target 🟥

### Intention

The reference volume and the moving frames have different brightness, and the
moving brightness *drifts over the recording* (bleaching, gain changes, etc.). If
the optimiser sees a brightness difference it can mistake it for motion. So the
pipeline warps the reference's intensity histogram to match the moving data —
and, because the moving brightness drifts, it is supposed to **re-learn that
mapping every 40 frames using the most recent real moving data** (this is exactly
§6 "Periodic reference intensity update" in `HighResolution.md`).

### The code

At **startup** the target is the raw moving data — correct:

```python
init_calibration_frames = [0, 1, 2, 3, 4]
init_target_stack = np.mean(
    F260517_mov_mem[init_calibration_frames].astype(np.float32), axis=0,
)   # mean of RAW MOVING frames 0..4
# -> learn_quantile_mapping(source=reference, target=init_target_stack)
```

But the **periodic** update (every 40 frames) uses a different, internal source.
The main loop collects "recent frames":

```python
calib_frames = sorted(registered_mem_mapped.keys())[-5:]
new_ref, sq, tq = update_ref_from_recent_frames(calib_frames)
```

and `update_ref_from_recent_frames` builds its target from
`registered_mem_mapped`:

```python
for fi in frame_indices:
    if fi in registered_mem_mapped:
        stacks.append(registered_mem_mapped[fi])   # <-- see below
...
target_stack = np.mean(np.stack(stacks, axis=0), axis=0)
# -> learn_quantile_mapping(source=reference, target=target_stack)
```

And `registered_mem_mapped` is filled with `mem_mapped`, the **reference
reconstruction**, not the moving data:

```python
# inside process_single_frame:
mem_mapped_zyx = mem_mapped.transpose(2, 1, 0)   # mem_mapped = ref_adj(φ)
...
registered_mem_mapped[i] = result["mem_mapped_zyx"]
```

### Why it doesn't make sense

Recall `mem_mapped = ref_mem_adj(φ)` — it is produced by sampling the **already
intensity-adjusted reference**. Its brightness histogram is, by construction,
approximately the histogram of `ref_mem_adj` itself (a geometric warp relocates
intensities but does not change their distribution, apart from mild interpolation
smoothing).

So the periodic update is doing this:

> learn a mapping from `reference` → (a warped copy of the
> already-adjusted `reference`)

That is self-referential. It re-derives roughly the **mapping that is already in
place** and learns **nothing from the actual moving frames at t = 35–39, 75–79**,
which are the frames that actually carry the brightness drift it is trying to
track. The one piece of new information that should drive the update — the current
real moving intensity — is never looked at.

The notebook's own logs confirm it. The high-percentile target value `tgt_q[max]`
goes `6505 → 6127 → 5578` across the three calibrations, and that downward drift
**tracks the rising no-coverage fraction** (`global_no_cov` climbs `0.05 → 0.24`
over the same frames), not a genuine change in moving-image brightness. In other
words the "recalibration" is mostly reacting to registration drift feeding back
on itself, which is the opposite of what it should do.

### What to do instead

Calibrate against the **raw moving frames** at those indices, exactly like the
startup path does. The data is already in memory as `F260517_mov_mem`.

Concretely, in `update_ref_from_recent_frames`, replace the target source:

```python
# BEFORE: target built from registered_mem_mapped (reference reconstruction)
stacks.append(registered_mem_mapped[fi])

# AFTER: target built from the raw moving stack at the same frame index
stacks.append(F260517_mov_mem[fi].astype(np.float32, copy=False))
```

(The moving stack is shape `(K, Y, X)`, which already matches the `(K, Y, X)`
shape the quantile-mapping helper expects via `z_idx`, so no transpose change is
needed — but verify the shapes print equal before running the full thing.)

**How to confirm the fix:** print `tgt_q` at each update as the code already does.
After the fix, `tgt_q` should reflect the moving frames' actual intensity
statistics and should *not* simply shrink in lockstep with the coverage drop.
A good sanity check: run 80 frames before and after, and compare the saved
`projected_mem` brightness stability over time — it should be flatter after the
fix.

### Side effects of the fix (so you don't get surprised)

I checked every use of `registered_mem_mapped` in the notebook. Its **stored
values** (the warped-reference stacks) are read in exactly one place — the
calibration target you are changing. Everywhere else only the dictionary's
*keys* (which frame indices were processed) or its *length* are used:

| Where | What it reads |
|---|---|
| `update_ref_from_recent_frames` (the line you're fixing) | the **value** → calibration target |
| `calib_frames = sorted(registered_mem_mapped.keys())[-5:]` | keys only |
| `len(registered_mem_mapped)` (final print) | count only |
| `processing_order.npy = sorted(registered_mem_mapped.keys())` | keys only |

So the one-line swap is a complete, drop-in fix — nothing else consumes those
values, and the shapes already match (`F260517_mov_mem[fi]` and the old
`registered_mem_mapped[fi]` are both `(K, Y, X) = (20, 1500, 630)`, which is what
`update_reference_intensity_mapping_from_target_stack` expects via its
`target_stack.shape[0] == len(z_idx)` check).

Two consequences worth knowing:

1. **After this fix, the *values* stored in `registered_mem_mapped` are dead** —
   only the keys still matter. The minimal fix is the one-line swap; an optional
   tidy-up is to stop storing `mem_mapped` altogether and keep just a list/set of
   processed frame indices for the `calib_frames` and `processing_order`
   bookkeeping.
2. **`registered_motion` is already fully dead** — it is written every frame but
   never read anywhere, so it just holds a `(X, Y, K, 3)` array per frame for
   nothing (a real memory cost over 200 frames). It is unrelated to this bug but
   safe to remove while you are here. (The temporal motion prior still works: that
   comes from `option["motion"] = 0.7 * motion_current`, set inside
   `process_single_frame`, not from `registered_motion`.)

---

## Concern 2 — The membrane channel is re-projected with max-splat scatter 🟧

### Intention

After registration, the moving signal must be drawn back onto a fixed set of
reference z-planes so every frame is viewed from the same vantage point (a stable
movie). Two channels are projected: the **membrane** (a dense, continuous-looking
structural channel) and the **sparse-cell** channel (isolated bright dots).

### The code

Both channels are projected with the same call,
`project_coords_to_fixed_planes_gpu` (`calFlowCrossResolution.py:854`):

```python
mem_proj = calFlowCrossResolution.project_coords_to_fixed_planes_gpu(
    coords_ref_xyk_xyz=phase_for_projection,
    values_xyk=mem_values_for_projection,     # values come from moving
    ...
    xy_splat_mode="subpixel_footprint",
)
```

This routine is a **forward (scatter) splat with max-combine**: each moving voxel
is thrown into the reference plane its `phase` points to, and where several voxels
land on the same output pixel, the **maximum** value wins (the GPU kernel uses
`atomicMax`, `calFlowCrossResolution.py:569`).

### Why it doesn't make sense (for the membrane)

Forward scatter has two well-known problems, both visible here:

1. **Holes.** Forward mapping does not guarantee every output pixel receives a
  sample, so gaps appear. The notebook is clearly aware of this — it computes a
   `coverage` map every frame and reports `no_coverage` rising from ~5% to ~24%
   (and up to ~50% on the worst plane). The 2× "supersurface" upsampling is a
   patch over the same symptom.
2. **Max-combine is wrong for a continuous channel.** Taking the brightest of
  several overlapping samples biases the membrane image upward and discards the
   others. Max is the right choice for **sparse bright dots** (you want to keep
   the dot), but for a dense membrane you want an **average**.

The repository already contains the better tool for the membrane:
`project_coords_to_fixed_planes_weighted_gpu` (`calFlowCrossResolution.py:676`),
which does bilinear XY splatting with a z-slab weight and **weighted-average**
combine. It is defined but unused.

### What to do instead

Use the weighted-average projector for the **membrane** channel and keep the
max-splat projector for the **sparse-cell** channel:

```python
# membrane: continuous -> weighted average
mem_proj, _w = calFlowCrossResolution.project_coords_to_fixed_planes_weighted_gpu(
    coords_ref_xyk_xyz=phase_for_projection,
    ref_volume=ref_mem_adj_for_projection,
    target_z_planes=fixed_target_z,
    values_xyk=mem_values_for_projection,
    ref_volume_order="xyz",
    z_window=projection_z_window,
    z_weight_mode="gaussian",
    return_numpy=True,
    output_order="zyx",
)

# sparse cells: keep max splat (project_coords_to_fixed_planes_gpu) unchanged
```

**How to confirm the fix:** the membrane `no_coverage` fraction should drop
substantially (bilinear splat fills 2×2 neighbourhoods and averages overlaps),
and the projected membrane should look smoother with fewer black speckles. Keep
the existing coverage print to measure the before/after difference on the same
frames.

> Note the weighted projector returns a `(projection, weight)` tuple, whereas the
> max projector returns a single array — adjust the call site accordingly.

---

## Concern 3 — Forward-only with a fixed anchor lets drift accumulate 🟧

### Intention

Register a long time-lapse efficiently by warm-starting each frame from the
previous one (adjacent frames move similarly), so the optimiser converges fast.

### The code

The initial map `option["phase"]` ($\phi_0$) is set once and never updated; each
frame only passes a decayed motion prior forward:

```python
option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)
```

### Why it doesn't make sense (as the *only* strategy)

`mem_err` creeps up across the run and the no-coverage fraction roughly
quadruples (0.05 → ~0.24) by frame 90. That is the classic signature of
**accumulated drift**: small per-frame errors compound because nothing ever
re-anchors the solution to an absolute target. The 0.7 decay slows it but cannot
remove a systematic bias, and $\phi_0$ being fixed means there is no correction
path once the chain has wandered.

This is not necessarily *wrong* — forward-only is a legitimate speed trade-off —
but it should be a conscious decision with a safety net, not the silent default.

### What to do instead (pick one, smallest first)

- **Cheap:** add a drift monitor — if `mem_err` or `no_coverage` exceeds a
threshold, re-register that frame from $\phi_0$ (cold start) instead of the
warm start, to break the chain.
- **Medium:** periodically (e.g. every N frames) re-register against the reference
from scratch and reset the motion prior, so error cannot compound indefinitely.
- **Larger:** a forward–backward pass (register the sequence in both directions
and blend) if drift remains a problem after the above.

**How to confirm:** plot `mem_err` vs frame and `global_no_cov` vs frame (the
notebook already records both to CSV in `diagnostics/`). A healthy run should be
roughly flat, not monotonically rising.

---

## Concern 4 — A hard-coded frame number changes behaviour mid-run 🟨

### Intention

Turn on the expensive "wrong-region correction" (the local-optimum escape, 3-pass
re-optimisation) only once the registration starts struggling.

### The code

```python
for i in range(5, T):
    if i > 75:
        option["wrong_region_enable"] = True
```

### Why it doesn't make sense

`75` is a magic number tuned to *this one recording*. It silently makes the second
half of every run behave differently from the first half, and it never turns off.
Anyone re-running on different data will get a behaviour change at frame 75 for no
discoverable reason. It reads like a debugging experiment left in by accident.

### What to do instead

Drive it from a named parameter, or better, from the drift monitor in Concern 3
(enable correction *when error is high*, not *after an arbitrary frame*):

```python
WRONG_REGION_AFTER_FRAME = None   # or an int; document why if set

if WRONG_REGION_AFTER_FRAME is not None:
    option["wrong_region_enable"] = (i > WRONG_REGION_AFTER_FRAME)
```

If the intent really is "always on after warmup," just set it once before the loop
and say so in a comment.

---

## Concern 5 — The notebook cannot run anywhere but the author's machine 🟧

### Intention

Develop quickly on a personal workstation.

### The code

```python
os.chdir('/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src/wholistic_registration')
from utils import IO                      # only resolves because of the chdir above
...
PROJECT_ROOT = Path("/home/cyf/wbi/Virginia/code/CoarseFlow").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from training.inference import (CoarseFlowInferenceConfig, CoarseFlowPredictor)  # never used

F260517_mov_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_exp_00001_TZCYX.ome.tiff"
base_out_dir     = "/home/cyf/wbi/Virginia/registrated_data/f260517/f260517_0609/"
cp.cuda.Device(1).use()
```

### Why it doesn't make sense

- Every path is absolute and points at `/home/cyf/...`; none exist on the lab
server. (This is the same "hardcoded user paths" issue already logged as finding
#5 in `AUDIT.md`.)
- `from utils import IO` only works because of the `os.chdir` above it — it is not
a package-relative import, so the notebook breaks the moment the working
directory differs.
- `CoarseFlowPredictor` is imported from an external repo but **never called** —
the z-initialisation actually uses the classical `FindInitZ_stack_global_fixed_spacing`.
This import will raise `ImportError` on any machine without `/home/cyf/.../CoarseFlow`,
which kills the whole notebook at cell 1 even though the model is unused.
(`HighResolution.md` §1 itself notes this model is "still training, haven't been
used for the data.")
- `cp.cuda.Device(1)` assumes a specific GPU index.

### What to do instead

- Put the three I/O paths and the GPU index in a small config block (or a `.toml`
under `configs/`) at the top of the notebook, and read input/output locations
relative to it. Keep these machine-specific values out of any shared commit.
- Replace `os.chdir(...) + from utils import ...` with package imports
(`from wholistic_registration.utils import ...`) after `pip install -e .`, so
the working directory no longer matters.
- **Delete the `CoarseFlow` import** until that model is actually wired in. It is
dead code today and an immediate crash on any other machine.

---

## Suggested order of work for the student

1. **Concern 5 first** (≈ 1 h) — you cannot test anything until the notebook runs
  on the lab server. Strip the hard-coded paths into a config block, fix the
   imports, delete the dead `CoarseFlow` import.
2. **Concern 1** (≈ 1 h) — the actual correctness bug. One-line target change plus
  a before/after `tgt_q` comparison.
3. **Concern 2** (≈ 1–3 h) — swap the membrane projector; measure the coverage
  improvement.
4. **Concern 4** (≈ 15 min) — remove the magic `i > 75`.
5. **Concern 3** (a few hours) — add the drift monitor / periodic re-anchor; this
  also gives Concern 4 a principled trigger.

For each change: run the **same first 80 frames** before and after, and compare the
two diagnostics the notebook already writes (`diagnostics/coverage_stats.csv` and
`mem_err` from the logs). Every fix above should make at least one of those
numbers better, and none should make `mem_err` worse.