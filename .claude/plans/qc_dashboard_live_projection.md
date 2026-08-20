# QC dashboard: live projection from source data

Supersedes the artifact-reading architecture in `native_qc_dashboard.md`.
Written 2026-08-20 after measuring the projection cost on real data.

## What the dashboard is

Three inputs, one operation, one picture:

| input | source of truth |
|---|---|
| raw moving | `260517_exp_00001_TZCYX.ome.tiff` |
| reference | `260517_anat_00003_TZCYX.ome.tiff` |
| displacement field | `phase_new_f{n}.npy`, written by the QC run |

The dashboard scatters the raw moving stack into reference space through the
displacement field, and draws the result over the reference anatomy, membrane
and sparse-cell channels both.

Everything else a QC run writes — `refspace_*`, `projected_*`, `raw_moving_*`,
`refspace_movie_*` — is an output of one past job. Reading those as the
dashboard's input is what made the previous design a patch: it inherited one
run's z crop, its lateral downsample, its stride, and its channel layout as
constants, and then needed a workaround whenever a run disagreed. They stay
loadable as the oracle the C scatter is checked against. They are not the
spine.

## The measurement

One frame, one channel, into the full reference grid (78, 1500, 630), K=4,
on the 80-core box. `phase_new_f0.npy` is 45 MB at K=4.

| | upsample=2 | upsample=1 |
|---|---|---|
| load field | 0.53 s | 0.53 s |
| load moving slices | 0.27 s | 0.03 s |
| scatter, 1 thread | 0.79 s | 0.37 s |
| scatter, 8 threads | 0.21 s | 0.12 s |
| accumulators (value + weight) | 590 MB | 590 MB |

Threading saturates around 8: 590 MB of accumulators makes the kernel
memory-bound, not compute-bound. The threaded figures come from a benchmark
whose threads race on a shared accumulator, so they are a floor rather than a
delivered number — a correct scheme needs per-thread z-slab ownership, paying
either a re-scan per slab or more memory. Bank on the single-threaded 0.79 s.

**The number that shapes the design: 11 projectable frames x 2 channels x
~1.0 s is roughly 15 seconds for the entire projectable movie in full space.**
Projection is therefore a load-time job, not an on-demand one. Project
everything once, scrub instantly, re-project when a parameter changes.

Memory forbids holding all of it at full resolution: 11 frames x 295 MB is
3.2 GB. At the movie's reduced grid (55, 750, 315) a frame is 52 MB and all 11
fit in 570 MB. So: full-resolution for the frame on screen, reduced for the
scrubbing set.

## The config contract

The raw data location belongs in a config file that both the QC run and the
dashboard read. Today it is a module-level constant in
`src/wholistic_registration/tests/run_F260517_0625_qc.py`
(`F260517_DATA_DIR + "/260517_exp_00001_TZCYX.ome.tiff"`), and the dashboard
would have to recover it from `projection_params.json`, which is an *output*.
An input the consumer reconstructs from an output is the patch in miniature.

One config, two consumers:

```
[data]
moving    = ".../260517_exp_00001_TZCYX.ome.tiff"
reference = ".../260517_anat_00003_TZCYX.ome.tiff"
[moving]
slices  = [8, 9, 10, 11]
z_init  = [25, 35, 45, 55]
[reference]
z0 = 165
z1 = 243
channel_mem = 1
channel_sparse = 0
[run]
out_dir = ".../registration_out/f260517_0625_qc_v5"
save_field_every = 1
```

**Format is undecided.** TOML is what this repo already uses for every other
config (`src/wholistic_registration/configs/*.toml`, read through
`utils/IO.py`), and `toml++` is already a fetched dependency of the native
build, so TOML costs nothing on either side. YAML would need a new C++
dependency. Both `toml` and `pyyaml` are installed in the Python env.
Recommendation: TOML, for consistency with the repo and zero new dependencies.

## Steps

1. **Config + QC run.** Define the config, teach `run_F260517_0625_qc.py` to
   read it instead of module constants, and set the field-saving stride to 1
   so every frame is projectable. At 45 MB a frame that is 4.5 GB per field
   type for 100 frames.
   *Exit check:* a QC run driven entirely by a config file, writing fields for
   every frame, with no data path in the script.

2. **Source readers.** Read the moving and reference OME-TIFFs directly:
   timepoint and channel indexing, the z crop as a parameter rather than a
   baked constant. The existing `tiff_volume.cpp` handles the page reads; this
   adds the `TZCYX` / `ZCYX` axis mapping.
   *Exit check:* a test asserting a plane read from the source file equals the
   corresponding page of the run's `raw_moving_*` output.

3. **The scatter kernel.** Trilinear splat into a caller-chosen grid, both
   channels, correctly parallel over output z slabs.
   *Exit check:* voxel-for-voxel against `refspace_movie_mem.ome.tif` on the
   frames where both exist. This is why the precomputed artifacts stay
   loadable — without an oracle a wrong scatter draws a plausible picture.

4. **Projection controls.** z crop, downsample, z-window, upsample, channel as
   live controls; re-project on change.
   *Exit check:* setting the controls to the pipeline's own recorded values
   reproduces step 3's agreement.

5. **The overlay panel.** Projected moving over reference anatomy, two
   channels, on `DISPLAY=:1`.
   *Exit check:* screenshot under `logs/dashboard_ui/`.

## What carries over

`Run` (per-family frame discovery, geometry, reference resolution), the npy /
npz / TIFF / CSV readers, and `projectable_frames()` all stand — they were
written against the data rather than against one run's conventions.
`MovieDescriptor` and the refspace/projected series demote from spine to
validation scaffolding.

## Decided 2026-08-20

- **Config format: TOML.** Consistent with the rest of the repo and no new
  dependency on either side.
- **Build against the timepoints that exist.** No QC re-run for now, so the
  11 frames carrying a displacement field are the working set and step 1
  shrinks to "the dashboard reads a TOML config to find the raw data". The QC
  pipeline keeps its current stride; raising it to 1 stays available whenever
  full-timepoint coverage is wanted.

This reorders the steps: 1 becomes config-for-the-dashboard only, and the
work moves to 2 and 3.

## Open

- `widgets.cpp` `EnterReturnsTrue`, still unfixed: aborts any non-NDEBUG
  build, silently dropped in Release.
