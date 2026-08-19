# QC dashboard for the F260517 registration run — plan

Status: proposal, nothing implemented. Written 2026-08-19.

Scope: visualise and diagnose the output of
`src/wholistic_registration/tests/run_F260517_0625_qc_cluster.sh`, starting with the
motion displacement fields.

---

## 1. The data this dashboard reads (all measured)

Target run: `/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/registration_out/f260517_0625_qc_v4`
— 13,307,383,969 bytes across 67 files (`find -printf %s`, summed). It is a strict
superset of `_qc`, `_v2`, `_v3`; only `_v4` contains the `refspace_*` family. `_v2`
carries three filename conventions at once and must not be used to define a contract.

Frames 0-4 exist for every per-frame artifact, with no gaps. Two index formats appear in
the same run: `phase_new_f0.npy` and `mask_mov_000000.npz`. A loader must accept both.

| Artifact | Shape | Axis order | Dtype | Bytes each |
|---|---|---|---|---|
| `diagnostics/phase_new_f{0..4}.npy` | (630, 1500, 20, 3) | xyz, last axis = (x,y,z) | float32 | 226,800,128 |
| `diagnostics/motion_current_f{0..4}.npy` | (630, 1500, 20, 3) | xyz | float32 | 226,800,128 |
| `diagnostics/mov_mem_f{0..4}.npy` | (20, 1500, 630) | zyx | float32 | 75,600,128 |
| `diagnostics/masks_mov/mask_mov_*.npz` | sparse coord lists | xyz | int64 | 53k-59k |
| `diagnostics/mask_ref.npz` | sparse, 538,614 voxels | xyz | int64 | 289,955 |
| `diagnostics/coverage/no_coverage_*.npz` | bit-packed (20,1500,630) | zyx | uint8 | 106k-117k |
| `diagnostics/ref_shape.npy` | (3,) = [220,1500,630] | zyx | int64 | 152 |
| `diagnostics/fixed_target_z.npy` | (20,) = 19.581 … 210.544 | — | float32 | 208 |
| `raw_moving_*/`, `projected_*/` tif | (20, 1500, 630) | zyx | float32 | 75,603,798 |
| `refspace_*/` tif | (220, 1500, 630) | zyx | float32 | 831,639,398 |
| `refspace_reference_mem.tif` | (220, 1500, 630) | zyx | float32 | 831,639,398 |

Four CSVs under `diagnostics/` carry the per-frame scalars: `errors_membrane.csv`
(MAE, nMAE, NCC, edge, hole_frac, elapsed_s), `errors_sparse.csv` (adds NN, recall,
precision), `hole_summary.csv` (23 columns: global plus `hole_frac_k00`..`k19`), and
`refspace_summary.csv` (non-NaN fraction, occupied planes, z range).

### Field semantics (verified)

`phase_new = base + motion_current` elementwise, where `base = (x_index, y_index,
z_init[k])` and `z_init[k] = 20 + 10k`. Verified on a 1-in-119 spatial subsample of
frame 0: x and y residuals are ±6.1e-5, which is float32 rounding at magnitude 1500;
the z residual reproduces `z_init[k] - k` exactly. The code path is
`calFlowCrossResolution.py:1910-1913` with the seed set once at
`run_F260517_0625_qc.py:762` and never reassigned in the loop.

`motion_current` is a **pull map**: `I_ref(phase_new[i,j,k]) ≈ I_mov[i,j,k]`, consumed by
`apply_H_to_matrix_gpu` at `calFlowCrossResolution.py:2095`. Warping moving data into
reference space is therefore a forward scatter, not a resample.

Units are voxel indices with no physical scaling in this run (`zRatio_HR = 1` at
`run_F260517_0625_qc.py:127`, `zRatio` left at its 1.0 default). If `zRatio_HR` is ever
set to something other than 1, the z units change meaning and this dashboard's z axis
label becomes wrong.

The saved arrays are per-voxel at full moving resolution, but the optimisation variables
live on a control-point grid of stride `2r+1 = 11` px in x and y (`r = 5` at
`run_F260517_0625_qc.py:122`), every slice in z, lifted back by trilinear
`interp3Grid`. Stride 11 is therefore the natural default quiver stride: it draws one
arrow per solved control point rather than per interpolated voxel.

Frame-0 value ranges (measured): `motion_current` spans -17.89 to +26.03 voxels with 0
NaNs; per component, x ∈ [-10.28, 11.61], y ∈ [-9.81, 26.03], z ∈ [-17.89, 13.47].

### Two display traps the numbers force

`projected_*` volumes use `fill_value = -200.0` and `refspace_*` volumes use NaN, per
`diagnostics/projection_params.json`. `refspace_summary.csv` reports a non-NaN fraction of
0.192, so 81% of each reference-space volume is empty. Percentile contrast computed
without excluding the fill value will be set by the fill, not by the tissue.

The motion field is defined over the whole 630x1500x20 box, but only 80,347-90,469 voxels
per frame lie inside `mask_mov` (measured, frames 0-4). Outside the mask the field is
extrapolated and carries no information. Rendering it unmasked makes the display a picture
of the extrapolation.

---

## 2. Environment constraints (measured)

| Host | Python | numpy / skimage / tifffile | matplotlib | plotly / panel / bokeh / dash / napari | flask / fastapi | cupy |
|---|---|---|---|---|---|---|
| This Mac (homebrew `python3`, conda `all`) | 3.14.6 | absent | absent | absent | absent | absent |
| cluster env `wholistic-registration` | 3.9.22 | present | 3.9.4 | absent | absent | 13.4.1 |

The cluster also has `cmake` 4.4.0 (in the `all` env), `g++` 11.5.0, and a conda env named
`xpra`. `xpra` is not on the login-node PATH.

Consequences: the 13 GB stays on `/nrs`; the 1.27 MB summary layer (CSVs, NPZ masks,
coverage, JSON, the two alignment PNGs) could sit anywhere. Any live recomputation of the
projection must run on the cluster, because the scatter is GPU-only.

## 3. Prior art

Both reference dashboards are the same house style: native C++20, Dear ImGui + ImPlot +
GLFW/OpenGL, reading flat `.npy` plus a JSON or TOML manifest, launched as
`./build/dashboard configs/exp3.toml`, viewed remotely through xpra.

- `icampsnfr/dashboard/src/views/volume_view.cpp` holds the tiling: `DisplayMode {Slice,
  Tile}`, `default_tile_cols` at :49-54, the tile paint loop at :478-494, the tile-to-voxel
  inverse hit-test at :1239-1310. It tiles **z-planes of one volume**, not time frames.
  There is no time-frame montage or movie scrubber in either repo.
- Its tile texture is `4·Z·Y·X` bytes with no cap and no `GL_MAX_TEXTURE_SIZE` guard. That
  is safe there because its volumes are (6,144,124). At our (220,1500,630) it is 831 MB.
- `nmf_holy/NMFDemo/scripts/export_run_bundle.py` writes a `dash/` bundle of `.npy` +
  `manifest.json`; the viewer never reads the pipeline's native output. Worth copying.
- Domain-free and liftable: `icampsnfr/dashboard/src/ui/widgets.*`, `src/util/colormap.*`,
  `src/util/gl_texture.*`, `src/util/twilight.hpp` (cyclic phase LUT), and the npy loader's
  endianness / Fortran-order / rank validators at `npy_loader.cpp:31-117`.
- Contrast convention in both: percentile clipping (5th/95th default, 3rd/97th for
  anatomy), diverging map symmetric about zero for signed quantities, and non-finite
  values rendered as a magenta sentinel rather than an extreme colour.
- Already in this repo: `utils/visualization.py:127` `quivermotion_py(template, r,
  motion_field)` (2D quiver, matplotlib) and `:180` `plot_deformed_grid_plotly(...)`.
  Plotly is not installed in the cluster env, so the second one does not currently run
  there.

---

## 4. Proposed architecture

Two pieces, separated the way both reference repos separate them.

**A. Exporter / server (Python, on the cluster).** Reads a run directory, resolves both
filename conventions, and emits a `manifest.json` describing every artifact with its
declared axis order. It memory-maps the large arrays and serves single slices on demand: a
motion-field plane is 630x1500x3 float32 = 11.3 MB, a reference-space z-plane read as one
TIFF page is 3.8 MB. Because it runs inside the `wholistic-registration` env it can also
call the existing GPU projection code directly, which is what makes the reprojection panel
possible at all.

**B. Viewer.** The stack is the open decision, see section 6.

## 5. Panels

1. **Motion field.** Quantity dropdown: |d|, dx, dy, dz. Sequential colormap for the norm,
   diverging and symmetric about zero for the signed components. Base layer is the moving
   membrane frame in grayscale; the field composites over it with an alpha slider.
   `mask_mov` gates the overlay to alpha zero outside tissue. Slice slider over k=0..19.
2. **Quiver mode.** In-plane (dx, dy) arrows at stride 11 by default, matching the control
   point grid; dz carried by arrow colour on a diverging map, since a 2D arrow cannot show
   the through-plane component.
3. **Tiling.** Montage over a chosen axis. Which axis is a decision, see section 6.
4. **Reference-space projection.** Fast path loads the precomputed
   `refspace_mem_{t}.tif` and composites it on `refspace_reference_mem.tif`, with a
   reference-z slider over 220 planes and NaN excluded from the contrast computation.
   Recompute path calls the pipeline's own projection with adjustable `z_window`,
   `upsample_factor`, and `xy_splat_mode`, and can apply frame i's field to frame j's
   image as a null test.
5. **Metrics.** The four CSVs as linked plots, plus `hole_summary.csv` as a frame-by-plane
   heatmap. Clicking a cell drives the image panels to that (frame, plane).
6. **Alignment QC.** `target_z_offset_per_plane.npy` (2,20) plus the two existing PNGs.

Note for panel 5: `ref_update_id` is 0 for all five frames, so the reference was never
updated in this run and any reference-refresh view has no variation to show yet.

## 6. Open decisions

1. Viewer stack: browser served from the cluster over an ssh tunnel, native C++ ImGui in
   the house style, or a static precomputed page.
2. "Tile the frames": over time (t=0..4 at fixed plane), over z (k=0..19 at fixed frame),
   or both axes selectable.
3. Reprojection: view the precomputed volumes only, or live recompute with adjustable
   parameters on a GPU node.

---

## 7. Host and stack — settled by measurement (supersedes §2, §4, §6.1, §6.3)

The dashboard runs on **`ruttenv-ws1.hhmi.org`** (ssh alias `ws1`), not on the Mac and not
on a cluster login node. Everything it needs is already there, verified 2026-08-19:

| Requirement | Status on ws1 |
|---|---|
| `/nrs/ahrens`, `/groups/ahrens` | mounted directly, no ssh hop for data |
| GPU | NVIDIA RTX A4000, 16,084 MiB, 15,530 MiB free |
| cupy | 13.4.1, device 0 acquired and a kernel executed |
| Python env | `/groups/ahrens/home/ruttenv/miniforge3/envs/wholistic-registration/bin/python`, 3.9.22, numpy 2.0.2, tifffile 2024.6.18 |
| Repo checkout | `/groups/ahrens/home/ruttenv/python_packages/wholistic_registration` |
| OS / cores / RAM | Ubuntu 22.04.4, 80 cores, 376 GB |
| Absent | `cmake`, `xpra`, conda under the local home, `DISPLAY` over ssh |

The cluster env is on the shared home, so it runs on ws1 unmodified. No package is
installed and the shared env is not written to.

**Stack: Python standard-library `http.server` plus PIL for PNG encoding, serving a plain
HTML/JS page.** Viewed at `localhost` when sitting at ws1, or through
`ssh -L 8787:localhost:8787 ws1` from anywhere; the same code serves both cases. Chosen
over the C++/ImGui house style because ws1 has no `cmake`, because the moving-to-reference
scatter is `cp.RawKernel` and so needs a Python process in either design, and because
zero packages need installing.

Live GPU reprojection is in scope from the first version, since the A4000 is in the same
machine as the viewer. The earlier "precomputed first, recompute later" phasing was a
workaround for a GPU-allocation constraint that does not apply on ws1.

### Layout change the exporter must make (measured on ws1)

`motion_current` is stored `(x, y, z, c)`, so reading one z-plane is a strided gather
across the full 216 MB file.

| Operation | Time |
|---|---|
| Plane read, native `(x,y,z,c)` | 1.929 s |
| Plane read, z-major `(z,x,y,c)` | 0.005 s |
| One-off rechunk and write, per frame | 2.225 s |
| Norm, percentile, PNG encode (222 KB) | 0.372 s |
| Plane switch end to end, z-major | 0.377 s |

The rechunked plane is byte-identical to the strided read (`np.array_equal` returned
True). The speedup is 386x measured; an earlier estimate of 60x in conversation was an
I/O-volume ratio, not a time, and is superseded by the table above.

Remaining render cost is dominated by the 0.372 s norm-plus-percentile-plus-encode step,
which is reducible by computing percentiles on a spatial subsample and caching the norm
per (frame, plane, quantity). Not yet implemented, not yet measured.
