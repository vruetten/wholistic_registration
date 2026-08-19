# QC dashboard — module seams and constants

Every constant below is measured from `f260517_0625_qc_v4` unless marked assumed.
Implementers must not re-derive them and must not guess axis orders.

## Host and run command

Runs on `ws1` (`ruttenv-ws1.hhmi.org`), where `/nrs` and `/groups` are mounted and an
RTX A4000 is present. Interpreter:
`/groups/ahrens/home/ruttenv/miniforge3/envs/wholistic-registration/bin/python` (3.9.22,
numpy 2.0.2, tifffile 2024.6.18, cupy 13.4.1). **Target Python 3.9 syntax** — no `match`,
no `X | Y` type unions, no `list[int]` in annotations evaluated at runtime.

Standard library only, plus numpy, tifffile, PIL, and (in `reproject.py` only) cupy.
Do not add a dependency. Do not write into the shared conda env.

## Canonical orientation rule

The single largest hazard in this dataset is that axis order differs per artifact and is
declared inside the files. **Every function in `qc_bundle.py` returns image planes as
`(Y, X) = (1500, 630)`**, and vector fields as `(Y, X, 3)` with the last axis ordered
`(x, y, z)` — component order is preserved from the source, spatial order is normalised.
No other module may transpose. Source layouts, for reference only:

| Artifact | On-disk shape | On-disk axis order |
|---|---|---|
| `motion_current_f{t}.npy`, `phase_new_f{t}.npy` | (630, 1500, 20, 3) | x, y, z, component |
| `mov_mem_f{t}.npy` | (20, 1500, 630) | z, y, x |
| `mask_mov_{t:06d}.npz`, `mask_ref.npz` | sparse coord lists | `axis_order` field = `'xyz'` |
| `no_coverage_{t:06d}.npz` | bit-packed, `shape` field = [20,1500,630] | `axis_order` field = `'zyx'` |
| `raw_moving_*/`, `projected_*/` `.tif` | (20, 1500, 630) | z, y, x |
| `refspace_*/`, `refspace_reference_mem.tif` | (220, 1500, 630) | z, y, x |
| `ref_shape.npy` | [220, 1500, 630] | z, y, x |

Read the `axis_order` field where one exists. Never assume it.

## Measured constants

- Frames present: 0, 1, 2, 3, 4. No gaps. Two filename conventions in one run:
  `phase_new_f{t}.npy` (bare int) and `mask_mov_{t:06d}.npz` (zero-padded). Support both.
- Reference grid (z, y, x) = (220, 1500, 630). Moving grid = (20, 1500, 630).
- `phase_new = base + motion_current`, `base = (x_index, y_index, z_init[k])`,
  `z_init[k] = 20 + 10*k`. Verified elementwise on a 1-in-119 subsample of frame 0;
  x and y residuals were +/- 6.1e-5, float32 rounding at magnitude 1500.
- `motion_current` is a pull map: `I_ref(phase_new[i,j,k]) ~= I_mov[i,j,k]`. Warping
  moving data into reference space is a forward scatter, not a resample.
- Units are voxel indices. `zRatio_HR = 1` in this run, so no physical scaling applies.
- Control-point stride is `2r+1 = 11` px in x and y, every slice in z. Use 11 as the
  default quiver stride: one arrow per solved control point.
- `motion_current` frame 0 range: -17.89 to +26.03 voxels, 0 NaNs.
  Per component x [-10.28, 11.61], y [-9.81, 26.03], z [-17.89, 13.47].
- `mask_mov` occupancy per frame: 90469, 80347, 88734, 84472, 82184 voxels
  (frames 0-4) out of 630*1500*20 = 18,900,000.
- `projected_*` fill value is -200.0. `refspace_*` fill is NaN.
- `refspace` non-NaN fraction is 0.192 over the whole volume; plane 110 measures 0.338.
- `fixed_target_z.npy` is (20,) spanning 19.581 to 210.544.
- `no_coverage` unpacks as `np.unpackbits(packed)[:np.prod(shape)].reshape(shape)`;
  its per-frame mean reproduces `hole_frac` in the CSVs to 6 decimals.
- `ref_update_id` is 0 for all 5 frames: the reference never updated in this run.

## Performance contract (measured on ws1)

Reading one z-plane from the native `(x,y,z,c)` layout costs 1.929 s because it is a
strided gather over the whole 216 MB file. A z-major `(z,x,y,c)` copy reduces that to
0.005 s, byte-identical (`np.array_equal` True), at a one-off cost of 2.225 s per frame.
`qc_bundle.py` MUST build and reuse that cache; a plane read that takes ~2 s is a defect.

Budget per plane switch: cache build excluded, target under 0.40 s end to end.
Current known cost of norm + percentile + PNG encode is 0.372 s and is the thing to
optimise (subsample the percentile, cache the norm per (frame, plane, quantity)).

## Module boundaries

Five files, one owner each. No file imports a sibling except as stated.

### 1. `dashboard/qc_bundle.py` — data access. Imports: numpy, tifffile, json, csv, os.
Owns run-directory discovery, the z-major cache, axis normalisation, mask/coverage
unpacking, CSV loading. Contains no rendering and no HTTP.

```
class QCBundle:
    def __init__(self, run_dir, cache_dir)
    frames            -> list of int
    ref_shape_zyx     -> (220, 1500, 630)
    mov_shape_zyx     -> (20, 1500, 630)
    projection_params -> dict            # from projection_params.json
    fixed_target_z    -> ndarray (20,)
    def motion_plane(frame, k)                    -> (1500, 630, 3) float32, comps (x,y,z)
    def phase_plane(frame, k)                     -> (1500, 630, 3) float32
    def mov_plane(frame, k)                       -> (1500, 630) float32
    def mask_mov_plane(frame, k)                  -> (1500, 630) bool
    def mask_ref_plane(z)                         -> (1500, 630) bool
    def coverage_plane(frame, k)                  -> (1500, 630) bool, True = NO coverage
    def raw_moving_plane(frame, k, channel)       -> (1500, 630) float32
    def projected_plane(frame, k, channel)        -> (1500, 630) float32, fill -200.0
    def refspace_plane(frame, z, channel)         -> (1500, 630) float32, fill NaN
    def refspace_reference_plane(z)               -> (1500, 630) float32
    def metrics()                                 -> dict of {name: {"columns": [...], "rows": [[...]]}}
    def manifest()                                -> JSON-serialisable dict
```
`channel` is `"mem"` or `"sparseCell"`. Note the directory naming is asymmetric:
`refspace_sparseCell/` holds files named `..._refspace_sparse_*`, while `projected_*` and
`raw_moving_*` use `sparseCell` in both directory and filename. Handle it.

### 2. `dashboard/render.py` — arrays to PNG. Imports: numpy, PIL. No file IO, no HTTP.
Pure functions. Takes arrays already in `(Y, X)` orientation and returns bytes or arrays.

```
def contrast_limits(a, mask=None, lo_pct=1.0, hi_pct=99.0, exclude_values=(-200.0,),
                    subsample=200000)                 -> (lo, hi)   # ignores non-finite
def apply_colormap(u01, cmap)                          -> (H, W, 3) uint8
def render_scalar(a, lo, hi, cmap, mask=None, nonfinite_rgb=(255, 0, 255))
                                                       -> (H, W, 3) uint8
def composite(base_rgb, over_rgb, alpha, over_mask=None) -> (H, W, 3) uint8
def quiver_overlay(rgb, dx, dy, dz=None, stride=11, scale=1.0, dz_cmap="diverging")
                                                       -> (H, W, 3) uint8
def montage(planes, cols=None, labels=None, max_edge_px=8192) -> (H, W, 3) uint8
def to_png(rgb)                                        -> bytes
```
Colormaps required: `"gray"`, `"magma"` or equivalent sequential, and a diverging map
symmetric about zero. Signed quantities (dx, dy, dz) use the diverging map with
`lo = -m, hi = +m`; the norm uses the sequential map. Non-finite pixels render magenta,
following the convention in both reference dashboards. `montage` must refuse or downsample
above `max_edge_px` — the icampsnfr montage has no such guard and would allocate
4*Z*Y*X bytes.

### 3. `dashboard/reproject.py` — GPU. Imports: numpy, cupy, and the repo's utils.
May import `qc_bundle`. Wraps the pipeline's own functions; does not reimplement them.

```
def project_to_planes(bundle, field_frame, image_frame, params) -> (20, 1500, 630) float32
def scatter_to_refspace(bundle, field_frame, image_frame, params) -> (220, 1500, 630) float32
```
`params` overrides `projection_params.json` keys: `z_window`, `upsample_factor`,
`xy_splat_mode`, `values_interp_order_mem`. Allowing `field_frame != image_frame` is
deliberate: applying frame i's field to frame j's image is the null test.
Existing entry points: `utils.calFlowCrossResolution.project_coords_to_fixed_planes_gpu`
and the scatter in `tests/run_F260517_0625_qc.py:307` (`_scatter_trilinear_to_refspace`).
Reconstruct `phase` for a frame as `base + motion_current` per the identity above, so a
modified field can be projected without re-running registration.

### 4. `dashboard/server.py` — stdlib `http.server`. Imports: qc_bundle, render, reproject.
Threaded server, one bundle instance, in-process LRU cache of rendered PNGs.
Routes return `image/png` or `application/json`. Query parameters carry frame, plane,
quantity, colormap, alpha, contrast percentiles, quiver on/off and stride, montage axis
and columns. Bind to `127.0.0.1` only.

### 5. `dashboard/static/index.html` — one file, inline CSS and JS, no build step, no CDN.
Panels: motion field, tiling montage, reference-space projection, metrics, alignment QC.
Controls per §5 of `plan/qc-dashboard.md`. The montage axis selector offers `time` and
`z`, defaulting to `time`.

## Verification required of every implementer

Code that has not been executed is a draft and must be reported as one. Run on ws1:

```
rsync -az dashboard/ ws1:/groups/ahrens/home/ruttenv/python_packages/wholistic_registration/dashboard/
ssh ws1 '/groups/ahrens/home/ruttenv/miniforge3/envs/wholistic-registration/bin/python \
  /groups/ahrens/home/ruttenv/python_packages/wholistic_registration/dashboard/<file> --selftest'
```

Report measured timings and observed output, not intentions. State plainly which paths
you executed and which you only wrote.

---

## Correction 1 — mask semantics (supersedes the mask guidance above)

Measured after the first implementer wave, and it reverses an instruction given earlier.

`mask_mov` is produced by `mask.getMask(mov_i, thresFactor)` followed by
`mask.bwareafilt3_wei(option["mask_mov"], maskRange)` (`run_F260517_0625_qc.py:645-646`);
`mask_ref` is the same pair applied once to the reference (`f260517_helpers.py:175-176`).
Both are intensity-threshold plus connected-component area filters, and both are **sparse**:

| Mask | True voxels | Box | Fraction |
|---|---|---|---|
| `mask_mov` frame 0 | 90,469 | 630 x 1500 x 20 = 18,900,000 | 0.479% |
| `mask_ref` | 538,614 | 630 x 1500 x 220 = 207,900,000 | 0.259% |

Consequences, both binding:

1. **Do not gate the motion-field display by `mask_mov` by default.** At 0.48% occupancy a
   gated plane is 99.5% empty and shows nothing about where displacement is large. The
   default motion view is ungated, drawn over the moving frame as grayscale anatomy. Mask
   gating is an explicit toggle, off by default.
2. **When gating is on, gate with alpha, not with the magenta sentinel.** Render the field
   unmasked via `render_scalar(...)` and then `composite(base, over, alpha,
   over_mask=mask)`, so the anatomy shows through where the mask is False. The
   `nonfinite_rgb` magenta path is reserved for genuinely invalid values (NaN, fill), which
   is a different statement than "outside the mask".
3. **Surface the metric support.** `compute_frame_metrics` and `compute_sparse_metrics`
   use `mask_mov` as their `valid` selector (`run_F260517_0625_qc.py:772-773`, `:792-793`),
   so every MAE, nMAE, NCC, recall and precision value in `errors_membrane.csv` and
   `errors_sparse.csv` is computed over that 0.48%. The metrics panel must state the
   support fraction next to the numbers, and the mask must be available as its own display
   layer, so a reader can see which voxels the error figures actually describe.

## Correction 2 — measured render costs (informational, from the completed render.py)

Measured on ws1 against the delivered `render.py`, plane size (1500, 630):

| Path | Time |
|---|---|
| `contrast_limits` + `render_scalar` + `to_png` | 0.172 s |
| `quiver_overlay`, stride 11, ~7752 arrows, dz-coloured | 0.264 s |
| `quiver_overlay`, stride 22, ~1904 arrows | 0.083 s |
| `quiver_overlay`, stride 44, ~476 arrows | 0.043 s |
| `quiver_overlay`, stride 11, `dz=None` | 0.047 s |
| `to_png` of a quiver result | 0.208 s |

The dz colouring costs 0.264 s against 0.047 s for the same arrow count because colour is
resolved per arrow. Batching arrows into a small number of dz bins and drawing each bin as
one pass would recover most of the difference. Not implemented, not measured.

Quiver mode at stride 11 therefore costs roughly 0.44 s end to end (render + quiver +
encode) against the 0.40 s budget in the performance contract. Either default the quiver
stride to 22, or bin the dz colours. Server owner: pick one and say which you did.

---

## Correction 3 — scale (supersedes "Frames present: 0, 1, 2, 3, 4. No gaps.")

The 5-frame run is a smoke test. A real run is **200 frames**: the moving stack
`260517_exp_00001_TZCYX.ome.tiff` measures `(200, 20, 2, 1500, 630)` int16, 14.1 GB,
8000 pages (`tifffile.TiffFile(...).series[0]`). Axes are T, Z, C, Y, X; C=2 is
membrane and sparse-cell.

At the measured 2.66 GB/frame, saving every diagnostic for 200 frames costs about 532 GB
(motion + phase 90.7 GB, refspace 332 GB, raw + projected TIFF 60 GB). Real runs will
therefore set `PHASE_SAVE_STRIDE` and `REF_SPACE_SAVE_STRIDE` above 1, and the selection
rule is `_frame_selected_for_stride(i, stride, final_idx)` =
`i == 0 or i == final_idx or i % stride == 0` (`run_F260517_0625_qc.py:232-236`).

### Binding consequences

1. **Frame sets are per artifact kind, and are neither contiguous nor arithmetic.**
   Stride 10 over 200 frames yields `{0, 10, 20, ..., 190, 199}` — 199 is the final frame
   and breaks the stride. `motion`/`phase` follow `PHASE_SAVE_STRIDE`; `refspace_*` follows
   `REF_SPACE_SAVE_STRIDE`; masks, coverage and the raw/projected TIFFs may follow neither.
   `QCBundle` must discover each kind's available frame indices independently by globbing,
   expose them separately, and never assume one kind's set covers another's.
   `frames` as a single list is no longer sufficient: add `frames_for(kind)`, and make
   every accessor raise a clear error naming the kind and index when asked for a frame
   that kind does not have.
2. **The z-major cache must be bounded.** At stride 1 it would reach 90.7 GB. Give it a
   size cap (default 20 GB, configurable), evict least-recently-used, and never evict a
   frame while it is being read. Cold build measured 3.788 s per frame on `/nrs`, so
   warming 200 frames costs about 12.6 minutes: build on demand, and report cache state
   through the manifest so the UI can show what is warm.
3. **A 200-tile montage cannot be served at full resolution.** 200 tiles of 1500x630 is
   189 Mpx; even clamped to an 8192 px edge it is roughly 188 MB of RGB. The montage path
   must downsample per tile before compositing, chosen from the tile count and a target
   output edge (default 2048 px), and must report the applied per-tile scale factor so the
   UI can label the view as decimated rather than let a reader assume full resolution.
4. **Add a per-frame summary layer, and make it the default time view.** Computing, once
   per run and caching to a small `.npz`, the per-(frame, plane) statistics of the
   displacement norm — mean, p95, max — gives a `(n_frames, 20)` array that is a few
   kilobytes and answers "when and where was motion large" without touching a 216 MB file.
   That heatmap, not a 200-tile montage, is the right entry point at this scale; clicking a
   cell drills into the full-resolution plane. Building it costs one full read of the
   motion files (about 43 GB at stride 1), so build it in the background and persist it.

### Interface additions to `qc_bundle.py`

```
frames_for(kind)        -> sorted list of int; kind in
                           {"motion", "phase", "mov", "mask_mov", "coverage",
                            "raw_moving", "projected", "refspace"}
plane_downsampled(kind, frame, k, factor, **kw) -> decimated plane, for montage and thumbnails
field_summary()         -> {"frames": [...], "mean": (F,20), "p95": (F,20), "max": (F,20)}
                           cached to disk; built lazily; states whether it is complete
cache_state()           -> which frames are warm, bytes used, bytes capped
```

---

## HTTP route table (authoritative; server.py implements, index.html consumes)

Bind `127.0.0.1` only. All image routes return `image/png`; all `/api/` routes return
`application/json`. Unknown or out-of-range parameters return HTTP 400 with a JSON body
`{"error": "..."}` naming the parameter — never a silent default, because a silently
substituted frame index produces a picture of the wrong frame.

```
GET  /                       -> static/index.html
GET  /static/<name>          -> static asset

GET  /api/manifest           -> {run_dir, ref_shape_zyx, mov_shape_zyx,
                                 kinds: {kind: [frame indices]},
                                 projection_params, fixed_target_z, cache_state}
GET  /api/metrics            -> {csv_name: {columns: [...], rows: [[...]]}}
                                 plus {mask_support_frac: {frame: float}} so the UI can
                                 state what fraction of voxels the errors describe
GET  /api/summary            -> {frames: [...], mean: [[...]], p95: [[...]], max: [[...]],
                                 complete: bool}   # (n_frames, 20) displacement-norm stats

GET  /img/motion             frame, k, quantity=norm|dx|dy|dz, cmap, lo_pct, hi_pct,
                             base=mov|none, alpha, mask=0|1, quiver=0|1, stride,
                             downsample
GET  /img/plane              kind=mov|raw_moving|projected, frame, k, channel, cmap,
                             lo_pct, hi_pct, downsample
GET  /img/refspace           frame, z, channel, source=precomputed|<job-id>,
                             overlay=0|1, alpha, cmap, lo_pct, hi_pct, downsample
GET  /img/coverage           frame, k, downsample
GET  /img/montage            axis=time|z, quantity, k (when axis=time),
                             frame (when axis=z), cols, max_edge, downsample
                             -> response header X-Tile-Scale carries the applied per-tile
                                scale factor; the UI must display it when it is not 1.0

POST /api/reproject          {field_frame, image_frame, mode=planes|refspace, params:{
                                z_window, upsample_factor, xy_splat_mode,
                                values_interp_order_mem}}
                             -> {job_id, shape, elapsed_s}
                             Result is held in memory and addressed as source=<job_id>
                             by /img/refspace. field_frame != image_frame is the null test
                             and must be allowed.
GET  /api/jobs               -> list of held jobs with their parameters and byte sizes
DELETE /api/jobs/<id>        -> free one
```

Defaults chosen from measurement, not taste: `lo_pct=1`, `hi_pct=99`, `stride=22` (stride
11 costs 0.264 s against a 0.40 s budget; see Correction 2 — if you instead bin the dz
colours and get stride 11 under budget, default to 11 and say so), `alpha=0.6`,
`mask=0` (see Correction 1), `cmap` sequential for `norm` and diverging for `dx|dy|dz`.

Signed quantities must use `diverging_limits` so the colour scale is symmetric about zero.
The server must not let a caller request a diverging map with asymmetric limits.
