# Native QC dashboard (`wrdash`)

Replaces the browser QC dashboard with a native Dear ImGui binary, following the
`nmf_holy/NMFDemo` generation: a domain-free `dashcore` library plus a
wholistic-specific application whose panels are declared in TOML.

Decisions taken 2026-08-19:

- Architecture: `wrdash` built on `dashcore`, not a single-layer icampsnfr-style app.
- Reprojection: the dashboard reads precomputed artifacts only. No cupy, no GPU,
  no subprocess. Changing projection parameters means re-running the pipeline.
- Display: native on physical `DISPLAY=:1`, per the `launch-dashboard-natively`
  memory. No browser, no xpra.

## Artifact inventory

Everything the panels need is already on disk under the run directory. Measured
against `f260517_0625_qc_v4`, whose `ref_shape_zyx` is (220, 1500, 630) and
`mov_shape_zyx` is (20, 1500, 630).

| Artifact | Path | Shape / dtype | Loader |
|---|---|---|---|
| motion field | `diagnostics/motion_current_f{n}.npy` | (20, 1500, 630, 3) float32 | cnpy |
| phase field | `diagnostics/phase_new_f{n}.npy` | (20, 1500, 630, 3) float32 | cnpy |
| moving membrane | `diagnostics/mov_mem_f{n}.npy` | (20, 1500, 630) float32 | cnpy |
| reference shape | `diagnostics/ref_shape.npy` | (3,) int | cnpy |
| target z | `diagnostics/fixed_target_z.npy` | (20,) | cnpy |
| per-plane z offset | `diagnostics/alignment_qc/target_z_offset_per_plane.npy` | (20,) | cnpy |
| moving mask | `diagnostics/masks_mov/mask_mov_{n:06d}.npz` | sparse bool | cnpy npz |
| reference mask | `diagnostics/mask_ref.npz` | sparse bool | cnpy npz |
| coverage | `diagnostics/coverage/no_coverage_{n:06d}.npz` | sparse bool | cnpy npz |
| projected volumes | `projected_{mem,sparseCell}/*.tif` | (Z, Y, X) | libtiff |
| reference-space volumes | `refspace_{mem,sparseCell}/*.tif` | (Z, Y, X) | libtiff |
| raw moving volumes | `raw_moving_{mem,sparseCell}/*.tif` | (Z, Y, X) | libtiff |
| reference anatomy | `refspace_reference_mem.tif` | (220, 1500, 630), 831 MB | libtiff |
| error tables | `diagnostics/errors_{membrane,sparse}.csv` | CSV | app-layer CSV reader |
| hole / refspace summaries | `diagnostics/{hole,refspace}_summary.csv` | CSV | app-layer CSV reader |
| projection parameters | `diagnostics/projection_params.json` | JSON | nlohmann/json |

Two loaders are new relative to `dashcore` as it stands in NMFDemo:

- **npz.** `cnpy` is already a fetched dependency and provides `npz_load`, so the
  three sparse mask artifacts need no new third-party code.
- **TIFF.** `libtiff` 6.1 and `tiffio.h` are present in the `all` conda env. The
  TIFF reader belongs in the app layer, not in `dashcore`, until a second
  application needs it.

## The dashcore copy

`nmf_holy` is not a standalone git repository — `git rev-parse --show-toplevel`
from inside it returns the home directory, which is itself a repo pointed at
`vruetten/wholistic.git`. A submodule is therefore not available, so `dashcore`
gets vendored into `dashboard/native/core/` with the source path and commit
recorded in a header comment.

The cost of a copy is divergence. Two things hold the seam: the vendored copy
keeps `core/tests/test_panel_seam.cpp` and the `no_domain_leak` ctest, and the
leak check's word list gains the wholistic domain terms — `frame`, `plane`,
`motion`, `refspace` — alongside the NMF ones it already greps for.

## Panels

Five, matching `dashboard/SPEC.md` line 149, each declared in TOML with a `kind`
and an `instance` so two can be opened side by side on different frames.

| Kind | dashcore base | Content |
|---|---|---|
| `motion_field` | `ImageCanvas` | moving-space plane, anatomy base + field overlay, quantity selector (norm, dx, dy, dz), quiver, mask overlay |
| `montage` | `ImageCanvas` | tiled planes across time or z for one quantity |
| `refspace` | `ImageCanvas` | reference-space plane, precomputed volume over reference anatomy |
| `metrics` | `TablePanel` | `errors_*.csv`, `hole_summary.csv`, `refspace_summary.csv` |
| `alignment_qc` | `TablePanel` + `ImageCanvas` | `target_z_offset_per_plane.npy`, `fixed_target_z.npy` |

The canonical orientation rule in `SPEC.md` — planes are (Y, X) = (1500, 630),
fields are (Y, X, 3) ordered (x, y, z), and no module below the loader
transposes — carries over unchanged and is the single most important thing to
preserve in the port.

## Sequencing

1. Vendor `dashcore` into `dashboard/native/core/`; build it and run its own
   ctest (`DISPLAY=:1 ctest --test-dir build`) before writing any wholistic code.
   Exit check: `dashcore_tests` and `no_domain_leak` both pass.
2. Add the run loader: npz + TIFF + CSV readers, and a `Run` struct that
   discovers frames from `phase_new_f*.npy` the way `qc_bundle._discover_frames`
   does. Exit check: a unit test asserting the discovered frame set and the
   loaded shapes against the real run directory.
3. `motion_field` panel alone, with a TOML carrying one run and one panel.
   Exit check: launched on `:1`, screenshot under `logs/dashboard_ui/`, compared
   against the browser dashboard's `/img/motion` PNG for the same frame and
   plane. The browser server stays available for exactly this comparison.
4. The remaining four panels, one per step, each with the same screenshot check.
5. Retire the Python dashboard: keep `SPEC.md`, delete `server.py`,
   `render.py` and `static/`, and keep `qc_bundle.py` only if something still
   consumes it.

## Open questions

- Frame scale. `SPEC.md` Correction 3 says real runs reach 200 frames with
  per-artifact frame sets from `PHASE_SAVE_STRIDE` and `REF_SPACE_SAVE_STRIDE`
  that are neither contiguous nor arithmetic. The native loader should read the
  frame set per artifact kind from the start rather than inheriting the
  browser version's 5-frame assumption.
- Memory. One `motion_current_f{n}.npy` is 227 MB and the reference anatomy is
  831 MB. Whether panels mmap or read is not yet decided; NMFDemo's async
  loading behind a modal is the precedent if reads turn out to block the frame.
