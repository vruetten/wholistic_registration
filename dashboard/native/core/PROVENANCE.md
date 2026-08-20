# Where dashcore came from

Copied on 2026-08-19 from

    /groups/ahrens/home/ruttenv/python_packages/nmf_holy/NMFDemo/dashboard/core

whose newest source file was last modified 2026-08-14. There is no upstream
commit to cite: `nmf_holy` is not a git repository of its own, and the home
directory repo that contains it (`vruetten/wholistic.git`, HEAD b0fedc9 at copy
time) does not track `NMFDemo/dashboard/core` at all. Provenance is a path and
an mtime, nothing stronger.

A copy diverges. Two things are meant to slow that down:

- `tests/check_no_domain_leak.sh` runs on every build and fails if a
  registration-domain word reaches `core/`. Its word list here adds `motion`,
  `refspace` and `wholistic` to the NMF-side list. It deliberately omits
  `frame` and `plane`, which are dashcore's own GUI vocabulary.
- `tests/test_panel_seam.cpp` drives `TablePanel`, `HeatmapPanel` and
  `ImageCanvas` over a grocery list, a random matrix and a solid RGB image,
  using nothing but dashcore's public headers.

## Changes made to the copy

- The domain-leak word list above.
- `npz.{cpp,hpp}` are new. The plan's artifact inventory lists three sparse mask
  artifacts stored as `.npz`, and the source copy reads `.npy` only.
- `detail/npy_header.{cpp,hpp}` are new, and `npy.cpp` was rewritten to call
  them. An `.npz` member arrives as a decompressed buffer rather than a file, so
  the header parse had to stop being file-shaped for both readers to share one
  parse rather than carry two copies of it. The rewrite also reads the 12-byte
  preamble before the declared header length, so a valid file with a long header
  no longer reports "header exceeds available bytes".
- `CMakeLists.txt` and `tests/CMakeLists.txt` list the files above.

## What the source has that this copy does not

`NMFDemo/dashboard/core` moved on 2026-08-20, the day after the copy, and this
copy has not been refreshed since. Absent here: `series_plot_panel.{cpp,hpp}`,
`ui/series_pick.hpp`, and the `ValueDomain::kSigned` / `percentile_range_signed`
pair that ranges a signed quantity symmetrically about zero. Nothing in the
five planned panels needs a series plot; `kSigned` is worth revisiting when the
`motion_field` panel displays a signed component (dx, dy, dz) on a diverging
colormap, because without it zero drifts off the colormap's centre.

Record further edits here, so a later diff against `NMFDemo/dashboard/core`
separates intentional divergence from drift.
