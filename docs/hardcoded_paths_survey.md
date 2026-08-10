# Hardcoded Path Survey (F5)

**Survey only — no code changes yet.** This document inventories all
hardcoded user paths under `src/wholistic_registration/` so they can be
removed in Phase 4 (per `REMEDIATION_PLAN.md`).

Excluded from this survey: `src/wholistic_registration/archive/` and
`__pycache__/` artifacts.

## Path roots observed

| Root | Where it comes from | Owner |
|---|---|---|
| `/home/cyf/wbi/Virginia/...` | Yunfeng Chi's workstation | personal |
| `/nrs/ahrens/Virginia_nrs/...` | Janelia NRS network storage | shared (Ahrens lab) |
| `/groups/ahrens/...` | Janelia groups network storage | shared (Ahrens lab) |

Once the package is released, **none of these paths should be visible to
external users**. They will 404 on every clone outside the lab and leak
filesystem layout otherwise.

## Affected files (30 source files)

### Entry-point scripts (top of the package)
| File | Notes |
|---|---|
| `pipeline.py`        | hardcoded `configFile` path inside `if __name__ == "__main__"` block |
| `pipeline_vmsr.py`   | hardcoded NRS input/output paths in `DefineParams` call |

### Demos (`src/wholistic_registration/demos/`)
| File |
|---|
| `demo2d.py` |
| `demo_338.py` |
| `test_edge_map.py` |
| `test_f2013.py` |
| `test_f2013_reference.py` |
| `demo_0805.ipynb` |
| `demo_0805_f2013.ipynb` |
| `demo_0822.ipynb` |
| `demo_0907.ipynb` |
| `demo_0912.ipynb` |
| `demo_f338.ipynb` |
| `generateSimulation.ipynb` |

### Test scripts / notebooks (`src/wholistic_registration/tests/`)
| File |
|---|
| `test.py` |
| `test_crossresolution_registration.py` (untracked) |
| `zarr_to_tiffseries.py` |
| `test_HR.ipynb` |
| `test_F260517.ipynb` |
| `test_mask.ipynb` |

### Configs (`src/wholistic_registration/configs/`)
TOML files contain absolute `input_path`/`registrated_path`/`mask_path`:
| File |
|---|
| `config_f338.toml` |
| `config_f338_0326.toml` |
| `config_f2013.toml` |
| `config_f2013_0206.toml` |
| `config_f2013_0225.toml` |

### Other
| File | Why |
|---|---|
| `code/wholistic_registration/configs/config_0120.toml` | Stray nested duplicate of `configs/` (slated for deletion in F14) |
| `macros/LoadTiffList.ijm` | ImageJ macro with `/nrs/ahrens/...` path |
| `simulations/average_eva.m` | MATLAB script with absolute output path |
| `simulations/generate_simulation.m` | MATLAB script with absolute output path |
| `pipeline/pipeline.md` | Internal TODO journal references `/nrs/ahrens/...` |

## Remediation strategy (deferred to Phase 4)

Per `REMEDIATION_PLAN.md` §F22:

1. **Scripts** (`pipeline.py`, `pipeline_vmsr.py`, `demos/*.py`):
   move out of the package to `scripts/`, replace hardcoded paths with
   `argparse`/`click` CLI args. Register as `console_scripts` entry
   points where appropriate.
2. **Notebooks** (`demos/*.ipynb`, `tests/*.ipynb`):
   move to `examples/notebooks/`, replace absolute paths with paths
   loaded from an env var (`WHOLISTIC_DATA_DIR`) or a small JSON config
   that ships with the notebook.
3. **Tests** (`tests/*.py`, `tests/*.ipynb`):
   replace with real pytest tests that use synthetic in-memory data
   (see F6 / `tests/conftest.py`).
4. **Config TOMLs** (`configs/config_*.toml`):
   either move out to a per-user `examples/configs/` directory with
   placeholder paths (`<INPUT_PATH>`, `<OUTPUT_PATH>`), or generate them
   at runtime with `DefineParams` so they never land in version control.
5. **ImageJ / MATLAB tooling** (`macros/*.ijm`, `simulations/*.m`):
   move to `tools/imagej/` and `tools/matlab/` (per F15 of audit), and
   parameterize paths via macro arguments or top-of-file variables that
   default to placeholders.

## Tracking

Open one GitHub issue per category above when this remediation is
scheduled. Resolve each by checking that:

```bash
grep -rn -E '/home/cyf|/nrs/ahrens|/groups/ahrens|/Users/' src/ tests/ examples/ \
  --include='*.py' --include='*.toml' --include='*.md' --include='*.ipynb'
```

returns nothing under the file categories you owned.
