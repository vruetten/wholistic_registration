# Pass 1 — bugs & errors

**Date:** 2026-08-10 · **Commit reviewed:** `b975648` (verified identical to working
tree for all cited files) · **Method:** 15 parallel reviewer subagents (whole-file /
chunked reads against a fixed bug-class checklist) → 114 pooled findings → 20
independent adversarial verifiers, one refute-by-default pass per finding, running
minimal repros in a local CPU venv wherever executable.
**Detailed per-finding evidence, repro transcripts, and verifier reasoning:**
[`pass-1-verification-log.md`](./pass-1-verification-log.md).

## Tally

- **114 findings reported → 96 confirmed · 18 refuted** (refuted entries kept below
  so they aren't re-found).
- Confirmed by severity: **13 🟥** (crash / wrong results on a main path) ·
  **~30 🟧** (wrong results on a plausible path) · rest 🟨 latent/edge and 🟦 minor.
- Verification quality: ~60% of confirmed findings are `CONFIRMED-run`
  (reproduced by executed snippet); the rest `CONFIRMED-read` (mostly GPU-only
  paths, tagged `[gpu-unverified]`, to re-verify on the Janelia server).

## Headline findings

1. **v2 registration has never run** — a six-layer dead cascade, each crash
   verified by fixing the previous layer: nonexistent `prep` import swallowed into
   a misleading RuntimeError (B-103) → wrong `getMotion` argument order (B-104) →
   option dict missing 4 required keys (B-105) → motion init shaped for v1's 2D
   fake-z path (B-106) → 5-axis transpose of a 4-D array (B-107) → 2D path dies in
   `getMask` (B-108). Zero tests exercise `register_batch`, which is how it shipped.
2. **v1 pipeline: every 3D run uses the hardcoded `zRatio=27.693`** regardless of
   the dataset's real anisotropy — the config value is read but never reaches the
   algorithm (B-071). Warm-start motion is also fed back transposed (Z,Y,X,3) into
   an (X,Y,Z,3) slot and silently `imresize`d into garbage every batch after the
   first (B-070).
3. **The GPU accel of cyf's motion module is dead**: `cupyx.scipy.ndi` is a
   misspelling of `ndimage`, silently caught → `HAS_CUPY` always False (B-001);
   consequently `close_gap_frames` gap-closing never runs anywhere (B-017), and
   the artifact filter that should drop whole-body-motion episodes correlates
   delta against cumulative motion and never fires — measured corr 0.06 where the
   fix gives 1.00 (B-018).
4. **calFlowCrossResolution can't even be imported on CPU** (module-level
   `cp.RawKernel` under the numpy fallback, B-034), and on GPU carries a
   wrong-axis unit conversion `neiDiff[:,:,2] *= zRatio_hr` — scales z-slice 2
   instead of the z-component, `IndexError` if nz≤2 (B-089).
5. **IO writes corrupted metadata/files** on several paths: zarr ZipStore archive
   is invalid at the moment `saveZarr_fast` returns (B-028); 5D TIFF downsampling
   silently no-ops while stamping downsampled pixel sizes (B-026); embedded TIFF
   spacing off by the downsample factor (B-027); single-channel 2D ND2s crash
   metadata reading (B-024).
6. **Three functions in `simulation.py`/`preprocess.py` cannot run at all**
   (NameErrors from missing imports: B-054, B-055, B-056) — direct evidence these
   paths have never been executed since the refactor.

## Confirmed findings

Severity · location · claim · status. `[gpu]` = [gpu-unverified] · "(latent)" =
mechanism proven but no in-repo caller currently triggers it.

### v1 core pipeline — `core/main_function.py`, `utils/registration.py`, `utils/reference.py`, pipeline scripts

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-066 | 🟥 | main_function.py:819 | `num_gpus` unbound after failed GPU probe → `parallel=True` (the default) always crashes; bare `from utils import cp` makes the probe fail on every normal install | run |
| B-067 | 🟥 | main_function.py:1536 | unconditional `import cupy` bypasses shim → serial CPU runs die after the middle block | read |
| B-068 | 🟥 | main_function.py:1108,1326 | z-slice list built from `SIZE[2]` (=C) not `SIZE[1]` (=Z) → raw volumes read with C planes (downsample utilities) | read |
| B-069 | 🟥 | pipeline_vmsr.py:22 | `referece_chunk` kwarg typo → TypeError before any work | run |
| B-070 | 🟧 | registration.py:328 | 3D warm-start motion re-ingested transposed; silently imresized → garbled init every batch after first | read `[gpu]` |
| B-071 | 🟧 | registration.py:262 | `option["zRatio"]` never set from config → hardcoded 27.693 for every pipeline 3D run | read |
| B-073 | 🟧 | registration.py:170,321 | `dict["k":v, ...]` is class subscription → errors output is `GenericAlias` garbage, silently discarded | run |
| B-074 | 🟨 | main_function.py:643-656 | mid window unclamped → negative wrap pulls end-frames into "middle", then IndexError (or silent duplicates with large stride) | run |
| B-075 | 🟨 | main_function.py:649/855/1437 | frame at `mid_end` registered twice (middle block + forward start), overwritten with different reference | run |
| B-076 | 🟨 | main_function.py:780-790 | `head_mem` filled only from batch 0 → shipped config_f2013 gets 100/400 frames (75% shortfall) for first backward reference | read |
| B-077 | 🟨 | reference.py:82-120 | 1-frame block → all-NaN reference, warning only (2-3 frames → degenerate single-frame ref); pathological configs only | run |

### calFlowCrossResolution.py (HR cross-resolution registration, active)

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-089 | 🟥 | :2063 | `neiDiff[:,:,2] *= zRatio_hr` scales z-slice 2 (all components) instead of z-component; IndexError if nz≤2; in-loop vs returned error use different metrics | read `[gpu]` |
| B-034 | 🟧 | :484,575 | module-level `cp.RawKernel` → unimportable under numpy fallback; kills all CPU consumers (exact traceback reproduced) | run |
| B-090 | 🟧 | :1342,1430 | `_make_ball` uses reciprocal of the module's zRatio convention → trap-mask ball 11 slices where 1 intended (and collapses at coarse layers — hits current zRatio_HR=1 runs) | read `[gpu]` |
| B-091 | 🟨 | :2192 | `option["tol"]` ignored whenever `wrong_region_enable=True`; actual victim test_F260517_v2 frames>75 run 1e-3 vs configured 1e-6 | read |
| B-092 | 🟨 | :2296 | "highresidual" mode excludes most-IMPROVED voxels, keeps never-improved — inverts name, docstring, and detection criterion | read |
| B-035 | 🟨 | :1214 | MAD=0 branch flags any value ≠ median as significant (latent — zero live callers) | run |
| B-036 | 🟨 | :443/452 | z_init scored with rint but returned floor-truncated (latent — call sites pass integers; float API intended) | run |
| B-037 | 🟨 | :8 | module-level `filterwarnings("ignore", UserWarning)` suppresses UserWarnings process-wide | run |
| B-038 | 🟦 | :832 | dead `cp.empty_like` allocation | read `[gpu]` |
| B-093 | 🟦 | :2038 | `phase_identity_cp` etc. built per call, used only by commented-out code | run |

### Flow primitives — `calFlow3d_Wei_v1.py`, `imresize.py`, `interp.py`, `calculate.py`

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-100 | 🟨 | calFlow3d:160-204 | boundary gradients exactly halved at volume faces (4× in tensor terms); no-op for 2-plane fake-3D, real for true volumes | run |
| B-096 | 🟨 | calFlow3d:972 | x-component divided by zRatio (z-scale) — sister module has the corrected copy, proving wrong axis (latent, triply masked) | read |
| B-097 | 🟨 | imresize.py:37 | 2D input crashes before the flag2D branch → advertised 2D support is dead API | run |
| B-098 | 🟨 | imresize.py:111 | `mode="org"` dim-2 path: wrong squeeze axis (ValueError) + wrong output slot (zero callers) | run |
| B-102 | 🟦 | calFlow3d:352 | error_log stores GPU-resident volumes against in-file rationale (minor; log never persisted) | read `[gpu]` |

### IO & caching — `utils/IO.py`, `converters.py`, `motion_stage_cache.py`

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-024 | 🟧 | IO.py:107 | single-channel single-z ND2 → IndexError in readMeta_new → readND2Frame/pipeline dead for such files (nd2 drops singleton axes — verified in library source) | read |
| B-026 | 🟧 | IO.py:548 | 5D TIFF (its own documented format) matches no downsample branch → full-res data with downsampled spacing metadata | run |
| B-027 | 🟧 | converters.py:19 | embedded TIFF metadata keeps pre-downsample spacing (resolution tag correct) → off by xy_downsample; the one real caller hits it | run |
| B-028 | 🟧 | IO.py:896 | ZipStore never closed → **archive invalid at function return** until a GC cycle; hard exit → BadZipFile | run |
| B-029 | 🟧 | IO.py:202,650 | `frames=-1` → `slice(-1,0)` empty selection returned silently (passes 5D check; latent — public reader API) | run |
| B-032 | 🟨 | IO.py:110 | framerate fallback unguarded inside bare except → single-frame ND2 crashes metadata read with unrelated IndexError | read |
| B-031 | 🟨 | IO.py:559 | 3D→5D pad puts z-stack in C slot → output "50-channel" image (reproduced: axes CYX, channels=50) | run |
| B-030 | 🟨 | motion_stage_cache.py:416 | distance matrix pickled AND saved to npz (2× storage); "without" variant assigned after save — dead | run |
| B-033 | 🟦 | IO.py:779 | saveTiff_new mutates caller's metadata dict | run |

### motion_correlation_pattern.py (cyf, active — 7,941 LOC)

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-001 | 🟧 | :49 | `cupyx.scipy.ndi` misspelling (real: `ndimage`) swallowed → HAS_CUPY always False; GPU paths of whole module silently dead | read `[gpu]` |
| B-017 | 🟧 | :1013 | CPU fallback ignores `close_gap_frames` → CPU/GPU divergence; with B-001, gap-closing dead on every path | read |
| B-018 | 🟧 | :1416 | artifact filter correlates delta vs cumulative → whole-sample-motion episodes score 0.06 (pass) instead of 1.0 (drop); filter can essentially never fire | run |
| B-012 | 🟧 | :3199 | misindented block: invalid-activation pairs get UnboundLocalError or the previous pair's stale distance marked compatible (latent trigger) | run |
| B-021 | 🟧 | :2069 | `use_velocity=True` subtracts cumulative global from velocity — 22.6× drift-ramp domination; precondition unsatisfiable via own pipeline (latent, documented public flag) | run |
| B-041 | 🟧 | :4349,3821 | `mode_model is None` guards never fire (init `{}`) → KeyError 'B' on unfitted episodes | run |
| B-042 | 🟧 | :4611,4170 | diagnostics hardcode motion_abs; stored `use_velocity` flag never read anywhere | read |
| B-043 | 🟧 | :4224 | `cm.get_cmap` removed in matplotlib≥3.9 (pin has no upper bound) → gallery crashes on current installs | run |
| B-039 | 🟧 | :4223 | gallery fallback background transposed + patch-units vs pixel-unit markers → misaligned overlay on default path | read |
| B-040 | 🟧 | :4248 vs 4131 | region arrows and mode quivers use opposite component conventions for the same field; per map_coordinates convention the mode-quiver "xy" default is the wrong side | read |
| B-046 | 🟧 | :6273 | `lag_mode` values/semantics incompatible between sibling APIs; "positive" doesn't restrict sign in one | run |
| B-008 | 🟧 | :2555 | `merge_small_regions=True` silent no-op (fragments pre-filtered before merge helper); metadata misreports strategy | run |
| B-002 | 🟨 | :620 | `_safe_corr` fallback re-wraps demeaned arrays → constant traces score 0 not ±1 | run |
| B-009 | 🟨 | :2477 | one NaN in response map → whole mode silently dropped (guard exists at 4655, missing here) | run |
| B-019 | 🟨 | :922 | gap closing one frame wider than documented + border erosion loses frames | run `[gpu]` |
| B-020 | 🟨 | :891-901 | explicit `use_gpu=True` without cupy → opaque AttributeError (independent of B-001) | read |
| B-044 | 🟨 | :3978,4294 | K_final=0 episodes (explicitly supported) crash `plt.subplots(0,4)` | run |
| B-049 | 🟨 | :5001 | MAD=0 on sparse class traces → `mad_k` inert across 3 orders of magnitude (sibling has the guard) | run |
| B-051 | 🟨 | :5533 | truncated boundary event windows vs full-length baseline → dependency_score biased down (conservative-only) | run |
| B-052 | 🟨 | :5458 | closing structure `merge_gap+2` → fills one more than the other two gap mechanisms; none documents semantics | run |
| B-013 | 🟨 | :3364 | `enforce_spatial_gate=False` dead (unconditional spatial_ok skip) | run |
| B-014 | 🟨 | :3744 | thresholds OR'd vs documented minimum semantics → min_iou inoperative at defaults (possibly deliberate; doc fix) | read |
| B-016 | 🟨 | :3639 | pattern_to_binary_mask reads only region_mask → union path dead for mode-unit patterns (area 25 vs true 650) | run |
| B-003 | 🟦 | :271 | empty MotionPattern half-initialized (AttributeError; unreachable from in-file callers) | run |
| B-011 | 🟦 | :2285,2335 | `_split_binary_support`/`_region_centroid` dead code duplicating live logic with divergent defaults | run |
| B-045 | 🟦 | :4294 etc. | 3 of 4 viz functions leak figures (4th returns fig — partially refuted); B-053: 4 more confirmed leakers | run |

### Small utils — `simulation.py`, `preprocess.py`, `reliableAnalysis.py`, `visualization.py`, `generate_demo_data.py`

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-054 | 🟥 | simulation.py:49 | `cupy_ndimge` typo → NameError; whole simulated-data path dead | run |
| B-055 | 🟥 | preprocess.py:343 | Yunfeng_edge_map uses 4 never-imported names → NameError on any call | run |
| B-056 | 🟥 | simulation.py:384 | `plt` never imported → NameError before any plotting | run |
| B-057 | 🟧 | reliableAnalysis.py:399 | `cp.asnumpy` on numpy fallback → kills reliability_map_v2/ComputeMask_v2 on CPU | run |
| B-058 | 🟧 | preprocess.py:307 | canny NMS folds angle with abs() → wrong diagonal for (−157.5,−112.5)∪(−67.5,−22.5); ~15% differing edge pixels on curved edges | run |
| B-060 | 🟧 | generate_demo_data.py:84 | `displacement=None` sentinel overwritten → ONE velocity for all cells and frames (1 draw where 9 expected); unit test doesn't catch it | run |
| B-061 | 🟨 | reliableAnalysis.py:306 | unclamped float32 variance → thousands of NaN pixels at realistic uint16 offsets (sibling clamps) | run |
| B-062 | 🟨 | reliableAnalysis.py:311 | `clip[1]-clip[0]` outside the None guard → TypeError on clip=None | run |
| B-063 | 🟨 | reliableAnalysis.py:447 | maximum-vs-minimum unresolved: commit b17778d's message says "set back to minimum" but the diff flips TO maximum; `# minimum?...` left in code | read |
| B-064 | 🟨 | reliableAnalysis.py:229 | debug branch `.get()` on numpy fallback → AttributeError | run |
| B-065 | 🟦 | visualization.py:57 | threshold param: identical if/else branches, dead parameter | read |

### v2 — readers, registration, reference, runner, config, tests

> **RESOLVED 2026-08-10 — v2 deleted entirely.** Virginia removed
> `src/wholistic_registration/v2/` (old code) after the review showed its
> registration path had never run (B-103…B-108) and nothing outside v2 imported
> it (Pass 0). All 23 findings below are **RESOLVED-by-deletion**; the table is
> kept as the record of what the tree contained and why it went. Two v1-relevant
> lessons survive the deletion: the B-109/B-112 reference-selection analysis
> (v1's own off-by-one at `utils/reference.py:84` NaNs at T=2 — see B-077), and
> the vacuous-test patterns (B-085/086) to avoid when writing the new test suite.

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| B-103 | 🟥 | v2/core/registration.py:97 | imports nonexistent `utils.prep`; swallowed → register_batch always RuntimeError | run |
| B-104 | 🟥 | :223 | getMotion wrong argument order (scalar subscripted at calFlow3d:314) | run |
| B-105 | 🟥 | :160 | option dict missing zRatio/tol/save_ite/smoothPenalty → KeyError | run |
| B-106 | 🟥 | :179 | motion init (X,Y,Z,2,3) from v1's 2D path; dead is_3d conditional → broadcast crash | run |
| B-078 | 🟥 | v2/io/readers.py:226 | ND2Reader misuses read_frame (plane sequence, not timepoints) → 3D ND2 silently scrambled; primary README path | read |
| B-107 | 🟧 | v2/core/registration.py:247 | 5-axis transpose of 4-D motion → ValueError when motion saving on | run |
| B-108 | 🟧 | :198 | 2D path passes bare 2-D arrays → dies in getMask (v1's fake-z stacking dropped) | run |
| B-109 | 🟧 | v2/core/reference.py:174 | v1's `len//2` cap dropped → averages 39/40 frames vs v1's 20/40 — top-correlated selection effectively disabled | run |
| B-110 | 🟧 | v2/pipeline/runner.py:157 | no clamp for short videos → **silent index-wraparound corruption** at boundary parity, crash otherwise; validate_frame_range never called | run |
| B-111 | 🟧 | :334,459 | short-chunk pad duplicates ONE frame (comment claims keep-previous) → duplicate gets 58% reference weight (live when t_chunk<window_size) | run |
| B-079 | 🟧 | v2/io/readers.py:610 | TiffSeriesReader metadata from pages[0] → n_z=1 for z-stack files (metadata.json/OME damage) | run |
| B-080 | 🟧 | :457 | no-T-axis 3D tiff → z-planes served as timepoints | run |
| B-081 | 🟧 | :471 | TCYX C≠1 → n_z = image height in metadata | run |
| B-082 | 🟨 | :339 | 5D zarr unpacked as TCZYX only; TZCYX (own writer's convention) → channel selects z-slice (needs external zarr) | run |
| B-083 | 🟨 | :167,480 | bare excepts silently default voxel size/framerate to 1.0 → stamped into all OME output; 2015-01 namespace parses silently wrong | run |
| B-112 | 🟨 | v2/core/reference.py:180 | scoring off-by-one vs v1 — v2 is arguably the *fix*; flagged as silent behavioral divergence | read |
| B-113 | 🟨 | :174 | window_size=1 (allowed by validation) → all-NaN reference silently | run |
| B-114 | 🟨 | v2/config/settings.py:141 | `intensity_range` documented as intensity, used as component-size-in-voxels | read |
| B-084 | 🟨 | v2/examples/synthetic_example.py:19 | sys.path bootstrap off by one dir → both documented invocations fail | run |
| B-085 | 🟨 | v2/tests/test_reference.py:59 | smoothness test passes for zero-averaging reference (ratio 1.003) | run |
| B-086 | 🟨 | v2/tests/test_synthetic_data.py:85 | suite passes 12/12 with all motion zeroed (no lower bound anywhere) | run |
| B-087 | 🟦 | v2/tests/test_io.py:225 | TimeIncrement assert vacuous if Pixels missing | read |
| B-088 | 🟦 | v2/tests/conftest.py:45 | fixture requested-but-unused (executes and discards); sample_config dead | read |

## Refuted (kept so they aren't re-found)

B-004 (dead code), B-005 (NaN-gate returns first), B-006 (shape mismatch can't
arise in one run), B-007 (caller is NaN-tolerant), B-010 (zero callers), B-015
(pipeline always sets unique episode_ids), B-022 (**not silent** — NaN crashes
loudly with LinAlgError at iteration 0 on both K-selection paths), B-023 (needs
Kmax>N; N≥30 enforced), B-025 (**nd2 pins C directly before Y/X regardless of
loop order** — the feared axis scramble is unproducible), B-047/B-048 (pipeline
clamps the triggering inputs), B-050 (magnitude premise inverted — realistic
strengths are 4–384, so the 1.0 fallback *under*-weights), B-059 (motion branch
matches the core channel↔axis convention; the cited "repo convention" is the
outlier — residual: undocumented convention split, → Pass 4), B-072 (all three
defects real; zero callers — notebooks define their own local variant), B-094/
B-095 (inside never-called getMapping), B-099 (zero output="valid" callers),
B-101 (zero callers use the mislabeled method names).

## Looks wrong but is fine (selection; full lists in per-reviewer outputs)

- calFlow3d LK solve has no minus sign — correct: residual convention is flipped.
- `Iz/zRatio` twice in calFlow3d — correct: to-physical then back-to-index.
- `imresize` kernel_width=4 for bilinear — matches upstream matlab_imresize; zero-weight taps drop out.
- 1e6 sentinel entering scipy linkage — safe under complete linkage + threshold.
- `batch_frames` comprehension reusing outer `i` — Python 3 comprehension scoping makes it correct.
- `distance_transform_edt(~mask_b)` min-over-mask_a — correct nearest-distance idiom.

## Coverage

All live+demo-only files from Pass 0 reviewed in full (reviewers report zero
unread ranges in scope): core/ (2), utils/ (18 incl. 7-chunk sweep of
motion_correlation_pattern.py and 2-chunk calFlowCrossResolution.py), v2/ (27),
pipeline scripts (2). Excluded per Pass-0 scope notes: `demos/`+src `tests/`
script roots, `archive/`, `ImmuneCell.py` (D-001). GPU/cupy execution paths
reviewed by reading only — every such finding tagged `[gpu-unverified]`.

## Open follow-ups (Pass 1 exit debt)

1. **Regression tests not yet written.** The skill requires a failing test per
   CONFIRMED-run finding (~55). Priority seeds: B-018, B-060, B-058, B-061,
   B-028, B-026, B-027, B-002, B-008, B-009, B-012, B-049, B-110, B-111, B-085/086
   (test-strengthening). → next session, `tests/regressions/`.
2. **`[gpu-unverified]` findings** (B-001, B-017 GPU half, B-019, B-034 GPU
   claims, B-038, B-070, B-089, B-090, B-102) to confirm on the Janelia server.
3. **Coordination with cyf** before fixing anything in
   `motion_correlation_pattern.py` / `calFlowCrossResolution.py` (his active
   branch may have moved).
4. Convention split (channel↔axis vs channel↔axis-1) and v1/v2 divergences
   (B-109/B-112) → Pass 4 architecture agenda.
