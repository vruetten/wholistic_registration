# Pass 1 pooled findings (working file; final ledger = audit/pass-1-bugs.md)

Reviewed at commit b975648.

## From R1 (motion_correlation_pattern.py 1–713)

- B-001 🟧 mcp.py:49 — GPU import misspelled (`cupyx.scipy.ndi` not a module) → HAS_CUPY always False; forced use_gpu=True crashes on None. [gpu-unverified]
- B-002 🟨 mcp.py:620 — _safe_corr fallback uses demeaned arrays (a0/b0 re-wrap demeaned a/b) → constant traces score 0 vs intended raw cosine 1.
- B-003 🟨 mcp.py:271 — MotionPattern._summarize early-returns on empty regions leaving attrs (activation_list, prototype_region_map…) undefined → AttributeError on empty pattern.
- B-004 🟨 mcp.py:632 — _resample_1d returns length-0 for empty input (np.repeat of empty) → IndexError in _compute_activation_feature h_rs[0].
- B-005 🟨 mcp.py:684 — _compute_spatial_stats: NaN in region_magnitude → strength=NaN passes `<= eps` guard, poisons all MotionPattern prototype weights.
- B-006 🟨 mcp.py:416 — prototype-region-map average: shape-mismatched maps skipped AFTER weight normalization → prototype systematically dimmed / built from one member if first map odd-shaped.
- B-007 🟨 mcp.py:587 — _zscore_1d nanmean/nanstd but no nan_to_num on output (sibling _zscore_time_matrix cleans) → NaN propagates into correlations on one path only.

Cross-chunk notes from R1: center_xy is (row,col) despite name (consumers at ~2594 & plotting must check swap); cupy_ndi=None crash sites at 792/803/891/909/1082.

## From R4 (motion_correlation_pattern.py 2280–2784)

- B-008 🟧 mcp.py:2555 — merge_small_regions=True silent no-op: fragments dropped by min_region_area filters (2494/2506/2539) BEFORE _merge_or_discard_small_region_masks runs → small_ids always empty; params inert; metadata misreports strategy.
- B-009 🟨 mcp.py:2477 — NaN in mode.response_strength → vmax=NaN → all comparisons False → split_mode_to_regions returns [] silently, whole mode dropped.
- B-010 🟨 mcp.py:2317 — _split_binary_support fallback returns whole disconnected support as one region when all components < min_region_area (latent — function currently uncalled).
- B-011 🟦 mcp.py:2285 — _split_binary_support + _region_centroid dead code, duplicating live splitting logic with divergent defaults (drift hazard).

Cross-chunk notes from R4: mode.activation_resampled never assigned anywhere (aliasing path dead; regions always get len-12 resample while _compute_activation_feature uses 16 — length mismatch if mixed); region_id unique only within episode.

## From R5 (motion_correlation_pattern.py 2785–3804)

- B-012 🟧 mcp.py:3199 — misindented block in compute_region_distance_matrix_simple: invalid_activation branch falls through into distance-combination; first such pair → UnboundLocalError on D_h; later pairs → stale D_h/D_b/sign_j from previous pair, info overwritten compatible=True.
- B-013 🟨 mcp.py:3364 — build_region_graph ignores enforce_spatial_gate=False: unconditional `continue` on spatial_ok False → flag dead.
- B-014 🟨 mcp.py:3744 — find_patterns_overlapping_region ORs its three min_* thresholds (docstring says AND semantics) → min_iou inoperative at defaults.
- B-015 🟨 mcp.py:3134 — same-episode gate: default episode_id -1==-1 marks all hand-built pairs incompatible → only singleton patterns, reason "same_episode".
- B-016 🟨 mcp.py:3639 — pattern_to_binary_mask only reads region_mask; MotionMode units use support_mask → union path silently dead for mode patterns, falls back to 20%-threshold prototype.

R5 confirms R1's B-002 (_safe_corr fallback) independently from the caller side (affects sign decisions at 3311-3312).

## From R2 (motion_correlation_pattern.py 714–1463)

- B-017 🟧 mcp.py:1013 — CPU fallback of getMotionUnit ignores close_gap_frames (GPU path binary_closing at 920-923; CPU path none) → CPU/GPU result divergence; and given B-001, the GPU path is dead anyway so gap-closing never happens at all.
- B-018 🟧 mcp.py:1416 — filter_episodes_artifacts criterion 3 correlates motion_delta vs cumulative global_motion → near-zero corr for pure-global motion (reviewer measured 0.063 sinusoid, 0.0 ramp) → artifact filter never triggers; should use motion_abs.
- B-019 🟨 mcp.py:922 — gap-closing structure ones(close_gap_frames+2) merges gaps 1 frame wider than doc'd AND binary_closing border erosion trims runs touching t=0 (scipy-verified). [gpu-unverified]
- B-020 🟨 mcp.py:900 — explicit use_gpu=True without cupy → AttributeError NoneType.asarray (891 gates only "auto" on HAS_CUPY); same at 792/801, 1082/1087. Overlaps B-001 crash consequence.

R2 independently re-derived B-001 (cross-chunk) and B-002's consequence for criterion 3.

## VERDICTS (V1: B-001..B-003)

- B-001 CONFIRMED-read [gpu-unverified]: cupyx/scipy/__init__ whitelist has no `ndi` (fetched from cupy repo). Crash actually at cp.asarray (~801) not cupy_ndi call. No external caller passes use_gpu → default silently CPU.
- B-002 CONFIRMED-run: _safe_corr([1,1,1],[1,1,1]) = 0.0 (should be ~1); [5,5,5] vs [-5,-5,-5] = 0.0 (should be -1).
- B-003 CONFIRMED-run but LATENT: empty MotionPattern unreachable from in-file callers (groups always ≥1 member); in-file reads use getattr(...,None). Keep 🟨→🟦-ish, API fragility.

## From R3 (motion_correlation_pattern.py 1464–2279)

- B-021 🟧 mcp.py:2069 — use_velocity=True subtracts cumulative global_motion from velocity motion_delta (global_motion only ever set cumulative at 1212/1272) → decomposition fits negated drift ramp; precondition in comment unsatisfiable via own pipeline.
- B-022 🟨 mcp.py:1760 — NaN input disables all fit-loop guards (NaN comparisons False) → 200 iterations, all-NaN modes returned silently (svd path: LinAlgError instead).
- B-023 🟨 mcp.py:2107 — K>N: seeder caps K=min(Kmax,N) silently while K_selected/selected_svd_r2 record uncapped → diagnostics describe unfit model. Latent (needs raised Kmax or tiny episode).

## From R12 (IO.py + motion_stage_cache.py + converters.py)

- B-024 🟧 IO.py:107 — readMeta_new IndexError on single-channel single-z ND2 (3-tuple shape, data_shape[3] OOB) → readND2Frame unusable for such files.
- B-025 🟧 IO.py:244 — readND2Frame assumes (T,Z,C,Y,X) order, never transposes (unlike ensure_5d_tzcyx which builds axis_map); (T,C,Z,...) file → slices picks channels, channel picks z, silently wrong data.
- B-026 🟧 IO.py:548 — downsample_tifs_dask: 5D TZCYX (its documented format) matches NO downsample branch → full-res output with spacing metadata ×downsample_xy → corrupted physical units.
- B-027 🟧 converters.py:19 — save_zarr_as_tiffs_simple writes caller's pre-downsample spacing in embedded metadata while resolution tag uses downsampled → embedded pixel size wrong by xy_downsample.
- B-028 🟧 IO.py:896 — saveZarr_fast(single_file=True): ZipStore never closed → zip central directory unwritten on crash/long-running → corrupt store.
- B-029 🟧 IO.py:202 — normalize_index(-1) → slice(-1,0) = empty selection → frames=-1 silently returns 0-length axis (also 650-652, and [-1] lists).
- B-030 🟨 motion_stage_cache.py:416 — distance_matrix popped from shallow copy AFTER pickling full obj → double storage; info_without_distance_matrix assigned after save, never persisted (dead code).
- B-031 🟨 IO.py:559 — 3D input padded to 5D by prepending axes → stack lands in C slot of TZCYX (z-stack becomes 50-channel image in ImageJ metadata).
- B-032 🟨 IO.py:110 — framerate fallback: bare except then frame_metadata(1) outside handler → single-frame ND2 crashes metadata read entirely, original exception swallowed.
- B-033 🟦 IO.py:779 — saveTiff_new mutates caller's metadata dict in place (data_shape overwritten).

## VERDICTS (V2: B-004..B-007) — all killed
- B-004 REFUTED-as-unreachable: _compute_activation_feature never called anywhere (dead code); IndexError confirmed in isolation.
- B-005 REFUTED-as-unreachable: sole caller's vmax NaN-gate returns [] before _compute_spatial_stats (real symptom is B-009's "NaN kills mode silently", already tracked).
- B-006 REFUTED-as-unreachable: missing renormalization reproduced, but mask_shape constant within a run; no in-file caller mixes shapes. Robustness nit.
- B-007 REFUTED: sole caller uses nanmean + isfinite guards; no NaN poisoning.

## From R8 (calFlowCrossResolution.py 1–1310)

- B-034 🟧 cfcr.py:484 — module-level cp.RawKernel(...) (also 575) → AttributeError at import when shim falls back cp=np → module unimportable on CPU-only machines, kills all consumers; cp.asnumpy at 452/850/1115 same class.
- B-035 🟨 cfcr.py:1214 — detect_significant_mad: MAD==0 branch flags ANY value != median as significant (no threshold) → well-registered foreground marked "wrong region" when >half CPs are background-zeroed.
- B-036 🟨 cfcr.py:452 — FindInitZ returns floor-truncated uint16 z_init while scoring used rint → off-by-one plane for fractional delta_ref_idx (latent; call sites pass 10).
- B-037 🟨 cfcr.py:8 — module-level warnings.filterwarnings("ignore", UserWarning) suppresses UserWarnings process-wide for all libraries.
- B-038 🟦 cfcr.py:832 — dead allocation out = cp.empty_like overwritten by cp.where.
- (cross-chunk flag for R9 range: cfcr.py:2063 `neiDiff[:,:,2] *= zRatio_hr` scales all components at z-plane 2 instead of z-component everywhere — awaiting R9; treat as candidate B-0xx if R9/verifier confirms.)

## From R6 (motion_correlation_pattern.py 3805–5251, diagnostics/viz)

- B-039 🟧 mcp.py:4223 — visualize_episode_regions fallback bg transposed AND in patch units while regions drawn in pixel units (×patch_size misalignment + flip); default path of save_episode_mode_region_gallery.
- B-040 🟧 mcp.py:4248 — region arrows use (v[1],v[0]) while mode quivers use comp0=dx ("xy") for the SAME response_field → the two gallery figures show perpendicular-swapped directions; one is wrong.
- B-041 🟧 mcp.py:4349 — mode_model is None guards never fire (init = {}) → KeyError 'B' instead of skip; _get_BH_from_episode same (3821).
- B-042 🟧 mcp.py:4611 — mode_incremental_contribution/reconstruction maps hardcode motion_abs even when fit used use_velocity=True (stored flag never read) → displacement-vs-velocity mismatch numbers.
- B-043 🟧 mcp.py:4224 — cm.get_cmap("tab20") removed in matplotlib≥3.9 (pin only >=3.7) → AttributeError on current installs.
- B-044 🟨 mcp.py:3978 — K_final==0 episodes → plt.subplots(0,4) ValueError in sources overview/compare (fit explicitly supports K=0).
- B-045 🟦 mcp.py:4294 — 4 viz functions no close/show=False path → figure leak in batch loops (sibling gallery functions do it right).

R6 cross-chunk: mid-file `import numpy as np` at 5249 + `_as_bool_mask_fallback` testing globals() suggests an appended second module — chunk 7 to check duplicate/shadowing defs; motion aligned to t+1 (752) vs Ca stack indexing → possible one-frame offset (chunks 2/7).

## VERDICTS (V3: B-008..B-011)
- B-008 CONFIRMED-run (spy-wrapped helper: small mask never reaches it; helper WOULD merge if fed directly; metadata lies).
- B-009 CONFIRMED-run (1 NaN pixel: 1 region → 0 regions silently; svd path crashes loudly instead; "fixed" path reaches it).
- B-010 REFUTED-as-unreachable (behavior real, zero callers — folds into B-011 cleanup).
- B-011 CONFIRMED-run (repo-wide grep: definitions only).

## From R7 (motion_correlation_pattern.py 4629–7941)
- B-046 🟧 mcp.py:6273 — lag_mode forwarded between functions with incompatible value sets/semantics ("both"/"negative" crash; "positive" doesn't restrict sign in one of them).
- B-047 🟨 mcp.py:5317 — _interval_from_motion_region_scheme_a: time_range (s,s) → interval length s (end = s + max(1,s)).
- B-048 🟨 mcp.py:4762 — unit_to_activation_trace truncates h from front instead of shifting when time_range starts <0 → trace shifted early.
- B-049 🟨 mcp.py:5001 — detect_activation_events_mad MAD=0 on sparse traces → thresh~1e-12, mad_k inert (sibling guards with nanstd fallback).
- B-050 🟨 mcp.py:4964 — non-positive weight replaced with 1.0 which DOMINATES typical 0.01-0.1 weights → degenerate unit dominates class trace.
- B-051 🟨 mcp.py:5533 — truncated boundary event windows counted full vs full-length-only baseline → dependency_score biased down.
- B-052 🟨 mcp.py:5458 — closing structure merge_gap+2 fills gaps ≤ merge_gap+1 (one more than same knob elsewhere: 5007 uses +1).
- B-053 🟦 mcp.py:6524 — 4 more viz functions never close figures (same class as B-045).

## From R13 (small utils batch)
- B-054 🟥 simulation.py:49 — `cupy_ndimge` typo (import is cupy_ndimage) → NameError, generateMotion + whole sim path dead.
- B-055 🟥 preprocess.py:343 — Yunfeng_edge_map uses cp/calculate/label/regionprops, none imported in module → NameError on any call.
- B-056 🟥 simulation.py:384 — plot_publication_metric uses plt, never imported → NameError.
- B-057 🟧 reliableAnalysis.py:399 — cp.asnumpy(R) crashes on np-fallback shim (np has no asnumpy) → kills structural_difference_map/ComputeMask_v2 on CPU.
- B-058 🟧 preprocess.py:307 — canny NMS folds angle with abs() instead of mod-180 → (-157.5,-22.5) diagonals suppressed against wrong diagonal.
- B-059 🟧 visualization.py:348 — plot_deformed_grid_plotly motion branch meshgrid(arange(H),arange(W)) X/Y transposed vs phase branch.
- B-060 🟧 generate_demo_data.py:84 — displacement=None sentinel overwritten by first draw → one velocity reused for ALL cells and frames.
- B-061 🟨 reliableAnalysis.py:306 — sqrt of unclamped variance → NaN pixels (sibling clamps at 320); NaN·0 mask keeps NaN.
- B-062 🟨 reliableAnalysis.py:311 — clip[1]-clip[0] outside `if clip is not None` guard → TypeError on clip=None.
- B-063 🟨 reliableAnalysis.py:447 — np.maximum used where comment block documents minimum required ("# minimum?..." unresolved).
- B-064 🟨 reliableAnalysis.py:229 — debug_dir branch calls .get() unconditionally → AttributeError on np fallback.
- B-065 🟦 visualization.py:57 — threshold param: if/else branches byte-identical, param ignored.

## From R10 (core orchestrator batch)
- B-066 🟥 main_function.py:819 — num_gpus referenced after swallowed exception that never assigns → NameError; advertised serial fallback unreachable on exception paths.
- B-067 🟥 main_function.py:1536 — process_directional_chunks hard `import cupy` (bypasses shim) → serial CPU run crashes after middle block; 1673 needs real cupy every batch.
- B-068 🟥 main_function.py:1108 — base_dsZ from SIZE[2] (=C) instead of SIZE[1] (=Z) (also v4 at 1326) → raw volumes read with C z-planes vs registered Z.
- B-069 🟥 pipeline_vmsr.py:22 — `referece_chunk` kwarg typo → TypeError at import/run.
- B-070 🟧 registration.py:328 — 3D returns motion (Z,Y,X,3) but re-ingests motion_init as (X,Y,Z,3); imresize silently stretches → garbled warm-start every batch after first. [gpu-unverified runtime]
- B-071 🟧 registration.py:262 — option["zRatio"] never set from config on Registration_v3 path → hardcoded 27.693 used regardless of dataset.
- B-072 🟧 registration.py:337 — register_one_frame passes args in wrong positional order to both callees (+ unpacks 2-tuple into one) → crashes on any invocation.
- B-073 🟧 registration.py:170 — `dict["initial_error":ie, ...]` is class subscription → GenericAlias, not dict (also 321) → errors output garbage.
- B-074 🟨 main_function.py:643 — mid window unclamped: window > dataset → negative mid_start wraps to end frames + IndexError.
- B-075 🟨 main_function.py:649 — frame at mid_end registered twice (middle block inclusive + forward start) → overwritten with different reference.
- B-076 🟨 main_function.py:780 — head_mem filled only from batch 0 (`if i == 0`) → backward seed under-filled when reference_chunk > batch_size.
- B-077 🟨 reference.py:82 — block ≤1 frame → ncorr=0 → mean over empty axis → all-NaN reference propagates silently.

## From R15 (v2 io/tests/examples)
- B-078 🟥 v2/io/readers.py:226 — ND2Reader.read_frames misuses nd2.read_frame: arg is T×Z plane-sequence index, returns 2D plane (C first) not (Z,Y,X) volume → 3D ND2 silently yields z-planes-as-timepoints / unselected channels (verified against nd2 lib source).
- B-079 🟧 v2/io/readers.py:610 — TiffSeriesReader reads pages[0].shape (2D page) → n_z=1 for multi-page z-stack files; metadata is_3d=False for 3D data.
- B-080 🟧 v2/io/readers.py:457 — TiffReader: no-T-axis 3D tiff → n_frames=len(pages) → z-planes become fake timepoints.
- B-081 🟧 v2/io/readers.py:471 — Z inference: TCYX with C≠1 → n_z=shape[2]=image height → shape_zyx (512,512,512), is_3d=True for 2D data.
- B-082 🟨 v2/io/readers.py:339 — ZarrReader unpacks all 5D as (T,C,Z,Y,X); own writer emits TZCYX → C/Z swapped, data[channel] selects z-plane.
- B-083 🟨 v2/io/readers.py:167,480 — bare excepts default voxel_size/framerate to 1.0 silently → wrong PhysicalSize stamped into all OME output.
- B-084 🟨 v2/examples/synthetic_example.py:19 — sys.path bootstrap off by one dir (3 parents not 4) → documented invocation crashes without pip install.
- B-085 🟨 v2/tests/test_reference.py:59 — smoothness test passes for zero-averaging reference (ref_var ≤ 1.5×frame var can't fail plausibly).
- B-086 🟨 v2/tests/test_synthetic_data.py:85 — motion test only upper-bounds; zero-motion regression passes suite.
- B-087 🟦 v2/tests/test_io.py:225 — TimeIncrement assert inside conditional loop → vacuous pass if Pixels missing.
- B-088 🟦 v2/tests/conftest.py:45 — synthetic_tiff_data requests fixture it never uses; sample_config used by zero tests.

## From R9 (calFlowCrossResolution.py 1311–2610)
- B-089 🟥 cfcr.py:2063 — neiDiff[:,:,2] *= zRatio_hr scales z-SLICE index 2 (all components) instead of z-component everywhere (siblings use [...,2] at 2100/2132); IndexError if nz≤2; final recompute at 2169 applies no scaling (objective inconsistency). Confirms R8's cross-chunk flag.
- B-090 🟧 cfcr.py:1342 — _make_ball uses reciprocal of module's zRatio convention (phys-per-index) → trap-mask ball anisotropic in wrong direction (±6 z-slices where 2 intended; collapses when zRatio_hr<1); margin at 1430 same.
- B-091 🟨 cfcr.py:2192 — correct_wrong_regions_one_layer has no tol param → option["tol"] ignored whenever wrong_region_enable=True (default); run_F260517 sets 1e-6, gets 1e-3.
- B-092 🟨 cfcr.py:2296 — "highresidual" mode thresholds on error IMPROVEMENT (init−last) not residual → never-improved voxels (the actual bad ones) kept, well-corrected ones excluded.
- B-093 🟦 cfcr.py:2038 — Xcp/Ycp/Zcp/phase_identity_cp computed per call, used only by commented-out code.

## VERDICTS (V5: B-017..B-020) — all confirmed
- B-017 CONFIRMED-read (partial non-parameterized compensation via extend_radius merges ≤2-frame gaps; still diverges for close_gap_frames≥2; no other caller).
- B-018 CONFIRMED-run (synthetic whole-sample motion: as-coded corr=0.063 → filter never fires; motion_abs corr=1.0 → correctly discarded).
- B-019 CONFIRMED-run (scipy: structure s closes gaps ≤ s−1 → +2 closes one wider than doc; border erosion loses (cgf+1)//2 frames, extend_radius restores 1). [gpu-unverified]
- B-020 CONFIRMED-read (independent of B-001: cp=None also when cupy genuinely absent; explicit use_gpu=True → opaque AttributeError).

## From R11 (calFlow3d_Wei_v1 + interp + imresize + calculate)
- B-094 🟧 calFlow3d_Wei_v1.py:975 — cupy_ndimage never imported in module → getMapping NameError on first use (no in-repo caller; getMotion unaffected).
- B-095 🟨 calFlow3d_Wei_v1.py:795 — option["phase"] set → phase_current assigned before existing → UnboundLocalError (stale shape on later layers).
- B-096 🟨 calFlow3d_Wei_v1.py:972 — generate_continuous_H_gpu divides component 0 (x) by zRatio (z scale) — latent, sole caller passes zRatio=1.
- B-097 🟨 imresize.py:37 — any 2D input → IndexError in deriveSizeFromScale before flag2D branch → advertised 2D support unreachable.
- B-098 🟨 imresize.py:111 — imresizemex mode="org" dim==2: wrong squeeze axis (ValueError) + wrong output slot — latent, all callers use "vec".
- B-099 🟨 calculate.py:59 — imfilter output="valid": size-1 leading kernel dim → result[0:0] empty; 3rd dim never cropped — latent, all callers "same".
- B-100 🟨 calFlow3d_Wei_v1.py:160 — boundary gradients halved (clipped coords still /2step) → structure tensor underestimated at faces; z worst (thin volumes: every voxel a z-boundary).
- B-101 🟨 interp.py:36 — "lanczos2"/"lanczos3" map to spline orders 4/5 (not Lanczos); "box"→0 — silent method mismatch, latent (callers use "linear").
- B-102 🟦 calFlow3d_Wei_v1.py:352 — error_log stores GPU-resident volumes (others converted to CPU per in-file rationale). [gpu-unverified]

## VERDICTS (V4: B-012..016)
- B-012 CONFIRMED-run (both modes reproduced: UnboundLocalError when invalid pair first; stale distance 0.0284 w/ compatible=True otherwise; latent trigger — needs activation-less unit).
- B-013 CONFIRMED-run (flag provably dead; no in-repo caller of build_region_graph — low practical impact).
- B-014 CONFIRMED-read (OR vs documented AND; min_iou inoperative at defaults; possibly deliberate — doc-fix-level).
- B-015 REFUTED-as-unreachable (pipeline always assigns unique episode_ids; zero hand-built constructions in repo).
- B-016 CONFIRMED-run (union path dead for mode-units: area 25 vs true union 650; reachable documented API).

## VERDICTS (V6: B-034..038) — all confirmed
- B-034 CONFIRMED-run (exact traceback at line 484 importing with cp=np; consumers f260517/run_F260517 die; package top-level import unaffected).
- B-035 CONFIRMED-run (60% zeros → all 400 foreground values flagged; 40% zeros → none) — but zero live callers at this commit (latent).
- B-036 CONFIRMED-run latent (rint [3,6,8] vs uint16 [3,5,8]; all call sites pass integer 10; float API intended).
- B-037 CONFIRMED-run (process-wide suppression demonstrated; no module= arg).
- B-038 CONFIRMED-read 🟦 [gpu-unverified].

## VERDICTS (V7: B-039..045)
- B-039 CONFIRMED-read (background patch-units+transposed vs markers pixel-units in same path; gallery always hits it).
- B-040 CONFIRMED-read (same data, perpendicular-swapped; adjudication: comp0=axis-0 per map_coordinates → visualize_episode_modes "xy" default + _B_column_to_maps quivers are the WRONG side).
- B-041 CONFIRMED-run (KeyError 'B' both sites; correct idiom `if not model` exists at 4602).
- B-042 CONFIRMED-read (use_velocity stored at 2256, never read anywhere — grep).
- B-043 CONFIRMED-run (venv matplotlib 3.11.1: AttributeError; pin has no upper bound; crashes gallery before B-039 renders).
- B-044 CONFIRMED-run (ValueError via real MotionEpisode with K=0 model; K=0 explicitly supported by producer; leaks first figure before crash).
- B-045 CONFIRMED 3/4 (visualize_episode_regions_overview returns fig → partially refuted for that one).

## From R14 (v2 core/config/pipeline/utils)
- B-103 🟥 v2/core/registration.py:97 — imports nonexistent `wholistic_registration.utils.prep` (only preprocess.py exists); except swallows → every register_batch raises RuntimeError "Legacy motion estimation modules not available" → v2 registration can NEVER run.
- B-104 🟥 v2/core/registration.py:223 — getMotion called (moving, ref, smooth_penalty, option) vs v1 signature (dat_mov, dat_ref, option, verbose) → float lands in option → TypeError; smoothPenalty never delivered.
- B-105 🟥 v2/core/registration.py:160 — option dict missing zRatio/tol/save_ite/smoothPenalty → KeyError 'zRatio' at calFlow3d:321.
- B-106 🟥 v2/core/registration.py:179 — motion init (X,Y,Z,2,3) (copy-paste from v1 2D fake-z path) vs required (X,Y,Z,3); both is_3d branches identical (dead conditional).
- B-107 🟧 v2/core/registration.py:247 — motion.transpose(2,1,0,3,4) on 4-D return → AxisError when save_motion=True.
- B-108 🟧 v2/core/registration.py:198 — 2D path passes bare 2-D arrays into strictly-3D algorithm (v1 stacks fake z=2; v2 dropped it).
- B-109 🟧 v2/core/reference.py:174 — v1's min(len//2,50) cap dropped → averages ~whole window instead of top-correlated half → all references differ from v1.
- B-110 🟧 v2/pipeline/runner.py:157 — no clamp: n_frames < initial_frames → negative middle start → range(-5,35) requested from reader; validate_frame_range helper exists but never called.
- B-111 🟧 v2/pipeline/runner.py:334 — short-chunk pad repeats ONE frame (ref_window_mem[-1:].repeat) though comment says keep-previous → duplicated frame dominates reference (same at 459-468).
- B-112 🟨 v2/core/reference.py:180 — cc_sorted[:,1:ncorr+1] vs v1's [:,1:ncorr] → off-by-one score → possibly different reference frame.
- B-113 🟨 v2/core/reference.py:174 — T=1 window → ncorr=0 → all-NaN reference silently (mirror of B-077).
- B-114 🟨 v2/config/settings.py:141 — intensity_range documented as intensity but used as component SIZE range (bwareafilt3_wei voxel counts).

## VERDICTS (B-060..065) — all confirmed
- B-060 CONFIRMED-run (1 draw instead of 9; all cells identical constant velocity; top-level test doesn't catch it).
- B-061 CONFIRMED-run (precision: exactly-flat doesn't trigger; near-flat/high-offset uint16-scale does — 8448-16038 NaN pixels of 65536; function is local_zscore_difference).
- B-062 CONFIRMED-run (TypeError on clip=None).
- B-063 CONFIRMED-read (git b17778d: commit msg "set back to minimum" but diff flips minimum→maximum + "# minimum?..." — self-documented unresolved state; no downstream compensation).
- B-064 CONFIRMED-run (debug-path only; cp is np here, .get() AttributeError).
- B-065 CONFIRMED-read (dead param; all callers commented out; 🟦).

## VERDICTS (B-046..049)
- B-046 CONFIRMED-run (ValueError cross-API; best_lag=-4 under "positive"; no in-repo caller passes non-default → trap, not active wrongness).
- B-047 REFUTED-as-unreachable (branch needs activation=None; pipeline always sets activation).
- B-048 REFUTED-as-unreachable (all builders clamp time_range start ≥0).
- B-049 CONFIRMED-run (mad_k inert across 3 orders of magnitude on exactly the sparse traces the sole caller produces).

## VERDICTS (B-029..033) — all confirmed
- B-029 CONFIRMED-run (empty selection passes 5D shape check silently; latent — no in-repo caller passes negatives; public reader API).
- B-030 CONFIRMED-run (matrix in pickle AND npz, ~2× storage; dead assignment never persisted).
- B-031 CONFIRMED-run (real output: axes CYX, channels=50 for a 50-plane z-stack; reachable from main_function 1359).
- B-032 CONFIRMED-read (wording fix: fallback is INSIDE except but unguarded; nd2 0.11.3 source confirms IndexError on 1-frame file).
- B-033 CONFIRMED-run (dict mutated; in-repo callers unharmed today; 🟦 stands).

## VERDICTS (B-084..088) — all confirmed
- B-084 CONFIRMED-run (both documented invocations fail with ModuleNotFoundError; bootstrap needs 4 parents).
- B-085 CONFIRMED-run (single-raw-frame "reference": ratio 1.0031 → passes; test can't detect zero averaging).
- B-086 CONFIRMED-run (monkeypatched zero-motion generator: 12/12 tests still pass).
- B-087 CONFIRMED-read (vacuous window = OME-XML without Pixels; sibling test at 209 does it right; 🟦).
- B-088 CONFIRMED-read (unused-but-executed fixture; sample_config dead; 🟦).

## VERDICTS (B-066..071) — all confirmed
- B-066 CONFIRMED-run (UnboundLocalError reproduced; aggravator: bare `from utils import cp` fails on ANY normal install so parallel=True always crashes; pipeline.py dodges via parallel=False).
- B-067 CONFIRMED-read (serial branch calls it twice with device_id=None; even the shim couldn't satisfy get_default_memory_pool at 1673).
- B-068 CONFIRMED-read (five convention sites prove (T,Z,C,Y,X); base_dsZ passed as slices= to Z axis; affects downsample utilities not the registration loop).
- B-069 CONFIRMED-run (AST-verified: no **kwargs; deterministic TypeError).
- B-070 CONFIRMED-read (airtight shape trace 1644→266→imresize; no transpose-back exists; 2D path self-consistent, confirming 3D anomaly; components also never permuted).
- B-071 CONFIRMED-read (only zRatio write is in a demo; main_function never touches utils.option; every pipeline 3D run uses 27.693).

## VERDICTS (B-021..023)
- B-021 CONFIRMED-run latent (22.6× ramp dominance demonstrated; global_motion provably always cumulative; documented public flag, zero current callers).
- B-022 REFUTED (not silent: NaN seeds sort to FRONT → HHt NaN → LinAlgError at line 1696 iteration 0 on BOTH paths; loud crash, not garbage).
- B-023 REFUTED-as-unreachable (needs Kmax>N; min_total_area=30 forces N≥30, doc Kmax≤8; diagnostics-only even when forced).

## VERDICTS (B-099..102)
- B-099 REFUTED-as-unreachable (mechanism confirmed by run; zero output="valid" callers).
- B-100 CONFIRMED-run (face gradients exactly halved for x/y/z on true 3D volumes; BUT the 2-plane fake-3D headline scenario is a no-op — Iz≡0 for duplicated planes; severity narrows to real multi-plane boundaries).
- B-101 REFUTED-as-unreachable (mislabels real; box→nearest defensible; zero callers use those names).
- B-102 CONFIRMED-read 🟦 [gpu-unverified] (error_logs never persisted — overwritten per frame; minor).

## VERDICTS (B-089..093) — all confirmed
- B-089 CONFIRMED-read [gpu-unverified] (shape-mock proves slice-vs-component; IndexError at nz=2 reproduced; git -L shows WIP refactor 7f6a1f5; objective vs returned error use different metrics).
- B-090 CONFIRMED-read [gpu-unverified] (numpy port: z_ratio=5 → ball spans 11 slices vs correct 1; hits current runs at coarse layers since zRatio_hr=1/2^layer<1 even with zRatio_HR=1).
- B-091 CONFIRMED-read (impact correction: run_F260517 dodges via wrong_region_enable=False; actual victim is test_F260517_v2.py frames>75 running 1e-3 vs configured 1e-6).
- B-092 CONFIRMED-read (top-20%-most-IMPROVED excluded, never-improved dropped from bad mask — contradicts name, docstring, AND detection stage's residual criterion).
- B-093 CONFIRMED-run (exhaustive grep: only defs + commented line).

## VERDICTS (B-094..098)
- B-094 REFUTED-as-unreachable (getMapping zero callers; NameError reproduced artificially; sister module imports correctly — stale copy).
- B-095 REFUTED-as-unreachable (inside never-called getMapping; UnboundLocalError reproduced).
- B-096 CONFIRMED-read latent (sister copy cfcr.py:474 has the corrected axis — proves wrong-axis; triply masked today).
- B-097 CONFIRMED-run (all three 2D entry variants IndexError; zero in-repo 2D callers — dead advertised API).
- B-098 CONFIRMED-run (ValueError reproduced; reviewer's intermediate shape slightly off but crash exact; zero mode="org" callers).

## VERDICTS (B-054..059)
- B-054 CONFIRMED-run (NameError; kills generateSimulation.ipynb chain).
- B-055 CONFIRMED-run (NameError cp at 343; no late imports).
- B-056 CONFIRMED-run (NameError plt before any loop).
- B-057 CONFIRMED-run (CPU-fallback only; GPU unaffected).
- B-058 CONFIRMED-run (range correction: wrong-neighbor range is (−157.5,−112.5)∪(−67.5,−22.5) — diagonals only; disk test: ~15% differing edge pixels, under-suppression).
- B-059 REFUTED (motion branch matches core channel↔axis convention — identical to phase branch on identity; cited preprocess convention is the repo outlier; residual: undocumented convention split at module boundary → Pass 4 note).

## VERDICTS (B-050..053)
- B-050 REFUTED (magnitude premise inverted: realistic strengths 3.8–384, so fallback 1.0 is SMALLER than legit weights; zero-strength regions can't arise from pipeline constructor).
- B-051 CONFIRMED-run (bias reproduced: dep −0.0378 vs analytic −0.0371; conservative-only — suppresses edge events, no false positives).
- B-052 CONFIRMED-run (off-by-one exact; framing corrected: three independent gap-merge mechanisms, none documents semantics, Ca path alone means ≤gap+1).
- B-053 CONFIRMED-run (3 figures accumulate; 4 confirmed outliers vs house style; 🟦).

## VERDICTS (B-072..077)
- B-072 REFUTED-as-unreachable (all 3 defects real; zero callers — notebooks define their own local register_one_frame; note as dead-broken code).
- B-073 CONFIRMED-run (silent GenericAlias garbage py≥3.9; all pipeline consumers discard via _; py3.8 would TypeError).
- B-074 CONFIRMED-run (IndexError at 656; negative wrap; large stride → 50 silent duplicate frames instead; config_f338 at exactly 800/800 is the boundary that just barely works).
- B-075 CONFIRMED-run (exactly one frame — mid_end — double-registered on forward seam, overwritten with different reference; backward clean).
- B-076 CONFIRMED-read (shipped config_f2013_0206: reference_chunk=400 frames vs batch 100 → 75% shortfall on first backward chunk).
- B-077 CONFIRMED-run T=1 only (all-NaN ref); T=2-3 degenerate single-frame ref not NaN; needs pathological config — low reachability.

## VERDICTS (B-024..028)
- B-024 CONFIRMED-read (nd2 source: sizes drops singleton axes → (T,Y,X) len-3 → IndexError; any 2D single-channel ND2 kills pipeline via readND2Frame→readMeta_new).
- B-025 REFUTED (nd2 pins C directly before Y,X regardless of loop order — (T,C,Z,...) unproducible; residual T↔Z hypothetical is a different, unestablished mechanism; dims/is_5d dead code note stands).
- B-026 CONFIRMED-run (5D file: data unchanged, metadata claims 4× coarser pixels; scope: pipeline-written tiffs squeeze to 3D and are safe — fires on genuine TZCYX files, its own documented format).
- B-027 CONFIRMED-run (resolution tag 0.8µm vs embedded JSON 0.4µm; the one real caller exercises the buggy branch).
- B-028 CONFIRMED-run (WORSE than claimed: zip is invalid AT FUNCTION RETURN until a gc cycle; hard exit after return → BadZipFile; normal exit repairs via finalizers).

## VERDICTS (B-103..108) — all CONFIRMED-run via layered monkeypatch cascade
- B-103 CONFIRMED-run (RuntimeError at guard 138; runner calls register_batch at 206/312/437 — not dead within v2).
- B-104 CONFIRMED-run (nuance: IndexError not TypeError — numpy scalar subscripted at calFlow3d:314).
- B-105 CONFIRMED-run (KeyError 'zRatio' at calFlow3d:321; no .get() anywhere).
- B-106 CONFIRMED-run (hard ValueError broadcast crash in imresize — not silent).
- B-107 CONFIRMED-run (ValueError at 247 iff return_motion; runner passes save_motion → triggers when motion saving on).
- B-108 CONFIRMED-run (dies even earlier than claimed: getMask binary_opening dimensionality at 156).
- Meta: zero tests exercise FrameRegistrar/register_batch — explains how a 6-layer-deep dead path shipped. With 103-106 fixed + motion saving off, register_batch completes on synthetic 3D.

## VERDICTS (B-078..083) — all confirmed
- B-078 CONFIRMED-read (nd2 source: read_frame = one 2D plane over T×Z sequence; both 3D cases wrong; correct asarray fallback unreachable; primary README example path).
- B-079 CONFIRMED-run (metadata n_z=1 vs read data (2,5,8,8); caveat: v2 registration derives 3D-ness from data ndim — damage is metadata.json/OME output + external consumers).
- B-080 CONFIRMED-run (all four writer variants: z-planes served as timepoints; ZYX axes also double-counts n_z).
- B-081 CONFIRMED-run (TCYX C=2: n_z=16=height; frame data itself correct — metadata-only damage, 🟧 fair).
- B-082 CONFIRMED-run (TZCYX zarr: data[channel] returns z-slice with all channels; caveat: no in-repo zarr round-trip — needs external 5D zarr).
- B-083 CONFIRMED-run (2015-01 namespace → silent (1,1,1) voxel size; zero logging on any default path; propagates into all OME output; registration math unaffected).
