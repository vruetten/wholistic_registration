"""Regression tests for utils/motion_correlation_pattern.py findings (audit pass 1)."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wholistic_registration.utils import motion_correlation_pattern as mcp


def _make_mode(response_strength, response_field, activation, episode_id=0, mode_id=0):
    """Build a MotionMode via __new__ + __dict__ (as in the verification log)."""
    mode = mcp.MotionMode.__new__(mcp.MotionMode)
    mode.__dict__.update(
        {
            "episode_id": episode_id,
            "mode_id": mode_id,
            "time_range": (0, len(activation) - 1),
            "activation": np.asarray(activation, dtype=np.float32),
            "response_field": np.asarray(response_field, dtype=np.float32),
            "response_strength": np.asarray(response_strength, dtype=np.float32),
        }
    )
    return mode


def test_b002_safe_corr_constant_traces():
    """_safe_corr scores constant traces via un-demeaned cosine: const/const ~1, const/-const ~-1, const/zero 0, and matches np.corrcoef on the normal path. Regression for B-002 (fixed in 5b3c725)."""
    assert mcp._safe_corr([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0, abs=1e-5)
    assert mcp._safe_corr([5.0, 5.0, 5.0], [-5.0, -5.0, -5.0]) == pytest.approx(-1.0, abs=1e-5)
    assert mcp._safe_corr([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    rng = np.random.default_rng(0)
    a = rng.standard_normal(50).astype(np.float32)
    b = (0.8 * a + 0.2 * rng.standard_normal(50)).astype(np.float32)
    expected = np.corrcoef(a, b)[0, 1]
    assert mcp._safe_corr(a, b) == pytest.approx(float(expected), abs=1e-4)


def test_b009_split_mode_survives_single_nan_pixel():
    """split_mode_to_regions yields the same region count for a response map with one NaN pixel as for the clean map. Regression for B-009 (fixed in c33e6ce)."""
    A = np.zeros((40, 40), dtype=np.float32)
    A[10:20, 10:20] = 1.0
    B = np.zeros((40, 40, 2), dtype=np.float32)
    B[10:20, 10:20, 0] = 0.5
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)

    clean = mcp.split_mode_to_regions(_make_mode(A, B, h))

    A_nan = A.copy()
    A_nan[12, 12] = np.nan
    with_nan = mcp.split_mode_to_regions(_make_mode(A_nan, B, h))

    assert len(clean) == 1
    assert len(with_nan) == len(clean)


def _make_region(episode_id, activation, rng):
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    field = np.zeros((20, 20, 2), dtype=np.float32)
    field[5:15, 5:15, :] = rng.standard_normal((10, 10, 2)).astype(np.float32)
    return mcp.MotionRegion(
        episode_id=episode_id,
        mode_id=0,
        region_id=episode_id,
        time_range=(0, 9),
        activation=activation,
        response_field=field,
        response_strength=np.linalg.norm(field, axis=-1).astype(np.float32),
        region_mask=mask,
    )


def test_b012_activationless_pair_gets_invalid_activation_not_stale_distance():
    """compute_region_distance_matrix_simple handles an activation-less pair (first or after a valid pair) with reason=invalid_activation and dist>=1e6, no UnboundLocalError. Regression for B-012 (fixed in 83fd55b)."""
    rng = np.random.default_rng(1)
    h = np.sin(np.linspace(0, np.pi, 10)).astype(np.float32)
    r_noact = _make_region(0, None, rng)
    r_valid1 = _make_region(1, h, rng)
    r_valid2 = _make_region(2, h * 1.1 + 0.05, rng)

    # Case 1: the activation-less pair is the FIRST pair examined.
    D, info = mcp.compute_region_distance_matrix_simple(
        [r_noact, r_valid1, r_valid2], verbose=False
    )
    assert info[(0, 1)]["reason"] == "invalid_activation"
    assert info[(0, 1)]["compatible"] is False
    assert D[0, 1] >= 1e6

    # Case 2: the activation-less pair comes AFTER a valid pair.
    D2, info2 = mcp.compute_region_distance_matrix_simple(
        [r_valid1, r_valid2, r_noact], verbose=False
    )
    assert info2[(0, 1)]["reason"] == "ok"
    assert info2[(0, 1)]["compatible"] is True
    assert np.isfinite(D2[0, 1]) and D2[0, 1] < 1e6
    for pair in [(0, 2), (1, 2)]:
        assert info2[pair]["reason"] == "invalid_activation"
        assert info2[pair]["compatible"] is False
        assert D2[pair] >= 1e6


def test_b018_artifact_filter_discards_pure_whole_body_motion():
    """filter_episodes_artifacts discards a pure whole-body-motion episode (identical cumulative sinusoid on every patch) at default max_global_corr. Regression for B-018 (fixed in fix/b018-artifact-filter)."""
    T = 40
    valid_mask = np.zeros((24, 24), dtype=bool)
    valid_mask[2:22, 2:22] = True

    region = np.zeros((24, 24), dtype=np.uint8)
    region[10:13, 10:13] = 1  # 9 interior patches, away from the edge zone
    N = int(region.sum())

    # motions_obtain-style cumulative sinusoid, identical across patches
    t = np.arange(T, dtype=np.float32)
    s = 5.0 * np.sin(2.0 * np.pi * t / T)  # cumulative displacement
    motion_abs = np.repeat(s[:, None, None], N, axis=1)
    motion_abs = np.concatenate([motion_abs, motion_abs], axis=2)  # (T, N, 2)
    delta = np.diff(s, prepend=0.0).astype(np.float32)
    motion_delta = np.repeat(delta[:, None, None], N, axis=1)
    motion_delta = np.concatenate([motion_delta, motion_delta], axis=2)  # (T, N, 2)
    global_motion = np.stack([s, s], axis=1)  # (T, 2) cumulative

    ep = mcp.MotionEpisode(
        time_range=(0, T - 1),
        region_mask=region,
        episode_id=0,
        motion_delta=motion_delta,
        motion_abs=motion_abs,
        global_motion=global_motion,
    )

    kept = mcp.filter_episodes_artifacts([ep], valid_mask, verbose=False)

    # the episode IS the whole sample moving together -> must be discarded
    assert len(kept) == 0


def test_b041_unfitted_episode_skipped_and_valueerror():
    """summarize_temporal_basis_likeness skips a default (mode_model={}) episode and _get_BH_from_episode raises ValueError on it. Regression for B-041 (fixed in 867e82c)."""
    ep = mcp.MotionEpisode()
    assert ep.mode_model == {}

    rows = mcp.summarize_temporal_basis_likeness([ep])
    assert len(rows) == 0  # skipped, no KeyError('B')

    with pytest.raises(ValueError):
        mcp._get_BH_from_episode(ep)


def test_b043_tab20_colormap_lookup_works():
    """visualize_episode_regions runs end-to-end on a one-region episode (no AttributeError from the removed cm.get_cmap) and paints its scatter with tab20 colour 0 at the expected patch centres. Regression for B-043 (fixed in 8bfd0e2)."""
    source = inspect.getsource(mcp)
    assert "cm.get_cmap" not in source

    cmap = plt.get_cmap("tab20")  # the exact lookup used at visualize_episode_regions
    assert cmap.N == 20

    # The checks above only exercise matplotlib and the source text.  Actually
    # call the repo function that does the lookup.  Backend is forced to Agg in
    # tests/regressions/conftest.py.
    plt.close("all")
    rng = np.random.default_rng(43)
    region = _make_region(0, np.sin(np.linspace(0, np.pi, 10)).astype(np.float32), rng)
    region.mean_response_vector = np.array([0.3, -0.2], dtype=np.float32)
    region.center_xy = (9.0, 9.0)

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    episode = mcp.MotionEpisode(time_range=(0, 9), region_mask=mask, episode_id=43)
    episode.regions = [region]

    fig = mcp.visualize_episode_regions(episode, patch_size=7, rel_thresh=0.10, show=False)

    assert fig is not None  # None means it bailed out before the colormap lookup

    # Independent recomputation of what the scatter must contain: the pixels of
    # the region's response_strength above 10% of its own max, mapped to patch
    # centres, all painted with tab20 entry 0.
    A = np.asarray(region.response_strength, dtype=np.float32)
    coords = np.argwhere(A > 0.10 * float(np.max(A)))
    assert len(coords) > 0  # guard: the oracle below must not be empty
    expected_xy = np.column_stack([coords[:, 1] * 7 + 3, coords[:, 0] * 7 + 3]).astype(float)

    scatters = [c for c in fig.axes[0].collections if c.get_offsets().shape[0] == len(coords)]
    assert len(scatters) == 1, f"expected exactly one scatter of {len(coords)} points"
    assert np.array_equal(np.asarray(scatters[0].get_offsets()), expected_xy)

    face = np.asarray(scatters[0].get_facecolor())
    assert face.shape == (len(coords), 4)
    assert np.allclose(face[:, :3], np.asarray(cmap(0))[:3]), (
        f"scatter is not painted with tab20 colour 0: got {face[0, :3]}"
    )
    assert face[:, 3].min() >= 0.25 and face[:, 3].max() <= 0.90

    plt.close("all")


def _make_k0_episode():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    N = int(mask.sum())
    T = 5
    ep = mcp.MotionEpisode(
        time_range=(0, T - 1),
        region_mask=mask,
        episode_id=7,
        motion_delta=np.zeros((T, N, 2), dtype=np.float32),
        motion_abs=np.zeros((T, N, 2), dtype=np.float32),
        global_motion=np.zeros((T, 2), dtype=np.float32),
    )
    ep.mode_model = {
        "B": np.zeros((2 * N, 0), dtype=np.float32),
        "H": np.zeros((0, T), dtype=np.float32),
    }
    return ep


def test_b044_k0_episode_source_viz_returns_cleanly():
    """Both source-visualization functions return cleanly on a K=0 mode_model and leave no open figures. Regression for B-044 (fixed in 603282b)."""
    plt.close("all")
    ep = _make_k0_episode()

    diag = mcp.visualize_episode_sources_overview(ep)
    assert diag == []

    result = mcp.compare_sources_to_observed_frames(ep)
    assert result is None

    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# B-017 / B-019 / B-020 — temporal gap-closing semantics
# ---------------------------------------------------------------------------


def _active_trace(pattern):
    """(T,1,1) motionMag where 1 -> active under restMotion=0.5, use_abs_dev=False."""
    a = np.asarray(pattern, dtype=np.float32).reshape(-1, 1, 1)
    return a, np.zeros_like(a) + 0.5


def _run_units(pattern, close_gap_frames, extend_radius=0):
    """Return the MotionUnits of the single patch (getMotionUnit yields (units_map, mask))."""
    mag, rest = _active_trace(pattern)
    motion = np.zeros((mag.shape[0], 1, 1, 2), dtype=np.float32)
    units_map, _mask = mcp.getMotionUnit(
        motion, mag, rest, extend_radius=extend_radius,
        use_gpu=False, close_gap_frames=close_gap_frames, use_abs_dev=False,
    )
    return units_map[0][0]


def _spans(units):
    return sorted((int(u.time_range[0]), int(u.time_range[1])) for u in units)


def test_b019_closes_exactly_gaps_up_to_close_gap_frames():
    """A gap of exactly n closes; a gap of n+1 stays open — for even and odd structure sizes. Regression for B-019."""
    for n in (1, 2, 3, 4):
        closed = [1, 1] + [0] * n + [1, 1]
        assert len(_run_units(closed, n)) == 1, f"gap {n} should close at n={n}"
        open_ = [1, 1] + [0] * (n + 1) + [1, 1]
        assert len(_run_units(open_, n)) == 2, f"gap {n + 1} should stay open at n={n}"


def test_b019_border_runs_are_not_eroded():
    """Border-touching activity survives AND a closeable gap still closes.

    Both halves are needed: asserting only that t=0/t=T-1 survive would also pass
    on code that does no closing at all (the pre-fix CPU path), so the trace also
    contains a gap of exactly n that MUST close. Regression for B-019 (erosion)
    and B-017 (CPU path did not close).
    """
    for n in (1, 2, 3, 5):
        # run at t=0 | closeable gap of n | run | wide gap (stays open) | run at t=T-1
        pattern = [1, 1, 1, 1] + [0] * n + [1, 1] + [0] * 12 + [1, 1, 1, 1]
        T = len(pattern)
        units = _run_units(pattern, n)
        spans = _spans(units)
        assert spans == [(0, 5 + n), (18 + n, T - 1)], f"n={n}: {spans}"
        assert spans[0][0] == 0, f"n={n}: activity at t=0 was eroded"
        assert spans[-1][1] == T - 1, f"n={n}: activity at t=T-1 was eroded"


def _raw_runs(pattern):
    """Independent oracle: the runs of a thresholded trace, with NO gap closing."""
    spans, t = [], 0
    pattern = list(pattern)
    while t < len(pattern):
        if pattern[t]:
            s = t
            while t + 1 < len(pattern) and pattern[t + 1]:
                t += 1
            spans.append((s, t))
        t += 1
    return spans


def test_b017_close_gap_frames_zero_is_a_strict_noop():
    """close_gap_frames=0 (the default) leaves the mask untouched — protects every existing config.

    Asserted against an INDEPENDENT oracle (the raw thresholded runs), not against
    a sibling call with a different argument: comparing 0 against None would be
    satisfied by code that closes gaps in *both* cases. Regression for B-017.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        pattern = (rng.random(30) > 0.5).astype(np.float32)
        expected = _raw_runs(pattern)
        assert _spans(_run_units(pattern, 0)) == expected
        assert _spans(_run_units(pattern, None)) == expected


def test_b017_gpu_branch_also_closes_gaps():
    """Both backends route through the shared closing helper — the GPU call site is not silently dropped.

    No test can execute the GPU branch on a CPU host, so removing its
    `_close_temporal_gaps(...)` call would otherwise leave the whole suite green
    while reintroducing B-017 on the path that actually runs in production.
    This pins the structure instead. Regression for B-017.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(mcp.getMotionUnit)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_close_temporal_gaps"
    ]
    use_gpu_args = sorted(
        ast.unparse(kw.value) for c in calls for kw in c.keywords if kw.arg == "use_gpu"
    )
    assert use_gpu_args == ["False", "True"], (
        f"expected one closing call per backend, got {use_gpu_args}"
    )


def test_b020_explicit_use_gpu_without_cupy_raises_clearly():
    """use_gpu=True without CuPy raises an actionable error, not an opaque NoneType AttributeError. Regression for B-020."""
    if mcp.HAS_CUPY:
        pytest.skip("CuPy present: the GPU path is real here")
    mag, rest = _active_trace([1, 0, 1])
    motion = np.zeros((3, 1, 1, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="CuPy is unavailable"):
        mcp.getMotionUnit(motion, mag, rest, use_gpu=True, use_abs_dev=False)
    # and an unrecognised string is rejected rather than silently selecting GPU
    with pytest.raises(ValueError, match="must be True, False or 'auto'"):
        mcp.getMotionUnit(motion, mag, rest, use_gpu="cpu", use_abs_dev=False)


# ---------------------------------------------------------------------------
# B-118 — getMotionPattern index correspondence across the pattern filters
# ---------------------------------------------------------------------------


def _b118_region(region_id, activation, angle):
    """A MotionRegion on a fixed 10x10 support, distinguished only by its activation and field angle."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:14, 4:14] = 1
    field = np.zeros((20, 20, 2), dtype=np.float32)
    field[4:14, 4:14, 0] = np.cos(angle)
    field[4:14, 4:14, 1] = np.sin(angle)
    return mcp.MotionRegion(
        episode_id=region_id,  # distinct per region: same-episode pairs never cluster
        mode_id=0,
        region_id=region_id,
        time_range=(0, len(activation) - 1),
        activation=np.asarray(activation, dtype=np.float32),
        response_field=field,
        response_strength=np.linalg.norm(field, axis=-1).astype(np.float32),
        region_mask=mask,
        # MotionRegion stores these verbatim; split_mode_to_regions is what
        # normally fills them, and filter_regions_for_patterns reads them.
        strength=1.0,
        area_effective=100.0,
    )


def _b118_episodes():
    """One episode of 8 regions drawn from a fixed seed, clustering into groups of size 2, 1, 2, 3.

    Every region shares one support mask, so the spatial gate admits every pair
    and the pairwise distances vary instead of all collapsing onto
    ``incompatible_dist``.  Tied distances make ``fcluster`` number the clusters
    in descending size, which would place every singleton last and leave the
    member filter shifting nothing; the guard in the test re-checks that.
    """
    rng = np.random.default_rng(4)
    cluster_sizes = (2, 1, 2, 1, 2)
    angles = (0.0, 1.0, 2.0, 2.6, 0.5)
    waves = [rng.standard_normal(24).astype(np.float32) for _ in cluster_sizes]

    regions = []
    region_id = 0
    for size, wave, angle in zip(cluster_sizes, waves, angles):
        for _ in range(size):
            regions.append(_b118_region(region_id, wave, angle))
            region_id += 1

    episode = mcp.MotionEpisode(time_range=(0, 23), episode_id=118)
    episode.regions = regions
    return [episode]


def _check_b118_correspondence(patterns, kept_units, groups, labels, where):
    """Assert groups[i] lists the kept_units indices of patterns[i]'s members, and labels agrees."""
    index_of_unit = {id(u): j for j, u in enumerate(kept_units)}

    for i, pattern in enumerate(patterns):
        assert i < len(groups), (
            f"{where}: patterns has {len(patterns)} entries but groups has "
            f"{len(groups)}, so patterns[{i}] has no group"
        )
        members = sorted(index_of_unit[id(r)] for r in pattern.regions)
        assert sorted(int(g) for g in groups[i]) == members, (
            f"{where}: groups[{i}] = {sorted(int(g) for g in groups[i])} does not "
            f"list the kept_units indices {members} of patterns[{i}]'s members"
        )

    assert len(groups) == len(patterns), (
        f"{where}: groups has {len(groups)} entries against {len(patterns)} patterns"
    )
    assert len(labels) == len(kept_units), (
        f"{where}: labels has {len(labels)} entries against "
        f"{len(kept_units)} kept_units"
    )

    grouped = {u for g in groups for u in map(int, g)}
    for unit_index, label in enumerate(map(int, labels)):
        assert label < len(groups), (
            f"{where}: labels[{unit_index}] = {label} is out of range for "
            f"{len(groups)} returned groups"
        )
        if label >= 0:
            assert unit_index in set(map(int, groups[label])), (
                f"{where}: labels[{unit_index}] = {label} but groups[{label}] = "
                f"{sorted(int(g) for g in groups[label])} does not contain unit "
                f"{unit_index}"
            )
        else:
            assert unit_index not in grouped, (
                f"{where}: labels[{unit_index}] = -1 but unit {unit_index} is "
                f"still listed in a returned group"
            )


def test_b118_groups_and_labels_track_the_filtered_patterns(monkeypatch):
    """getMotionPattern returns groups[i] holding patterns[i]'s member indices, and labels indexing the returned patterns, after the member pre-filter and after the unified-mode quality filter. Regression for B-118."""
    episodes = _b118_episodes()

    # Guard: the clustering must place a singleton BEFORE the last multi-member
    # cluster, otherwise dropping singletons shifts nothing and the assertions
    # below would hold for the unfixed code too.
    _, _, groups_all, _, _ = mcp.getMotionPattern(
        episodes, min_pattern_members=1, compute_unified=False, verbose=False
    )
    sizes = [len(g) for g in groups_all]
    singletons = [i for i, s in enumerate(sizes) if s < 2]
    multis = [i for i, s in enumerate(sizes) if s >= 2]
    assert singletons and multis, f"cluster sizes {sizes} exercise no filtering"
    assert min(singletons) < max(multis), (
        f"cluster order {sizes} does not shift any surviving pattern"
    )
    n_surviving = len(multis)

    # Site 1: the min_pattern_members pre-filter (default 2) drops the singletons.
    patterns, kept_units, groups, labels, info = mcp.getMotionPattern(
        episodes, compute_unified=False, verbose=False
    )
    assert len(patterns) == n_surviving, (
        f"pre-filter kept {len(patterns)} patterns, expected {n_surviving}"
    )
    _check_b118_correspondence(patterns, kept_units, groups, labels, "pre-filter")

    # Site 2: the post-hoc quality filter.  compute_pattern_unified_mode is
    # replaced by a stub whose mask area depends only on member count, so
    # min_unified_area drops exactly the two singletons and nothing else.
    def _stub_unified_mode(pattern, episodes_arg, **kwargs):
        area = 10 if pattern.n_members >= 2 else 1
        mask = np.zeros((20, 20), dtype=bool)
        mask.reshape(-1)[:area] = True
        h = np.ones(24, dtype=np.float32)
        B = np.zeros((20, 20, 2), dtype=np.float32)
        return h, B, mask, {"member_info": []}

    monkeypatch.setattr(mcp, "compute_pattern_unified_mode", _stub_unified_mode)

    patterns2, kept_units2, groups2, labels2, info2 = mcp.getMotionPattern(
        episodes,
        min_pattern_members=1,
        compute_unified=True,
        min_unified_area=5,
        verbose=False,
    )
    assert len(patterns2) == n_surviving, (
        f"quality filter kept {len(patterns2)} patterns, expected {n_surviving}"
    )
    _check_b118_correspondence(patterns2, kept_units2, groups2, labels2, "quality-filter")
