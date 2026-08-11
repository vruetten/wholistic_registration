"""Regression tests for the CPU (NumPy-fallback) import path.

Both findings made code unusable on machines without CuPy/CUDA — the exact
configuration CI and every laptop runs in, which is why they survived so long.
"""

import ast
import inspect
from pathlib import Path

import pytest

from wholistic_registration.utils import CUPY_AVAILABLE, cp

SRC = Path(__file__).resolve().parents[2] / "src" / "wholistic_registration"


def test_b067_directional_chunks_has_no_hard_cupy_import():
    """process_directional_chunks routes through the cp shim, not a hard `import cupy`, so serial CPU runs survive. Regression for B-067."""
    tree = ast.parse((SRC / "core" / "main_function.py").read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "process_directional_chunks"
    )

    hard_imports = [
        alias.name
        for node in ast.walk(fn)
        for alias in getattr(node, "names", [])
        if isinstance(node, ast.Import) and alias.name.split(".")[0] == "cupy"
    ]
    assert hard_imports == [], f"hard cupy import(s) back in the serial path: {hard_imports}"


def test_b067_gpu_only_calls_are_guarded():
    """The device-selection and memory-pool calls run only when CuPy IS available.

    Checks the guard semantically via AST, including polarity: a substring search
    for "CUPY_AVAILABLE" near the call would also pass with the guard inverted
    (`if not CUPY_AVAILABLE:`), which reintroduces B-067 in its worst form —
    running GPU-only calls exclusively on CPU-only machines. Regression for B-067.
    """
    import textwrap

    from wholistic_registration.core import main_function

    tree = ast.parse(textwrap.dedent(inspect.getsource(main_function.process_directional_chunks)))
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    targets = {"cp.cuda.Device", "cp.get_default_memory_pool"}
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        src = ast.unparse(node)
        if src not in targets:
            continue
        tests, parent = [], node
        while getattr(parent, "parent", None) is not None:
            parent = parent.parent
            if isinstance(parent, ast.If):
                tests.append(ast.unparse(parent.test))
        found[src] = tests

    assert set(found) == targets, f"GPU-only calls not found: {targets - set(found)}"
    for call, tests in found.items():
        assert tests, f"{call} is unguarded"
        positive = [t for t in tests if "CUPY_AVAILABLE" in t and "not CUPY_AVAILABLE" not in t]
        assert positive, f"{call} guard polarity is wrong or missing: {tests}"


def test_b034_calflow_cross_resolution_imports_without_cupy():
    """calFlowCrossResolution imports under the NumPy fallback (CUDA kernels are compiled lazily, not at module scope). Regression for B-034."""
    from wholistic_registration.utils import calFlowCrossResolution as cf

    # a pure-CPU function from the module is reachable, which was the whole
    # point: the import bomb took every CPU-safe function down with it
    assert callable(cf.correctMotion)
    assert callable(cf.getNeiDiff)


@pytest.mark.skipif(CUPY_AVAILABLE, reason="CuPy present: kernels compile for real")
def test_b034_kernel_request_raises_clear_error_on_cpu():
    """Asking for a CUDA kernel on the NumPy fallback fails with an explanatory RuntimeError, not AttributeError at import. Regression for B-034."""
    from wholistic_registration.utils import calFlowCrossResolution as cf

    with pytest.raises(RuntimeError, match="requires CuPy with a CUDA device"):
        cf._nanmax_kernel()
    with pytest.raises(RuntimeError, match="requires CuPy with a CUDA device"):
        cf._weighted_avg_kernel()


def test_b034_kernel_sources_are_intact(monkeypatch):
    """The lazy refactor preserved both kernel sources, no cp.RawKernel is constructed at module scope (AST), and each entry point compiles ITS OWN source under its own name and caches it. Regression for B-034.

    The RawKernel construction itself cannot run here (no CUDA), so the
    compile step is driven through a stand-in `cp` — what is pinned is the
    wiring (which source goes with which name, and that it happens lazily),
    not the CUDA compilation.
    """
    import types

    from wholistic_registration.utils import calFlowCrossResolution as cf

    assert "project_to_planes_nanmax_kernel" in cf._PROJECT_TO_PLANES_NANMAX_SRC
    assert "project_to_planes_weighted_avg_kernel" in cf._PROJECT_TO_PLANES_WEIGHTED_AVG_SRC
    assert 'extern "C" __global__' in cf._PROJECT_TO_PLANES_NANMAX_SRC
    assert 'extern "C" __global__' in cf._PROJECT_TO_PLANES_WEIGHTED_AVG_SRC

    # String containment says nothing about *where* the kernels get built, which
    # is the entire finding.  Assert structurally that no RawKernel is
    # constructed at module scope -- a substring search for "RawKernel" would
    # pass with the module-level construction back in place.
    tree = ast.parse((SRC / "utils" / "calFlowCrossResolution.py").read_text())

    def _is_rawkernel_call(sub):
        # match both `cp.RawKernel(...)` and a bare `RawKernel(...)` -- keying
        # only on the Attribute form leaves an `from cupy import RawKernel`
        # (or any local alias) free to reintroduce B-034 undetected
        if not isinstance(sub, ast.Call):
            return False
        func = sub.func
        return (isinstance(func, ast.Attribute) and func.attr == "RawKernel") or (
            isinstance(func, ast.Name) and func.id == "RawKernel"
        )

    module_scope_rawkernels = [
        ast.unparse(node)
        for node in tree.body  # top level only, not ast.walk
        for sub in ast.walk(node)
        if _is_rawkernel_call(sub)
        and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert module_scope_rawkernels == [], (
        f"cp.RawKernel back at module scope (B-034): {module_scope_rawkernels}"
    )

    # And drive the lazy path with a stand-in cp so the source/name pairing is
    # checked, not assumed.  _KERNEL_CACHE is module state shared with
    # test_b034_kernel_request_raises_clear_error_on_cpu, so restore it.
    saved_cache = dict(cf._KERNEL_CACHE)
    cf._KERNEL_CACHE.clear()
    try:
        built = []

        def fake_raw_kernel(source, name):
            built.append((name, source))
            return f"compiled::{name}"

        monkeypatch.setattr(cf, "cp", types.SimpleNamespace(RawKernel=fake_raw_kernel))

        assert cf._nanmax_kernel() == "compiled::project_to_planes_nanmax_kernel"
        assert cf._weighted_avg_kernel() == "compiled::project_to_planes_weighted_avg_kernel"

        assert built == [
            ("project_to_planes_nanmax_kernel", cf._PROJECT_TO_PLANES_NANMAX_SRC),
            ("project_to_planes_weighted_avg_kernel", cf._PROJECT_TO_PLANES_WEIGHTED_AVG_SRC),
        ], "entry point compiled the wrong source or under the wrong name"

        # second call is served from the cache, not recompiled
        assert cf._nanmax_kernel() == "compiled::project_to_planes_nanmax_kernel"
        assert len(built) == 2
    finally:
        cf._KERNEL_CACHE.clear()
        cf._KERNEL_CACHE.update(saved_cache)


def test_shim_is_numpy_when_cupy_unavailable():
    """Sanity anchor for the two tests above: on a CPU-only machine the shim is numpy."""
    if not CUPY_AVAILABLE:
        assert cp.__name__ == "numpy"
