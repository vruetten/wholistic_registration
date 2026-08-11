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


def test_b034_kernel_sources_are_intact():
    """The lazy refactor preserved both kernel sources and their entry-point names. Regression for B-034."""
    from wholistic_registration.utils import calFlowCrossResolution as cf

    assert "project_to_planes_nanmax_kernel" in cf._PROJECT_TO_PLANES_NANMAX_SRC
    assert "project_to_planes_weighted_avg_kernel" in cf._PROJECT_TO_PLANES_WEIGHTED_AVG_SRC
    assert 'extern "C" __global__' in cf._PROJECT_TO_PLANES_NANMAX_SRC
    assert 'extern "C" __global__' in cf._PROJECT_TO_PLANES_WEIGHTED_AVG_SRC


def test_shim_is_numpy_when_cupy_unavailable():
    """Sanity anchor for the two tests above: on a CPU-only machine the shim is numpy."""
    if not CUPY_AVAILABLE:
        assert cp.__name__ == "numpy"
