"""Regression tests for utils/IO.py, converters.py and motion_stage_cache.py findings."""

import inspect
import json
import pickle
import textwrap
import zipfile

import dask
import numpy as np
import pytest
import tifffile
import zarr

from wholistic_registration.utils import IO, converters, motion_stage_cache


def test_b026_downsample_tifs_dask_handles_5d(tmp_path):
    """downsample_tifs_dask actually downsamples a 5D (TZCYX) tiff instead of passing it through at full resolution. Regression for B-026 (fixed in 8eab9cf)."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, size=(2, 3, 2, 16, 16), dtype=np.uint16)
    tifffile.imwrite(str(in_dir / "vol.tif"), data)

    results = IO.downsample_tifs_dask(
        str(in_dir), str(out_dir), downsample_xy=4, n_workers=1, verbose=False
    )
    assert results == ["vol.tif"]

    out = tifffile.imread(str(out_dir / "vol.tif"))
    assert out.shape == (2, 3, 2, 4, 4)
    np.testing.assert_array_equal(out, data[:, :, :, ::4, ::4])


def test_b027_converter_embeds_downsample_corrected_spacing(tmp_path):
    """save_zarr_as_tiffs_simple embeds the same downsample-corrected spacing as the resolution tag and leaves the caller's metadata dict unmutated. Regression for B-027 (fixed in 9b326e9)."""
    out_dir = tmp_path / "tiffs"
    out_dir.mkdir()

    rng = np.random.default_rng(1)
    stack = rng.random((2, 8, 8)).astype(np.float32)
    metadata = {"spacing_x": 0.4, "spacing_y": 0.4}
    snapshot = dict(metadata)

    tasks = converters.save_zarr_as_tiffs_simple(
        stack, str(out_dir), xy_downsample=2, metadata=metadata
    )
    paths = dask.compute(*tasks)

    assert metadata == snapshot  # caller dict unmutated

    with tifffile.TiffFile(paths[0]) as tif:
        page = tif.pages[0]
        num, den = page.tags["XResolution"].value
        tag_spacing = den / num
        embedded = json.loads(page.description)

    assert tag_spacing == pytest.approx(0.8)
    assert embedded["spacing_x"] == pytest.approx(0.8)
    assert embedded["spacing_y"] == pytest.approx(0.8)
    # embedded spacing must equal the resolution-tag spacing
    assert embedded["spacing_x"] == pytest.approx(tag_spacing)


def test_b028_savezarr_fast_single_file_valid_at_return(tmp_path):
    """saveZarr_fast(single_file=True) leaves a valid zip archive at function return, with all datasets readable. Regression for B-028 (fixed in aba9ee0)."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('name = "regression"\n')

    rng = np.random.default_rng(2)
    mem = rng.random((4, 8, 8)).astype(np.float32)
    ca = rng.random((4, 8, 8)).astype(np.float32)
    ref = rng.random((8, 8)).astype(np.float32)
    save_path = str(tmp_path / "store")

    IO.saveZarr_fast(
        mem, ca, ref, str(config_path), save_path, chunks=(2, 4, 4), single_file=True
    )

    zip_path = save_path + ".zip"
    # The archive must be valid IMMEDIATELY at return (no gc/exit finalizer needed)
    assert zipfile.is_zipfile(zip_path)

    store = zarr.ZipStore(zip_path, mode="r")
    try:
        root = zarr.open(store, mode="r")
        np.testing.assert_array_equal(root["membrane"][:], mem)
        np.testing.assert_array_equal(root["calcium"][:], ca)
        np.testing.assert_array_equal(root["reference"][:], ref)
        assert json.loads(root.attrs["config"])["name"] == "regression"
    finally:
        store.close()


def _extract_normalize_index():
    """Pull readND2Frame's inner normalize_index out via exec (no ND2 file needed)."""
    src = inspect.getsource(IO.readND2Frame)
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("def normalize_index"))
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        block.append(ln)
    ns = {"np": np}
    exec(textwrap.dedent("\n".join(block)), ns)
    return ns["normalize_index"]


def test_b029_normalize_index_negative_one_selects_last_element():
    """readND2Frame's normalize_index maps -1 and [-1] to non-empty last-element slices (0/-2/[2,-1] unchanged). Regression for B-029 (fixed in 6725eb8)."""
    ni = _extract_normalize_index()

    assert ni(-1, "frames") == slice(-1, None)
    assert ni([-1], "frames") == slice(-1, None)
    assert ni(0, "frames") == slice(0, 1)
    assert ni(-2, "frames") == slice(-2, -1)
    assert ni([2, -1], "frames") == [2, -1]
    assert ni(None, "frames") == slice(None)

    # the observable: -1 selects the LAST element, not an empty axis
    arr = np.arange(6)
    np.testing.assert_array_equal(arr[ni(-1, "frames")], [5])
    np.testing.assert_array_equal(arr[ni(-2, "frames")], [4])
    np.testing.assert_array_equal(arr[ni([2, -1], "frames")], [2, 5])


def test_b030_save_patterns_stage_matrix_only_in_npz(tmp_path):
    """save_patterns_stage keeps the distance matrix out of the pickle, stores it in the npz, and load_patterns_stage round-trips it. Regression for B-030 (fixed in 49cf6c7)."""
    rng = np.random.default_rng(3)
    dist = rng.random((20, 20)).astype(np.float64)
    info = {"distance_matrix": dist, "n_clusters": 4}

    out_dir = motion_stage_cache.save_patterns_stage(tmp_path, patterns=[], info=info)

    # caller's dict untouched
    assert "distance_matrix" in info

    with open(out_dir / "objects.pkl", "rb") as f:
        raw = pickle.load(f)
    assert "distance_matrix" not in raw["info"]  # matrix absent from pickle
    assert raw["info"]["n_clusters"] == 4

    npz = np.load(out_dir / "distance_matrix.npz")
    np.testing.assert_array_equal(npz["distance_matrix"], dist)

    loaded = motion_stage_cache.load_patterns_stage(tmp_path)
    np.testing.assert_array_equal(loaded["info"]["distance_matrix"], dist)


def test_b033_savetiff_new_does_not_mutate_caller_metadata(tmp_path):
    """saveTiff_new leaves the caller's metadata dict unchanged while still saving a readable file. Regression for B-033 (fixed in bd1ee1a)."""
    img = np.random.default_rng(4).random((1, 2, 1, 8, 8)).astype(np.float32)
    metadata = {"spacing_x": 0.5, "spacing_y": 0.5}
    snapshot = dict(metadata)
    path = str(tmp_path / "out.tif")

    IO.saveTiff_new(img, path, metadata=metadata, verbose=False)

    assert metadata == snapshot  # no data_shape injected into the caller's dict
    out = tifffile.imread(path)
    np.testing.assert_array_equal(np.asarray(out).reshape(img.shape), img)
