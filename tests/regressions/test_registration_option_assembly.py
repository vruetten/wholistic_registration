"""Regression tests for the option dict wbi_registration_* hands to the flow solver."""

import numpy as np
import toml

from wholistic_registration.utils import calFlow3d_Wei_v1, option, registration


def _write_config(tmp_path, meta):
    config = {
        "MetaData": meta,
        "reference": {"pick_reference_auto": False, "chunk_size": 2},
        "channels": {"dual_channel": False, "k": 0, "function": "raw"},
        "mask": {"maskRange": [1, 500], "thresFactor": 5},
        "pyramid": {"r": 2, "layer": 1, "iter": 2, "tolerance": 1e-4, "smoothPenalty": 0.05},
    }
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(config))
    return str(path)


def _run_capturing_option(monkeypatch, config_path, preset_zratio):
    """Run wbi_registration_3d on a tiny volume with the solver stubbed out; return the option seen by getMotion.

    `option` is a module-level dict shared by every call, so seed it with a known
    value (and let monkeypatch restore it) instead of relying on test ordering.
    """
    monkeypatch.setitem(option, "zRatio", preset_zratio)
    seen = {}

    def fake_getMotion(dat_mov, dat_ref, opt):
        seen.update(opt)
        return np.zeros(dat_mov.shape + (3,), dtype=np.float32), 0.0, None, {}

    monkeypatch.setattr(calFlow3d_Wei_v1, "getMotion", fake_getMotion)
    monkeypatch.setattr(calFlow3d_Wei_v1, "correctMotion", lambda data, motion: data)

    rng = np.random.default_rng(0)
    moving = rng.uniform(0.0, 1000.0, size=(1, 4, 12, 12)).astype(np.float32)
    reference_image = moving[0]

    registration.wbi_registration_3d(
        moving, config_path, reference_image=reference_image, verbose=False
    )
    return seen


def test_b071_zratio_reaches_the_flow_solver(tmp_path, monkeypatch):
    """wbi_registration_3d copies the dataset zRatio from the config into option, instead of leaving the module default in place. Regression for B-071."""
    config_path = _write_config(tmp_path, {"zRatio": 5.5, "Dim": 3})

    seen = _run_capturing_option(monkeypatch, config_path, preset_zratio=27.693)

    assert seen["zRatio"] == 5.5


def test_b071_zratio_falls_back_to_the_module_default_when_absent(tmp_path, monkeypatch):
    """A config with no MetaData.zRatio keeps the in-place default rather than raising KeyError mid-run."""
    config_path = _write_config(tmp_path, {"Dim": 3})

    seen = _run_capturing_option(monkeypatch, config_path, preset_zratio=27.693)

    assert seen["zRatio"] == 27.693
