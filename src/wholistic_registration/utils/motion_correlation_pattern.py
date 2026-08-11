"""
motion_correlation_pattern.py

A cleaned, source/mode-response based motion-pattern analysis pipeline.

Core idea
---------
For each MotionEpisode, remove global background cumulative displacement and decompose
patch-wise cumulative displacement into shared-activation motion modes:

    Y_i(t) ~= sum_k h_k(t) * b_ik

where:
    h_k(t): scalar temporal activation of mode k
    b_ik:   2D response vector of patch i to mode k

Then split each mode's response support into spatially coherent MotionRegions.
MotionPattern is built across episodes from MotionRegions whose spatial support,
activation h_k(t), and response vector b are similar.

This file intentionally avoids the old "MotionSource = real biological source" interpretation.
Here MotionMode is only a data-driven source-response/correlation mode.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import exposure
from tifffile import imwrite

try:
    import networkx as nx
except Exception:
    nx = None
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cupy_ndi
    HAS_CUPY = True
except Exception:
    cp = None
    cupy_ndi = None
    HAS_CUPY = False

# =============================================================================
# Data structures
# =============================================================================


class MotionUnit:
    """A patch-level active temporal interval."""

    def __init__(self, time_range=(-1, -1), spatial_coor=(-1, -1), motion=None):
        self.time_range = list(time_range)
        self.spatial_coor = list(spatial_coor)
        self.motion = motion
        self.T = 0 if motion is None else len(motion)

        if motion is not None:
            if self.time_range[1] - self.time_range[0] + 1 != self.T:
                raise ValueError("Time range and motion length conflict.")


class MotionEpisode:
    """A spatiotemporal motion event built from patch-level MotionUnits."""

    def __init__(
        self,
        time_range=(-1, -1),
        region_mask=None,
        episode_id=-1,
        motion_delta=None,
        motion_abs=None,
        global_motion=None,
        global_motion_mode="median",
    ):
        self.time_range = list(time_range)
        self.episode_id = int(episode_id)
        self.spatial_region = (
            np.zeros((1, 1), dtype=np.uint8)
            if region_mask is None
            else np.asarray(region_mask).astype(np.uint8)
        )

        # delta motion is used for event detection / optional debugging
        self.motion_delta = motion_delta
        # alias for compatibility with previous code
        self.motion = motion_delta

        # cumulative displacement is used for mode decomposition
        self.motion_abs = motion_abs
        self.global_motion = global_motion
        self.global_motion_mode = global_motion_mode

        self.modes: List[MotionMode] = []
        self.regions: List[MotionRegion] = []
        self.mode_model: Dict[str, Any] = {}

    def __repr__(self):
        return f"MotionEpisode(id={self.episode_id}, time={self.time_range}, area={int(np.sum(self.spatial_region > 0))})"

    def decompose_motion_modes(self, **kwargs):
        modes = decompose_episode_motion_modes(self, **kwargs)
        self.modes = modes
        return modes

    def split_motion_regions(self, **kwargs):
        regions = split_episode_modes_to_regions(self, **kwargs)
        self.regions = regions
        return regions


class MotionMode:
    """
    A shared-activation source-response mode inside one episode.

    Important: this is NOT assumed to be a real biophysical force source.
    It is a data-driven correlation/mode object represented by h_k(t) and B_k(x).
    """

    def __init__(
        self,
        episode_id=-1,
        mode_id=-1,
        time_range=None,
        activation=None,             # h_k, shape (T,)
        response_field=None,         # B_k(x), shape (X, Y, 2)
        response_strength=None,      # ||B_k(x)||, shape (X, Y)
        response_direction=None,     # normalized B_k(x), shape (X, Y, 2)
        support_mask=None,           # thresholded response support, shape (X, Y)
        compact_response_field=None, # compact B over episode mask, shape (N, 2)
        valid_coords=None,           # compact coords, shape (N, 2)
        background_motion=None,
        reconstructed_motion=None,
        residual_motion=None,
        explained_energy=0.0,
        residual_energy=0.0,
        mode_mass=0.0,
        seed_position=None,
        confidence=1.0,
        metadata=None,
    ):
        self.episode_id = episode_id
        self.mode_id = mode_id
        self.time_range = time_range

        self.activation = activation
        self.response_field = response_field
        self.response_strength = response_strength
        self.response_direction = response_direction
        self.support_mask = support_mask

        self.compact_response_field = compact_response_field
        self.valid_coords = valid_coords

        self.background_motion = background_motion
        self.reconstructed_motion = reconstructed_motion
        self.residual_motion = residual_motion

        self.explained_energy = float(explained_energy)
        self.residual_energy = float(residual_energy)
        self.mode_mass = float(mode_mass)
        self.seed_position = seed_position
        self.confidence = float(confidence)
        self.metadata = metadata or {}

        self.regions: List[MotionRegion] = []


class MotionRegion:
    """
    Spatially coherent region split from a MotionMode response support.

    This is the recommended basic unit for downstream MotionPattern clustering.
    """

    def __init__(
        self,
        episode_id=-1,
        mode_id=-1,
        region_id=-1,
        time_range=None,
        activation=None,
        response_field=None,      # region-local full map, (X, Y, 2)
        response_strength=None,   # region-local full map, (X, Y)
        region_mask=None,         # full map, bool/uint8, (X, Y)
        induced_motion=None,      # representative h(t)*mean_b, (T, 2)
        mean_response_vector=None,
        center_xy=None,
        spatial_cov=None,
        area_effective=0.0,
        strength=0.0,
        metadata=None,
    ):
        self.episode_id = episode_id
        self.mode_id = mode_id
        self.region_id = region_id
        self.component_id = region_id  # compatibility
        self.time_range = time_range

        self.activation = activation
        self.response_field = response_field
        self.response_strength = response_strength
        self.region_mask = region_mask

        self.induced_motion = induced_motion
        self.mean_response_vector = mean_response_vector

        self.center_xy = center_xy
        self.spatial_cov = spatial_cov
        self.area_effective = float(area_effective)
        self.strength = float(strength)
        self.duration = 0 if time_range is None else int(time_range[1] - time_range[0] + 1)
        self.metadata = metadata or {}

        # Compatibility with old MotionComponent / MotionPattern code.
        self.region_magnitude = response_strength
        self.base_func = induced_motion
        self.base_func_resampled = None
        self.temporal_feature = None
        self.activation_resampled = None
        self.activation_feature = None
        self.response_feature = None
        self.time_length_resampled = None

    def show_activation(self, figsize=(7, 4)):
        h = np.asarray(self.activation, dtype=np.float32)
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(h)), h, marker="o")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"Region {self.region_id} activation, ep={self.episode_id}, mode={self.mode_id}")
        plt.xlabel("relative frame")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class MotionPattern:
    """A cross-episode pattern built from MotionRegion objects."""

    def __init__(self, pattern_id: int, regions: Sequence[MotionRegion]):
        self.pattern_id = int(pattern_id)
        self.regions = list(regions)
        self.components = self.regions  # compatibility
        self.n_members = len(self.regions)
        self.episode_ids: List[int] = []
        self.region_keys: List[Tuple[int, int, int]] = []

        self.prototype_activation = None
        self.prototype_induced_motion = None
        self.prototype_region = None
        self.prototype_response_vector = None
        self.center_xy = None
        self.spatial_cov = None
        self.total_strength = 0.0

        self._summarize()

    def _summarize(self, eps=1e-12):
        if len(self.regions) == 0:
            return

        # ------------------------------------------------------------
        # Basic identifiers
        # ------------------------------------------------------------
        self.region_keys = [
            (
                getattr(r, "episode_id", None),
                getattr(r, "mode_id", None),
                getattr(r, "region_id", None),
            )
            for r in self.regions
        ]

        self.episode_ids = sorted(
            list(set([getattr(r, "episode_id", None) for r in self.regions]))
        )

        weights = np.asarray(
            [
                max(float(getattr(r, "strength", 1.0)), 0.0)
                for r in self.regions
            ],
            dtype=np.float32,
        )

        if float(np.sum(weights)) < eps:
            weights[:] = 1.0

        w = weights / np.maximum(np.sum(weights), eps)
        self.total_strength = float(np.sum(weights))

        # ------------------------------------------------------------
        # 1. Activation summary: variable-length, medoid prototype
        # ------------------------------------------------------------
        activations = []
        activation_weights = []
        activation_region_indices = []

        for idx, (wi, r) in enumerate(zip(w, self.regions)):
            h = getattr(r, "activation", None)

            # fallback
            if h is None:
                h = getattr(r, "activation_resampled", None)

            if h is None:
                continue

            h = np.asarray(h, dtype=np.float32).reshape(-1)
            if len(h) == 0:
                continue

            activations.append(h)
            activation_weights.append(float(wi))
            activation_region_indices.append(idx)

        self.activation_list = activations

        if len(activations) > 0:
            activation_weights = np.asarray(activation_weights, dtype=np.float32)
            activation_weights = activation_weights / np.maximum(
                np.sum(activation_weights), eps
            )

            local_medoid_idx = _find_activation_medoid_index(
                activations,
                weights=activation_weights,
                eps=eps,
            )

            global_medoid_idx = activation_region_indices[local_medoid_idx]
            medoid_region = self.regions[global_medoid_idx]
            medoid_activation = activations[local_medoid_idx].copy()

            self.prototype_region_object = medoid_region
            self.prototype_activation = medoid_activation
            self.prototype_activation_type = "medoid"
            self.prototype_activation_region_index = int(global_medoid_idx)

            # Optional fixed-length version only for plotting / feature export.
            self.prototype_activation_resampled = _resample_1d(
                medoid_activation,
                16,
            )
        else:
            self.prototype_region_object = None
            self.prototype_activation = None
            self.prototype_activation_type = "none"
            self.prototype_activation_region_index = None
            self.prototype_activation_resampled = None

        # ------------------------------------------------------------
        # 2. Prototype induced motion
        #    Use medoid region; do not average variable-length curves.
        # ------------------------------------------------------------
        if self.prototype_region_object is not None:
            proto_motion = getattr(self.prototype_region_object, "induced_motion", None)
            if proto_motion is not None:
                self.prototype_induced_motion = np.asarray(
                    proto_motion,
                    dtype=np.float32,
                )
                try:
                    self.prototype_induced_motion_resampled = _resample_vector_func(
                        self.prototype_induced_motion,
                        16,
                    )
                except Exception:
                    self.prototype_induced_motion_resampled = None
            else:
                self.prototype_induced_motion = None
                self.prototype_induced_motion_resampled = None
        else:
            self.prototype_induced_motion = None
            self.prototype_induced_motion_resampled = None

        # ------------------------------------------------------------
        # 3. Prototype region map
        #    Spatial maps have the same image shape, so weighted averaging is OK.
        # ------------------------------------------------------------
        region_maps = []
        wr = []

        for wi, r in zip(w, self.regions):
            rg = getattr(r, "response_strength", None)
            if rg is None:
                continue

            rg = np.asarray(rg, dtype=np.float32)
            s = float(np.sum(rg))

            if s > eps:
                rg = rg / s

            region_maps.append(rg)
            wr.append(wi)

        if len(region_maps) > 0:
            wr = np.asarray(wr, dtype=np.float32)
            wr = wr / np.maximum(np.sum(wr), eps)

            proto_rg = np.zeros_like(region_maps[0], dtype=np.float32)
            for wi, rg in zip(wr, region_maps):
                if rg.shape != proto_rg.shape:
                    # Defensive check. This should usually not happen within one dataset.
                    continue
                proto_rg += wi * rg

            # Keep old field name for compatibility.
            self.prototype_region = proto_rg

            # Clearer name for future code.
            self.prototype_region_map = proto_rg
        else:
            self.prototype_region = None
            self.prototype_region_map = None

        # ------------------------------------------------------------
        # 4. Prototype response vector
        #    Use weighted average, sign-aligned to medoid activation.
        # ------------------------------------------------------------
        vecs = []
        wv = []

        ref_h = self.prototype_activation

        for wi, r in zip(w, self.regions):
            v = getattr(r, "mean_response_vector", None)
            if v is None:
                continue

            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.shape != (2,):
                continue

            h = getattr(r, "activation", None)
            if h is None:
                h = getattr(r, "activation_resampled", None)

            if ref_h is not None and h is not None:
                sign = _sign_by_resampled_corr(ref_h, h, target_len=16)
                v = sign * v

            vecs.append(v)
            wv.append(wi)

        if len(vecs) > 0:
            wv = np.asarray(wv, dtype=np.float32)
            wv = wv / np.maximum(np.sum(wv), eps)

            proto_v = np.zeros(2, dtype=np.float32)
            for wi, v in zip(wv, vecs):
                proto_v += wi * v

            self.prototype_response_vector = proto_v.astype(np.float32)
        else:
            self.prototype_response_vector = None

        # ------------------------------------------------------------
        # 5. Spatial center / covariance
        # ------------------------------------------------------------
        centers = []
        wc = []

        for wi, r in zip(w, self.regions):
            c = getattr(r, "center_xy", None)
            if c is None:
                continue

            c = np.asarray(c, dtype=np.float32)

            if c.shape == (2,) and np.all(np.isfinite(c)):
                centers.append(c)
                wc.append(wi)

        if len(centers) > 0:
            centers = np.stack(centers, axis=0)
            wc = np.asarray(wc, dtype=np.float32)
            wc = wc / np.maximum(np.sum(wc), eps)

            c = np.sum(centers * wc[:, None], axis=0)

            self.center_xy = c.astype(np.float32)

            dc = centers - c[None, :]
            self.spatial_cov = (dc.T @ (dc * wc[:, None])).astype(np.float32)
        else:
            self.center_xy = None
            self.spatial_cov = None

    def summary_dict(self):
        return {
            "pattern_id": self.pattern_id,
            "n_members": self.n_members,
            "episode_ids": self.episode_ids,
            "region_keys": self.region_keys,
            "center_xy": None if self.center_xy is None else self.center_xy.tolist(),
            "total_strength": self.total_strength,
        }


# =============================================================================
# Generic helpers
# =============================================================================
def _safe_float(x, default=0.0):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _as_bool_mask(mask):
    mask = np.asarray(mask)
    return mask.astype(bool)


def _normalize_1d(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m = np.nanmax(np.abs(x)) if x.size > 0 else 0.0
    if not np.isfinite(m) or m < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


def _smooth_1d(x, win=3):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if win is None or win <= 1:
        return x.copy()

    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)

    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _compute_dff_stack(ca_patch_stack, baseline_percentile=20, eps=1e-6):
    """
    Compute dF/F along time axis.

    Supports:
        ca_patch_stack: (T, X, Y)
        ca_patch_stack: (T, Z, X, Y)
        or more generally: (T, *spatial_shape)
    """
    ca = np.asarray(ca_patch_stack, dtype=np.float32)

    if ca.ndim < 2:
        raise ValueError(f"ca_patch_stack should have shape (T, *spatial), got {ca.shape}")

    base = np.nanpercentile(
        ca,
        baseline_percentile,
        axis=0,
        keepdims=True,
    )

    dff = (ca - base) / (np.abs(base) + eps)
    dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)

    return dff.astype(np.float32)


def _zscore_1d(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    mu = np.nanmean(x)
    sd = np.nanstd(x)

    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mu) / (sd + eps)).astype(np.float32)


def _zscore_time_matrix(X, eps=1e-12):
    """
    X: (T, P)
    z-score each spatial trace over time.
    """
    X = np.asarray(X, dtype=np.float32)

    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)

    Z = (X - mu) / (sd + eps)
    Z[:, (sd.reshape(-1) < eps)] = 0.0
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    return Z.astype(np.float32)


def _safe_corr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        L = min(a.size, b.size)
        a = a[:L]
        b = b[:L]
    a0 = a
    b0 = b
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        # fall back to cosine without demeaning (on the original inputs)
        return float(np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + eps))
    return float(np.dot(a, b) / (na * nb + eps))


def _resample_1d(x, target_len=16):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = len(x)
    if T == target_len:
        return x.copy()
    if T <= 1:
        return np.repeat(x, target_len).astype(np.float32)
    old = np.linspace(0.0, 1.0, T)
    new = np.linspace(0.0, 1.0, target_len)
    return np.interp(new, old, x).astype(np.float32)


def _resample_vector_func(v, target_len=16):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"Expected vector function shape (T,2), got {v.shape}")
    T = v.shape[0]
    if T == target_len:
        return v.copy()
    if T <= 1:
        return np.repeat(v, target_len, axis=0).astype(np.float32)
    old = np.linspace(0.0, 1.0, T)
    new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, 2), dtype=np.float32)
    out[:, 0] = np.interp(new, old, v[:, 0])
    out[:, 1] = np.interp(new, old, v[:, 1])
    return out


def _normalize_energy(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v.reshape(-1))
    return v / max(float(n), eps)


def _compute_temporal_feature_from_induced_motion(induced_motion, resampled_len=16, eps=1e-12):
    bf = np.asarray(induced_motion, dtype=np.float32)
    bf_norm = _normalize_energy(bf, eps=eps)
    bf_rs = _resample_vector_func(bf_norm, target_len=resampled_len)
    mag = np.linalg.norm(bf, axis=1)
    mag_rs = _resample_1d(mag / max(float(np.sum(mag)), eps), target_len=resampled_len)
    feat = np.concatenate([bf_rs.reshape(-1), mag_rs], axis=0).astype(np.float32)
    return feat, bf_rs


def _compute_activation_feature(h, resampled_len=16, eps=1e-12):
    h = np.asarray(h, dtype=np.float32).reshape(-1)
    h_rs = _resample_1d(h, target_len=resampled_len)
    h_rs = h_rs / max(float(np.linalg.norm(h_rs)), eps)
    dh = np.diff(h_rs, prepend=h_rs[0])
    feat = np.concatenate([h_rs, dh], axis=0).astype(np.float32)
    return feat, h_rs.astype(np.float32)


def _compute_spatial_stats(region_magnitude, eps=1e-12):
    R = np.asarray(region_magnitude, dtype=np.float32)
    if R.ndim != 2:
        raise ValueError(f"region_magnitude must be 2D, got {R.shape}")
    strength = float(np.sum(np.maximum(R, 0.0)))
    if strength <= eps:
        return (
            np.array([np.nan, np.nan], dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            0.0,
            0.0,
        )
    coords = np.argwhere(R > 0)
    w = R[R > 0].astype(np.float32)
    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)
    ws = float(np.sum(w))
    cx = float(np.sum(w * x) / max(ws, eps))
    cy = float(np.sum(w * y) / max(ws, eps))
    dx = x - cx
    dy = y - cy
    cov = np.array(
        [
            [float(np.sum(w * dx * dx) / max(ws, eps)), float(np.sum(w * dx * dy) / max(ws, eps))],
            [float(np.sum(w * dx * dy) / max(ws, eps)), float(np.sum(w * dy * dy) / max(ws, eps))],
        ],
        dtype=np.float32,
    )
    area_eff = float((ws ** 2) / max(float(np.sum(w ** 2)), eps))
    return np.array([cx, cy], dtype=np.float32), cov, area_eff, strength


# =============================================================================
# Patch motion and episode extraction
# =============================================================================


def motions_obtain(motion, mask, patch_size, return_abs=False):
    """
    Compute patch-level frame-to-frame motion and cumulative displacement.

    Parameters
    ----------
    motion : ndarray, shape (T,H,W,2)
        Cumulative displacement relative to reference.
    mask : ndarray, shape (H,W)
    patch_size : int
    return_abs : bool
        If True, return (motion_delta, motion_abs_aligned, mask_patched).
        motion_delta[t] = patch_abs[t+1] - patch_abs[t]
        motion_abs_aligned[t] = patch_abs[t+1]

    Returns
    -------
    motion_delta : (T-1, Xp, Yp, 2)
    motion_abs_aligned : (T-1, Xp, Yp, 2), optional
    mask_patched : (Xp, Yp), bool
    """
    # motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 4 or motion.shape[-1] != 2:
        raise ValueError(f"Expected motion shape (T,H,W,2), got {motion.shape}")

    T, H, W, _ = motion.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    H_ = n_rows * patch_size
    W_ = n_cols * patch_size

    motion = motion[:, :H_, :W_, :]
    blocks = motion.reshape(T, n_rows, patch_size, n_cols, patch_size, 2)
    patch_abs = blocks.mean(axis=(2, 4))  # (T, Xp, Yp, 2)
    patch_delta = patch_abs[1:] - patch_abs[:-1]
    patch_abs_aligned = patch_abs[1:]

    mask = np.asarray(mask)
    mask_patched = (
        mask[:H_, :W_].reshape(n_rows, patch_size, n_cols, patch_size).mean(axis=(1, 3)) > 0.3
    )

    if return_abs:
        return patch_delta.astype(np.float32), patch_abs_aligned.astype(np.float32), mask_patched.astype(bool)
    return patch_delta.astype(np.float32), mask_patched.astype(bool)


def estimate_rest_state_motion(
    motionMag_patched,
    window_size_t=21,
    window_size_xy=5,
    scale=5.0,
    use_gpu="auto",
    max_mad_t=21,
    return_median=False,
):
    """
    Estimate resting-state motion fluctuation.

    Uses Median Absolute Deviation (MAD) from a local spatiotemporal median:
        restMotion = scale * 1.4826 * median(|motionMag - median_local|)

    If CuPy is available, use GPU median filters, matching the old implementation.

    Parameters
    ----------
    max_mad_t : int
        Cap on the MAD temporal window. Default 21 prevents the MAD filter
        from becoming excessively expensive on long recordings (where
        T//4 could be 100+).
    return_median : bool
        If True, also return the local median motion (needed by getMotionUnit
        when use_abs_dev=True, so abs_dev = |motionMag - median_local| can be
        compared against restMotion instead of raw motionMag).
    """
    use_gpu = (HAS_CUPY if use_gpu == "auto" else bool(use_gpu))

    motionMag_np = np.asarray(motionMag_patched, dtype=np.float32)
    T, X, Y = motionMag_np.shape

    wt = int(max(3, min(window_size_t, T if T % 2 == 1 else max(T - 1, 3))))
    mad_t = min(max(3, T // 4), int(max_mad_t))

    if use_gpu:
        motion_gpu = cp.asarray(motionMag_np)

        median_local = cupy_ndi.median_filter(
            motion_gpu,
            size=(wt, window_size_xy, window_size_xy),
            mode="reflect",
        )

        abs_dev = cp.abs(motion_gpu - median_local)

        mad_local = cupy_ndi.median_filter(
            abs_dev,
            size=(mad_t, window_size_xy + 2, window_size_xy + 2),
            mode="reflect",
        )

        rest_state_motion = scale * 1.4826 * mad_local
        result = cp.asnumpy(rest_state_motion).astype(np.float32)
        if return_median:
            return result, cp.asnumpy(median_local).astype(np.float32)
        return result

    # CPU fallback
    median_local = ndi.median_filter(
        motionMag_np,
        size=(wt, window_size_xy, window_size_xy),
        mode="reflect",
    )

    abs_dev = np.abs(motionMag_np - median_local)

    mad_local = ndi.median_filter(
        abs_dev,
        size=(mad_t, window_size_xy + 2, window_size_xy + 2),
        mode="reflect",
    )

    result = (scale * 1.4826 * mad_local).astype(np.float32)
    if return_median:
        return result, median_local
    return result


def getMotionUnit(
    motion_patched,
    motionMag_patched,
    restMotion,
    extend_radius=1,
    save_motion=False,
    use_gpu="auto",
    close_gap_frames=0,
    use_abs_dev=True,
    median_local=None,
    median_window_t=21,
    median_window_xy=5,
):
    """
    Extract active intervals for each patch.

    GPU path follows the old implementation:
    - active mask on GPU
    - start/end interval detection on GPU
    - only interval grouping on CPU

    save_motion=False is recommended for the new pipeline.

    Parameters
    ----------
    close_gap_frames : int
        If > 0, apply temporal binary closing to the active mask before
        start/end detection. This merges nearby active intervals separated
        by gaps <= close_gap_frames, dramatically reducing CPU loop time
        when using noisy signals like cumulative displacement.
    use_abs_dev : bool
        If True (default), the active condition is:
            |motionMag - median_local| > restMotion
        i.e. deviation from local baseline exceeds MAD threshold.
        If False (old behavior), the condition is:
            motionMag > restMotion
    median_local : ndarray or None
        Precomputed local median of motionMag, shape (T, X, Y). Only used
        when use_abs_dev=True. If None, it is computed inside this function.
        Pass the second return value of estimate_rest_state_motion(..., return_median=True).
    median_window_t : int
        Temporal window for median_local computation (only used when
        use_abs_dev=True and median_local is not provided).
    median_window_xy : int
        Spatial window for median_local computation (only used when
        use_abs_dev=True and median_local is not provided).
    """
    use_gpu = (HAS_CUPY if use_gpu == "auto" else bool(use_gpu))

    motion_np = np.asarray(motion_patched, dtype=np.float32)
    motionMag_np = np.asarray(motionMag_patched, dtype=np.float32)
    rest_np = np.asarray(restMotion, dtype=np.float32)

    T, X, Y, _ = motion_np.shape

    if use_gpu:
        motionMag_gpu = cp.asarray(motionMag_np)
        rest_gpu = cp.asarray(rest_np)

        if use_abs_dev:
            # active: deviation from local baseline exceeds threshold
            if median_local is not None:
                median_gpu = cp.asarray(median_local, dtype=np.float32)
            else:
                wt = int(max(3, min(int(median_window_t), T if T % 2 == 1 else max(T - 1, 3))))
                median_gpu = cupy_ndi.median_filter(
                    motionMag_gpu,
                    size=(wt, int(median_window_xy), int(median_window_xy)),
                    mode="reflect",
                )
            abs_dev_gpu = cp.abs(motionMag_gpu - median_gpu)
            active = abs_dev_gpu > rest_gpu  # (T, X, Y)
        else:
            active = motionMag_gpu > rest_gpu  # (T, X, Y)

        # Merge nearby active gaps on GPU to reduce CPU loop iterations
        if close_gap_frames is not None and close_gap_frames > 0:
            import cupy as _cp
            structure = _cp.ones((int(close_gap_frames) + 2, 1, 1), dtype=bool)
            active = cupy_ndi.binary_closing(active, structure=structure)

        prev_active = cp.zeros_like(active)
        prev_active[1:] = active[:-1]

        next_active = cp.zeros_like(active)
        next_active[:-1] = active[1:]

        start_mask = active & (~prev_active)
        end_mask = active & (~next_active)

        start_t, start_x, start_y = [cp.asnumpy(a) for a in cp.where(start_mask)]
        end_t, end_x, end_y = [cp.asnumpy(a) for a in cp.where(end_mask)]

        start_lin = start_x * Y + start_y
        end_lin = end_x * Y + end_y

        start_order = np.argsort(start_lin, kind="stable")
        end_order = np.argsort(end_lin, kind="stable")

        start_lin = start_lin[start_order]
        start_t = start_t[start_order]

        end_lin = end_lin[end_order]
        end_t = end_t[end_order]

        units_map = [[[] for _ in range(Y)] for _ in range(X)]
        active_mask = np.zeros((T, X, Y), dtype=bool)

        i0 = 0
        j0 = 0
        n_start = len(start_lin)
        n_end = len(end_lin)

        while i0 < n_start:
            lin_id = start_lin[i0]

            i1 = i0
            while i1 < n_start and start_lin[i1] == lin_id:
                i1 += 1

            while j0 < n_end and end_lin[j0] < lin_id:
                j0 += 1

            j1 = j0
            while j1 < n_end and end_lin[j1] == lin_id:
                j1 += 1

            starts_patch = start_t[i0:i1]
            ends_patch = end_t[j0:j1]

            m = min(len(starts_patch), len(ends_patch))
            if m == 0:
                i0 = i1
                j0 = j1
                continue

            x = int(lin_id // Y)
            y = int(lin_id % Y)

            intervals = []
            for s, e in zip(starts_patch[:m], ends_patch[:m]):
                s2 = max(0, int(s) - int(extend_radius))
                e2 = min(T - 1, int(e) + int(extend_radius))
                intervals.append([s2, e2])

            intervals.sort()
            merged = []
            for s, e in intervals:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)

            for s, e in merged:
                active_mask[s:e + 1, x, y] = True
                mseg = motion_np[s:e + 1, x, y, :].copy() if save_motion else None
                units_map[x][y].append(
                    MotionUnit(
                        time_range=[s, e],
                        spatial_coor=[x, y],
                        motion=mseg,
                    )
                )

            i0 = i1
            j0 = j1

        return units_map, active_mask

    # CPU fallback
    if use_abs_dev:
        if median_local is not None:
            median_np = np.asarray(median_local, dtype=np.float32)
        else:
            wt = int(max(3, min(int(median_window_t), T if T % 2 == 1 else max(T - 1, 3))))
            median_np = ndi.median_filter(
                motionMag_np,
                size=(wt, int(median_window_xy), int(median_window_xy)),
                mode="reflect",
            )
        abs_dev_np = np.abs(motionMag_np - median_np)
        active = abs_dev_np > rest_np
    else:
        active = motionMag_np > rest_np

    units_map = [[[] for _ in range(Y)] for _ in range(X)]
    active_mask = np.zeros((T, X, Y), dtype=bool)

    for x in range(X):
        for y in range(Y):
            a = active[:, x, y]
            if not np.any(a):
                continue

            padded = np.concatenate([[False], a, [False]])
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            intervals = []
            for s, e in zip(starts, ends):
                s2 = max(0, int(s) - int(extend_radius))
                e2 = min(T - 1, int(e) + int(extend_radius))
                intervals.append([s2, e2])

            intervals.sort()
            merged = []
            for s, e in intervals:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)

            for s, e in merged:
                active_mask[s:e + 1, x, y] = True
                mseg = motion_np[s:e + 1, x, y, :].copy() if save_motion else None
                units_map[x][y].append(
                    MotionUnit(
                        time_range=[s, e],
                        spatial_coor=[x, y],
                        motion=mseg,
                    )
                )

    return units_map, active_mask


def filterMotionUnits(
    units_map,
    active_mask,
    k=3,
    n=7,
    thresh=0.5,
    use_gpu="auto",
):
    """
    Remove isolated MotionUnits using local spatiotemporal active support.
    """
    use_gpu = (HAS_CUPY if use_gpu == "auto" else bool(use_gpu))

    active_np = np.asarray(active_mask, dtype=np.float32)

    if use_gpu:
        active_gpu = cp.asarray(active_np)
        smoothed_gpu = cupy_ndi.uniform_filter(
            active_gpu,
            size=(k, n, n),
            mode="nearest",
        )
        smoothed = cp.asnumpy(smoothed_gpu)
    else:
        smoothed = ndi.uniform_filter(
            active_np,
            size=(k, n, n),
            mode="nearest",
        )

    T, X, Y = active_np.shape
    refined = [[[] for _ in range(Y)] for _ in range(X)]

    for x in range(X):
        for y in range(Y):
            for unit in units_map[x][y]:
                t0, t1 = unit.time_range
                if float(smoothed[t0:t1 + 1, x, y].mean()) >= thresh:
                    refined[x][y].append(unit)

    return refined

def _compute_global_motion_series(motion_full_abs, valid_mask=None, mode="median"):
    arr = np.asarray(motion_full_abs, dtype=np.float32)  # (T,X,Y,2)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"motion_full_abs should have shape (T,X,Y,2), got {arr.shape}")
    T, X, Y, C = arr.shape
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape != (X, Y):
            raise ValueError(f"valid_mask shape {vm.shape} != motion spatial shape {(X,Y)}")
        vals = arr[:, vm, :]  # (T,N,2)
    else:
        vals = arr.reshape(T, X * Y, C)
    if vals.shape[1] == 0:
        vals = arr.reshape(T, X * Y, C)
    if mode == "median":
        g = np.median(vals, axis=1)
    elif mode == "mean":
        g = np.mean(vals, axis=1)
    elif mode == "zero":
        g = np.zeros((T, 2), dtype=np.float32)
    else:
        raise ValueError(f"Unknown global_motion_mode: {mode}")
    return g.astype(np.float32)


def getMotionEpisode(
    motion_units,
    motion_full,
    motion_full_abs=None,
    global_valid_mask=None,
    global_motion_mode="median",
    tolerant_time=1,
    min_total_area=30,
    expand_frames=0,
    repair_mask=True,
    closing_iter=1,
    min_cc_area=4,
    dilation_iter=3,
    overlap_threshold=0.3,
):
    """Build MotionEpisodes by grouping MotionUnits with similar time and overlapping region.

    Parameters
    ----------
    expand_frames : int
        Number of frames to expand each episode's time window in both directions
        (before the first frame and after the last frame). This gives mode decomposition
        more temporal context without changing the spatial region.
    """
    if nx is None:
        raise ImportError("networkx is required for getMotionEpisode().")

    motion_full = np.asarray(motion_full, dtype=np.float32)
    if motion_full_abs is None:
        motion_full_abs = motion_full
    motion_full_abs = np.asarray(motion_full_abs, dtype=np.float32)

    Xp, Yp = len(motion_units), len(motion_units[0])
    time_dict = defaultdict(list)

    for x in range(Xp):
        for y in range(Yp):
            for mu in motion_units[x][y]:
                time_dict[tuple(mu.time_range)].append(tuple(mu.spatial_coor))

    preliminary_masks = []
    time_ranges = []
    areas = []
    bboxes = []
    structure = np.ones((3, 3), dtype=bool)

    for time_range, coords in time_dict.items():
        mask = np.zeros((Xp, Yp), dtype=bool)
        for x, y in coords:
            mask[int(x), int(y)] = True
        if repair_mask:
            mask = ndi.binary_closing(mask, structure=structure, iterations=int(closing_iter))
            mask = ndi.binary_fill_holes(mask)
            lab, num = ndi.label(mask, structure=structure)
            if num > 0:
                sizes = np.bincount(lab.ravel())
                keep_ids = np.where(sizes >= int(min_cc_area))[0]
                keep_ids = keep_ids[keep_ids != 0]
                mask = np.isin(lab, keep_ids) if len(keep_ids) > 0 else np.zeros_like(mask)
        area = int(mask.sum())
        if area == 0:
            continue
        xs, ys = np.nonzero(mask)
        preliminary_masks.append(mask)
        time_ranges.append([int(time_range[0]), int(time_range[1])])
        areas.append(area)
        bboxes.append((int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))

    n = len(preliminary_masks)
    if n == 0:
        return []

    time_ranges = np.asarray(time_ranges, dtype=np.int32)
    areas = np.asarray(areas, dtype=np.float32)
    global_motion_all = _compute_global_motion_series(motion_full_abs, valid_mask=global_valid_mask, mode=global_motion_mode)

    dilated_masks = [ndi.binary_dilation(m, iterations=int(dilation_iter)) for m in preliminary_masks]

    time_bucket = defaultdict(list)
    for idx, (t0, t1) in enumerate(time_ranges):
        time_bucket[(int(t0), int(t1))].append(idx)

    def bbox_overlap(box1, box2, margin):
        x1min, x1max, y1min, y1max = box1
        x2min, x2max, y2min, y2max = box2
        return not (
            x1max + margin < x2min or x2max + margin < x1min or
            y1max + margin < y2min or y2max + margin < y1min
        )

    G = nx.Graph()
    G.add_nodes_from(range(n))
    dt_pairs = [(ds, de) for ds in range(-tolerant_time, tolerant_time + 1)
                for de in range(-tolerant_time, tolerant_time + 1)]

    for i in range(n):
        t0, t1 = time_ranges[i]
        candidates = []
        for ds, de in dt_pairs:
            candidates.extend(time_bucket.get((int(t0 + ds), int(t1 + de)), []))
        for j in candidates:
            if j <= i:
                continue
            if not bbox_overlap(bboxes[i], bboxes[j], margin=int(dilation_iter)):
                continue
            inter = np.count_nonzero(dilated_masks[i] & preliminary_masks[j])
            if inter == 0:
                continue
            ratio = float(inter) / max(float(min(areas[i], areas[j])), 1.0)
            if ratio > overlap_threshold:
                G.add_edge(i, j)

    T_total = motion_full.shape[0]
    episodes = []
    for comp in nx.connected_components(G):
        comp = list(comp)
        t_min = int(time_ranges[comp, 0].min())
        t_max = int(time_ranges[comp, 1].max())
        mask = np.zeros_like(preliminary_masks[0], dtype=bool)
        for idx in comp:
            mask |= preliminary_masks[idx]
        if repair_mask:
            mask = ndi.binary_closing(mask, structure=structure, iterations=int(closing_iter))
            mask = ndi.binary_fill_holes(mask)
        if int(mask.sum()) < min_total_area:
            continue

        # ---- time expansion: add context frames before and after ----
        ef = int(expand_frames)
        t_min_exp = max(0, t_min - ef)
        t_max_exp = min(T_total - 1, t_max + ef)

        motion_delta_seg = motion_full[t_min_exp:t_max_exp + 1][:, mask, :]
        motion_abs_seg = motion_full_abs[t_min_exp:t_max_exp + 1][:, mask, :]
        global_motion_seg = global_motion_all[t_min_exp:t_max_exp + 1]
        ep = MotionEpisode(
            time_range=[t_min_exp, t_max_exp],
            region_mask=mask.astype(np.uint8),
            episode_id=len(episodes),
            motion_delta=motion_delta_seg,
            motion_abs=motion_abs_seg,
            global_motion=global_motion_seg,
            global_motion_mode=global_motion_mode,
        )
        episodes.append(ep)
    return episodes


def filter_episodes_artifacts(
    episodes: List[MotionEpisode],
    valid_mask: np.ndarray,
    max_fov_fraction: float = 0.5,
    min_duration: int = 3,
    max_duration: Optional[int] = None,
    max_global_corr: Optional[float] = 0.90,
    max_edge_fraction: Optional[float] = 0.80,
    edge_width: int = 3,
    verbose: bool = True,
) -> List[MotionEpisode]:
    """
    Filter out likely motion-artifact episodes.

    Four criteria (an episode is discarded if ANY criterion triggers):

    1. **FOV coverage**: spatial_region area > max_fov_fraction * valid_area
       → Whole-FOV drift is unlikely to be biological.

    2. **Duration bounds**: duration < min_duration or > max_duration
       → Too-short episodes are often noise spikes.
         Too-long episodes may be slow drift.

    3. **Global motion correlation**: mean patch motion highly correlated
       with global motion → episode is just the whole sample moving together.

    4. **Edge concentration**: high fraction of active patches at the
       boundary of the valid mask → registration boundary artifacts.

    Parameters
    ----------
    episodes : list of MotionEpisode
    valid_mask : np.ndarray, bool, shape (Xp, Yp)
        Patch-level valid mask (True = valid tissue).
    max_fov_fraction : float
        Max fraction of valid FOV an episode can cover.
    min_duration : int
        Minimum episode duration in frames.
    max_duration : int, optional
        Maximum episode duration in frames (None = no limit).
    max_global_corr : float, optional
        If mean correlation between episode mean motion and global motion
        exceeds this, discard. None = skip check.
    max_edge_fraction : float, optional
        If fraction of episode area within `edge_width` of valid_mask boundary
        exceeds this, discard. None = skip check.
    edge_width : int
        Width (in patches) of the boundary zone.
    verbose : bool

    Returns
    -------
    kept : list of MotionEpisode
    """
    valid_mask = np.asarray(valid_mask).astype(bool)
    valid_area = int(valid_mask.sum())

    if valid_area == 0:
        raise ValueError("valid_mask is empty.")

    # Precompute edge mask (patches near the boundary of valid_mask)
    edge_mask = None
    if max_edge_fraction is not None:
        # Distance from each valid patch to the nearest invalid patch
        dist_to_boundary = ndi.distance_transform_edt(valid_mask)
        edge_mask = (dist_to_boundary > 0) & (dist_to_boundary <= int(edge_width))

    kept = []
    reasons = {
        "fov_coverage": 0,
        "duration_short": 0,
        "duration_long": 0,
        "global_corr": 0,
        "edge_fraction": 0,
    }

    for ep in episodes:
        # ------------------------------------------------------------
        # Criterion 1: FOV coverage
        # ------------------------------------------------------------
        ep_mask = np.asarray(ep.spatial_region).astype(bool)
        ep_area = int(ep_mask.sum())

        if ep_area > max_fov_fraction * valid_area:
            reasons["fov_coverage"] += 1
            if verbose:
                fov_frac = ep_area / max(valid_area, 1)
                print(
                    f"[artifact filter] ep={ep.episode_id} DISCARD: "
                    f"FOV coverage={fov_frac:.3f} > {max_fov_fraction}"
                )
            continue

        # ------------------------------------------------------------
        # Criterion 2: Duration
        # ------------------------------------------------------------
        duration = int(ep.time_range[1] - ep.time_range[0] + 1)

        if duration < min_duration:
            reasons["duration_short"] += 1
            if verbose:
                print(
                    f"[artifact filter] ep={ep.episode_id} DISCARD: "
                    f"duration={duration} < {min_duration}"
                )
            continue

        if max_duration is not None and duration > max_duration:
            reasons["duration_long"] += 1
            if verbose:
                print(
                    f"[artifact filter] ep={ep.episode_id} DISCARD: "
                    f"duration={duration} > {max_duration}"
                )
            continue

        # ------------------------------------------------------------
        # Criterion 3: Global motion correlation
        # ------------------------------------------------------------
        if max_global_corr is not None:
            gm = ep.global_motion
            md = ep.motion_delta

            if gm is not None and md is not None:
                gm = np.asarray(gm, dtype=np.float32)  # (T, 2)
                md = np.asarray(md, dtype=np.float32)   # (T, N, 2)
                T_ep = md.shape[0]

                if gm.shape[0] == T_ep and md.shape[1] > 0:
                    # Mean motion across all patches in the episode
                    mean_motion = np.mean(md, axis=1)  # (T, 2)

                    # Correlation per component (x, y)
                    corr_x = _safe_corr(mean_motion[:, 0], gm[:, 0])
                    corr_y = _safe_corr(mean_motion[:, 1], gm[:, 1])
                    mean_corr = (corr_x + corr_y) / 2.0

                    if mean_corr > max_global_corr:
                        reasons["global_corr"] += 1
                        if verbose:
                            print(
                                f"[artifact filter] ep={ep.episode_id} DISCARD: "
                                f"global_corr={mean_corr:.4f} > {max_global_corr}"
                            )
                        continue

        # ------------------------------------------------------------
        # Criterion 4: Edge concentration
        # ------------------------------------------------------------
        if max_edge_fraction is not None and edge_mask is not None:
            if ep_mask.shape == edge_mask.shape:
                edge_patches = int((ep_mask & edge_mask).sum())
                edge_frac = edge_patches / max(ep_area, 1)

                if edge_frac > max_edge_fraction:
                    reasons["edge_fraction"] += 1
                    if verbose:
                        print(
                            f"[artifact filter] ep={ep.episode_id} DISCARD: "
                            f"edge_fraction={edge_frac:.3f} > {max_edge_fraction}"
                        )
                    continue

        # All checks passed
        kept.append(ep)

    if verbose:
        n_total = len(episodes)
        n_kept = len(kept)
        n_discarded = n_total - n_kept
        print(
            f"[artifact filter] {n_total} episodes → {n_kept} kept "
            f"({n_discarded} discarded): {reasons}"
        )

    return kept


# =============================================================================
# Motion mode decomposition: Y_i(t) ~= sum h_k(t) b_ik
# =============================================================================


def _initialize_modes_spatial_seed(Y_data, valid_coords, Kmax=4, rng=None, min_seed_dist=3.0, eps=1e-12):
    if rng is None:
        rng = np.random.default_rng(0)
    T, N, _ = Y_data.shape
    K = min(int(Kmax), N)
    energy = np.sum(Y_data ** 2, axis=(0, 2))
    order = np.argsort(energy)[::-1]
    seeds = []
    for idx in order:
        if len(seeds) >= K:
            break
        xy = valid_coords[idx].astype(np.float32)
        if len(seeds) == 0:
            seeds.append(int(idx))
        else:
            prev = valid_coords[np.asarray(seeds)].astype(np.float32)
            d = np.linalg.norm(prev - xy[None, :], axis=1)
            if np.min(d) >= min_seed_dist:
                seeds.append(int(idx))
    while len(seeds) < K:
        cand = int(rng.integers(0, N))
        if cand not in seeds:
            seeds.append(cand)

    H = np.zeros((K, T), dtype=np.float32)
    for k, idx in enumerate(seeds):
        traj = Y_data[:, idx, :]  # (T,2)
        mean_vec = np.mean(traj, axis=0)
        nrm = np.linalg.norm(mean_vec)
        if nrm < eps:
            h = np.linalg.norm(traj, axis=1)
        else:
            h = traj @ (mean_vec / nrm)
        hn = np.linalg.norm(h)
        if hn < eps:
            h = rng.normal(size=T).astype(np.float32)
            hn = np.linalg.norm(h)
        H[k] = h / max(float(hn), eps)

    M = np.concatenate([Y_data[:, :, 0].T, Y_data[:, :, 1].T], axis=0).astype(np.float32)
    HHt = H @ H.T + 1e-6 * np.eye(K, dtype=np.float32)
    B = M @ H.T @ np.linalg.inv(HHt)
    return H.astype(np.float32), B.astype(np.float32), seeds


def _normalize_spatial_coords(valid_coords, mask_shape):
    """Map patch coordinates to [0, 1]^2 for a dimensionless compactness loss."""
    coords = np.asarray(valid_coords, dtype=np.float64)
    scale = np.maximum(np.asarray(mask_shape, dtype=np.float64) - 1.0, 1.0)
    return (coords / scale[None, :]).astype(np.float64)


def _normalize_H_rows(H, reference=None, eps=1e-12):
    """Retract every row of H onto the unit sphere without rescaling B."""
    H = np.asarray(H, dtype=np.float64).copy()
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    bad = norms[:, 0] <= eps
    if np.any(bad):
        if reference is None:
            raise ValueError("Cannot normalize a zero row of H without a reference.")
        H[bad] = np.asarray(reference, dtype=np.float64)[bad]
        norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(norms, eps)


def _compute_mode_centroids(B, spatial_coords, previous=None, eps=1e-12):
    """Return the amplitude-weighted spatial centroid of every mode."""
    B = np.asarray(B, dtype=np.float64)
    coords = np.asarray(spatial_coords, dtype=np.float64)
    N = B.shape[0] // 2
    amplitude = np.hypot(B[:N], B[N:])
    mass = np.sum(amplitude, axis=0)
    centers = np.empty((B.shape[1], 2), dtype=np.float64)
    for k in range(B.shape[1]):
        if mass[k] > eps:
            centers[k] = np.sum(amplitude[:, k, None] * coords, axis=0) / mass[k]
        elif previous is not None:
            centers[k] = np.asarray(previous, dtype=np.float64)[k]
        else:
            centers[k] = np.mean(coords, axis=0)
    return centers


def _spatial_compact_penalty(
    B,
    spatial_coords,
    centers,
    rho=1.0,
    kappa=4.0,
    B_scale=1.0,
    eps=1e-12,
):
    """Dimensionless sparse-compact penalty and its per-patch weights."""
    B = np.asarray(B, dtype=np.float64)
    coords = np.asarray(spatial_coords, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    N = B.shape[0] // 2
    K = B.shape[1]
    amplitude = np.hypot(B[:N], B[N:])
    distance_sq = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    weights = float(rho) + float(kappa) * distance_sq
    raw = float(np.sum(weights * amplitude) / max(N * K * B_scale, eps))
    return raw, weights


def _group_soft_threshold_B(B, thresh, eps=1e-12):
    """Weighted group soft-thresholding of each vector b_ik=(bx_ik, by_ik)."""
    B = np.asarray(B, dtype=np.float64).copy()
    N2, K = B.shape
    N = N2 // 2
    Bx = B[:N, :]
    By = B[N:, :]
    norm = np.hypot(Bx, By)
    thresh = np.asarray(thresh, dtype=np.float64)
    scale = np.maximum(0.0, 1.0 - thresh / np.maximum(norm, eps))
    B[:N, :] = Bx * scale
    B[N:, :] = By * scale
    return B


def _sparse_compact_objective(
    M,
    B,
    H,
    spatial_coords,
    centers,
    lambda_sc,
    rho,
    kappa,
    B_scale,
    total_energy,
    eps=1e-12,
):
    residual = np.asarray(M, dtype=np.float64) - B @ H
    recon = float(np.sum(residual ** 2) / total_energy)
    compact_raw, _ = _spatial_compact_penalty(
        B,
        spatial_coords,
        centers,
        rho=rho,
        kappa=kappa,
        B_scale=B_scale,
        eps=eps,
    )
    compact = float(lambda_sc) * compact_raw
    return recon + compact, recon, compact, compact_raw


def _fit_motion_modes_minimal(
    M,
    B,
    H,
    spatial_coords,
    lambda_sc=0.05,
    rho=1.0,
    kappa=4.0,
    max_iter=200,
    tol=1e-4,
    verbose=True,
    eps=1e-12,
    B_scale=None,
    initial_centers=None,
    max_backtracking=30,
):
    """Fit reconstruction + sparse-compact loss with ||H[k]||_2 = 1.

    The optimized objective is

        ||M - B H||_F^2 / ||M||_F^2
        + lambda_sc/(N K s_B) * sum_{i,k}
          (rho + kappa ||r_i - mu_k||_2^2) ||b_ik||_2,

    where r_i are normalized patch coordinates and mu_k is the
    amplitude-weighted centroid of mode k.  There is deliberately no temporal
    smoothness term, Ridge term, graph-Laplacian term, or mode-column penalty.

    B is updated by weighted group-Lasso proximal gradient.  The centroids are
    then updated exactly.  H is updated on a product of unit spheres using a
    tangent gradient and row-normalization retraction.  Backtracking rejects
    any block step that would increase the objective.
    """
    M = np.asarray(M, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64).copy()
    H = _normalize_H_rows(H, eps=eps)
    spatial_coords = np.asarray(spatial_coords, dtype=np.float64)

    N2, T = M.shape
    N = N2 // 2
    K = H.shape[0]

    if K == 0:
        return B.astype(np.float32), H.astype(np.float32), [], np.zeros((0, 2), dtype=np.float32)
    if N2 != 2 * len(spatial_coords):
        raise ValueError("spatial_coords must contain one coordinate for each B patch.")
    if lambda_sc < 0 or rho < 0 or kappa < 0:
        raise ValueError("lambda_sc, rho, and kappa must be non-negative.")

    total_energy = float(np.sum(M ** 2)) + eps
    if B_scale is None:
        B_scale = float(np.sqrt(T) * np.sqrt(np.mean(M ** 2)) + eps)
    B_scale = max(float(B_scale), eps)

    if initial_centers is None:
        centers = _compute_mode_centroids(B, spatial_coords, eps=eps)
    else:
        centers = _compute_mode_centroids(
            B,
            spatial_coords,
            previous=np.asarray(initial_centers, dtype=np.float64),
            eps=eps,
        )

    loss, recon, compact, compact_raw = _sparse_compact_objective(
        M, B, H, spatial_coords, centers, lambda_sc, rho, kappa,
        B_scale, total_energy, eps=eps,
    )
    loss_history = []

    for it in range(max_iter):
        B_old = B.copy()
        H_old = H.copy()
        centers_old = centers.copy()
        loss_old = loss

        # B block: proximal gradient with the centroid held fixed.
        HHt = H @ H.T
        grad_B = (2.0 / total_energy) * (B @ H - M) @ H.T
        lip_B = (2.0 / total_energy) * float(np.linalg.norm(HHt, ord=2))
        step_B = 1.0 / (lip_B + eps)
        _, spatial_weights = _spatial_compact_penalty(
            B, spatial_coords, centers, rho=rho, kappa=kappa,
            B_scale=B_scale, eps=eps,
        )
        fixed_center_loss = loss
        accepted_B = False
        for _ in range(max_backtracking):
            threshold = (
                step_B * float(lambda_sc) * spatial_weights
                / max(N * K * B_scale, eps)
            )
            B_trial = _group_soft_threshold_B(B - step_B * grad_B, threshold, eps=eps)
            trial_loss, _, _, _ = _sparse_compact_objective(
                M, B_trial, H, spatial_coords, centers, lambda_sc, rho,
                kappa, B_scale, total_energy, eps=eps,
            )
            if trial_loss <= fixed_center_loss + 1e-12:
                B = B_trial
                accepted_B = True
                break
            step_B *= 0.5
        if not accepted_B:
            step_B = 0.0

        # mu block: this weighted mean is the exact minimizer for fixed B.
        centers = _compute_mode_centroids(B, spatial_coords, previous=centers, eps=eps)

        # H block: Riemannian gradient step on ||H[k]||_2 = 1.
        grad_H = (2.0 / total_energy) * B.T @ (B @ H - M)
        grad_H_tangent = grad_H - np.sum(grad_H * H, axis=1, keepdims=True) * H
        grad_H_norm_sq = float(np.sum(grad_H_tangent ** 2))
        lip_H = (2.0 / total_energy) * float(np.linalg.norm(B.T @ B, ord=2))
        step_H = 1.0 / (lip_H + eps)
        before_H_loss, _, _, _ = _sparse_compact_objective(
            M, B, H, spatial_coords, centers, lambda_sc, rho, kappa,
            B_scale, total_energy, eps=eps,
        )
        accepted_H = grad_H_norm_sq <= eps
        if not accepted_H:
            for _ in range(max_backtracking):
                H_trial = _normalize_H_rows(
                    H - step_H * grad_H_tangent,
                    reference=H,
                    eps=eps,
                )
                trial_loss, _, _, _ = _sparse_compact_objective(
                    M, B, H_trial, spatial_coords, centers, lambda_sc,
                    rho, kappa, B_scale, total_energy, eps=eps,
                )
                armijo_rhs = before_H_loss - 1e-4 * step_H * grad_H_norm_sq
                if trial_loss <= armijo_rhs + 1e-12:
                    H = H_trial
                    accepted_H = True
                    break
                step_H *= 0.5
        if not accepted_H:
            step_H = 0.0

        loss, recon, compact, compact_raw = _sparse_compact_objective(
            M, B, H, spatial_coords, centers, lambda_sc, rho, kappa,
            B_scale, total_energy, eps=eps,
        )
        if loss > loss_old + 1e-10:
            B, H, centers = B_old, H_old, centers_old
            loss, recon, compact, compact_raw = _sparse_compact_objective(
                M, B, H, spatial_coords, centers, lambda_sc, rho, kappa,
                B_scale, total_energy, eps=eps,
            )
            step_B = 0.0
            step_H = 0.0

        delta = float(np.mean(np.abs(B - B_old)) + np.mean(np.abs(H - H_old)))
        relative_decrease = float((loss_old - loss) / max(abs(loss_old), eps))
        h_norm_error = float(np.max(np.abs(np.linalg.norm(H, axis=1) - 1.0)))
        loss_history.append({
            "loss": float(loss),
            "recon": float(recon),
            "sparse_compact": float(compact),
            "sparse_compact_raw": float(compact_raw),
            "step_B": float(step_B),
            "step_H": float(step_H),
            "delta": delta,
            "relative_decrease": relative_decrease,
            "H_norm_error": h_norm_error,
        })

        if verbose:
            print(
                f"[mode fit] iter={it:03d}, "
                f"loss={loss:.6e}, recon={recon:.6e}, "
                f"sparse_compact={compact:.6e}, "
                f"delta={delta:.3e}, H_norm_error={h_norm_error:.2e}"
            )

        if relative_decrease <= tol and delta <= np.sqrt(tol):
            break

    return (
        B.astype(np.float32),
        H.astype(np.float32),
        loss_history,
        centers.astype(np.float32),
    )

def _hard_threshold_B(B, support_rel_thresh=0.08, eps=1e-12):
    """Keep the existing post-fit relative hard-thresholding behavior."""
    B = np.asarray(B, dtype=np.float32).copy()
    N = B.shape[0] // 2
    amplitude = np.hypot(B[:N], B[N:])
    max_amplitude = np.max(amplitude, axis=0, keepdims=True)
    support_mask = amplitude > support_rel_thresh * max_amplitude
    support_mask[:, max_amplitude[0] <= eps] = False
    drop = ~support_mask
    removed_energy = float(np.sum(B[:N][drop] ** 2 + B[N:][drop] ** 2))
    B[:N][drop] = 0.0
    B[N:][drop] = 0.0
    return B, support_mask, int(np.sum(drop)), removed_energy


def _build_motion_modes_from_BH(
    B,
    H,
    M,
    Y_data,
    valid_coords,
    mask_shape,
    episode_id,
    time_range,
    global_motion,
    support_rel_thresh=0.10,
    eps=1e-12,
):
    """Convert every post-threshold B/H component into a MotionMode."""
    N2, T = M.shape
    N = N2 // 2
    K = H.shape[0]
    X, Y = mask_shape
    modes = []
    total_energy = float(np.sum(M ** 2)) + eps
    for k in range(K):
        response_compact = np.stack([B[:N, k], B[N:, k]], axis=1).astype(np.float32)
        strength = np.linalg.norm(response_compact, axis=1)
        mass = float(np.sum(strength))
        Mk = B[:, [k]] @ H[[k], :]
        explained = float(np.sum(Mk ** 2) / total_energy)
        field = np.zeros((X, Y, 2), dtype=np.float32)
        A = np.zeros((X, Y), dtype=np.float32)
        for idx, (r, c) in enumerate(valid_coords):
            field[r, c, :] = response_compact[idx]
            A[r, c] = strength[idx]
        support = A > eps
        direction = np.zeros_like(field, dtype=np.float32)
        direction[support] = field[support] / A[support, None]
        h = H[k].astype(np.float32)
        recon_k = h[:, None, None] * response_compact[None, :, :]
        resid_k = Y_data - recon_k
        residual_energy = float(np.sum(resid_k ** 2) / total_energy)
        seed_position = tuple(np.unravel_index(np.argmax(A), A.shape)) if np.any(A > 0) else None
        mode = MotionMode(
            episode_id=episode_id,
            mode_id=len(modes),
            time_range=list(time_range),
            activation=h,
            response_field=field,
            response_strength=A,
            response_direction=direction,
            support_mask=support.astype(np.uint8),
            compact_response_field=response_compact,
            valid_coords=valid_coords,
            background_motion=global_motion,
            reconstructed_motion=recon_k,
            residual_motion=resid_k,
            explained_energy=explained,
            residual_energy=residual_energy,
            mode_mass=mass,
            seed_position=seed_position,
            confidence=1.0,
            metadata={
                "support_rel_thresh": support_rel_thresh,
                "model": "motion_mode_sparse_compact",
                "postprocessing": "hard_threshold_only",
            },
        )
        modes.append(mode)
    return modes


def _compute_recon_loss(M, B, H, eps=1e-12):
    """
    M: (D, T)
    B: (D, K)
    H: (K, T)
    """
    R = M - B @ H
    return 0.5 * float(np.sum(R * R))


def _compute_recon_r2(M, B, H, eps=1e-12):
    loss = _compute_recon_loss(M, B, H, eps=eps)
    energy = 0.5 * float(np.sum(M * M)) + eps
    return 1.0 - loss / energy


def _select_K_by_svd_energy(
    M,
    target_r2=0.85,
    Kmax=None,
    K_min=1,
    eps=1e-12,
):
    """
    Select K by cumulative SVD explained energy.

    M: (2N, T)

    Return
    ------
    selected_K : int
    info : dict
    """
    M = np.asarray(M, dtype=np.float32)
    if M.ndim != 2 or min(M.shape) == 0:
        raise ValueError("M must be a non-empty two-dimensional matrix.")

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    energy = S ** 2
    total_energy = float(np.sum(energy)) + eps
    cum_r2 = np.cumsum(energy) / total_energy

    rank_max = len(S)

    if Kmax is None:
        Kmax_eff = rank_max
    else:
        Kmax_eff = min(int(Kmax), rank_max)

    K_min = min(max(1, int(K_min)), rank_max)
    Kmax_eff = max(K_min, Kmax_eff)

    selected_K = Kmax_eff

    for k in range(K_min, Kmax_eff + 1):
        if cum_r2[k - 1] >= target_r2:
            selected_K = k
            break

    records = []
    for k in range(1, rank_max + 1):
        records.append(
            {
                "K": int(k),
                "svd_r2": float(cum_r2[k - 1]),
                "singular_value": float(S[k - 1]),
                "energy_fraction": float(energy[k - 1] / total_energy),
            }
        )

    info = {
        "method": "svd_cumulative_energy",
        "target_r2": float(target_r2),
        "selected_K": int(selected_K),
        "K_min": int(K_min),
        "Kmax_input": None if Kmax is None else int(Kmax),
        "Kmax_eff": int(Kmax_eff),
        "rank_max": int(rank_max),
        "singular_values": S.astype(np.float32),
        "cum_r2": cum_r2.astype(np.float32),
        "selected_svd_r2": float(cum_r2[selected_K - 1]),
        "records": records,
    }

    return int(selected_K), info

def decompose_episode_motion_modes(
    episode: MotionEpisode,
    Kmax=4,
    lambda_sc=0.05,
    rho=1.0,
    kappa=4.0,
    max_iter=200,
    tol=1e-4,
    support_rel_thresh=0.08,

    # K selection
    K_selection_method="svd",   # "svd" or "fixed"
    K_min=1,

    # SVD K selection
    svd_target_r2=0.85,

    verbose=True,
    random_state=0,
    use_velocity=False,
):
    """
    Decompose one MotionEpisode with reconstruction + sparse-compact loss.

    Optimization model:
        min_{B,H,mu} ||M-BH||_F^2 / ||M||_F^2
            + lambda_sc/(N K s_B) sum_{i,k}
              (rho + kappa ||r_i-mu_k||_2^2) ||b_ik||_2
        subject to ||H[k]||_2 = 1 for every k.

    Pipeline:
        1. Prepare the episode motion matrix M.
        2. Select K and initialize B/H from spatial seeds.
        3. Alternate weighted group-Lasso B updates, exact centroid updates,
           and unit-sphere H updates.  No temporal smoothness is used.
        4. Keep the existing relative hard-threshold on B after optimization.
        5. Convert all K components to MotionMode objects.

    Model dimensions:
        M: (2N, T)
        B: (2N, K)
        H: (K, T)
    """

    # ------------------------------------------------------------------
    # 1. Validate and select the motion field
    # ------------------------------------------------------------------
    if use_velocity:
        if episode.motion_delta is None:
            raise ValueError(
                "episode.motion_delta is required when use_velocity=True."
            )
        motion_data = np.asarray(episode.motion_delta, dtype=np.float32)
        motion_field_name = "motion_delta (velocity)"
    else:
        if episode.motion_abs is None:
            raise ValueError(
                "episode.motion_abs is required when use_velocity=False."
            )
        motion_data = np.asarray(episode.motion_abs, dtype=np.float32)
        motion_field_name = "motion_abs (cumulative displacement)"

    if episode.global_motion is None:
        raise ValueError(
            "episode.global_motion is required for motion mode decomposition."
        )

    rng = np.random.default_rng(random_state)
    eps = 1e-12
    if int(Kmax) < 1:
        raise ValueError("Kmax must be at least 1.")

    # ------------------------------------------------------------------
    # 2. Prepare M from the episode data
    # ------------------------------------------------------------------
    mask = np.asarray(episode.spatial_region).astype(bool)
    X, Y = mask.shape
    valid_coords = np.argwhere(mask)
    spatial_coords = _normalize_spatial_coords(valid_coords, (X, Y))

    global_motion = np.asarray(episode.global_motion, dtype=np.float32)

    T, N, C = motion_data.shape
    if C != 2 or len(valid_coords) != N:
        raise ValueError(
            f"motion_data shape {motion_data.shape} inconsistent with "
            f"episode mask count {len(valid_coords)}"
        )

    if global_motion.shape != (T, 2):
        raise ValueError(
            f"global_motion should have shape {(T, 2)}, "
            f"got {global_motion.shape}"
        )

    # The physical units of global_motion must match motion_data. In
    # particular, use_velocity=True requires a velocity-like global motion.
    Y_data = motion_data - global_motion[:, None, :]  # (T, N, 2)

    M = np.concatenate(
        [
            Y_data[:, :, 0].T,
            Y_data[:, :, 1].T,
        ],
        axis=0,
    ).astype(np.float32)  # (2N, T)

    total_energy = float(np.sum(M ** 2)) + eps
    # Since ||H[k]||_2=1, a coefficient in B naturally scales as
    # sqrt(T) times a per-frame motion amplitude.
    B_scale = float(np.sqrt(T) * np.sqrt(np.mean(M ** 2)) + eps)

    if verbose:
        print(
            f"[motion data] episode={episode.episode_id}, "
            f"using {motion_field_name}, "
            f"T={T}, N={N}, total_energy={total_energy:.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Select K and initialize B/H
    # ------------------------------------------------------------------
    K_select_info = None
    selected_K = None

    if K_selection_method == "svd":
        # SVD selects only K; B/H are still initialized from spatial seeds.
        selected_K, K_select_info = _select_K_by_svd_energy(
            M,
            target_r2=svd_target_r2,
            Kmax=Kmax,
            K_min=K_min,
            eps=eps,
        )

        H, B, seeds = _initialize_modes_spatial_seed(
            Y_data,
            valid_coords,
            Kmax=selected_K,
            rng=rng,
            eps=eps,
        )

        if verbose:
            print(
                f"[mode K selection: SVD] episode={episode.episode_id}, "
                f"Kmax={Kmax}, selected_K={selected_K}, "
                f"svd_target_r2={svd_target_r2}, "
                f"selected_svd_R2={K_select_info['selected_svd_r2']:.6f}, "
                f"rank_max={K_select_info['rank_max']}"
            )

    elif K_selection_method == "fixed":
        selected_K = min(int(Kmax), N, T)

        H, B, seeds = _initialize_modes_spatial_seed(
            Y_data,
            valid_coords,
            Kmax=selected_K,
            rng=rng,
            eps=eps,
        )

        K_select_info = {
            "method": "fixed",
            "selected_K": int(selected_K),
        }

        if verbose:
            print(
                f"[mode K selection: fixed] episode={episode.episode_id}, "
                f"K={selected_K}"
            )

    else:
        raise ValueError(
            f"Unknown K_selection_method: {K_selection_method}. "
            "Use 'svd' or 'fixed'."
        )

    K_init = H.shape[0]

    # ------------------------------------------------------------------
    # 4. Optimize B/H and the auxiliary spatial centroids.
    # ------------------------------------------------------------------
    initial_centers = spatial_coords[np.asarray(seeds, dtype=np.int64)]
    B, H, loss_history, mode_centers_normalized = _fit_motion_modes_minimal(
        M,
        B,
        H,
        spatial_coords=spatial_coords,
        lambda_sc=lambda_sc,
        rho=rho,
        kappa=kappa,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        eps=eps,
        B_scale=B_scale,
        initial_centers=initial_centers,
    )

    K_final = H.shape[0]
    optimized_B = B.copy()
    optimized_recon = _compute_recon_loss(M, optimized_B, H, eps=eps)
    optimized_r2 = _compute_recon_r2(M, optimized_B, H, eps=eps)

    # Existing downstream post-processing: intentionally unchanged in scope.
    B, support_mask, removed_count, removed_energy = _hard_threshold_B(
        optimized_B,
        support_rel_thresh=support_rel_thresh,
        eps=eps,
    )
    final_recon = _compute_recon_loss(M, B, H, eps=eps)
    final_r2 = _compute_recon_r2(M, B, H, eps=eps)

    if verbose:
        print(
            f"[mode fit] episode={episode.episode_id}, K={K_final}, "
            f"optimized_R2={optimized_r2:.6f}, "
            f"post_threshold_R2={final_r2:.6f}, "
            f"removed_patch_modes={removed_count}"
        )

    # ------------------------------------------------------------------
    # 5. Convert every fitted component to a MotionMode object.
    # ------------------------------------------------------------------
    if K_final > 0:
        modes = _build_motion_modes_from_BH(
            B,
            H,
            M,
            Y_data,
            valid_coords,
            (X, Y),
            episode.episode_id,
            episode.time_range,
            global_motion,
            support_rel_thresh=support_rel_thresh,
            eps=eps,
        )
    else:
        modes = []

    # ------------------------------------------------------------------
    # 6. Save the final model and diagnostics
    # ------------------------------------------------------------------
    episode.modes = modes
    episode.mode_model = {
        "B": B,
        "B_before_hard_threshold": optimized_B,
        "H": H,
        "loss_history": loss_history,
        "mode_centers_normalized": mode_centers_normalized,
        "mode_centers_patch": mode_centers_normalized * np.maximum(
            np.asarray((X, Y), dtype=np.float32) - 1.0,
            1.0,
        )[None, :],

        "lambda_sc": float(lambda_sc),
        "rho": float(rho),
        "kappa": float(kappa),

        "Kmax": Kmax,
        "K_init": K_init,
        "K_final": K_final,
        "K_modes": len(modes),
        "seeds": seeds,

        "total_energy": total_energy,
        "optimized_recon": optimized_recon,
        "optimized_r2": optimized_r2,
        "final_recon": final_recon,
        "final_r2": final_r2,
        "hard_threshold_support_mask": support_mask,
        "hard_threshold_removed_count": removed_count,
        "hard_threshold_removed_energy": removed_energy,

        "K_selection_method": K_selection_method,
        "K_selected": selected_K,
        "K_select_info": K_select_info,
        "svd_target_r2": svd_target_r2,

        "B_scale": B_scale,
        "use_velocity": use_velocity,
        "motion_field_used": motion_field_name,
        "H_constraint": "row_l2_norm_equals_1",
        "objective": "reconstruction_plus_sparse_compact",
        "postprocessing": "hard_threshold_only",
    }

    if verbose:
        print(
            f"[motion modes] episode={episode.episode_id}, "
            f"selected_K={selected_K}, built_modes={len(modes)}, "
            f"final_R2={final_r2:.6f}, postprocessing=hard_threshold_only"
        )

    return modes

def getMotionModes(motion_episodes: Sequence[MotionEpisode], **kwargs):
    all_modes = []
    for ep in motion_episodes:
        modes = decompose_episode_motion_modes(ep, **kwargs)
        all_modes.extend(modes)
    return all_modes


# =============================================================================
# Mode -> MotionRegion split
# =============================================================================


def _split_binary_support(
    support_original,
    min_region_area=5,
    split_mode="gap_tolerant",
    gap_close_iter=1,
    gap_dilation_iter=2,
):
    support_original = np.asarray(support_original).astype(bool)
    if not np.any(support_original):
        return []
    if int(support_original.sum()) < min_region_area:
        return []
    structure = np.ones((3, 3), dtype=bool)
    if split_mode == "loose":
        return [support_original]
    elif split_mode == "strict":
        support_for_label = support_original
    elif split_mode == "gap_tolerant":
        support_for_label = support_original.copy()
        if gap_close_iter and gap_close_iter > 0:
            support_for_label = ndi.binary_closing(support_for_label, structure=structure, iterations=int(gap_close_iter))
        if gap_dilation_iter and gap_dilation_iter > 0:
            support_for_label = ndi.binary_dilation(support_for_label, structure=structure, iterations=int(gap_dilation_iter))
    else:
        raise ValueError("split_mode should be 'strict', 'gap_tolerant', or 'loose'.")
    lab, num = ndi.label(support_for_label, structure=structure)
    masks = []
    for rid in range(1, num + 1):
        # Use repaired/dilated mask only for grouping, final region keeps original support pixels.
        m = support_original & (lab == rid)
        if int(m.sum()) >= min_region_area:
            masks.append(m)
    if len(masks) == 0 and int(support_original.sum()) >= min_region_area:
        masks = [support_original]
    return masks


def _weighted_mean_response_vector(B, A, mask, eps=1e-8):
    if mask is None or not np.any(mask):
        return np.zeros(2, dtype=np.float32)

    w = A[mask].astype(np.float32)
    b = B[mask].astype(np.float32)

    if np.sum(w) <= eps:
        return b.mean(axis=0).astype(np.float32)

    return (np.sum(w[:, None] * b, axis=0) / (np.sum(w) + eps)).astype(np.float32)


def _region_centroid(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    return pts.mean(axis=0).astype(np.float32)


def _region_min_distance(mask_a, mask_b):
    """
    Minimum pixel/patch distance from mask_a to mask_b.
    """
    if not np.any(mask_a) or not np.any(mask_b):
        return np.inf

    dist_to_b = ndi.distance_transform_edt(~mask_b)
    return float(np.min(dist_to_b[mask_a]))


def _cosine_similarity(v1, v2, eps=1e-8):
    v1 = np.asarray(v1, dtype=np.float32).reshape(-1)
    v2 = np.asarray(v2, dtype=np.float32).reshape(-1)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < eps or n2 < eps:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2 + eps))


def _merge_or_discard_small_region_masks(
    region_masks,
    A,
    B,
    min_region_area=10,
    discard_region_area=3,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,
):
    """
    Merge small regions into nearby larger regions, or discard isolated tiny ones.

    region_masks: list of bool masks
    A: response_strength, (X, Y)
    B: response_field, (X, Y, 2)
    """
    if len(region_masks) == 0:
        return []

    region_masks = [m.astype(bool).copy() for m in region_masks]
    areas = np.array([int(m.sum()) for m in region_masks], dtype=np.int32)

    # If everything is small, keep only the largest one if it is not too tiny.
    large_ids = np.where(areas >= min_region_area)[0].tolist()
    small_ids = np.where(areas < min_region_area)[0].tolist()

    if len(large_ids) == 0:
        best = int(np.argmax(areas))
        if areas[best] >= discard_region_area:
            return [region_masks[best]]
        return []

    kept = [region_masks[i].copy() for i in large_ids]

    for sid in small_ids:
        small_mask = region_masks[sid]
        small_area = int(small_mask.sum())

        if small_area < discard_region_area:
            continue

        small_vec = _weighted_mean_response_vector(B, A, small_mask)

        best_j = None
        best_score = np.inf

        for j, big_mask in enumerate(kept):
            d = _region_min_distance(small_mask, big_mask)
            if d > merge_small_max_dist:
                continue

            big_vec = _weighted_mean_response_vector(B, A, big_mask)
            cos = _cosine_similarity(small_vec, big_vec)

            if cos < merge_small_vector_cos:
                continue

            # Prefer nearest, then stronger directional consistency.
            score = d - 0.25 * cos
            if score < best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            kept[best_j] = np.logical_or(kept[best_j], small_mask)
        # else discard it

    return kept


def split_mode_to_regions(
    mode,
    support_rel_thresh=0.10,

    # main spatial tolerance
    split_mode="gap_tolerant_discard_small",
    gap_dilation_iter=4,
    gap_close_iter=4,

    # small fragment handling
    min_region_area=20,
    discard_region_area=3,
    merge_small_regions=False,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,

    # output
    region_id_start=0,
):
    """
    Split one MotionMode into spatial MotionRegions.

    Recommended mode:
        split_mode="gap_tolerant_discard_small"

    Logic:
        1. response_strength -> raw_support
        2. use closing/dilation only to connect nearby fragments
        3. connected components are computed on the tolerant grouping mask
        4. final region masks only contain raw_support pixels
        5. small regions are directly discarded
    """
    A = np.asarray(mode.response_strength, dtype=np.float32)
    B = np.asarray(mode.response_field, dtype=np.float32)
    h = np.asarray(mode.activation, dtype=np.float32)

    if A.ndim != 2:
        raise ValueError(f"mode.response_strength should be 2D, got {A.shape}")
    if B.ndim != 3 or B.shape[-1] != 2:
        raise ValueError(f"mode.response_field should be (X,Y,2), got {B.shape}")

    vmax = float(np.nanmax(A)) if A.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0:
        return []

    # NaN pixels compare False and are excluded from the support automatically
    raw_support = A > (support_rel_thresh * vmax)

    if not np.any(raw_support):
        return []

    structure = np.ones((3, 3), dtype=bool)

    # ------------------------------------------------------------
    # 1. Build region masks
    # ------------------------------------------------------------
    if split_mode in ["whole", "mode_support", "no_split", "loose"]:
        # One mode produces at most one region.
        # If it is too small, discard it.
        if int(raw_support.sum()) >= min_region_area:
            region_masks = [raw_support]
        else:
            region_masks = []

    elif split_mode == "strict":
        # Strict connected components on raw support.
        label_map, num = ndi.label(raw_support, structure=structure)

        region_masks = []
        for rid in range(1, num + 1):
            m = label_map == rid
            if int(m.sum()) >= min_region_area:
                region_masks.append(m)

    elif split_mode in ["gap_tolerant", "gap_tolerant_discard_small"]:
        # Use closing/dilation only for grouping.
        # Final masks still only contain raw_support pixels.
        group_mask = raw_support.copy()

        # closing fills small holes / bridges tiny gaps
        if gap_close_iter is not None and gap_close_iter > 0:
            group_mask = ndi.binary_closing(
                group_mask,
                structure=structure,
                iterations=int(gap_close_iter),
            )

        # dilation makes the connectivity more tolerant
        if gap_dilation_iter is not None and gap_dilation_iter > 0:
            group_mask = ndi.binary_dilation(
                group_mask,
                structure=structure,
                iterations=int(gap_dilation_iter),
            )

        label_map, num = ndi.label(group_mask, structure=structure)

        region_masks = []
        for rid in range(1, num + 1):
            # Keep only original support pixels.
            # Do not include dilated fake pixels.
            m = raw_support & (label_map == rid)

            area = int(m.sum())
            if area >= min_region_area:
                region_masks.append(m)

        # For this mode, small regions are already discarded above.
        # No additional small-fragment merging is needed.
        if split_mode == "gap_tolerant_discard_small":
            merge_small_regions = False

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # ------------------------------------------------------------
    # 2. Optional old behavior: merge small regions
    # ------------------------------------------------------------
    # Current recommendation: keep merge_small_regions=False.
    # If you explicitly set merge_small_regions=True, this restores old behavior.
    if merge_small_regions and len(region_masks) > 0:
        region_masks = _merge_or_discard_small_region_masks(
            region_masks=region_masks,
            A=A,
            B=B,
            min_region_area=min_region_area,
            discard_region_area=discard_region_area,
            merge_small_max_dist=merge_small_max_dist,
            merge_small_vector_cos=merge_small_vector_cos,
        )
    else:
        # Defensive filtering.
        region_masks = [
            m for m in region_masks
            if int(m.sum()) >= min_region_area
        ]

    if len(region_masks) == 0:
        return []

    # ------------------------------------------------------------
    # 3. Build MotionRegion objects
    # ------------------------------------------------------------
    regions = []

    for local_id, region_mask in enumerate(region_masks):
        area = int(region_mask.sum())
        if area < min_region_area:
            continue

        response_strength = np.where(region_mask, A, 0.0).astype(np.float32)
        response_field = np.where(region_mask[:, :, None], B, 0.0).astype(np.float32)

        mean_b = _weighted_mean_response_vector(B, A, region_mask)

        # Representative induced motion of this region:
        # u_r(t) = h(t) * mean_b
        induced_motion = h[:, None] * mean_b[None, :]

        center_xy, spatial_cov, area_effective, strength = _compute_spatial_stats(
            response_strength
        )

        region = MotionRegion(
            episode_id=mode.episode_id,
            mode_id=mode.mode_id,
            region_id=region_id_start + local_id,
            time_range=mode.time_range,

            activation=h.copy(),
            response_field=response_field,
            response_strength=response_strength,
            region_mask=region_mask.astype(np.uint8),

            induced_motion=induced_motion.astype(np.float32),
            mean_response_vector=mean_b.astype(np.float32),

            center_xy=center_xy,
            spatial_cov=spatial_cov,
            area_effective=float(area_effective),
            strength=float(strength),
            metadata={
                "support_rel_thresh": support_rel_thresh,
                "split_mode": split_mode,
                "gap_dilation_iter": gap_dilation_iter,
                "gap_close_iter": gap_close_iter,
                "min_region_area": min_region_area,
                "discard_region_area": discard_region_area,
                "merge_small_regions": merge_small_regions,
                "merge_small_max_dist": merge_small_max_dist,
                "merge_small_vector_cos": merge_small_vector_cos,
                "raw_area": area,
                "small_region_strategy": (
                    "discard" if not merge_small_regions else "merge_or_discard"
                ),
                "final_region_mask": "raw_support_only",
            },
        )

        region.duration = int(len(h))
        if getattr(mode, "activation_resampled", None) is not None:
            region.activation_resampled = np.asarray(
                mode.activation_resampled,
                dtype=np.float32,
            )
        else:
            try:
                region.activation_resampled = _resample_1d_simple(h, target_len=12)
            except NameError:
                region.activation_resampled = h.copy().astype(np.float32)

        region.activation_feature = region.activation_resampled.copy()

        regions.append(region)

    return regions


def split_episode_modes_to_regions(
    episode: MotionEpisode,
    support_rel_thresh=0.05,
    split_mode="gap_tolerant_discard_small",
    gap_dilation_iter=4,
    gap_close_iter=4,
    min_region_area=20,
    discard_region_area=4,
    merge_small_regions=False,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,
    verbose=False,
):
    """
    Split all MotionModes in one episode into MotionRegions.

    Current recommended logic:
        - still split according to spatial connectivity;
        - use gap-tolerant connectivity to avoid over-fragmentation;
        - directly discard small regions;
        - do not merge small fragments into nearby regions.
    """
    regions_all = []
    region_id = 0

    modes = getattr(episode, "modes", None)
    if modes is None:
        modes = []

    for mode in modes:
        # Important: clear old regions to avoid duplicate append when rerunning.
        mode.regions = []

        regs = split_mode_to_regions(
            mode,
            support_rel_thresh=support_rel_thresh,
            split_mode=split_mode,
            gap_dilation_iter=gap_dilation_iter,
            gap_close_iter=gap_close_iter,
            min_region_area=min_region_area,
            discard_region_area=discard_region_area,
            merge_small_regions=merge_small_regions,
            merge_small_max_dist=merge_small_max_dist,
            merge_small_vector_cos=merge_small_vector_cos,
            region_id_start=region_id,
        )

        for r in regs:
            r.region_id = region_id
            r.component_id = region_id
            regions_all.append(r)
            mode.regions.append(r)
            region_id += 1

    episode.regions = regions_all

    if verbose:
        print(
            f"[split_episode_modes_to_regions] "
            f"episode={getattr(episode, 'episode_id', None)}, "
            f"modes={len(modes)}, regions={len(regions_all)}"
        )

    return regions_all


def getMotionRegions(
    motion_episodes,
    support_rel_thresh=0.05,

    split_mode="gap_tolerant_discard_small",
    gap_dilation_iter=4,
    gap_close_iter=4,

    min_region_area=20,
    discard_region_area=4,

    # Current recommendation: do not merge small regions.
    # Small fragments are more likely noise / episode extraction artifacts.
    merge_small_regions=False,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,

    verbose=True,
):
    """
    Convert MotionModes into MotionRegions for all episodes.

    Recommended behavior:
        1. Use tolerant connectivity to avoid excessive fragmentation.
        2. Discard small regions directly.
        3. Do not merge small fragments into nearby large regions.
        4. Reset previous episode.regions and mode.regions every time.
    """
    all_regions = []

    for ep in motion_episodes:
        ep_regions = split_episode_modes_to_regions(
            ep,
            support_rel_thresh=support_rel_thresh,
            split_mode=split_mode,
            gap_dilation_iter=gap_dilation_iter,
            gap_close_iter=gap_close_iter,
            min_region_area=min_region_area,
            discard_region_area=discard_region_area,
            merge_small_regions=merge_small_regions,
            merge_small_max_dist=merge_small_max_dist,
            merge_small_vector_cos=merge_small_vector_cos,
            verbose=False,
        )

        all_regions.extend(ep_regions)

        if verbose:
            modes = getattr(ep, "modes", []) or []
            print(
                f"[getMotionRegions] episode={getattr(ep, 'episode_id', None)}, "
                f"modes={len(modes)}, regions={len(ep_regions)}"
            )

    if verbose:
        n_eps = len(motion_episodes)
        n_modes = sum(len(getattr(ep, "modes", []) or []) for ep in motion_episodes)
        n_regions = len(all_regions)

        print(
            f"[getMotionRegions] total episodes={n_eps}, "
            f"total modes={n_modes}, total regions={n_regions}, "
            f"regions/mode={n_regions / max(n_modes, 1):.3f}"
        )

    return all_regions
# =============================================================================
# MotionRegion pattern clustering
# =============================================================================


def collect_regions_from_episodes(motion_episodes):
    out = []
    for ep in motion_episodes:
        out.extend(getattr(ep, "regions", []) or [])
    return out


def collect_modes_from_episodes(motion_episodes):
    out = []
    for ep in motion_episodes:
        out.extend(getattr(ep, "modes", []) or [])
    return out


def collect_units_from_episodes(motion_episodes, unit_type="region"):
    if unit_type == "mode":
        return collect_modes_from_episodes(motion_episodes)
    return collect_regions_from_episodes(motion_episodes)


def filter_regions_for_patterns(regions, min_strength=0.0, min_area=0.0, min_duration=1):
    """Filter units (MotionRegion or MotionMode) for pattern clustering.

    Missing attributes on MotionMode are computed from response_strength / activation.
    """
    kept = []
    for r in regions:
        # strength
        s = getattr(r, "strength", None)
        if s is None:
            A = getattr(r, "response_strength", None)
            s = float(np.sum(A)) if A is not None else 0.0
        if s < min_strength:
            continue

        # area_effective
        a = getattr(r, "area_effective", None)
        if a is None:
            A = getattr(r, "response_strength", None)
            if A is not None:
                A = np.asarray(A, dtype=np.float32)
                w = A[A > 0]
                a = float(np.sum(w) ** 2 / max(np.sum(w ** 2), 1e-12)) if len(w) > 0 else 0.0
            else:
                a = 0.0
        if a < min_area:
            continue

        # duration
        d = getattr(r, "duration", None)
        if d is None:
            h = getattr(r, "activation", None)
            d = len(np.asarray(h).reshape(-1)) if h is not None else 0
        if d < min_duration:
            continue

        # mean_response_vector
        mrv = getattr(r, "mean_response_vector", None)
        if mrv is None:
            B = getattr(r, "response_field", None)
            A = getattr(r, "response_strength", None)
            if B is not None and A is not None:
                B = np.asarray(B, dtype=np.float32)
                A = np.asarray(A, dtype=np.float32)
                mask = A > 0
                if np.any(mask):
                    w = A[mask]
                    b = B[mask]
                    mrv = (np.sum(w[:, None] * b, axis=0) / max(np.sum(w), 1e-12)).astype(np.float32)
        if mrv is None:
            continue
        # Attach computed mrv for downstream use
        r.mean_response_vector = mrv

        kept.append(r)
    return kept


def _region_support_mask(region):
    # Works for both MotionRegion and MotionMode
    m = getattr(region, "region_mask", None)
    if m is not None:
        return np.asarray(m).astype(bool)
    m = getattr(region, "support_mask", None)
    if m is not None:
        return np.asarray(m).astype(bool)
    A = np.asarray(region.response_strength, dtype=np.float32)
    return A > 0


def _region_iou(r1, r2):
    m1 = _region_support_mask(r1)
    m2 = _region_support_mask(r2)
    if m1.shape != m2.shape:
        return 0.0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / (union + 1e-12))


def _unit_centroid(unit):
    """Compute spatial centroid from response_strength (fallback if no center_xy)."""
    c = getattr(unit, "center_xy", None)
    if c is not None:
        c = np.asarray(c, dtype=np.float32)
        if c.shape == (2,) and np.all(np.isfinite(c)):
            return c
    # Compute from response_strength
    A = np.asarray(unit.response_strength, dtype=np.float32)
    if A.ndim == 2 and np.max(A) > 0:
        pts = np.argwhere(A > 0)
        if len(pts) > 0:
            w = A[A > 0].astype(np.float32)
            c = np.average(pts.astype(np.float32), axis=0, weights=w)
            return c.astype(np.float32)
    return np.array([np.nan, np.nan], dtype=np.float32)


def _region_centroid_distance(r1, r2):
    c1 = _unit_centroid(r1)
    c2 = _unit_centroid(r2)
    if np.any(~np.isfinite(c1)) or np.any(~np.isfinite(c2)):
        return np.inf
    return float(np.linalg.norm(c1 - c2))

def _regions_spatially_compatible(
    r1,
    r2,
    spatial_rule="iou_or_centroid",
    iou_thresh=0.10,
    centroid_dist_thresh=3.0,
):
    iou = _region_iou(r1, r2)
    cd = _region_centroid_distance(r1, r2)
    if spatial_rule == "iou":
        ok = iou >= iou_thresh
    elif spatial_rule == "centroid":
        ok = cd <= centroid_dist_thresh
    elif spatial_rule == "iou_and_centroid":
        ok = (iou >= iou_thresh) and (cd <= centroid_dist_thresh)
    elif spatial_rule == "iou_or_centroid":
        ok = (iou >= iou_thresh) or (cd <= centroid_dist_thresh)
    else:
        raise ValueError(f"Unknown spatial_rule: {spatial_rule}")
    return bool(ok), {"iou": iou, "centroid_distance": cd, "spatial_ok": bool(ok)}

def _sign_aware_activation_dtw(h1, h2):
    """
    Compare h1 with h2 and -h2.
    Return smaller DTW distance and sign for aligning h2/B2.
    """
    d_pos = _dtw_distance_1d(h1, h2)
    d_neg = _dtw_distance_1d(h1, -np.asarray(h2))

    if d_neg < d_pos:
        return float(d_neg), -1.0
    else:
        return float(d_pos), 1.0

def _get_region_activation(region):
    if hasattr(region, "activation") and region.activation is not None:
        return np.asarray(region.activation, dtype=np.float32).reshape(-1)
    if hasattr(region, "activation_resampled") and region.activation_resampled is not None:
        return np.asarray(region.activation_resampled, dtype=np.float32).reshape(-1)
    return None

def _get_region_mask_simple(region):
    # Works for both MotionRegion and MotionMode
    if hasattr(region, "region_mask") and region.region_mask is not None:
        mask = np.asarray(region.region_mask).astype(bool)
        if mask.ndim == 2 and np.any(mask):
            return mask

    if hasattr(region, "support_mask") and region.support_mask is not None:
        mask = np.asarray(region.support_mask).astype(bool)
        if mask.ndim == 2 and np.any(mask):
            return mask

    if hasattr(region, "response_strength") and region.response_strength is not None:
        A = np.asarray(region.response_strength, dtype=np.float32)
        if A.ndim == 2 and np.max(A) > 0:
            return A > 0

    return None

def _get_region_response_field(region):
    if hasattr(region, "response_field") and region.response_field is not None:
        B = np.asarray(region.response_field, dtype=np.float32)
        if B.ndim == 3 and B.shape[-1] == 2:
            return B
    return None

def _mask_iou(mask1, mask2, eps=1e-8):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / (union + eps))

def _response_field_distance_on_overlap(region1, region2, sign2=1.0, eps=1e-8):
    """
    Compare b_ik vector values on the common region only.

    D_b = ||B1 - sign*B2|| / (||B1|| + ||B2||)
    """
    mask1 = _get_region_mask_simple(region1)
    mask2 = _get_region_mask_simple(region2)
    B1 = _get_region_response_field(region1)
    B2 = _get_region_response_field(region2)

    if mask1 is None or mask2 is None or B1 is None or B2 is None:
        return np.inf

    if mask1.shape != mask2.shape or B1.shape != B2.shape:
        return np.inf

    common = np.logical_and(mask1, mask2)
    if not np.any(common):
        return np.inf

    v1 = B1[common]                    # (N, 2)
    v2 = sign2 * B2[common]            # (N, 2)

    numerator = np.sqrt(np.sum((v1 - v2) ** 2))
    denom = np.sqrt(np.sum(v1 ** 2)) + np.sqrt(np.sum(v2 ** 2)) + eps

    return float(numerator / denom)


def _response_field_correlation_on_overlap(region1, region2, sign2=1.0, eps=1e-8):
    """Compare B vector fields using correlation on the overlap region.

    Flattens the 2D vectors on the common mask into (N_common*2,) arrays
    and computes Pearson r. Returns 1 - |r| (sign-insensitive, 0=identical).

    D_b = 1 - |pearson_r(v1, sign2 * v2)|
    """
    mask1 = _get_region_mask_simple(region1)
    mask2 = _get_region_mask_simple(region2)
    B1 = _get_region_response_field(region1)
    B2 = _get_region_response_field(region2)

    if mask1 is None or mask2 is None or B1 is None or B2 is None:
        return np.inf

    if mask1.shape != mask2.shape or B1.shape != B2.shape:
        return np.inf

    common = np.logical_and(mask1, mask2)
    if np.sum(common) < 3:   # need at least 3 points for meaningful correlation
        return np.inf

    v1 = B1[common].ravel().astype(np.float64)           # (N*2,)
    v2 = (sign2 * B2[common]).ravel().astype(np.float64)  # (N*2,)

    # Pearson r
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)
    v1_demean = v1 - v1_mean
    v2_demean = v2 - v2_mean
    num = np.dot(v1_demean, v2_demean)
    denom = np.sqrt(np.sum(v1_demean ** 2) * np.sum(v2_demean ** 2)) + eps

    r = num / denom
    r = np.clip(r, -1.0, 1.0)

    # Sign-insensitive: |r| close to 1 = highly correlated
    return float(1.0 - abs(r))

def _resample_1d_simple(x, target_len=12):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if len(x) == target_len:
        return x.copy()

    if len(x) == 1:
        return np.full(target_len, float(x[0]), dtype=np.float32)

    old_grid = np.linspace(0.0, 1.0, len(x))
    new_grid = np.linspace(0.0, 1.0, target_len)

    return np.interp(new_grid, old_grid, x).astype(np.float32)

def _resample_time_2d_simple(x, target_len=12):
    """
    x: (T, C), e.g. induced_motion with C=2
    return: (target_len, C)
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"x should be 2D, got {x.shape}")

    T, C = x.shape

    if T == 0:
        return np.zeros((target_len, C), dtype=np.float32)

    if T == target_len:
        return x.copy()

    if T == 1:
        return np.repeat(x, target_len, axis=0).astype(np.float32)

    old_grid = np.linspace(0.0, 1.0, T)
    new_grid = np.linspace(0.0, 1.0, target_len)

    out = np.zeros((target_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(new_grid, old_grid, x[:, c])

    return out

def compute_region_distance_matrix_simple(
    regions,
    min_iou=0.10,
    omega=1.0,
    mu=1.0,
    b_distance="l2",         # "l2" or "correlation"
    incompatible_dist=1e6,
    verbose=True,
):
    """
    Pairwise distance for MotionRegionPattern.

    Hard gate:
        IoU(region_i, region_j) >= min_iou

    Soft distance:
        D = omega * D_h + mu * D_b

    where D_h is sign-aware DTW of activations, and D_b is either:
      - "l2":  normalized L2 distance on overlap (Eq: ||B1-sign*B2|| / (||B1||+||B2||))
      - "correlation": 1 - |pearson_r(B1, sign*B2)| on overlap
    """
    n = len(regions)
    D = np.zeros((n, n), dtype=np.float32)
    pair_info = {}

    for i in range(n):
        for j in range(i + 1, n):
            # Hard constraint: never cluster two regions from the SAME episode
            # (they are already distinct spatiotemporal units)
            if getattr(regions[i], "episode_id", None) == getattr(regions[j], "episode_id", None):
                dist = incompatible_dist
                info = {
                    "compatible": False,
                    "reason": "same_episode",
                    "iou": 0.0,
                    "D_h": np.inf,
                    "D_b": np.inf,
                    "distance": dist,
                    "sign": 1.0,
                }

            else:
                mask_i = _get_region_mask_simple(regions[i])
                mask_j = _get_region_mask_simple(regions[j])

                if mask_i is None or mask_j is None or mask_i.shape != mask_j.shape:
                    dist = incompatible_dist
                    info = {
                        "compatible": False,
                        "reason": "invalid_mask",
                        "iou": 0.0,
                        "D_h": np.inf,
                        "D_b": np.inf,
                        "distance": dist,
                        "sign": 1.0,
                    }
                else:
                    iou = _mask_iou(mask_i, mask_j)

                    if iou < min_iou:
                        dist = incompatible_dist
                        info = {
                            "compatible": False,
                            "reason": "low_iou",
                            "iou": float(iou),
                            "D_h": np.inf,
                            "D_b": np.inf,
                            "distance": dist,
                            "sign": 1.0,
                        }
                    else:
                        h_i = _get_region_activation(regions[i])
                        h_j = _get_region_activation(regions[j])

                        if h_i is None or h_j is None:
                            dist = incompatible_dist
                            info = {
                                "compatible": False,
                                "reason": "invalid_activation",
                                "iou": float(iou),
                                "D_h": np.inf,
                                "D_b": np.inf,
                                "distance": dist,
                                "sign": 1.0,
                            }
                        else:
                            D_h, sign_j = _sign_aware_activation_dtw(h_i, h_j)
                            if b_distance == "correlation":
                                D_b = _response_field_correlation_on_overlap(
                                    regions[i], regions[j], sign2=sign_j)
                            else:
                                D_b = _response_field_distance_on_overlap(
                                    regions[i], regions[j], sign2=sign_j)

                            if not np.isfinite(D_h) or not np.isfinite(D_b):
                                dist = incompatible_dist
                                compatible = False
                                reason = "invalid_distance"
                            else:
                                dist = omega * D_h + mu * D_b
                                compatible = True
                                reason = "ok"

                            info = {
                                "compatible": bool(compatible),
                                "reason": reason,
                                "iou": float(iou),
                                "D_h": float(D_h),
                                "D_b": float(D_b),
                                "distance": float(dist),
                                "sign": float(sign_j),
                            }

            D[i, j] = dist
            D[j, i] = dist
            pair_info[(i, j)] = info

    if verbose:
        finite = D[(D > 0) & (D < incompatible_dist)]
        if finite.size > 0:
            print(
                f"[compute_region_distance_matrix_simple] n={n}, "
                f"finite_pairs={finite.size // 2}, "
                f"dist min/median/max="
                f"{finite.min():.4f}/{np.median(finite):.4f}/{finite.max():.4f}"
            )
        else:
            print(f"[compute_region_distance_matrix_simple] n={n}, no compatible pairs.")

    return D, pair_info

def cluster_regions_hierarchical(
    dist_mat,
    cluster_dist_thresh=0.8,
    linkage_method="complete",
    incompatible_dist=1e6,
    verbose=True,
):
    """
    Hierarchical clustering from precomputed distance matrix.
    Complete linkage is recommended to avoid graph-chain effect.
    """
    dist_mat = np.asarray(dist_mat, dtype=np.float32)
    n = dist_mat.shape[0]

    if n == 0:
        return [], np.array([], dtype=np.int32)

    if n == 1:
        return [[0]], np.array([0], dtype=np.int32)

    D = dist_mat.copy()
    D[~np.isfinite(D)] = incompatible_dist
    np.fill_diagonal(D, 0.0)

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)

    raw_labels = fcluster(
        Z,
        t=cluster_dist_thresh,
        criterion="distance",
    )

    unique_labels = sorted(np.unique(raw_labels))
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    labels = np.array([label_map[x] for x in raw_labels], dtype=np.int32)

    groups = []
    for gid in range(len(unique_labels)):
        groups.append(np.where(labels == gid)[0].tolist())

    if verbose:
        print(
            f"[cluster_regions_hierarchical] "
            f"linkage={linkage_method}, "
            f"cluster_dist_thresh={cluster_dist_thresh}, "
            f"n_groups={len(groups)}"
        )
        print("[cluster_regions_hierarchical] group sizes:", [len(g) for g in groups])

    return groups, labels

def region_pair_distance(
    r1,
    r2,
    w_activation=1.0,
    w_response=1.0,
    w_space=0.0,
    spatial_rule="iou_or_centroid",
    iou_thresh=0.10,
    centroid_dist_thresh=3.0,
    enforce_spatial_gate=True,
    incompatible_dist=1e6,
):
    spatial_ok, spatial_info = _regions_spatially_compatible(
        r1, r2,
        spatial_rule=spatial_rule,
        iou_thresh=iou_thresh,
        centroid_dist_thresh=centroid_dist_thresh,
    )
    if enforce_spatial_gate and not spatial_ok:
        return float(incompatible_dist), {**spatial_info, "D_activation": None, "D_response": None, "D_space": None}

    h1 = np.asarray(r1.activation_resampled, dtype=np.float32)
    h2 = np.asarray(r2.activation_resampled, dtype=np.float32)
    corr_h = _safe_corr(h1, h2)
    sign = 1.0 if corr_h >= 0 else -1.0
    D_h = 1.0 - abs(float(corr_h))

    b1 = np.asarray(r1.mean_response_vector, dtype=np.float32)
    b2 = sign * np.asarray(r2.mean_response_vector, dtype=np.float32)
    cos_b = _cosine_similarity(b1, b2)
    D_b = 1.0 - np.clip(cos_b, -1.0, 1.0)

    D_s = 1.0 - spatial_info["iou"] if np.isfinite(spatial_info["iou"]) else 1.0
    D = w_activation * D_h + w_response * D_b + w_space * D_s
    return float(D), {
        **spatial_info,
        "D_activation": float(D_h),
        "activation_corr": float(corr_h),
        "sign": float(sign),
        "D_response": float(D_b),
        "response_cosine": float(cos_b),
        "D_space": float(D_s),
        "D_total": float(D),
    }

def build_region_graph(
    regions,
    spatial_rule="iou_or_centroid",
    iou_thresh=0.10,
    centroid_dist_thresh=3.0,
    activation_dist_thresh=0.35,
    response_dist_thresh=0.45,
    total_dist_thresh=None,
    w_activation=1.0,
    w_response=1.0,
    w_space=0.0,
    enforce_spatial_gate=True,
    verbose=True,
):
    n = len(regions)
    neighbors = [[] for _ in range(n)]
    edge_info = {}
    stats = {"pairs": 0, "spatial_pass": 0, "activation_pass": 0, "response_pass": 0, "edge": 0}
    for i in range(n):
        for j in range(i + 1, n):
            stats["pairs"] += 1
            D, info = region_pair_distance(
                regions[i], regions[j],
                w_activation=w_activation,
                w_response=w_response,
                w_space=w_space,
                spatial_rule=spatial_rule,
                iou_thresh=iou_thresh,
                centroid_dist_thresh=centroid_dist_thresh,
                enforce_spatial_gate=enforce_spatial_gate,
            )
            if not info.get("spatial_ok", True):
                continue
            stats["spatial_pass"] += 1
            if info["D_activation"] is None or info["D_activation"] > activation_dist_thresh:
                continue
            stats["activation_pass"] += 1
            if info["D_response"] is None or info["D_response"] > response_dist_thresh:
                continue
            stats["response_pass"] += 1
            if total_dist_thresh is not None and D > total_dist_thresh:
                continue
            neighbors[i].append(j)
            neighbors[j].append(i)
            edge_info[(i, j)] = info
            stats["edge"] += 1
    if verbose:
        print(f"[region graph] nodes={n}, pairs={stats['pairs']}, spatial={stats['spatial_pass']}, "
              f"activation={stats['activation_pass']}, response={stats['response_pass']}, edges={stats['edge']}")
    return neighbors, edge_info, stats

def _dtw_distance_1d(x, y, eps=1e-8):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.inf

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    return float(dp[n, m] / (n + m + eps))

def _find_activation_medoid_index(activations, weights=None, eps=1e-8):
    """
    Select the region whose activation is most representative of the pattern.
    This does not require equal-length activations.
    """
    n = len(activations)
    if n == 0:
        return None
    if n == 1:
        return 0

    if weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

    weights = weights / (np.sum(weights) + eps)

    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            d = _dtw_distance_1d(activations[i], activations[j], eps=eps)
            s += weights[j] * d
        scores[i] = s

    return int(np.argmin(scores))

def _sign_by_resampled_corr(ref_h, h, target_len=16):
    """
    Only for sign alignment. Raw activations can still keep original length.
    """
    ref_h = np.asarray(ref_h, dtype=np.float32).reshape(-1)
    h = np.asarray(h, dtype=np.float32).reshape(-1)

    if len(ref_h) == 0 or len(h) == 0:
        return 1.0

    ref_rs = _resample_1d(ref_h, target_len)
    h_rs = _resample_1d(h, target_len)

    return 1.0 if _safe_corr(ref_rs, h_rs) >= 0 else -1.0

def connected_components_from_graph(neighbors):
    n = len(neighbors)
    visited = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        group = []
        while stack:
            u = stack.pop()
            group.append(u)
            for v in neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(sorted(group))
    return groups

def build_motion_patterns_from_groups(regions, groups):
    """
    Build MotionPattern objects from clustered MotionRegion groups.

    Parameters
    ----------
    regions : list[MotionRegion]
        The filtered MotionRegion list.
    groups : list[list[int]]
        Each group contains indices into `regions`.

    Returns
    -------
    patterns : list[MotionPattern]
    """
    patterns = []

    for pid, group in enumerate(groups):
        group_regions = [regions[i] for i in group]
        pattern = MotionPattern(
            pattern_id=pid,
            regions=group_regions,
        )
        patterns.append(pattern)

    return patterns

def getMotionPattern(
    motion_episodes,
    unit_type="region",        # "region" or "mode"
    min_strength=0.0,
    min_area=5,
    min_duration=1,

    min_iou=0.10,
    omega=1.0,
    mu=1.0,
    b_distance="l2",           # "l2" or "correlation"

    cluster_dist_thresh=0.8,
    linkage_method="complete",
    incompatible_dist=1e6,

    verbose=True,
):
    """Build MotionPatterns from MotionRegions or MotionModes.

    Parameters
    ----------
    unit_type : str
        "region" - cluster MotionRegions (default).
        "mode"   - cluster MotionModes directly, skipping spatial splitting.
    """
    # Collect units
    all_units = collect_units_from_episodes(motion_episodes, unit_type=unit_type)

    if verbose:
        print(f"[getMotionPattern] collected {unit_type}s: {len(all_units)}")

    kept_units = filter_regions_for_patterns(
        all_units,
        min_strength=min_strength,
        min_area=min_area,
        min_duration=min_duration,
    )

    if verbose:
        print(f"[getMotionPattern] kept {unit_type}s: {len(kept_units)}")

    if len(kept_units) == 0:
        return [], [], [], np.array([], dtype=np.int32), {}

    dist_mat, pair_info = compute_region_distance_matrix_simple(
        kept_units,
        min_iou=min_iou,
        omega=omega,
        mu=mu,
        b_distance=b_distance,
        incompatible_dist=incompatible_dist,
        verbose=verbose,
    )

    groups, labels = cluster_regions_hierarchical(
        dist_mat,
        cluster_dist_thresh=cluster_dist_thresh,
        linkage_method=linkage_method,
        incompatible_dist=incompatible_dist,
        verbose=verbose,
    )

    patterns = build_motion_patterns_from_groups(
        kept_units,
        groups,
    )

    info = {
        "distance_matrix": dist_mat,
        "pair_info": pair_info,
        "labels": labels,
        "unit_type": unit_type,
        "params": {
            "unit_type": unit_type,
            "min_strength": min_strength,
            "min_area": min_area,
            "min_duration": min_duration,
            "min_iou": min_iou,
            "omega": omega,
            "mu": mu,
            "cluster_dist_thresh": cluster_dist_thresh,
            "linkage_method": linkage_method,
            "incompatible_dist": incompatible_dist,
        },
    }

    return patterns, kept_units, groups, labels, info


# Backward-compatible alias
def getMotionRegionPattern(
    motion_episodes,
    min_strength=0.0,
    min_area=5,
    min_duration=1,
    min_iou=0.10,
    omega=1.0,
    mu=1.0,
    b_distance="l2",
    cluster_dist_thresh=0.8,
    linkage_method="complete",
    incompatible_dist=1e6,
    verbose=True,
):
    """Backward-compatible wrapper - delegates to getMotionPattern(unit_type='region')."""
    return getMotionPattern(
        motion_episodes,
        unit_type="region",
        min_strength=min_strength,
        min_area=min_area,
        min_duration=min_duration,
        min_iou=min_iou,
        omega=omega,
        mu=mu,
        b_distance=b_distance,
        cluster_dist_thresh=cluster_dist_thresh,
        linkage_method=linkage_method,
        incompatible_dist=incompatible_dist,
        verbose=verbose,
    )

def pattern_to_binary_mask(
    pattern,
    support_rel_thresh=0.20,
    use_region_union=True,
    keep_largest_cc=False,
):
    """
    Convert one MotionPattern to a binary spatial mask.

    Priority:
        1. union of member region masks;
        2. prototype_region_map / prototype_region fallback.
    """
    masks = []

    if use_region_union:
        for r in getattr(pattern, "regions", []) or []:
            m = getattr(r, "region_mask", None)
            if m is not None:
                masks.append(np.asarray(m).astype(bool))

    if len(masks) > 0:
        mask = np.any(np.stack(masks, axis=0), axis=0)
    else:
        region_map = getattr(pattern, "prototype_region_map", None)
        if region_map is None:
            region_map = getattr(pattern, "prototype_region", None)

        if region_map is None:
            return None

        region_map = np.asarray(region_map, dtype=np.float32)
        vmax = float(np.nanmax(region_map))

        if vmax <= 0:
            return None

        mask = region_map > support_rel_thresh * vmax

    if keep_largest_cc:
        labeled, num = ndi.label(mask)
        if num > 0:
            areas = np.array([(labeled == rid).sum() for rid in range(1, num + 1)])
            rid = int(np.argmax(areas)) + 1
            mask = labeled == rid

    return mask.astype(bool)

def find_patterns_overlapping_region(
    patterns,
    query_mask,
    min_iou=0.05,
    min_query_covered=0.10,
    min_pattern_covered=0.01,
    support_rel_thresh=0.20,
    use_region_union=True,
    keep_largest_cc=False,
    sort_by="query_covered",
):
    """
    Find all motion patterns overlapping with a user-specified region.

    Parameters
    ----------
    patterns:
        list of MotionPattern.

    query_mask:
        boolean array, same spatial shape as pattern masks.

    min_iou:
        minimum IoU between pattern mask and query mask.

    min_query_covered:
        require at least this fraction of the query region covered by pattern.

    min_pattern_covered:
        require at least this fraction of the pattern covered by query.

    Returns
    -------
    rows:
        list of dicts, each containing pattern_id and overlap statistics.
    """
    query_mask = np.asarray(query_mask).astype(bool)

    if query_mask.sum() == 0:
        raise ValueError("query_mask is empty.")

    rows = []

    for p in patterns:
        pid = getattr(p, "pattern_id", None)

        pmask = pattern_to_binary_mask(
            p,
            support_rel_thresh=support_rel_thresh,
            use_region_union=use_region_union,
            keep_largest_cc=keep_largest_cc,
        )

        if pmask is None:
            continue

        if pmask.shape != query_mask.shape:
            raise ValueError(
                f"Pattern {pid} mask shape {pmask.shape} != query_mask shape {query_mask.shape}"
            )

        inter = np.logical_and(pmask, query_mask).sum()
        union = np.logical_or(pmask, query_mask).sum()

        pattern_area = int(pmask.sum())
        query_area = int(query_mask.sum())

        if union == 0 or pattern_area == 0 or query_area == 0:
            continue

        iou = inter / union
        query_covered = inter / query_area
        pattern_covered = inter / pattern_area

        if (
            iou >= min_iou
            or query_covered >= min_query_covered
            or pattern_covered >= min_pattern_covered
        ):
            rows.append({
                "pattern_id": pid,
                "n_regions": len(getattr(p, "regions", []) or []),
                "pattern_area": pattern_area,
                "query_area": query_area,
                "intersection": int(inter),
                "iou": float(iou),
                "query_covered": float(query_covered),
                "pattern_covered": float(pattern_covered),
                "pattern": p,
                "pattern_mask": pmask,
            })

    if sort_by is not None:
        rows = sorted(rows, key=lambda x: x[sort_by], reverse=True)

    return rows

def make_query_mask_from_point(shape, point_xy, radius=5):
    """
    Make a circular query mask around one point.

    point_xy:
        (x, y) or (row, col) index in patch-coordinate system.
        Must match your pattern mask coordinate convention.
    """
    X, Y = shape
    x0, y0 = point_xy

    xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    dist2 = (xx - x0) ** 2 + (yy - y0) ** 2

    return dist2 <= radius ** 2

def make_query_mask_from_bbox(shape, x0, x1, y0, y1):
    """
    Make rectangular query mask.

    Coordinates are in patch-coordinate system:
        x in [x0, x1-1]
        y in [y0, y1-1]
    """
    X, Y = shape

    x0 = max(0, int(x0))
    x1 = min(X, int(x1))
    y0 = max(0, int(y0))
    y1 = min(Y, int(y1))

    mask = np.zeros(shape, dtype=bool)
    mask[x0:x1, y0:y1] = True

    return mask



# =============================================================================
# Visualization
# =============================================================================


def _auto_contrast(img):
    img = np.asarray(img)
    if img.size == 0:
        return img
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return img
    return np.clip((img - p1) / (p99 - p1), 0, 1)


def _get_BH_from_episode(ep):
    model = getattr(ep, "mode_model", None)
    if not model or "B" not in model:
        raise ValueError("episode.mode_model is missing or empty (episode not fitted).")
    B = np.asarray(ep.mode_model["B"], dtype=np.float32)  # (2N, K)
    H = np.asarray(ep.mode_model["H"], dtype=np.float32)  # (K, T)
    return B, H


def _episode_motion_matrix(ep, use_global_subtracted=True):
    motion_abs = np.asarray(ep.motion_abs, dtype=np.float32)  # (T,N,2)

    if use_global_subtracted:
        global_motion = np.asarray(ep.global_motion, dtype=np.float32)  # (T,2)
        Y = motion_abs - global_motion[:, None, :]
    else:
        Y = motion_abs

    M = np.concatenate(
        [Y[:, :, 0].T, Y[:, :, 1].T],
        axis=0,
    ).astype(np.float32)  # (2N,T)

    return M, Y


def _B_column_to_maps(ep, b_col):
    """
    b_col: (2N,)
    return:
        bx_map, by_map, mag_map, mask
    """
    mask = np.asarray(ep.spatial_region).astype(bool)
    coords = np.argwhere(mask)
    N = len(coords)

    bx = b_col[:N]
    by = b_col[N:]

    bx_map = np.zeros(mask.shape, dtype=np.float32)
    by_map = np.zeros(mask.shape, dtype=np.float32)

    bx_map[coords[:, 0], coords[:, 1]] = bx
    by_map[coords[:, 0], coords[:, 1]] = by

    mag_map = np.sqrt(bx_map ** 2 + by_map ** 2)

    return bx_map, by_map, mag_map, mask


def _frame_vector_to_maps(ep, frame_vec):
    """
    frame_vec: (2N,)
    """
    return _B_column_to_maps(ep, frame_vec)


def _safe_norm(x, eps=1e-12):
    return float(np.sqrt(np.sum(np.asarray(x) ** 2)) + eps)


def diagnose_temporal_basis_like_sources(ep, eps=1e-12):
    B, H = _get_BH_from_episode(ep)

    K, T = H.shape

    rows = []

    for k in range(K):
        h = H[k].astype(np.float32)
        abs_h = np.abs(h)

        l1 = float(np.sum(abs_h)) + eps
        l2_sq = float(np.sum(h ** 2)) + eps

        peak_frame = int(np.argmax(abs_h))
        peak_value = float(abs_h[peak_frame])

        # 越接近 1，越像单帧 one-hot；越接近 1/T，越分散
        peak_fraction = peak_value / l1

        # participation ratio:
        # one-hot -> 1
        # uniform over T frames -> T
        temporal_pr = (l1 ** 2) / l2_sq

        # entropy normalized to [0,1]
        p = abs_h / l1
        entropy = -float(np.sum(p * np.log(p + eps))) / np.log(T + eps)

        b_norm = _safe_norm(B[:, k], eps=eps)

        rows.append({
            "mode": k,
            "T": T,
            "peak_frame": peak_frame,
            "peak_fraction": peak_fraction,
            "temporal_participation_ratio": temporal_pr,
            "temporal_entropy": entropy,
            "B_norm": b_norm,
        })

    return rows


def visualize_episode_sources_overview(
    ep,
    max_modes=20,
    sort_by="B_norm",   # "B_norm", "peak_frame", None
    quiver_step=3,
    figsize=(16, 10),
):
    B, H = _get_BH_from_episode(ep)
    K, T = H.shape

    diag = diagnose_temporal_basis_like_sources(ep)

    if sort_by == "B_norm":
        order = sorted(range(K), key=lambda k: diag[k]["B_norm"], reverse=True)
    elif sort_by == "peak_frame":
        order = sorted(range(K), key=lambda k: diag[k]["peak_frame"])
    else:
        order = list(range(K))

    order = order[:min(max_modes, K)]

    n = len(order)
    if n == 0:
        print("[visualize_episode_sources_overview] no modes to visualize (K=0)")
        return diag
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    # --------- Activation heatmap ---------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    H_show = H[order]
    im = ax.imshow(H_show, aspect="auto", interpolation="nearest")
    ax.set_title("Source activations H[k,t]")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("source index after sorting")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(k) for k in order])
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    for kk in order:
        h = H[kk]
        ax.plot(np.arange(T), h, marker="o", linewidth=1, label=f"k={kk}")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Activation curves")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("h_k(t)")
    ax.grid(True, alpha=0.3)
    if len(order) <= 10:
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()

    # --------- Spatial response maps ---------
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)

    for ax_i, k in enumerate(order):
        ax = axes[ax_i]
        bx_map, by_map, mag_map, mask = _B_column_to_maps(ep, B[:, k])

        ax.imshow(mask, cmap="gray", alpha=0.2)
        im = ax.imshow(np.ma.masked_where(mag_map <= 0, mag_map), cmap="magma", alpha=0.9)

        yy, xx = np.where(mag_map > 0)
        if len(yy) > 0:
            keep = (yy % quiver_step == 0) & (xx % quiver_step == 0)
            yy2 = yy[keep]
            xx2 = xx[keep]
            ax.quiver(
                xx2,
                yy2,
                bx_map[yy2, xx2],
                by_map[yy2, xx2],
                angles="xy",
                scale_units="xy",
            )

        d = diag[k]
        title = (
            f"k={k}, peak={d['peak_frame']}, "
            f"PR={d['temporal_participation_ratio']:.2f}, "
            f"pf={d['peak_fraction']:.2f}"
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for j in range(len(order), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Episode {getattr(ep, 'episode_id', None)} source spatial responses",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    return diag


def visualize_frame_source_contributions(ep, figsize=(10, 5)):
    B, H = _get_BH_from_episode(ep)
    K, T = H.shape

    B_norm_sq = np.sum(B ** 2, axis=0)  # (K,)
    E = (H ** 2) * B_norm_sq[:, None]   # (K,T)

    E_sum_t = np.sum(E, axis=0, keepdims=True) + 1e-12
    frac = E / E_sum_t  # 每一帧由每个 source 解释的比例

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    im = ax.imshow(frac, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Frame-wise source contribution fraction")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("source k")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    dominant_k = np.argmax(frac, axis=0)
    dominance = np.max(frac, axis=0)
    ax.plot(np.arange(T), dominant_k, marker="o", label="dominant source")
    ax2 = ax.twinx()
    ax2.plot(np.arange(T), dominance, marker="s", linestyle="--", label="dominance")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("dominant source k")
    ax2.set_ylabel("dominance fraction")
    ax.set_title("Dominant source per frame")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("dominant source per frame:", dominant_k)
    print("dominance fraction:", dominance)

    return frac, dominant_k, dominance


def visualize_episode_modes(
    episode: MotionEpisode,
    ref_img=None,
    patch_size=7,
    max_modes=8,
    rel_thresh=0.10,
    arrow_step=2,
    arrow_scale=None,
    vector_order="xy",
    flip_y_vector=False,
    figsize_per_row=3.2,
    save_path=None,
    show=True,
):
    modes = getattr(episode, "modes", [])
    if len(modes) == 0:
        print(f"Episode {episode.episode_id}: no modes.")
        return None
    modes = modes[:max_modes]
    n_rows = 1 + len(modes)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * figsize_per_row), squeeze=False)
    mask = np.asarray(episode.spatial_region).astype(bool)
    ax = axes[0, 0]
    ax.imshow(mask, cmap="gray")
    ax.set_title(f"Episode {episode.episode_id}\ntime={episode.time_range}, modes={len(getattr(episode, 'modes', []))}")
    ax.axis("off")
    # observed/recon/resid maps
    obs_map, rec_map, res_map = compute_episode_mode_reconstruction_maps(episode)
    for col, (title, M) in enumerate([("Observed RMS", obs_map), ("Reconstructed RMS", rec_map), ("Residual RMS", res_map)], start=1):
        ax = axes[0, col]
        if M is not None:
            im = ax.imshow(M, cmap="magma")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(mask, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    for row, mode in enumerate(modes, start=1):
        A = np.asarray(mode.response_strength, dtype=np.float32)
        B = np.asarray(mode.response_field, dtype=np.float32)
        h = np.asarray(mode.activation, dtype=np.float32)
        vmax = float(np.max(A)) if A.size > 0 else 0.0
        support = A > rel_thresh * vmax if vmax > 0 else np.zeros_like(A, dtype=bool)
        ax = axes[row, 0]
        im = ax.imshow(A, cmap="magma")
        if np.any(support):
            ax.contour(support.astype(float), levels=[0.5], linewidths=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Mode {mode.mode_id}\nE={mode.explained_energy:.3f}, mass={mode.mode_mass:.2f}")
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(mask, cmap="gray", alpha=0.35)
        ax.imshow(support.astype(float), cmap="viridis", alpha=0.75)
        ax.set_title(f"support area={int(support.sum())}")
        ax.axis("off")

        ax = axes[row, 2]
        ax.imshow(A, cmap="gray", alpha=0.45)
        rr, cc = np.where(support)
        if len(rr) > 0:
            keep = np.arange(len(rr))
            if arrow_step and arrow_step > 1:
                keep = keep[((rr[keep] % arrow_step) == 0) & ((cc[keep] % arrow_step) == 0)]
            rr2, cc2 = rr[keep], cc[keep]
            if vector_order == "xy":
                U, V = B[rr2, cc2, 0], B[rr2, cc2, 1]
            elif vector_order == "yx":
                U, V = B[rr2, cc2, 1], B[rr2, cc2, 0]
            else:
                raise ValueError("vector_order must be 'xy' or 'yx'.")
            if flip_y_vector:
                V = -V
            if arrow_scale is None:
                ax.quiver(cc2, rr2, U, V, angles="xy", scale_units="xy")
            else:
                ax.quiver(cc2, rr2, U, V, angles="xy", scale_units="xy", scale=arrow_scale)
        ax.set_xlim(-0.5, A.shape[1] - 0.5)
        ax.set_ylim(A.shape[0] - 0.5, -0.5)
        ax.set_title("response vector field")
        ax.axis("off")

        ax = axes[row, 3]
        ax.plot(np.arange(len(h)), h, marker="o")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("activation h(t)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compute_episode_mode_reconstruction_maps(episode: MotionEpisode):
    if episode.motion_abs is None or episode.global_motion is None:
        return None, None, None
    mask = np.asarray(episode.spatial_region).astype(bool)
    coords = np.argwhere(mask)
    motion_abs = np.asarray(episode.motion_abs, dtype=np.float32)
    global_motion = np.asarray(episode.global_motion, dtype=np.float32)
    Y_data = motion_abs - global_motion[:, None, :]
    recon = np.zeros_like(Y_data, dtype=np.float32)
    for mode in getattr(episode, "modes", []):
        h = np.asarray(mode.activation, dtype=np.float32)
        B = np.asarray(mode.response_field, dtype=np.float32)
        Bc = B[coords[:, 0], coords[:, 1], :]
        recon += h[:, None, None] * Bc[None, :, :]
    resid = Y_data - recon
    obs_mag = np.sqrt(np.mean(np.sum(Y_data ** 2, axis=-1), axis=0))
    rec_mag = np.sqrt(np.mean(np.sum(recon ** 2, axis=-1), axis=0))
    res_mag = np.sqrt(np.mean(np.sum(resid ** 2, axis=-1), axis=0))
    return (
        _compact_to_full(obs_mag, coords, mask.shape),
        _compact_to_full(rec_mag, coords, mask.shape),
        _compact_to_full(res_mag, coords, mask.shape),
    )


def _compact_to_full(values, coords, shape, fill=0.0):
    values = np.asarray(values)
    coords = np.asarray(coords)
    if values.ndim == 1:
        out = np.full(shape, fill, dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            out[r, c] = values[i]
        return out
    else:
        out = np.full((*shape, values.shape[1]), fill, dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            out[r, c, :] = values[i]
        return out


def visualize_episode_regions(
    episode: MotionEpisode,
    ref_img=None,
    patch_size=7,
    max_regions=20,
    rel_thresh=0.10,
    figsize=(10, 8),
    save_path=None,
    show=True,
):
    regions = getattr(episode, "regions", [])
    if len(regions) == 0:
        print(f"Episode {episode.episode_id}: no regions.")
        return None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if ref_img is not None:
        ax.imshow(_auto_contrast(ref_img), cmap="gray", origin="upper")
    else:
        ax.imshow(np.asarray(episode.spatial_region).T, cmap="gray", origin="upper")
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(regions[:max_regions]):
        A = np.asarray(r.response_strength, dtype=np.float32)
        vmax = float(np.max(A))
        if vmax <= 0:
            continue
        mask = A > rel_thresh * vmax
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        vals = A[mask]
        vals = vals / (vmax + 1e-12)
        color = np.array(cmap(i % 20))
        X = coords[:, 0]
        Y = coords[:, 1]
        col = Y * patch_size + patch_size // 2
        row = X * patch_size + patch_size // 2
        rgba = np.tile(color, (len(vals), 1))
        rgba[:, 3] = 0.25 + 0.65 * vals
        ax.scatter(col, row, s=15 + 45 * vals, c=rgba, edgecolors="none", label=f"R{i}/M{r.mode_id}")
        v = np.asarray(r.mean_response_vector, dtype=np.float32)
        if np.all(np.isfinite(v)):
            # draw one representative arrow at region center
            cx, cy = r.center_xy
            ax.arrow(cy * patch_size + patch_size // 2, cx * patch_size + patch_size // 2,
                     v[1] * 5.0, v[0] * 5.0, color=color, head_width=3, head_length=4,
                     length_includes_head=True, alpha=0.8)
    ax.set_title(f"Episode {episode.episode_id}: MotionRegions n={len(regions)}")
    ax.set_aspect("equal")
    if ref_img is not None:
        H, W = ref_img.shape[:2]
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
    else:
        ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compare_sources_to_observed_frames(
    ep,
    max_modes=20,
    sort_by_peak=True,
    quiver_step=3,
    figsize_per_row=(12, 3),
):
    B, H = _get_BH_from_episode(ep)
    M, Y = _episode_motion_matrix(ep, use_global_subtracted=True)

    K, T = H.shape
    diag = diagnose_temporal_basis_like_sources(ep)

    if sort_by_peak:
        order = sorted(range(K), key=lambda k: diag[k]["peak_frame"])
    else:
        order = list(range(K))

    order = order[:min(max_modes, K)]

    n = len(order)
    if n == 0:
        print("[compare_sources_to_observed_frames] no modes to visualize (K=0)")
        return None
    fig, axes = plt.subplots(n, 3, figsize=(figsize_per_row[0], figsize_per_row[1] * n))
    if n == 1:
        axes = axes[None, :]

    for row, k in enumerate(order):
        h = H[k]
        t_peak = int(np.argmax(np.abs(h)))

        # source response B_k
        bx_b, by_b, mag_b, mask = _B_column_to_maps(ep, B[:, k])

        # source contribution at peak frame: h_k(t_peak) * B_k
        contrib_vec = B[:, k] * h[t_peak]
        bx_c, by_c, mag_c, _ = _frame_vector_to_maps(ep, contrib_vec)

        # observed motion at peak frame
        obs_vec = M[:, t_peak]
        bx_o, by_o, mag_o, _ = _frame_vector_to_maps(ep, obs_vec)

        maps = [
            (mag_b, bx_b, by_b, f"B_k response | k={k}, peak={t_peak}"),
            (mag_c, bx_c, by_c, f"h_k(t_peak) B_k"),
            (mag_o, bx_o, by_o, f"Observed M[:, {t_peak}]"),
        ]

        for col, (mag, bx, by, title) in enumerate(maps):
            ax = axes[row, col]
            ax.imshow(mask, cmap="gray", alpha=0.2)
            ax.imshow(np.ma.masked_where(mag <= 0, mag), cmap="magma", alpha=0.9)

            yy, xx = np.where(mag > 0)
            if len(yy) > 0:
                keep = (yy % quiver_step == 0) & (xx % quiver_step == 0)
                yy2 = yy[keep]
                xx2 = xx[keep]
                ax.quiver(
                    xx2,
                    yy2,
                    bx[yy2, xx2],
                    by[yy2, xx2],
                    angles="xy",
                    scale_units="xy",
                )

            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def summarize_temporal_basis_likeness(episodes):
    rows = []

    for ei, ep in enumerate(episodes):
        model = getattr(ep, "mode_model", None)
        if not model or "B" not in model:
            continue

        B, H = _get_BH_from_episode(ep)
        K, T = H.shape

        diag = diagnose_temporal_basis_like_sources(ep)

        peak_fracs = np.array([d["peak_fraction"] for d in diag], dtype=np.float32)
        prs = np.array([d["temporal_participation_ratio"] for d in diag], dtype=np.float32)
        ent = np.array([d["temporal_entropy"] for d in diag], dtype=np.float32)
        peaks = np.array([d["peak_frame"] for d in diag], dtype=np.int32)

        unique_peak_count = len(np.unique(peaks))

        rows.append({
            "episode_index": ei,
            "episode_id": getattr(ep, "episode_id", None),
            "T": T,
            "K": K,
            "mean_peak_fraction": float(np.mean(peak_fracs)),
            "median_peak_fraction": float(np.median(peak_fracs)),
            "mean_temporal_PR": float(np.mean(prs)),
            "median_temporal_PR": float(np.median(prs)),
            "mean_entropy": float(np.mean(ent)),
            "unique_peak_count": unique_peak_count,
            "unique_peak_count_over_T": unique_peak_count / max(T, 1),
        })

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows





def save_episode_mode_region_gallery(
    motion_episodes,
    out_dir,
    episode_ids=None,
    max_episodes=None,
    patch_size=7,
    max_modes=8,
    max_regions=20,
    rel_thresh=0.10,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if episode_ids is None:
        eps = list(motion_episodes)
    else:
        ids = set(episode_ids)
        eps = [ep for ep in motion_episodes if ep.episode_id in ids]
    if max_episodes is not None:
        eps = eps[:max_episodes]
    saved = []
    for ep in eps:
        if len(getattr(ep, "modes", [])) > 0:
            p = out_dir / f"episode_{ep.episode_id:04d}_modes.png"
            visualize_episode_modes(ep, patch_size=patch_size, max_modes=max_modes, rel_thresh=rel_thresh, save_path=p, show=False)
            saved.append(p)
        if len(getattr(ep, "regions", [])) > 0:
            p = out_dir / f"episode_{ep.episode_id:04d}_regions.png"
            visualize_episode_regions(ep, patch_size=patch_size, max_regions=max_regions, rel_thresh=rel_thresh, save_path=p, show=False)
            saved.append(p)
    print(f"Saved {len(saved)} figures to {out_dir}")
    return saved


def _region_center_from_mask(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    return pts.mean(axis=0)


def _get_region_mask(region):
    if hasattr(region, "region_mask") and region.region_mask is not None:
        return np.asarray(region.region_mask).astype(bool)

    if hasattr(region, "response_strength") and region.response_strength is not None:
        A = np.asarray(region.response_strength, dtype=np.float32)
        if A.size > 0 and np.max(A) > 0:
            return A > 0

    return None


def _get_region_strength(region):
    if hasattr(region, "response_strength") and region.response_strength is not None:
        return np.asarray(region.response_strength, dtype=np.float32)

    mask = _get_region_mask(region)
    if mask is None:
        return None

    return mask.astype(np.float32)


def diagnose_episode_svd_rank(ep, use_global_subtracted=True, ranks=(1, 2, 3, 5, 10, 20)):
    """
    Check the best possible low-rank reconstruction upper bound for one episode.
    """
    if use_global_subtracted:
        Y = np.asarray(ep.motion_abs, dtype=np.float32) - np.asarray(ep.global_motion, dtype=np.float32)[:, None, :]
    else:
        Y = np.asarray(ep.motion_abs, dtype=np.float32)

    # Y: (T, N, 2)
    T, N, _ = Y.shape

    # M: (2N, T)
    M = np.concatenate([Y[:, :, 0].T, Y[:, :, 1].T], axis=0)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    total_energy = np.sum(S ** 2) + 1e-12

    result = {}
    for r in ranks:
        r = min(r, len(S))
        explained = np.sum(S[:r] ** 2) / total_energy
        result[r] = float(explained)

    return result


def visualize_episode_regions_overview(
    episode,
    max_regions=80,
    min_area=0,
    color_by="region",   # "region" or "mode"
    show_id=True,
    show_center=True,
    figsize=(8, 8),
    title=None,
):
    """
    Visualize all MotionRegions in one episode.

    color_by:
        "region": each region has a different label/color
        "mode": regions from the same parent mode have the same label/color
    """
    regions = getattr(episode, "regions", [])
    if len(regions) == 0:
        print(f"Episode {getattr(episode, 'episode_id', None)} has no regions.")
        return None

    mask_ep = np.asarray(episode.spatial_region).astype(bool)
    label_map = np.zeros(mask_ep.shape, dtype=np.int32)

    region_records = []

    for idx, r in enumerate(regions):
        mask = _get_region_mask(r)
        if mask is None:
            continue

        area = int(mask.sum())
        if area < min_area:
            continue

        region_records.append((idx, r, mask, area))

    # sort by area, keep largest max_regions
    region_records = sorted(region_records, key=lambda x: x[3], reverse=True)
    region_records = region_records[:max_regions]

    for display_idx, (idx, r, mask, area) in enumerate(region_records, start=1):
        if color_by == "mode":
            val = int(getattr(r, "mode_id", 0)) + 1
        else:
            val = display_idx

        label_map[mask] = val

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(mask_ep, cmap="gray", alpha=0.25)
    im = ax.imshow(np.ma.masked_where(label_map == 0, label_map), cmap="tab20", alpha=0.85)

    if show_id or show_center:
        for display_idx, (idx, r, mask, area) in enumerate(region_records, start=1):
            center = _region_center_from_mask(mask)
            if center is None:
                continue

            y, x = center
            if show_center:
                ax.scatter([x], [y], s=15, c="white", edgecolors="black", linewidths=0.5)

            if show_id:
                mode_id = getattr(r, "mode_id", -1)
                region_id = getattr(r, "region_id", idx)
                txt = f"{mode_id}:{region_id}" if color_by == "mode" else str(display_idx)
                ax.text(
                    x, y, txt,
                    color="white",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"),
                )

    ep_id = getattr(episode, "episode_id", None)
    tr = getattr(episode, "time_range", None)
    if title is None:
        title = f"Episode {ep_id} regions | time={tr} | n={len(region_records)}"

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig
# =============================================================================
# Diagnostics
# =============================================================================


def mode_activation_corr_matrix(ep: MotionEpisode):
    modes = getattr(ep, "modes", [])
    if len(modes) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    H = np.stack([np.asarray(m.activation).reshape(-1) for m in modes], axis=0)
    H = H - H.mean(axis=1, keepdims=True)
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return (H @ H.T).astype(np.float32)


def mode_response_iou_matrix(ep: MotionEpisode, rel_thresh=0.10):
    modes = getattr(ep, "modes", [])
    n = len(modes)
    out = np.zeros((n, n), dtype=np.float32)
    masks = []
    for m in modes:
        A = np.asarray(m.response_strength)
        if A.size == 0 or np.max(A) <= 0:
            masks.append(np.zeros_like(A, dtype=bool))
        else:
            masks.append(A > rel_thresh * np.max(A))
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            out[i, j] = inter / (union + 1e-12)
    return out


def mode_incremental_contribution(ep: MotionEpisode):
    model = getattr(ep, "mode_model", None)
    if not model or "B" not in model or "H" not in model:
        return np.array([], dtype=np.float32)
    B = model["B"]
    H = model["H"]
    motion_abs = np.asarray(ep.motion_abs, dtype=np.float32)
    global_motion = np.asarray(ep.global_motion, dtype=np.float32)
    Y_data = motion_abs - global_motion[:, None, :]
    M = np.concatenate([Y_data[:, :, 0].T, Y_data[:, :, 1].T], axis=0)
    err_full = float(np.sum((M - B @ H) ** 2))
    total = float(np.sum(M ** 2)) + 1e-12
    vals = []
    for k in range(H.shape[0]):
        B_wo = B.copy()
        B_wo[:, k] = 0.0
        err_wo = float(np.sum((M - B_wo @ H) ** 2))
        vals.append((err_wo - err_full) / total)
    return np.asarray(vals, dtype=np.float32)


# =============================================================================
# Analysis with Ca channel
# =============================================================================



# Convert MotionRegion / MotionMode to spatial mask
def motion_region_to_mask(region):
    """
    Use MotionRegion.region_mask.
    """
    m = getattr(region, "region_mask", None)
    if m is None:
        return None
    return _as_bool_mask(m)


def motion_mode_to_mask(
    mode,
    support_rel_thresh=0.05,
):
    """
    Build support mask from MotionMode.response_strength.
    """
    A = getattr(mode, "response_strength", None)
    if A is None:
        return None

    A = np.asarray(A, dtype=np.float32)
    if A.ndim != 2:
        return None

    vmax = float(np.nanmax(A))
    if not np.isfinite(vmax) or vmax <= 0:
        return None

    return A > support_rel_thresh * vmax


def unit_to_mask(
    unit,
    unit_type="region",
    mode_support_rel_thresh=0.05,
):
    if unit_type == "region":
        return motion_region_to_mask(unit)
    elif unit_type == "mode":
        return motion_mode_to_mask(
            unit,
            support_rel_thresh=mode_support_rel_thresh,
        )
    else:
        raise ValueError(f"Unknown unit_type: {unit_type}")


# Get unit activation and absolute frame range
def _infer_unit_time_indices(unit, n_frames=None):
    """
    Infer absolute frame indices for a unit from unit.time_range and unit.activation.

    Supports both:
        time_range = (start, end_exclusive)
    and:
        time_range = (start, end_inclusive)

    Returns
    -------
    frames : ndarray, shape (len(h),)
    """
    h = getattr(unit, "activation", None)
    if h is None:
        return None

    h = np.asarray(h, dtype=np.float32).reshape(-1)
    L = len(h)

    tr = getattr(unit, "time_range", None)
    if tr is None:
        return None

    start = int(tr[0])
    end = int(tr[1])

    # Case 1: [start, end) has length L
    if end - start == L:
        frames = np.arange(start, end, dtype=int)

    # Case 2: [start, end] has length L
    elif end - start + 1 == L:
        frames = np.arange(start, end + 1, dtype=int)

    else:
        # Fallback: assume start + len(h)
        frames = np.arange(start, start + L, dtype=int)

    if n_frames is not None:
        valid = (frames >= 0) & (frames < int(n_frames))
        frames = frames[valid]

    return frames

def unit_to_activation_trace(
    unit,
    n_frames,
    use_abs=True,
    normalize_unit=True,
    smooth_win=None,
):
    """
    Put a region/mode activation back onto full video time axis.

    Returns
    -------
    trace : (n_frames,)
    """
    h = getattr(unit, "activation", None)
    if h is None:
        return np.zeros(int(n_frames), dtype=np.float32)

    h = np.asarray(h, dtype=np.float32).reshape(-1)

    if use_abs:
        h = np.abs(h)

    if smooth_win is not None and smooth_win > 1:
        h = _smooth_1d(h, win=smooth_win)

    if normalize_unit:
        h = _normalize_1d(h)

    frames = _infer_unit_time_indices(unit, n_frames=n_frames)
    trace = np.zeros(int(n_frames), dtype=np.float32)

    if frames is None:
        return trace

    L = min(len(frames), len(h))
    if L <= 0:
        return trace

    frames = frames[:L]
    h = h[:L]

    valid = (frames >= 0) & (frames < int(n_frames))
    trace[frames[valid]] = h[valid]

    return trace


# ROI-overlap classification
def classify_motion_units_by_roi(
    units,
    roi_mask,
    valid_mask=None,
    unit_type="region",
    mode_support_rel_thresh=0.05,

    min_intersection=1,
    min_roi_coverage=0.01,

    local_locality_thresh=0.50,
    local_globalness_max=0.10,

    global_globalness_thresh=0.25,
    global_locality_max=0.25,

    roi_dilation_iter=0,
):
    """
    Classify motion regions/modes overlapping a user-specified ROI into:
        local / global / mixed

    Definitions:
        roi_coverage = |S ∩ R| / |R|
        locality     = |S ∩ R| / |S|
        globalness   = |S| / |valid spatial area|

    Parameters
    ----------
    units:
        list of MotionRegion or MotionMode.

    roi_mask:
        bool mask, shape (X,Y), in patch coordinates.

    valid_mask:
        optional bool mask. If None, whole FOV is used.

    unit_type:
        "region" or "mode".
    """
    roi_mask = _as_bool_mask(roi_mask)

    if roi_dilation_iter is not None and roi_dilation_iter > 0:
        roi_eval = ndi.binary_dilation(
            roi_mask,
            iterations=int(roi_dilation_iter),
        )
    else:
        roi_eval = roi_mask.copy()

    if valid_mask is None:
        valid_mask = np.ones_like(roi_eval, dtype=bool)
    else:
        valid_mask = _as_bool_mask(valid_mask)

    roi_area = int((roi_eval & valid_mask).sum())
    valid_area = int(valid_mask.sum())

    if roi_area <= 0:
        raise ValueError("ROI mask is empty after applying valid_mask.")

    if valid_area <= 0:
        raise ValueError("valid_mask is empty.")

    records = []
    grouped = {
        "local": [],
        "global": [],
        "mixed": [],
    }

    for idx, unit in enumerate(units):
        mask = unit_to_mask(
            unit,
            unit_type=unit_type,
            mode_support_rel_thresh=mode_support_rel_thresh,
        )

        if mask is None:
            continue

        mask = _as_bool_mask(mask)

        if mask.shape != roi_eval.shape:
            raise ValueError(
                f"Unit mask shape {mask.shape} != roi_mask shape {roi_eval.shape}"
            )

        mask = mask & valid_mask

        support_area = int(mask.sum())
        if support_area <= 0:
            continue

        inter = int((mask & roi_eval).sum())
        if inter < min_intersection:
            continue

        roi_coverage = inter / max(roi_area, 1)
        locality = inter / max(support_area, 1)
        globalness = support_area / max(valid_area, 1)

        if roi_coverage < min_roi_coverage:
            continue

        # Classification
        if (locality >= local_locality_thresh) and (globalness <= local_globalness_max):
            cls = "local"
        elif (globalness >= global_globalness_thresh) or (locality <= global_locality_max):
            cls = "global"
        else:
            cls = "mixed"

        strength = _safe_float(getattr(unit, "strength", 1.0), default=1.0)

        rec = {
            "index": idx,
            "unit": unit,
            "class": cls,
            "support_area": support_area,
            "intersection": inter,
            "roi_area": roi_area,
            "valid_area": valid_area,
            "roi_coverage": float(roi_coverage),
            "locality": float(locality),
            "globalness": float(globalness),
            "strength": float(strength),
            "episode_id": getattr(unit, "episode_id", None),
            "mode_id": getattr(unit, "mode_id", None),
            "region_id": getattr(unit, "region_id", None),
            "pattern_id": getattr(unit, "pattern_id", None),
            "mask": mask,
        }

        records.append(rec)
        grouped[cls].append(rec)

    return records, grouped

# Build class activation traces
def build_class_activation_traces(
    grouped_records,
    n_frames,
    weight_mode="roi_strength",
    use_abs=True,
    normalize_unit=True,
    smooth_unit_win=None,
    normalize_class=True,
):
    """
    Build one full-time activation trace for each class:
        global / local / mixed

    weight_mode:
        "uniform"
        "strength"
        "roi_coverage"
        "roi_strength" = strength * roi_coverage
        "intersection"
    """
    traces = {}

    for cls, records in grouped_records.items():
        out = np.zeros(int(n_frames), dtype=np.float32)
        wsum = 0.0

        for rec in records:
            unit = rec["unit"]

            h_full = unit_to_activation_trace(
                unit,
                n_frames=n_frames,
                use_abs=use_abs,
                normalize_unit=normalize_unit,
                smooth_win=smooth_unit_win,
            )

            if weight_mode == "uniform":
                w = 1.0
            elif weight_mode == "strength":
                w = rec["strength"]
            elif weight_mode == "roi_coverage":
                w = rec["roi_coverage"]
            elif weight_mode == "roi_strength":
                w = rec["strength"] * rec["roi_coverage"]
            elif weight_mode == "intersection":
                w = rec["intersection"]
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

            if not np.isfinite(w) or w <= 0:
                w = 1.0

            out += float(w) * h_full
            wsum += float(w)

        if wsum > 0:
            out = out / wsum

        if normalize_class:
            out = _normalize_1d(out)

        traces[cls] = out.astype(np.float32)

    # Make sure all keys exist
    for cls in ["global", "local", "mixed"]:
        traces.setdefault(cls, np.zeros(int(n_frames), dtype=np.float32))

    return traces

# Event detection from activation trace
def detect_activation_events_mad(
    trace,
    mad_k=3.0,
    min_len=1,
    merge_gap=1,
    eps=1e-12,
):
    """
    Detect active periods from a 1D activation trace using MAD threshold.

    Returns list of dicts:
        start, end, peak_frame, peak_value, duration
    """
    x = np.asarray(trace, dtype=np.float32).reshape(-1)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    sigma = 1.4826 * mad

    # Fallback for majority-zero / nearly constant traces.
    # Class activation traces are mostly zeros, so MAD collapses to 0 and every
    # nonzero frame would clear the threshold regardless of mad_k.
    if not (sigma > eps):
        sigma = np.nanstd(x)

    sigma = sigma + eps
    thresh = med + mad_k * sigma

    active = x > thresh

    if merge_gap is not None and merge_gap > 0:
        structure = np.ones(int(merge_gap) + 1, dtype=bool)
        active = ndi.binary_closing(active, structure=structure)

    labeled, num = ndi.label(active)

    events = []
    for rid in range(1, num + 1):
        idx = np.where(labeled == rid)[0]
        if len(idx) < min_len:
            continue

        vals = x[idx]
        peak_local = int(np.argmax(vals))
        peak_frame = int(idx[peak_local])

        events.append({
            "start": int(idx[0]),
            "end": int(idx[-1]),
            "peak_frame": peak_frame,
            "peak_value": float(x[peak_frame]),
            "duration": int(len(idx)),
            "threshold": float(thresh),
        })

    return events

# Ca lag correlation
def compute_lagged_ca_correlation_map(
    activation_trace,
    ca_patch_stack,
    max_lag=10,
    lag_mode="positive",
    use_dff=True,
    ca_smooth_win=None,
    min_std=1e-6,
):
    """
    Compute lagged correlation between one activation trace A(t)
    and Ca at each spatial location C(t, ...).

    Supports:
        ca_patch_stack: (T, X, Y)
        ca_patch_stack: (T, Z, X, Y)
        or generally: (T, *spatial_shape)

    lag definition:
        corr(A(t), C(t + lag, ...))

    Therefore:
        lag < 0: Ca precedes motion activation
        lag = 0: Ca is synchronous with motion activation
        lag > 0: Ca follows motion activation

    lag_mode:
        "positive": choose lag with maximum correlation
        "absolute": choose lag with maximum absolute correlation

    Returns
    -------
    dict:
        corr_by_lag: (n_lags, *spatial_shape)
        lags: (n_lags,)
        best_corr: (*spatial_shape,)
        best_lag: (*spatial_shape,)
        spatial_shape: tuple
    """
    A = np.asarray(activation_trace, dtype=np.float32).reshape(-1)
    ca = np.asarray(ca_patch_stack, dtype=np.float32)

    if ca.ndim < 3:
        raise ValueError(
            "ca_patch_stack should be at least 3D: "
            "(T, X, Y) or (T, Z, X, Y). "
            f"Got shape {ca.shape}"
        )

    T = ca.shape[0]
    spatial_shape = ca.shape[1:]

    if len(A) != T:
        raise ValueError(
            f"activation_trace length {len(A)} != ca_patch_stack time length {T}"
        )

    if use_dff:
        ca = _compute_dff_stack(ca)

    # Optional temporal smoothing for each spatial trace.
    # Works for both 2D and 3D Ca, because we flatten space.
    ca_flat = ca.reshape(T, -1).astype(np.float32)

    if ca_smooth_win is not None and ca_smooth_win > 1:
        ca_smooth = np.zeros_like(ca_flat, dtype=np.float32)
        for j in range(ca_flat.shape[1]):
            ca_smooth[:, j] = _smooth_1d(ca_flat[:, j], win=ca_smooth_win)
        ca_flat = ca_smooth

    lags = np.arange(-int(max_lag), int(max_lag) + 1, dtype=int)

    n_lags = len(lags)
    n_sites = ca_flat.shape[1]

    corr_by_lag_flat = np.full(
        (n_lags, n_sites),
        np.nan,
        dtype=np.float32,
    )

    for li, lag in enumerate(lags):
        if lag >= 0:
            a_seg = A[:T - lag]
            c_seg = ca_flat[lag:T, :]
        else:
            a_seg = A[-lag:T]
            c_seg = ca_flat[:T + lag, :]

        if len(a_seg) < 3:
            continue

        a_z = _zscore_1d(a_seg)

        if np.nanstd(a_z) < min_std:
            continue

        C_z = _zscore_time_matrix(c_seg)

        corr = np.nanmean(C_z * a_z[:, None], axis=0)
        corr_by_lag_flat[li, :] = corr.astype(np.float32)

    corr_by_lag = corr_by_lag_flat.reshape(
        (n_lags,) + tuple(spatial_shape)
    )

    # ------------------------------------------------------------
    # Choose best lag for each spatial location
    # ------------------------------------------------------------
    if lag_mode == "positive":
        score_flat = corr_by_lag_flat.copy()
    elif lag_mode == "absolute":
        score_flat = np.abs(corr_by_lag_flat)
    else:
        raise ValueError(f"Unknown lag_mode: {lag_mode}")

    finite_any = np.any(np.isfinite(score_flat), axis=0)

    score_safe = np.where(np.isfinite(score_flat), score_flat, -np.inf)

    best_idx_flat = np.zeros(n_sites, dtype=np.int32)
    best_idx_flat[finite_any] = np.argmax(score_safe[:, finite_any], axis=0)

    best_corr_flat = np.full(n_sites, np.nan, dtype=np.float32)
    best_lag_flat = np.full(n_sites, 0, dtype=np.int32)

    idx_sites = np.arange(n_sites)

    best_corr_flat[finite_any] = corr_by_lag_flat[
        best_idx_flat[finite_any],
        idx_sites[finite_any],
    ]

    best_lag_flat[finite_any] = lags[best_idx_flat[finite_any]]

    best_corr = best_corr_flat.reshape(spatial_shape).astype(np.float32)
    best_lag = best_lag_flat.reshape(spatial_shape).astype(np.int32)

    return {
        "corr_by_lag": corr_by_lag.astype(np.float32),
        "lags": lags,
        "best_corr": best_corr,
        "best_lag": best_lag,
        "spatial_shape": tuple(spatial_shape),
        "ca_ndim": ca.ndim,
    }


def get_top_ca_sites_from_corr_map(
    corr_map,
    lag_map=None,
    top_n=20,
    min_corr=None,
):
    """
    Get top Ca sites from 2D or 3D correlation map.

    Supports:
        corr_map: (X, Y)
        corr_map: (Z, X, Y)
        or generally: (*spatial_shape,)

    Returns
    -------
    rows:
        list of dicts with:
            coord: tuple
            corr: float
            lag: int, optional
            plus convenience keys for 2D/3D
    """
    corr = np.asarray(corr_map, dtype=np.float32)
    flat = corr.reshape(-1)

    valid = np.isfinite(flat)

    if min_corr is not None:
        valid &= flat >= float(min_corr)

    idx_all = np.where(valid)[0]

    if len(idx_all) == 0:
        return []

    order = idx_all[np.argsort(flat[idx_all])[::-1]]
    order = order[:int(top_n)]

    rows = []

    for idx in order:
        coord = np.unravel_index(int(idx), corr.shape)
        coord = tuple(int(c) for c in coord)

        row = {
            "coord": coord,
            "corr": float(corr[coord]),
        }

        # Backward-compatible convenience fields
        if corr.ndim == 2:
            row["x"] = coord[0]
            row["y"] = coord[1]
        elif corr.ndim == 3:
            row["z"] = coord[0]
            row["x"] = coord[1]
            row["y"] = coord[2]

        if lag_map is not None:
            lag = np.asarray(lag_map)
            row["lag"] = int(lag[coord])

        rows.append(row)

    return rows

import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def _as_bool_mask_fallback(mask):
    if "_as_bool_mask" in globals():
        return _as_bool_mask(mask)
    return np.asarray(mask).astype(bool)


def _extract_unit_from_record(record, units=None):
    """
    Make this compatible with different possible formats returned by
    classify_motion_units_by_roi.

    Expected common formats:
        record["unit"]
        record["region"]
        record["motion_region"]
        record["unit_index"]
        record["idx"]
        or record itself is the unit.
    """
    if isinstance(record, dict):
        for k in ["unit", "region", "motion_region", "motion_unit", "obj"]:
            if k in record:
                return record[k]

        if units is not None:
            for k in ["unit_index", "idx", "index", "id"]:
                if k in record:
                    idx = int(record[k])
                    if 0 <= idx < len(units):
                        return units[idx]

    return record


def _interval_from_motion_region_scheme_a(region, n_frames):
    """
    Motion region binarization scheme A:
    Do not use activation magnitude.
    Only use region.time_range.

    Returns half-open interval [start, end).
    """
    tr = getattr(region, "time_range", None)
    h = getattr(region, "activation", None)

    if tr is None:
        return None

    start = int(tr[0])

    if h is not None:
        end = start + len(np.asarray(h).reshape(-1))
    elif len(tr) >= 2:
        second = int(tr[1])

        # Compatible with two possible conventions:
        #   time_range = [start, end]
        #   time_range = [start, duration]
        if second > start:
            end = second
        else:
            end = start + max(1, second)
    else:
        end = start + 1

    start = max(0, min(int(n_frames), start))
    end = max(0, min(int(n_frames), end))

    if end <= start:
        return None

    return (start, end)


def _merge_intervals(intervals, merge_gap=1):
    """
    Merge overlapping or nearby intervals.

    intervals:
        list of (start, end), half-open [start, end)
    """
    if len(intervals) == 0:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]

    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]

        if s <= last_e + int(merge_gap):
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])

    return [tuple(x) for x in merged]


def _make_lags(max_lag=10, lag_mode="positive"):
    max_lag = int(max_lag)

    if lag_mode == "positive":
        return np.arange(0, max_lag + 1, dtype=np.int32)
    elif lag_mode == "negative":
        return np.arange(-max_lag, 1, dtype=np.int32)
    elif lag_mode in ["both", "symmetric"]:
        return np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    else:
        raise ValueError("lag_mode should be 'positive', 'negative', or 'both'.")


def _binomial_tail_pvalue(k, n, p):
    """
    Right-tail p-value:
        P(X >= k), X ~ Binomial(n, p)

    This is a fast AQuA2-like approximate p-value.
    For final statistical claims, circular-shift permutation is still safer.
    """
    if n <= 0:
        return np.nan

    k = int(k)
    n = int(n)
    p = float(np.clip(p, 0.0, 1.0))

    if k <= 0:
        return 1.0

    try:
        from scipy.stats import binom
        return float(binom.sf(k - 1, n, p))
    except Exception:
        from math import comb

        val = 0.0
        for i in range(k, n + 1):
            val += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        return float(np.clip(val, 0.0, 1.0))


# ============================================================
# Ca active-state construction
# ============================================================

def _prepare_ca_active_chunk_fast(
    ca_chunk,
    use_dff=True,
    baseline_percentile=20,
    smooth_win=3,
    mad_k=3.0,
    min_len=2,
    merge_gap=1,
    eps=1e-6,
):
    """
    ca_chunk:
        shape (T, N_chunk)

    Returns:
        active:
            bool array, shape (T, N_chunk)
    """
    x = np.asarray(ca_chunk, dtype=np.float32)

    if use_dff:
        f0 = np.nanpercentile(x, baseline_percentile, axis=0).astype(np.float32)
        x = (x - f0[None, :]) / (np.abs(f0[None, :]) + eps)

    if smooth_win is not None and smooth_win > 1:
        try:
            from scipy.ndimage import uniform_filter1d
            x = uniform_filter1d(
                x,
                size=int(smooth_win),
                axis=0,
                mode="nearest",
            )
        except Exception:
            kernel = np.ones(int(smooth_win), dtype=np.float32) / int(smooth_win)
            for i in range(x.shape[1]):
                x[:, i] = np.convolve(x[:, i], kernel, mode="same")

    med = np.nanmedian(x, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(x - med[None, :]), axis=0).astype(np.float32)

    robust_std = 1.4826 * mad

    # Fallback for nearly constant traces.
    std = np.nanstd(x, axis=0).astype(np.float32)
    robust_std = np.where(robust_std > eps, robust_std, std)

    threshold = med + float(mad_k) * robust_std
    active = x > threshold[None, :]

    # Make Ca activation more continuous.
    # This is much faster than looping over every Ca site.
    try:
        from scipy.ndimage import binary_closing, binary_opening

        if merge_gap is not None and merge_gap > 0:
            # Fill short inactive gaps.
            structure = np.ones((int(merge_gap) + 2, 1), dtype=bool)
            active = binary_closing(active, structure=structure)

        if min_len is not None and min_len > 1:
            # Remove very short active bursts.
            structure = np.ones((int(min_len), 1), dtype=bool)
            active = binary_opening(active, structure=structure)

    except Exception:
        # If scipy is unavailable, return thresholded active state directly.
        pass

    return active.astype(bool)


def _baseline_window_prob_from_cumsum(
    csum,
    window_len,
    T,
    time_chunk_size=512,
):
    """
    Probability that a random window of length window_len contains
    at least one Ca-active frame.

    csum:
        shape (T + 1, N_chunk)
    """
    L = int(window_len)
    L = max(1, min(L, T))

    n_windows = T - L + 1
    N = csum.shape[1]

    if n_windows <= 0:
        return np.zeros(N, dtype=np.float32)

    hit_total = np.zeros(N, dtype=np.int64)

    for s0 in range(0, n_windows, int(time_chunk_size)):
        s1 = min(n_windows, s0 + int(time_chunk_size))
        starts = np.arange(s0, s1, dtype=np.int32)
        ends = starts + L

        count = csum[ends, :] - csum[starts, :]
        hit_total += np.sum(count > 0, axis=0)

    return (hit_total.astype(np.float32) / float(n_windows)).astype(np.float32)


def _event_window_hit_rate_from_cumsum(
    csum,
    starts,
    window_len,
    T,
):
    """
    For each Ca site, compute how many event windows contain at least
    one Ca-active frame.

    starts:
        event window start frames, shape (n_events,)
    """
    N = csum.shape[1]
    L = int(window_len)

    valid = []
    for s in starts:
        s = int(s)
        e = s + L

        if e <= 0 or s >= T:
            continue

        s = max(0, s)
        e = min(T, e)

        if e > s:
            valid.append((s, e))

    if len(valid) == 0:
        return (
            np.zeros(N, dtype=np.float32),
            np.zeros(N, dtype=np.int32),
            0,
        )

    hit_count = np.zeros(N, dtype=np.int32)

    for s, e in valid:
        has_ca = (csum[e, :] - csum[s, :]) > 0
        hit_count += has_ca.astype(np.int32)

    n_valid = len(valid)
    hit_rate = hit_count.astype(np.float32) / float(n_valid)

    return hit_rate, hit_count, n_valid


# ============================================================
# Class-level motion event construction
# ============================================================

def _build_local_mixed_intervals_from_grouped(
    grouped,
    regions,
    n_frames,
    classes=("local", "mixed"),
    merge_gap=1,
    merge_motion_events=True,
):
    """
    Build one class-level event sequence for local and mixed.

    For example:
        all local region time_ranges
        -> merge overlapping intervals
        -> local class motion event sequence
    """
    class_intervals = {}

    for cls in classes:
        intervals = []

        for rec in grouped.get(cls, []):
            unit = _extract_unit_from_record(rec, units=regions)
            interval = _interval_from_motion_region_scheme_a(unit, n_frames)

            if interval is not None:
                intervals.append(interval)

        if merge_motion_events:
            class_intervals[cls] = _merge_intervals(
                intervals,
                merge_gap=merge_gap,
            )
        else:
            class_intervals[cls] = intervals

    return class_intervals


# ============================================================
# Main function
# ============================================================
def _binomial_tail_pvalue_array(hit_count, n_windows, baseline_prob):
    """
    Vectorized right-tail p-value:
        P(X >= hit_count), X ~ Binomial(n_windows, baseline_prob)

    hit_count:     array
    n_windows:     array or scalar
    baseline_prob: array
    """
    hit_count = np.asarray(hit_count)
    n_windows = np.asarray(n_windows)
    baseline_prob = np.asarray(baseline_prob, dtype=np.float64)

    pvals = np.full(hit_count.shape, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(baseline_prob)
        & (n_windows > 0)
        & (hit_count >= 0)
        & (baseline_prob >= 0)
        & (baseline_prob <= 1)
    )

    if not np.any(valid):
        return pvals

    try:
        from scipy.stats import binom

        pvals[valid] = binom.sf(
            hit_count[valid].astype(np.int64) - 1,
            n_windows[valid].astype(np.int64),
            baseline_prob[valid],
        ).astype(np.float32)

    except Exception:
        # Fallback: slower, but works.
        for idx in np.where(valid)[0]:
            pvals[idx] = _binomial_tail_pvalue(
                k=int(hit_count[idx]),
                n=int(n_windows[idx]),
                p=float(baseline_prob[idx]),
            )

    return pvals

def analyze_roi_local_mixed_ca_dependency(
    roi_mask,
    regions,
    ca_patch_stack,

    # Time
    n_frames=None,

    # Optional spatial masks
    valid_mask=None,
    ca_valid_mask=None,

    # ROI classification parameters
    roi_dilation_iter=0,
    min_intersection=1,
    min_roi_coverage=0.01,

    local_locality_thresh=0.50,
    local_globalness_max=0.10,
    global_globalness_thresh=0.25,
    global_locality_max=0.25,

    # Class event construction
    motion_merge_gap=1,
    merge_motion_events=True,
    # Ca active-state construction
    ca_use_dff=True,
    ca_baseline_percentile=20,
    ca_smooth_win=3,
    ca_mad_k=3.0,
    ca_min_len=2,
    ca_merge_gap=1,

    # Dependency test
    max_lag=10,
    lag_mode="positive",
    window_len=3,

    # Ranking
    rank_by="dependency_score",
    # "dependency_score": hit_rate - baseline_prob
    # "hit_rate": raw probability of Ca activation after motion events
    # "enrichment": hit_rate / baseline_prob

    top_n_ca=20,
    min_rank_score=None,

    # Speed / memory
    chunk_size=20000,
    store_maps=True,
    compute_top_p=True,

    compute_p_map=True,
    p_map_eps=1e-300,

    verbose=True,

):
    """
    ROI-guided local/mixed motion-Ca event dependency analysis.

    Inputs
    ------
    roi_mask:
        Spatial ROI mask used to classify motion regions.

    regions:
        Existing motion regions.

    ca_patch_stack:
        Shape can be:
            (T, H, W)
            (T, Z, H, W)

    Main outputs
    ------------
    result["results"]["local"]["top_sites"]
    result["results"]["mixed"]["top_sites"]

    Each top site contains:
        coord
        probability / hit_rate
        baseline_prob
        dependency_score
        enrichment
        lag
        optional p_binom
    """
    roi_mask = _as_bool_mask_fallback(roi_mask)
    regions = list(regions)

    ca = np.asarray(ca_patch_stack, dtype=np.float32)

    if ca.ndim < 3:
        raise ValueError("ca_patch_stack should have shape (T,H,W) or (T,Z,H,W).")

    if n_frames is None:
        n_frames = int(ca.shape[0])
    else:
        n_frames = int(n_frames)

    if ca.shape[0] != n_frames:
        raise ValueError(
            f"ca_patch_stack time length {ca.shape[0]} != n_frames {n_frames}"
        )

    T = int(n_frames)
    spatial_shape = ca.shape[1:]
    N = int(np.prod(spatial_shape))
    ca_flat = ca.reshape(T, N)

    # ------------------------------------------------------------
    # 1. Classify motion regions by ROI
    # ------------------------------------------------------------
    records, grouped = classify_motion_units_by_roi(
        units=regions,
        roi_mask=roi_mask,
        valid_mask=valid_mask,
        unit_type="region",

        min_intersection=min_intersection,
        min_roi_coverage=min_roi_coverage,

        local_locality_thresh=local_locality_thresh,
        local_globalness_max=local_globalness_max,

        global_globalness_thresh=global_globalness_thresh,
        global_locality_max=global_locality_max,

        roi_dilation_iter=roi_dilation_iter,
    )

    # ------------------------------------------------------------
    # 2. Build class-level event intervals
    # ------------------------------------------------------------
    classes = ("local", "mixed")

    class_intervals = _build_local_mixed_intervals_from_grouped(
        grouped=grouped,
        regions=regions,
        n_frames=T,
        classes=classes,
        merge_gap=motion_merge_gap,
        merge_motion_events=merge_motion_events
    )

    class_onsets = {
        cls: np.asarray([s for s, e in class_intervals[cls]], dtype=np.int32)
        for cls in classes
    }

    lags = _make_lags(max_lag=max_lag, lag_mode=lag_mode)

    # ------------------------------------------------------------
    # 3. Valid Ca sites
    # ------------------------------------------------------------
    if ca_valid_mask is None:
        valid_ca_flat = np.ones(N, dtype=bool)
    else:
        valid_ca_flat = np.asarray(ca_valid_mask).astype(bool).reshape(-1)
        if valid_ca_flat.size != N:
            raise ValueError(
                f"ca_valid_mask shape does not match ca spatial shape {spatial_shape}"
            )

    valid_indices = np.where(valid_ca_flat)[0]

    # ------------------------------------------------------------
    # 4. Allocate outputs
    # ------------------------------------------------------------
    results = {}

    for cls in classes:
        results[cls] = {
            "n_regions": len(grouped.get(cls, [])),
            "n_events": len(class_onsets[cls]),
            "intervals": class_intervals[cls],
            "onsets": class_onsets[cls],

            "best_rank_score": np.full(N, -np.inf, dtype=np.float32),
            "best_probability": np.full(N, np.nan, dtype=np.float32),
            "best_hit_rate": np.full(N, np.nan, dtype=np.float32),
            "best_baseline_prob": np.full(N, np.nan, dtype=np.float32),
            "best_dependency_score": np.full(N, np.nan, dtype=np.float32),
            "best_enrichment": np.full(N, np.nan, dtype=np.float32),
            "best_lag": np.full(N, 0, dtype=np.int32),

            # Used only for top-site p-value.
            "best_hit_count": np.zeros(N, dtype=np.int32),
            "best_n_windows": np.zeros(N, dtype=np.int32),
        }

    # ------------------------------------------------------------
    # 5. Chunked Ca processing
    # ------------------------------------------------------------
    for c0 in range(0, len(valid_indices), int(chunk_size)):
        c1 = min(len(valid_indices), c0 + int(chunk_size))
        idx_chunk = valid_indices[c0:c1]

        if verbose:
            print(f"[Ca valid chunk] {c0}:{c1} / {len(valid_indices)}")

        ca_chunk = ca_flat[:, idx_chunk]

        ca_active = _prepare_ca_active_chunk_fast(
            ca_chunk,
            use_dff=ca_use_dff,
            baseline_percentile=ca_baseline_percentile,
            smooth_win=ca_smooth_win,
            mad_k=ca_mad_k,
            min_len=ca_min_len,
            merge_gap=ca_merge_gap,
        )

        csum = np.concatenate(
            [
                np.zeros((1, ca_active.shape[1]), dtype=np.int32),
                np.cumsum(ca_active.astype(np.int32), axis=0),
            ],
            axis=0,
        )

        baseline_prob = _baseline_window_prob_from_cumsum(
            csum=csum,
            window_len=window_len,
            T=T,
        )

        baseline_safe = np.maximum(baseline_prob, 1e-6)

        for cls in classes:
            onsets = class_onsets[cls]

            if len(onsets) == 0:
                continue

            for lag in lags:
                starts = onsets + int(lag)

                hit_rate, hit_count, n_valid = _event_window_hit_rate_from_cumsum(
                    csum=csum,
                    starts=starts,
                    window_len=window_len,
                    T=T,
                )

                if n_valid == 0:
                    continue

                dependency_score = hit_rate - baseline_prob
                enrichment = hit_rate / baseline_safe

                if rank_by == "dependency_score":
                    rank_score = dependency_score
                elif rank_by == "hit_rate":
                    rank_score = hit_rate
                elif rank_by == "enrichment":
                    rank_score = enrichment
                else:
                    raise ValueError(
                        "rank_by should be 'dependency_score', 'hit_rate', or 'enrichment'."
                    )

                old_score = results[cls]["best_rank_score"][idx_chunk]
                update = rank_score > old_score

                if not np.any(update):
                    continue

                idx_update = idx_chunk[update]

                results[cls]["best_rank_score"][idx_update] = rank_score[update]
                results[cls]["best_probability"][idx_update] = hit_rate[update]
                results[cls]["best_hit_rate"][idx_update] = hit_rate[update]
                results[cls]["best_baseline_prob"][idx_update] = baseline_prob[update]
                results[cls]["best_dependency_score"][idx_update] = dependency_score[update]
                results[cls]["best_enrichment"][idx_update] = enrichment[update]
                results[cls]["best_lag"][idx_update] = int(lag)
                results[cls]["best_hit_count"][idx_update] = hit_count[update]
                results[cls]["best_n_windows"][idx_update] = int(n_valid)

    # ------------------------------------------------------------
    # 6. Extract top sites
    # ------------------------------------------------------------
    final_results = {}

    for cls in classes:
        r = results[cls]

        score_flat = r["best_rank_score"]
        valid = np.isfinite(score_flat) & valid_ca_flat

        if min_rank_score is not None:
            valid &= score_flat >= float(min_rank_score)

        candidate_idx = np.where(valid)[0]

        if candidate_idx.size > 0:
            order = np.argsort(score_flat[candidate_idx])[::-1]
            order = order[:int(top_n_ca)]
            top_idx = candidate_idx[order]
        else:
            top_idx = np.array([], dtype=np.int64)

        top_sites = []

        for flat_i in top_idx:
            coord = tuple(int(v) for v in np.unravel_index(int(flat_i), spatial_shape))

            item = {
                "coord": coord,

                # The most direct quantity requested by you:
                # probability that Ca appears after this class of motion events.
                "probability": float(r["best_probability"][flat_i]),
                "hit_rate": float(r["best_hit_rate"][flat_i]),

                # Baseline and dependency metrics.
                "baseline_prob": float(r["best_baseline_prob"][flat_i]),
                "dependency_score": float(r["best_dependency_score"][flat_i]),
                "enrichment": float(r["best_enrichment"][flat_i]),

                # Best temporal lag.
                "lag": int(r["best_lag"][flat_i]),

                # Class-level counts.
                "n_regions": int(r["n_regions"]),
                "n_events": int(r["n_events"]),
                "hit_count": int(r["best_hit_count"][flat_i]),
                "n_windows": int(r["best_n_windows"][flat_i]),

                # Ranking score used internally.
                "rank_score": float(r["best_rank_score"][flat_i]),
            }

            if compute_top_p:
                item["p_binom"] = _binomial_tail_pvalue(
                    k=item["hit_count"],
                    n=item["n_windows"],
                    p=item["baseline_prob"],
                )

            top_sites.append(item)

        out = {
            "n_regions": int(r["n_regions"]),
            "n_events": int(r["n_events"]),
            "intervals": r["intervals"],
            "onsets": r["onsets"],
            "top_sites": top_sites,
        }

        if store_maps:
            out["best_probability"] = r["best_probability"].reshape(spatial_shape)
            out["best_hit_rate"] = r["best_hit_rate"].reshape(spatial_shape)
            out["best_baseline_prob"] = r["best_baseline_prob"].reshape(spatial_shape)
            out["best_dependency_score"] = r["best_dependency_score"].reshape(spatial_shape)
            out["best_enrichment"] = r["best_enrichment"].reshape(spatial_shape)
            out["best_lag"] = r["best_lag"].reshape(spatial_shape)
            out["best_rank_score"] = r["best_rank_score"].reshape(spatial_shape)

            # Optional diagnostic maps.
            out["best_hit_count"] = r["best_hit_count"].reshape(spatial_shape)
            out["best_n_windows"] = r["best_n_windows"].reshape(spatial_shape)

            if compute_p_map:
                p_flat = _binomial_tail_pvalue_array(
                    hit_count=r["best_hit_count"],
                    n_windows=r["best_n_windows"],
                    baseline_prob=r["best_baseline_prob"],
                )

                neglog10_p_flat = -np.log10(np.maximum(p_flat, p_map_eps))

                out["best_p_binom"] = p_flat.reshape(spatial_shape)
                out["best_neglog10_p"] = neglog10_p_flat.reshape(spatial_shape)

        final_results[cls] = out

        if verbose:
            print(f"\n[{cls}]")
            print(f"regions: {out['n_regions']}")
            print(f"merged events: {out['n_events']}")

            if len(top_sites) > 0:
                s0 = top_sites[0]
                print(
                    f"top coord={s0['coord']}, "
                    f"prob={s0['probability']:.3f}, "
                    f"baseline={s0['baseline_prob']:.3f}, "
                    f"dep={s0['dependency_score']:.3f}, "
                    f"enrich={s0['enrichment']:.2f}, "
                    f"lag={s0['lag']}"
                )
            else:
                print("no top Ca sites")

    result = {
        "roi_mask": roi_mask,
        "valid_mask": valid_mask,
        "ca_valid_mask": ca_valid_mask,

        "records": records,
        "grouped": grouped,

        "class_intervals": class_intervals,
        "results": final_results,

        "summary": {
            "n_regions_total": len(regions),
            "n_regions_matched": len(records),
            "n_local": len(grouped.get("local", [])),
            "n_mixed": len(grouped.get("mixed", [])),
            "n_global_ignored": len(grouped.get("global", [])),
            "n_frames": int(T),
            "ca_spatial_shape": spatial_shape,
            "n_valid_ca_sites": int(len(valid_indices)),
        },

        "params": {
            "motion_merge_gap": motion_merge_gap,
            "ca_use_dff": ca_use_dff,
            "ca_baseline_percentile": ca_baseline_percentile,
            "ca_smooth_win": ca_smooth_win,
            "ca_mad_k": ca_mad_k,
            "ca_min_len": ca_min_len,
            "ca_merge_gap": ca_merge_gap,
            "max_lag": max_lag,
            "lag_mode": lag_mode,
            "window_len": window_len,
            "rank_by": rank_by,
            "top_n_ca": top_n_ca,
            "compute_p_map": compute_p_map,
            "p_map_eps": p_map_eps,
        },
    }

    return result



# Main ROI-guided function
def analyze_roi_motion_activation_ca(
    roi_mask,

    # Existing motion decomposition results
    motion_regions=None,
    motion_modes=None,

    # Ca
    ca_patch_stack=None,

    # Unit choice
    unit_type="region",  # "region" or "mode"

    # Time length
    n_frames=None,

    # Spatial masks
    valid_mask=None,
    roi_dilation_iter=0,

    # Mode mask construction
    mode_support_rel_thresh=0.05,

    # ROI overlap filtering
    min_intersection=1,
    min_roi_coverage=0.01,

    # Classification thresholds
    local_locality_thresh=0.50,
    local_globalness_max=0.10,
    global_globalness_thresh=0.25,
    global_locality_max=0.25,

    # Activation trace construction
    weight_mode="roi_strength",
    use_abs_activation=True,
    normalize_unit_activation=True,
    smooth_unit_win=None,
    normalize_class_activation=True,

    # Event detection
    detect_events=True,
    event_mad_k=3.0,
    event_min_len=1,
    event_merge_gap=1,

    # Ca correlation
    max_lag=10,
    lag_mode="positive",
    ca_use_dff=True,
    ca_smooth_win=None,
    top_n_ca=20,
    min_top_corr=None,

    verbose=True,
):
    """
    ROI-guided motion activation and Ca coupling analysis.

    This function uses existing MotionRegion or MotionMode results.

    It does NOT require semantic segmentation.

    Main outputs:
        - ROI-overlapping units
        - global/local/mixed classification
        - class activation traces A_global/A_local/A_mixed
        - optional Ca lag-correlation maps for each class
    """
    roi_mask = _as_bool_mask(roi_mask)

    if unit_type == "region":
        if motion_regions is None:
            raise ValueError("motion_regions is required when unit_type='region'.")
        units = list(motion_regions)
    elif unit_type == "mode":
        if motion_modes is None:
            raise ValueError("motion_modes is required when unit_type='mode'.")
        units = list(motion_modes)
    else:
        raise ValueError("unit_type should be 'region' or 'mode'.")

    if n_frames is None:
        if ca_patch_stack is not None:
            n_frames = int(np.asarray(ca_patch_stack).shape[0])
        else:
            # Infer from unit time ranges
            max_frame = 0
            for u in units:
                h = getattr(u, "activation", None)
                tr = getattr(u, "time_range", None)
                if h is None or tr is None:
                    continue
                h = np.asarray(h).reshape(-1)
                start = int(tr[0])
                max_frame = max(max_frame, start + len(h))
            n_frames = max_frame

    if n_frames is None or n_frames <= 0:
        raise ValueError("Could not infer n_frames. Please provide n_frames explicitly.")

    # ------------------------------------------------------------
    # 1. Classify ROI-overlapping units
    # ------------------------------------------------------------
    records, grouped = classify_motion_units_by_roi(
        units=units,
        roi_mask=roi_mask,
        valid_mask=valid_mask,
        unit_type=unit_type,
        mode_support_rel_thresh=mode_support_rel_thresh,

        min_intersection=min_intersection,
        min_roi_coverage=min_roi_coverage,

        local_locality_thresh=local_locality_thresh,
        local_globalness_max=local_globalness_max,

        global_globalness_thresh=global_globalness_thresh,
        global_locality_max=global_locality_max,

        roi_dilation_iter=roi_dilation_iter,
    )

    # ------------------------------------------------------------
    # 2. Build class activation traces
    # ------------------------------------------------------------
    class_traces = build_class_activation_traces(
        grouped_records=grouped,
        n_frames=n_frames,
        weight_mode=weight_mode,
        use_abs=use_abs_activation,
        normalize_unit=normalize_unit_activation,
        smooth_unit_win=smooth_unit_win,
        normalize_class=normalize_class_activation,
    )

    # ------------------------------------------------------------
    # 3. Detect class events
    # ------------------------------------------------------------
    class_events = {}

    if detect_events:
        for cls, trace in class_traces.items():
            class_events[cls] = detect_activation_events_mad(
                trace,
                mad_k=event_mad_k,
                min_len=event_min_len,
                merge_gap=event_merge_gap,
            )
    else:
        for cls in ["global", "local", "mixed"]:
            class_events[cls] = []

    # ------------------------------------------------------------
    # 4. Ca lag-correlation maps
    # ------------------------------------------------------------
    ca_results = {}

    if ca_patch_stack is not None:
        ca_patch_stack = np.asarray(ca_patch_stack, dtype=np.float32)

        if ca_patch_stack.shape[0] != n_frames:
            raise ValueError(
                f"ca_patch_stack time length {ca_patch_stack.shape[0]} != n_frames {n_frames}"
            )

        for cls, trace in class_traces.items():
            if np.nanstd(trace) < 1e-8:
                ca_results[cls] = {
                    "corr_by_lag": None,
                    "lags": None,
                    "best_corr": None,
                    "best_lag": None,
                    "top_sites": [],
                    "note": "activation trace is nearly constant",
                }
                continue

            corr_res = compute_lagged_ca_correlation_map(
                activation_trace=trace,
                ca_patch_stack=ca_patch_stack,
                max_lag=max_lag,
                lag_mode=lag_mode,
                use_dff=ca_use_dff,
                ca_smooth_win=ca_smooth_win,
            )

            top_sites = get_top_ca_sites_from_corr_map(
                corr_res["best_corr"],
                lag_map=corr_res["best_lag"],
                top_n=top_n_ca,
                min_corr=min_top_corr,
            )

            corr_res["top_sites"] = top_sites
            ca_results[cls] = corr_res

    # ------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------
    summary = {
        "n_units_total": len(units),
        "n_units_matched": len(records),
        "n_global": len(grouped["global"]),
        "n_local": len(grouped["local"]),
        "n_mixed": len(grouped["mixed"]),
        "unit_type": unit_type,
        "n_frames": int(n_frames),
    }

    if verbose:
        print("[ROI motion activation analysis]")
        print(f"unit_type: {unit_type}")
        print(f"total units: {len(units)}")
        print(f"matched units: {len(records)}")
        print(f"global: {len(grouped['global'])}")
        print(f"local:  {len(grouped['local'])}")
        print(f"mixed:  {len(grouped['mixed'])}")

        if ca_patch_stack is not None:
            for cls in ["global", "local", "mixed"]:
                top = ca_results.get(cls, {}).get("top_sites", [])

                if len(top) > 0:
                    coord = top[0].get("coord", None)

                    print(
                        f"[Ca {cls}] top corr={top[0]['corr']:.3f}, "
                        f"coord={coord}, "
                        f"lag={top[0].get('lag', None)}"
                    )
                else:
                    print(f"[Ca {cls}] no top sites")

    result = {
        "roi_mask": roi_mask,
        "valid_mask": valid_mask,

        "records": records,
        "grouped": grouped,

        "class_traces": class_traces,
        "class_events": class_events,

        "ca_results": ca_results,

        "summary": summary,
        "params": {
            "unit_type": unit_type,
            "mode_support_rel_thresh": mode_support_rel_thresh,
            "min_intersection": min_intersection,
            "min_roi_coverage": min_roi_coverage,
            "local_locality_thresh": local_locality_thresh,
            "local_globalness_max": local_globalness_max,
            "global_globalness_thresh": global_globalness_thresh,
            "global_locality_max": global_locality_max,
            "weight_mode": weight_mode,
            "use_abs_activation": use_abs_activation,
            "normalize_unit_activation": normalize_unit_activation,
            "max_lag": max_lag,
            "lag_mode": lag_mode,
        },
    }

    return result


def _region_record_label(r):
    """Build a compact label for one region record."""
    ep = r.get("episode_id", None)
    m = r.get("mode_id", None)
    rid = r.get("region_id", None)

    if ep is not None and m is not None and rid is not None:
        return f"ep{ep}-m{m}-r{rid}"
    elif rid is not None:
        return f"r{rid}"
    else:
        return "region"


def _mask_boundary(mask, iterations=1):
    """Return boundary pixels of a binary mask."""
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    dil = ndi.binary_dilation(mask, iterations=iterations)
    ero = ndi.binary_erosion(mask, iterations=iterations)
    return dil ^ ero


def visualize_regions_of_one_class(
    result,
    cls="mixed",
    max_regions=50,
    start=0,
    sort_by="roi_coverage",
    descending=True,
    background=None,
    show_roi=True,
    show_labels=True,
    label_fontsize=7,
    alpha=0.55,
    boundary_alpha=0.95,
    figsize=(8, 8),
    cmap_name="tab20",
    print_table=True,
):
    """
    Visualize all regions from one ROI-guided class using different colors.

    Parameters
    ----------
    result:
        Output from analyze_roi_motion_activation_ca().

    cls:
        One of {"global", "local", "mixed"}.

    max_regions:
        Maximum number of regions to display in one figure.

    start:
        Start index after sorting. Useful when there are many regions.

    sort_by:
        Which metric to sort by before plotting.
        Common choices:
            "roi_coverage"
            "locality"
            "globalness"
            "strength"
            "support_area"

    background:
        Optional 2D image to show underneath.
        Should have the same shape as the region masks.
        Examples:
            np.mean(ca_patch_stack, axis=0)
            np.mean(motionMag_patched, axis=0)

    show_roi:
        Whether to draw query ROI boundary.

    show_labels:
        Whether to write region labels at region centroids.

    Returns
    -------
    shown_records:
        The records displayed in the plot.
    """
    if cls not in result["grouped"]:
        raise ValueError(f"Unknown class {cls}. Available: {list(result['grouped'].keys())}")

    records = list(result["grouped"][cls])

    if len(records) == 0:
        print(f"No regions found in class: {cls}")
        return []

    if sort_by is not None:
        records = sorted(
            records,
            key=lambda r: r.get(sort_by, 0.0),
            reverse=descending,
        )

    shown_records = records[start:start + max_regions]

    if len(shown_records) == 0:
        print(f"No regions to show for cls={cls}, start={start}, max_regions={max_regions}")
        return []

    roi_mask = np.asarray(result["roi_mask"]).astype(bool)

    # Infer spatial shape
    first_mask = np.asarray(shown_records[0]["mask"]).astype(bool)
    H, W = first_mask.shape

    if roi_mask.shape != (H, W):
        raise ValueError(f"roi_mask shape {roi_mask.shape} != region mask shape {(H, W)}")

    # ------------------------------------------------------------
    # Prepare background RGB
    # ------------------------------------------------------------
    if background is None:
        base = np.ones((H, W), dtype=np.float32)
    else:
        base = np.asarray(background, dtype=np.float32)
        if base.shape != (H, W):
            raise ValueError(f"background shape {base.shape} != region mask shape {(H, W)}")

        lo, hi = np.nanpercentile(base, [1, 99])
        base = (base - lo) / (hi - lo + 1e-12)
        base = np.clip(base, 0, 1)

    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    # ------------------------------------------------------------
    # Draw each region with a different color
    # ------------------------------------------------------------
    cmap = plt.get_cmap(cmap_name)
    n = len(shown_records)

    colors = [cmap(i % cmap.N)[:3] for i in range(n)]

    for i, (r, color) in enumerate(zip(shown_records, colors)):
        mask = np.asarray(r["mask"]).astype(bool)

        if mask.shape != (H, W):
            raise ValueError(f"Region mask shape {mask.shape} != expected {(H, W)}")

        color = np.asarray(color, dtype=np.float32)

        # Transparent fill
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

        # Stronger boundary for each region
        bd = _mask_boundary(mask, iterations=1)
        rgb[bd] = (1 - boundary_alpha) * rgb[bd] + boundary_alpha * color

    # ------------------------------------------------------------
    # Draw ROI boundary in red
    # ------------------------------------------------------------
    if show_roi:
        roi_bd = _mask_boundary(roi_mask, iterations=1)
        rgb[roi_bd] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title(
        f"{cls} regions | showing {len(shown_records)} / {len(records)} "
        f"| sorted by {sort_by}",
        fontsize=12,
    )
    ax.axis("off")

    # ------------------------------------------------------------
    # Add region labels at centroids
    # ------------------------------------------------------------
    if show_labels:
        for i, r in enumerate(shown_records):
            mask = np.asarray(r["mask"]).astype(bool)
            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            cy = float(np.mean(ys))
            cx = float(np.mean(xs))

            label = str(i)
            ax.text(
                cx,
                cy,
                label,
                color="white",
                fontsize=label_fontsize,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="black",
                    alpha=0.55,
                    edgecolor="none",
                    boxstyle="round,pad=0.15",
                ),
            )

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------
    if print_table:
        print("\nDisplayed regions:")
        print("-" * 100)
        print(
            f"{'idx':>4} | {'label':>18} | {'roi_cov':>7} | {'locality':>8} | "
            f"{'global':>7} | {'area':>6} | {'strength':>9}"
        )
        print("-" * 100)

        for i, r in enumerate(shown_records):
            label = _region_record_label(r)
            print(
                f"{i:4d} | {label:>18} | "
                f"{r.get('roi_coverage', np.nan):7.3f} | "
                f"{r.get('locality', np.nan):8.3f} | "
                f"{r.get('globalness', np.nan):7.3f} | "
                f"{r.get('support_area', np.nan):6} | "
                f"{r.get('strength', np.nan):9.3f}"
            )

    return shown_records


def _select_corr_lag_slice(
    corr_map,
    lag_map,
    z=None,
    z_select="max",
):
    """
    Convert 2D or 3D corr/lag map into one 2D slice for visualization.

    corr_map:
        2D: (H, W)
        3D: (Z, H, W)

    If 3D:
        z is not None:
            use corr_map[z]
        z is None:
            choose slice by z_select.
    """
    corr = np.asarray(corr_map, dtype=np.float32)
    lag = np.asarray(lag_map)

    if corr.shape != lag.shape:
        raise ValueError(f"corr_map shape {corr.shape} != lag_map shape {lag.shape}")

    if corr.ndim == 2:
        return corr, lag, None

    if corr.ndim != 3:
        raise ValueError(
            "corr_map should be 2D (H,W) or 3D (Z,H,W). "
            f"Got shape {corr.shape}"
        )

    Z = corr.shape[0]

    if z is None:
        if z_select == "max":
            # choose z with largest finite correlation
            z_scores = []
            for zi in range(Z):
                vals = corr[zi]
                if np.isfinite(vals).any():
                    z_scores.append(float(np.nanmax(vals)))
                else:
                    z_scores.append(-np.inf)
            z = int(np.argmax(z_scores))

        elif z_select == "mean_top1":
            z_scores = []
            for zi in range(Z):
                vals = corr[zi]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    z_scores.append(-np.inf)
                else:
                    z_scores.append(float(np.nanpercentile(vals, 99)))
            z = int(np.argmax(z_scores))

        else:
            raise ValueError(f"Unknown z_select: {z_select}")

    z = int(z)

    if z < 0 or z >= Z:
        raise ValueError(f"z={z} is outside valid range [0, {Z - 1}]")

    return corr[z], lag[z], z


def visualize_top_corr_components_with_lag(
    corr_map,
    lag_map,
    top_n=10,
    min_corr=None,
    percentile=99,
    min_area=3,
    connectivity=2,
    close_iter=1,
    dilate_iter=0,

    # 3D support
    z=None,
    z_select="max",

    transpose_display=True,
    background="corr",   # "corr", None, 2D array, or 3D array
    background_z=None,
    background_projection="slice",  # "slice", "max", "mean"
    cmap_bg="magma",
    cmap_regions="tab20",
    alpha_region=0.55,
    show_boundary=True,
    figsize=(8, 6),
    title="Top Ca regions correlated with ROI motion",
    print_table=True,
):
    """
    Detect and visualize top connected high-correlation Ca regions.

    Supports:
        corr_map: (H, W)
        lag_map:  (H, W)

    and:
        corr_map: (Z, H, W)
        lag_map:  (Z, H, W)

    For 3D maps, this function visualizes one z-slice at a time.
    If z is None, it automatically selects a representative z-slice.
    """

    # ------------------------------------------------------------
    # 0. Select 2D slice if corr_map is 3D
    # ------------------------------------------------------------
    corr_full = np.asarray(corr_map, dtype=np.float32)
    lag_full = np.asarray(lag_map)

    corr, lag, used_z = _select_corr_lag_slice(
        corr_full,
        lag_full,
        z=z,
        z_select=z_select,
    )

    valid = np.isfinite(corr)

    if not np.any(valid):
        raise ValueError("Selected corr_map slice has no finite values.")

    # ------------------------------------------------------------
    # 1. Threshold high-correlation pixels
    # ------------------------------------------------------------
    if min_corr is None:
        thr = float(np.nanpercentile(corr[valid], percentile))
    else:
        thr = float(min_corr)

    high = valid & (corr >= thr)

    if close_iter is not None and close_iter > 0:
        high = ndi.binary_closing(high, iterations=int(close_iter))

    if dilate_iter is not None and dilate_iter > 0:
        high = ndi.binary_dilation(high, iterations=int(dilate_iter))

    # ------------------------------------------------------------
    # 2. Connected components on the selected 2D slice
    # ------------------------------------------------------------
    if connectivity == 1:
        structure = ndi.generate_binary_structure(2, 1)
    else:
        structure = ndi.generate_binary_structure(2, 2)

    labeled, num = ndi.label(high, structure=structure)

    components = []

    for cid in range(1, num + 1):
        mask = labeled == cid
        area = int(mask.sum())

        if area < min_area:
            continue

        vals = corr[mask]
        lags = lag[mask]

        if vals.size == 0:
            continue

        coords = np.argwhere(mask)  # (n, 2), columns are h,w
        max_idx_local = int(np.nanargmax(vals))
        peak_h, peak_w = coords[max_idx_local]

        max_corr = float(corr[peak_h, peak_w])
        peak_lag = int(lag[peak_h, peak_w])

        weights = vals - np.nanmin(vals)
        if np.sum(weights) > 1e-12:
            weighted_lag = float(np.sum(weights * lags) / np.sum(weights))
        else:
            weighted_lag = float(np.nanmean(lags))

        w_cent = np.maximum(vals, 0)
        if np.sum(w_cent) > 1e-12:
            centroid_h = float(np.sum(coords[:, 0] * w_cent) / np.sum(w_cent))
            centroid_w = float(np.sum(coords[:, 1] * w_cent) / np.sum(w_cent))
        else:
            centroid_h = float(np.mean(coords[:, 0]))
            centroid_w = float(np.mean(coords[:, 1]))

        comp = {
            "component_id": cid,
            "mask": mask,
            "area": area,
            "max_corr": max_corr,
            "mean_corr": float(np.nanmean(vals)),
            "median_corr": float(np.nanmedian(vals)),
            "peak_h": int(peak_h),
            "peak_w": int(peak_w),
            "centroid_h": centroid_h,
            "centroid_w": centroid_w,
            "peak_lag": peak_lag,
            "mean_lag": float(np.nanmean(lags)),
            "median_lag": float(np.nanmedian(lags)),
            "weighted_lag": weighted_lag,
            "lag_min": int(np.nanmin(lags)),
            "lag_max": int(np.nanmax(lags)),
        }

        if used_z is not None:
            comp["z"] = int(used_z)
            comp["peak_coord"] = (int(used_z), int(peak_h), int(peak_w))
        else:
            comp["peak_coord"] = (int(peak_h), int(peak_w))

        components.append(comp)

    components = sorted(components, key=lambda d: d["max_corr"], reverse=True)
    components = components[:top_n]

    # ------------------------------------------------------------
    # 3. Prepare background
    # ------------------------------------------------------------
    if isinstance(background, str) and background == "corr":
        bg = corr.copy()

    elif background is None:
        bg = np.zeros_like(corr, dtype=np.float32)

    else:
        bg_arr = np.asarray(background, dtype=np.float32)

        if bg_arr.ndim == 2:
            bg = bg_arr

        elif bg_arr.ndim == 3:
            # background is also (Z,H,W)
            if used_z is None:
                bz = 0 if background_z is None else int(background_z)
            else:
                bz = used_z if background_z is None else int(background_z)

            if background_projection == "slice":
                bg = bg_arr[bz]

            elif background_projection == "max":
                bg = np.nanmax(bg_arr, axis=0)

            elif background_projection == "mean":
                bg = np.nanmean(bg_arr, axis=0)

            else:
                raise ValueError(
                    "background_projection should be 'slice', 'max', or 'mean'."
                )

        else:
            raise ValueError(
                "background should be None, 'corr', 2D array, or 3D array."
            )

        if bg.shape != corr.shape:
            raise ValueError(
                f"background 2D shape {bg.shape} != selected corr slice shape {corr.shape}"
            )

    # ------------------------------------------------------------
    # 4. Display helper
    # ------------------------------------------------------------
    def disp(A):
        return A.T if transpose_display else A

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(disp(bg), cmap=cmap_bg)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Best lagged correlation")

    cmap = plt.get_cmap(cmap_regions)

    for i, comp in enumerate(components):
        mask = comp["mask"]
        color = cmap(i % cmap.N)

        show_mask = disp(mask)

        overlay = np.zeros((*show_mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = color[0]
        overlay[..., 1] = color[1]
        overlay[..., 2] = color[2]
        overlay[..., 3] = show_mask.astype(np.float32) * alpha_region
        ax.imshow(overlay)

        if show_boundary:
            bd = ndi.binary_dilation(mask) ^ ndi.binary_erosion(mask)
            bd_show = disp(bd)

            boundary = np.zeros((*bd_show.shape, 4), dtype=np.float32)
            boundary[..., 0] = color[0]
            boundary[..., 1] = color[1]
            boundary[..., 2] = color[2]
            boundary[..., 3] = bd_show.astype(np.float32) * 0.95
            ax.imshow(boundary)

        centroid_h = comp["centroid_h"]
        centroid_w = comp["centroid_w"]

        # Important coordinate convention:
        # imshow(A):   display x = w, display y = h
        # imshow(A.T): display x = h, display y = w
        if transpose_display:
            text_x, text_y = centroid_h, centroid_w
        else:
            text_x, text_y = centroid_w, centroid_h

        ax.text(
            text_x,
            text_y,
            f"{i+1}\nlag={comp['peak_lag']}",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        )

    z_text = "" if used_z is None else f" | z={used_z}"
    ax.set_title(
        f"{title}{z_text}\nthreshold={thr:.3f}, components={len(components)}",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 5. Print table
    # ------------------------------------------------------------
    if print_table:
        print("\nTop connected Ca regions:")
        print("-" * 135)

        if used_z is None:
            print(
                f"{'idx':>3} | {'area':>5} | {'max_corr':>8} | {'mean_corr':>9} | "
                f"{'peak_lag':>8} | {'w_lag':>7} | {'lag_range':>12} | {'peak_hw':>14}"
            )
        else:
            print(
                f"{'idx':>3} | {'z':>3} | {'area':>5} | {'max_corr':>8} | {'mean_corr':>9} | "
                f"{'peak_lag':>8} | {'w_lag':>7} | {'lag_range':>12} | {'peak_zhw':>18}"
            )

        print("-" * 135)

        for i, comp in enumerate(components):
            lag_range = f"[{comp['lag_min']},{comp['lag_max']}]"

            if used_z is None:
                peak_str = f"({comp['peak_h']},{comp['peak_w']})"
                print(
                    f"{i+1:3d} | "
                    f"{comp['area']:5d} | "
                    f"{comp['max_corr']:8.3f} | "
                    f"{comp['mean_corr']:9.3f} | "
                    f"{comp['peak_lag']:8d} | "
                    f"{comp['weighted_lag']:7.2f} | "
                    f"{lag_range:>12} | "
                    f"{peak_str:>14}"
                )
            else:
                peak_str = f"({used_z},{comp['peak_h']},{comp['peak_w']})"
                print(
                    f"{i+1:3d} | "
                    f"{used_z:3d} | "
                    f"{comp['area']:5d} | "
                    f"{comp['max_corr']:8.3f} | "
                    f"{comp['mean_corr']:9.3f} | "
                    f"{comp['peak_lag']:8d} | "
                    f"{comp['weighted_lag']:7.2f} | "
                    f"{lag_range:>12} | "
                    f"{peak_str:>18}"
                )

    return components

def _select_2d_slice_from_3d_map(
    score_map,
    lag_map,
    z=None,
    z_select="max",
):
    """
    Select one 2D slice from 2D or 3D score/lag maps.

    score_map:
        (H, W) or (Z, H, W)
    lag_map:
        same shape as score_map

    z_select:
        "max": select z with largest finite max score
        "mean": select z with largest finite mean score
        "sum": select z with largest finite sum score
    """
    score_full = np.asarray(score_map, dtype=np.float32)
    lag_full = np.asarray(lag_map)

    if score_full.shape != lag_full.shape:
        raise ValueError(
            f"score_map shape {score_full.shape} != lag_map shape {lag_full.shape}"
        )

    if score_full.ndim == 2:
        return score_full, lag_full, None

    if score_full.ndim != 3:
        raise ValueError(
            "score_map should be 2D (H,W) or 3D (Z,H,W)."
        )

    Z = score_full.shape[0]

    if z is not None:
        used_z = int(z)
        if used_z < 0 or used_z >= Z:
            raise ValueError(f"z={used_z} out of range [0, {Z-1}]")
        return score_full[used_z], lag_full[used_z], used_z

    z_scores = []

    for zz in range(Z):
        sl = score_full[zz]
        valid = np.isfinite(sl)

        if not np.any(valid):
            z_scores.append(-np.inf)
            continue

        if z_select == "max":
            z_scores.append(float(np.nanmax(sl[valid])))
        elif z_select == "mean":
            z_scores.append(float(np.nanmean(sl[valid])))
        elif z_select == "sum":
            z_scores.append(float(np.nansum(sl[valid])))
        else:
            raise ValueError("z_select should be 'max', 'mean', or 'sum'.")

    used_z = int(np.argmax(z_scores))

    return score_full[used_z], lag_full[used_z], used_z

def _select_background_2d(
    background,
    used_z,
    target_shape,
    background_z=None,
    background_projection="slice",
):
    """
    Prepare 2D background image.
    """
    if background is None:
        return np.zeros(target_shape, dtype=np.float32)

    if isinstance(background, str):
        raise ValueError(
            "String background should be handled outside this helper."
        )

    bg_arr = np.asarray(background, dtype=np.float32)

    if bg_arr.ndim == 2:
        bg = bg_arr

    elif bg_arr.ndim == 3:
        if background_projection == "slice":
            if background_z is not None:
                bz = int(background_z)
            elif used_z is not None:
                bz = int(used_z)
            else:
                bz = 0
            bg = bg_arr[bz]

        elif background_projection == "max":
            bg = np.nanmax(bg_arr, axis=0)

        elif background_projection == "mean":
            bg = np.nanmean(bg_arr, axis=0)

        else:
            raise ValueError(
                "background_projection should be 'slice', 'max', or 'mean'."
            )

    else:
        raise ValueError(
            "background should be None, 2D array, or 3D array."
        )

    if bg.shape != target_shape:
        raise ValueError(
            f"background 2D shape {bg.shape} != target map shape {target_shape}"
        )

    return bg


def visualize_top_dependency_components_with_lag(
    res_dep=None,
    cls="local",

    # Direct map input, optional.
    score_map=None,
    lag_map=None,

    # Which metric to visualize from res_dep
    metric="best_dependency_score",
    # alternatives:
    #   "best_probability"
    #   "best_hit_rate"
    #   "best_enrichment"
    #   "best_rank_score"

    top_n=10,
    min_score=None,
    percentile=99,
    min_area=3,
    connectivity=2,
    close_iter=1,
    dilate_iter=0,

    # 3D support
    z=None,
    z_select="max",

    # Display
    transpose_display=True,
    background="score",
    # background:
    #   "score": use selected score map
    #   None: blank
    #   2D/3D array: custom background, e.g. reference image or mean Ca
    background_z=None,
    background_projection="slice",

    cmap_bg="magma",
    cmap_regions="tab20",
    alpha_region=0.55,
    show_boundary=True,

    # colorbar/title
    colorbar_label=None,
    title=None,
    figsize=(8, 6),

    # Text label content
    label_mode="rank_lag",
    # "rank_lag":       rank + lag
    # "rank_lag_score": rank + lag + score
    # "rank_only":      rank only

    print_table=True,
):
    """
    Visualize top connected high-dependency Ca regions.

    This is adapted for output from:
        analyze_roi_local_mixed_ca_dependency(...)

    Typical usage:
        visualize_top_dependency_components_with_lag(
            res_dep,
            cls="local",
            metric="best_dependency_score",
            z=10,
            percentile=99,
        )

    It detects connected components on a selected 2D slice:
        high = score >= threshold

    Then ranks components by peak score and annotates:
        component rank
        best lag at peak position
    """

    # ------------------------------------------------------------
    # 0. Resolve input maps
    # ------------------------------------------------------------
    if res_dep is not None:
        if cls not in res_dep["results"]:
            raise ValueError(
                f"cls={cls!r} not found in res_dep['results']. "
                f"Available: {list(res_dep['results'].keys())}"
            )

        if metric not in res_dep["results"][cls]:
            raise ValueError(
                f"metric={metric!r} not found in res_dep['results'][{cls!r}]."
            )

        score_full = np.asarray(res_dep["results"][cls][metric], dtype=np.float32)
        lag_full = np.asarray(res_dep["results"][cls]["best_lag"])

        if title is None:
            title = f"Top Ca regions coupled with {cls} ROI motion"

    else:
        if score_map is None or lag_map is None:
            raise ValueError(
                "Either provide res_dep, or provide score_map and lag_map."
            )

        score_full = np.asarray(score_map, dtype=np.float32)
        lag_full = np.asarray(lag_map)

        if title is None:
            title = "Top Ca regions coupled with ROI motion"

    if colorbar_label is None:
        if metric == "best_dependency_score":
            colorbar_label = "Best event dependency score"
        elif metric in ["best_probability", "best_hit_rate"]:
            colorbar_label = "Ca activation probability"
        elif metric == "best_enrichment":
            colorbar_label = "Activation enrichment"
        elif metric == "best_rank_score":
            colorbar_label = "Best rank score"
        else:
            colorbar_label = metric

    # ------------------------------------------------------------
    # 1. Select 2D slice
    # ------------------------------------------------------------
    score, lag, used_z = _select_2d_slice_from_3d_map(
        score_full,
        lag_full,
        z=z,
        z_select=z_select,
    )

    valid = np.isfinite(score)

    if not np.any(valid):
        raise ValueError("Selected score_map slice has no finite values.")

    # ------------------------------------------------------------
    # 2. Threshold high-score pixels
    # ------------------------------------------------------------
    if min_score is None:
        thr = float(np.nanpercentile(score[valid], percentile))
    else:
        thr = float(min_score)

    high = valid & (score >= thr)

    if close_iter is not None and close_iter > 0:
        high = ndi.binary_closing(high, iterations=int(close_iter))

    if dilate_iter is not None and dilate_iter > 0:
        high = ndi.binary_dilation(high, iterations=int(dilate_iter))

    # ------------------------------------------------------------
    # 3. Connected components
    # ------------------------------------------------------------
    if connectivity == 1:
        structure = ndi.generate_binary_structure(2, 1)
    else:
        structure = ndi.generate_binary_structure(2, 2)

    labeled, num = ndi.label(high, structure=structure)

    components = []

    for cid in range(1, num + 1):
        mask = labeled == cid
        area = int(mask.sum())

        if area < min_area:
            continue

        vals = score[mask]
        lags = lag[mask]

        if vals.size == 0:
            continue

        coords = np.argwhere(mask)  # (n, 2), h,w

        max_idx_local = int(np.nanargmax(vals))
        peak_h, peak_w = coords[max_idx_local]

        peak_score = float(score[peak_h, peak_w])
        peak_lag = int(lag[peak_h, peak_w])

        # score-weighted centroid
        w_cent = np.maximum(vals, 0)
        if np.sum(w_cent) > 1e-12:
            centroid_h = float(np.sum(coords[:, 0] * w_cent) / np.sum(w_cent))
            centroid_w = float(np.sum(coords[:, 1] * w_cent) / np.sum(w_cent))
        else:
            centroid_h = float(np.mean(coords[:, 0]))
            centroid_w = float(np.mean(coords[:, 1]))

        # score-weighted lag
        lag_weights = vals - np.nanmin(vals)
        if np.sum(lag_weights) > 1e-12:
            weighted_lag = float(np.sum(lag_weights * lags) / np.sum(lag_weights))
        else:
            weighted_lag = float(np.nanmean(lags))

        comp = {
            "component_id": int(cid),
            "mask": mask,
            "area": int(area),

            "peak_score": peak_score,
            "max_score": peak_score,
            "mean_score": float(np.nanmean(vals)),
            "median_score": float(np.nanmedian(vals)),

            "peak_h": int(peak_h),
            "peak_w": int(peak_w),
            "centroid_h": centroid_h,
            "centroid_w": centroid_w,

            "peak_lag": peak_lag,
            "mean_lag": float(np.nanmean(lags)),
            "median_lag": float(np.nanmedian(lags)),
            "weighted_lag": weighted_lag,
            "lag_min": int(np.nanmin(lags)),
            "lag_max": int(np.nanmax(lags)),
        }

        if used_z is not None:
            comp["z"] = int(used_z)
            comp["peak_coord"] = (int(used_z), int(peak_h), int(peak_w))
        else:
            comp["peak_coord"] = (int(peak_h), int(peak_w))

        components.append(comp)

    components = sorted(components, key=lambda d: d["peak_score"], reverse=True)
    components = components[:top_n]

    # ------------------------------------------------------------
    # 4. Background
    # ------------------------------------------------------------
    if isinstance(background, str):
        if background == "score":
            bg = score.copy()
        else:
            raise ValueError("String background only supports 'score'.")
    else:
        bg = _select_background_2d(
            background=background,
            used_z=used_z,
            target_shape=score.shape,
            background_z=background_z,
            background_projection=background_projection,
        )

    # ------------------------------------------------------------
    # 5. Display helper
    # ------------------------------------------------------------
    def disp(A):
        return A.T if transpose_display else A

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(disp(bg), cmap=cmap_bg)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(colorbar_label)

    cmap = plt.get_cmap(cmap_regions)

    for i, comp in enumerate(components):
        mask = comp["mask"]
        color = cmap(i % cmap.N)

        show_mask = disp(mask)

        overlay = np.zeros((*show_mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = color[0]
        overlay[..., 1] = color[1]
        overlay[..., 2] = color[2]
        overlay[..., 3] = show_mask.astype(np.float32) * alpha_region
        ax.imshow(overlay)

        if show_boundary:
            bd = ndi.binary_dilation(mask) ^ ndi.binary_erosion(mask)
            bd_show = disp(bd)

            boundary = np.zeros((*bd_show.shape, 4), dtype=np.float32)
            boundary[..., 0] = color[0]
            boundary[..., 1] = color[1]
            boundary[..., 2] = color[2]
            boundary[..., 3] = bd_show.astype(np.float32) * 0.95
            ax.imshow(boundary)

        centroid_h = comp["centroid_h"]
        centroid_w = comp["centroid_w"]

        if transpose_display:
            text_x, text_y = centroid_h, centroid_w
        else:
            text_x, text_y = centroid_w, centroid_h

        if label_mode == "rank_lag":
            label = f"{i+1}\nlag={comp['peak_lag']}"
        elif label_mode == "rank_lag_score":
            label = f"{i+1}\nlag={comp['peak_lag']}\n{comp['peak_score']:.2f}"
        elif label_mode == "rank_only":
            label = f"{i+1}"
        else:
            raise ValueError(
                "label_mode should be 'rank_lag', 'rank_lag_score', or 'rank_only'."
            )

        ax.text(
            text_x,
            text_y,
            label,
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        )

    z_text = "" if used_z is None else f" | z={used_z}"
    ax.set_title(
        f"{title}{z_text}\nthreshold={thr:.3f}, components={len(components)}",
        fontsize=12,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 6. Print table
    # ------------------------------------------------------------
    if print_table:
        print("\nTop connected Ca dependency regions:")
        print("-" * 145)

        if used_z is None:
            print(
                f"{'idx':>3} | {'area':>5} | {'peak_score':>10} | {'mean_score':>10} | "
                f"{'peak_lag':>8} | {'w_lag':>7} | {'lag_range':>12} | {'peak_hw':>14}"
            )
        else:
            print(
                f"{'idx':>3} | {'z':>3} | {'area':>5} | {'peak_score':>10} | {'mean_score':>10} | "
                f"{'peak_lag':>8} | {'w_lag':>7} | {'lag_range':>12} | {'peak_zhw':>18}"
            )

        print("-" * 145)

        for i, comp in enumerate(components):
            lag_range = f"[{comp['lag_min']},{comp['lag_max']}]"

            if used_z is None:
                peak_str = f"({comp['peak_h']},{comp['peak_w']})"
                print(
                    f"{i+1:3d} | "
                    f"{comp['area']:5d} | "
                    f"{comp['peak_score']:10.3f} | "
                    f"{comp['mean_score']:10.3f} | "
                    f"{comp['peak_lag']:8d} | "
                    f"{comp['weighted_lag']:7.2f} | "
                    f"{lag_range:>12} | "
                    f"{peak_str:>14}"
                )
            else:
                peak_str = f"({used_z},{comp['peak_h']},{comp['peak_w']})"
                print(
                    f"{i+1:3d} | "
                    f"{used_z:3d} | "
                    f"{comp['area']:5d} | "
                    f"{comp['peak_score']:10.3f} | "
                    f"{comp['mean_score']:10.3f} | "
                    f"{comp['peak_lag']:8d} | "
                    f"{comp['weighted_lag']:7.2f} | "
                    f"{lag_range:>12} | "
                    f"{peak_str:>18}"
                )

    return components

def visualize_top_dependency_components_with_lag_and_p(
    res_dep=None,
    cls="local",

    # direct map input (optional)
    score_map=None,
    lag_map=None,
    p_map=None,

    # which metric to use from res_dep
    metric="best_dependency_score",
    p_metric="best_p_binom",          # preferred if available
    neglog10_p_metric="best_neglog10_p",  # fallback if best_p_binom not saved

    top_n=10,
    min_score=None,
    percentile=99,
    min_area=3,
    connectivity=2,
    close_iter=1,
    dilate_iter=0,

    # 3D support
    z=None,
    z_select="max",

    # display
    transpose_display=True,
    background="score",
    background_z=None,
    background_projection="slice",
    cmap_bg="magma",
    cmap_regions="tab20",
    alpha_region=0.55,
    show_boundary=True,

    # labels
    show_p=True,
    p_display_mode="peak",   # "peak" or "min"
    p_format=".1e",
    label_mode="rank_lag_p",  # "rank_lag_p", "rank_lag", "rank_only"

    colorbar_label=None,
    title=None,
    figsize=(8, 6),

    print_table=True,
):
    """
    Visualize top connected high-dependency Ca regions on the dependency_score map,
    and annotate lag + p-value on the same plot.

    p-value shown:
        - peak: p-value at the peak dependency pixel of each component
        - min:  minimum p-value inside each component
    """

    # ------------------------------------------------------------
    # helper
    # ------------------------------------------------------------
    def _select_2d_slice(arr, z=None, z_select="max"):
        arr = np.asarray(arr)

        if arr.ndim == 2:
            return arr, None

        if arr.ndim != 3:
            raise ValueError("Input map must be 2D or 3D.")

        Z = arr.shape[0]

        if z is not None:
            zz = int(z)
            if zz < 0 or zz >= Z:
                raise ValueError(f"z={zz} out of range [0, {Z-1}]")
            return arr[zz], zz

        scores = []
        for zz in range(Z):
            sl = arr[zz]
            valid = np.isfinite(sl)

            if not np.any(valid):
                scores.append(-np.inf)
                continue

            if z_select == "max":
                scores.append(float(np.nanmax(sl[valid])))
            elif z_select == "mean":
                scores.append(float(np.nanmean(sl[valid])))
            elif z_select == "sum":
                scores.append(float(np.nansum(sl[valid])))
            else:
                raise ValueError("z_select should be 'max', 'mean', or 'sum'.")

        zz = int(np.argmax(scores))
        return arr[zz], zz

    def _select_background_2d(background, used_z, target_shape,
                              background_z=None, background_projection="slice"):
        if background is None:
            return np.zeros(target_shape, dtype=np.float32)

        if isinstance(background, str):
            raise ValueError("String background should be handled outside helper.")

        bg_arr = np.asarray(background, dtype=np.float32)

        if bg_arr.ndim == 2:
            bg = bg_arr
        elif bg_arr.ndim == 3:
            if background_projection == "slice":
                if background_z is not None:
                    bz = int(background_z)
                elif used_z is not None:
                    bz = int(used_z)
                else:
                    bz = 0
                bg = bg_arr[bz]
            elif background_projection == "max":
                bg = np.nanmax(bg_arr, axis=0)
            elif background_projection == "mean":
                bg = np.nanmean(bg_arr, axis=0)
            else:
                raise ValueError("background_projection should be 'slice', 'max', or 'mean'.")
        else:
            raise ValueError("background must be None, 2D array, or 3D array.")

        if bg.shape != target_shape:
            raise ValueError(f"background shape {bg.shape} != target shape {target_shape}")

        return bg

    def disp(A):
        return A.T if transpose_display else A

    # ------------------------------------------------------------
    # 0. resolve input maps
    # ------------------------------------------------------------
    if res_dep is not None:
        if cls not in res_dep["results"]:
            raise ValueError(f"{cls!r} not found in res_dep['results']")

        score_full = np.asarray(res_dep["results"][cls][metric], dtype=np.float32)
        lag_full = np.asarray(res_dep["results"][cls]["best_lag"])

        # p map
        if p_map is None:
            if p_metric in res_dep["results"][cls]:
                p_full = np.asarray(res_dep["results"][cls][p_metric], dtype=np.float32)
            elif neglog10_p_metric in res_dep["results"][cls]:
                neglog_full = np.asarray(res_dep["results"][cls][neglog10_p_metric], dtype=np.float32)
                p_full = 10.0 ** (-neglog_full)
            else:
                p_full = None
        else:
            p_full = np.asarray(p_map, dtype=np.float32)

        if title is None:
            title = f"Top Ca regions coupled with {cls} ROI motion"

    else:
        if score_map is None or lag_map is None:
            raise ValueError("Either provide res_dep, or provide score_map + lag_map.")

        score_full = np.asarray(score_map, dtype=np.float32)
        lag_full = np.asarray(lag_map)

        if p_map is not None:
            p_full = np.asarray(p_map, dtype=np.float32)
        else:
            p_full = None

        if title is None:
            title = "Top Ca regions coupled with ROI motion"

    if colorbar_label is None:
        colorbar_label = "Best event dependency score"

    # ------------------------------------------------------------
    # 1. select 2D slice
    # ------------------------------------------------------------
    score, used_z = _select_2d_slice(score_full, z=z, z_select=z_select)
    lag, _ = _select_2d_slice(lag_full, z=used_z, z_select=z_select)

    if p_full is not None:
        p2d, _ = _select_2d_slice(p_full, z=used_z, z_select=z_select)
    else:
        p2d = None

    valid = np.isfinite(score)
    if not np.any(valid):
        raise ValueError("Selected score map slice has no finite values.")

    # ------------------------------------------------------------
    # 2. threshold high-score pixels
    # ------------------------------------------------------------
    if min_score is None:
        thr = float(np.nanpercentile(score[valid], percentile))
    else:
        thr = float(min_score)

    high = valid & (score >= thr)

    if close_iter is not None and close_iter > 0:
        high = ndi.binary_closing(high, iterations=int(close_iter))

    if dilate_iter is not None and dilate_iter > 0:
        high = ndi.binary_dilation(high, iterations=int(dilate_iter))

    # ------------------------------------------------------------
    # 3. connected components
    # ------------------------------------------------------------
    if connectivity == 1:
        structure = ndi.generate_binary_structure(2, 1)
    else:
        structure = ndi.generate_binary_structure(2, 2)

    labeled, num = ndi.label(high, structure=structure)

    components = []

    for cid in range(1, num + 1):
        mask = labeled == cid
        area = int(mask.sum())

        if area < min_area:
            continue

        vals = score[mask]
        lags = lag[mask]
        coords = np.argwhere(mask)

        if vals.size == 0:
            continue

        peak_idx_local = int(np.nanargmax(vals))
        peak_h, peak_w = coords[peak_idx_local]

        peak_score = float(score[peak_h, peak_w])
        peak_lag = int(lag[peak_h, peak_w])

        # component p-values
        if p2d is not None:
            peak_p = float(p2d[peak_h, peak_w])
            min_p = float(np.nanmin(p2d[mask]))
        else:
            peak_p = np.nan
            min_p = np.nan

        # score-weighted centroid
        w_cent = np.maximum(vals, 0)
        if np.sum(w_cent) > 1e-12:
            centroid_h = float(np.sum(coords[:, 0] * w_cent) / np.sum(w_cent))
            centroid_w = float(np.sum(coords[:, 1] * w_cent) / np.sum(w_cent))
        else:
            centroid_h = float(np.mean(coords[:, 0]))
            centroid_w = float(np.mean(coords[:, 1]))

        comp = {
            "component_id": int(cid),
            "mask": mask,
            "area": int(area),

            "peak_score": peak_score,
            "mean_score": float(np.nanmean(vals)),
            "median_score": float(np.nanmedian(vals)),

            "peak_h": int(peak_h),
            "peak_w": int(peak_w),
            "centroid_h": centroid_h,
            "centroid_w": centroid_w,

            "peak_lag": peak_lag,
            "mean_lag": float(np.nanmean(lags)),
            "median_lag": float(np.nanmedian(lags)),

            "peak_p": peak_p,
            "min_p": min_p,
        }

        if used_z is not None:
            comp["z"] = int(used_z)
            comp["peak_coord"] = (int(used_z), int(peak_h), int(peak_w))
        else:
            comp["peak_coord"] = (int(peak_h), int(peak_w))

        components.append(comp)

    components = sorted(components, key=lambda d: d["peak_score"], reverse=True)
    components = components[:top_n]

    # ------------------------------------------------------------
    # 4. background
    # ------------------------------------------------------------
    if isinstance(background, str):
        if background == "score":
            bg = score.copy()
        else:
            raise ValueError("String background only supports 'score'.")
    else:
        bg = _select_background_2d(
            background=background,
            used_z=used_z,
            target_shape=score.shape,
            background_z=background_z,
            background_projection=background_projection,
        )

    # ------------------------------------------------------------
    # 5. display
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(disp(bg), cmap=cmap_bg)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(colorbar_label)

    cmap = plt.get_cmap(cmap_regions)

    for i, comp in enumerate(components):
        mask = comp["mask"]
        color = cmap(i % cmap.N)

        show_mask = disp(mask)

        overlay = np.zeros((*show_mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = color[0]
        overlay[..., 1] = color[1]
        overlay[..., 2] = color[2]
        overlay[..., 3] = show_mask.astype(np.float32) * alpha_region
        ax.imshow(overlay)

        if show_boundary:
            bd = ndi.binary_dilation(mask) ^ ndi.binary_erosion(mask)
            bd_show = disp(bd)

            boundary = np.zeros((*bd_show.shape, 4), dtype=np.float32)
            boundary[..., 0] = color[0]
            boundary[..., 1] = color[1]
            boundary[..., 2] = color[2]
            boundary[..., 3] = bd_show.astype(np.float32) * 0.95
            ax.imshow(boundary)

        centroid_h = comp["centroid_h"]
        centroid_w = comp["centroid_w"]

        if transpose_display:
            text_x, text_y = centroid_h, centroid_w
        else:
            text_x, text_y = centroid_w, centroid_h

        if show_p and label_mode == "rank_lag_p":
            p_to_show = comp["peak_p"] if p_display_mode == "peak" else comp["min_p"]

            if np.isfinite(p_to_show):
                p_str = format(p_to_show, p_format)
                label = f"{i+1}\nlag={comp['peak_lag']}\np={p_str}"
            else:
                label = f"{i+1}\nlag={comp['peak_lag']}"
        elif label_mode == "rank_lag":
            label = f"{i+1}\nlag={comp['peak_lag']}"
        elif label_mode == "rank_only":
            label = f"{i+1}"
        else:
            label = f"{i+1}\nlag={comp['peak_lag']}"

        ax.annotate(
            label,
            xy=(text_x, text_y),                 # 箭头指向 component 的中心
            xytext=(-50, -50),                     # 标签相对中心偏移，单位是 points
            textcoords="offset points",
            color="white",
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="white",
                alpha=0.8,
                linewidth=0.8,
            ),
        )

    z_text = "" if used_z is None else f" | z={used_z}"
    ax.set_title(
        f"{title}{z_text}\nthreshold={thr:.3f}, components={len(components)}",
        fontsize=12,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 6. print table
    # ------------------------------------------------------------
    if print_table:
        print("\nTop connected Ca dependency regions:")
        print("-" * 150)
        if used_z is None:
            print(
                f"{'idx':>3} | {'area':>5} | {'peak_score':>10} | {'peak_lag':>8} | "
                f"{'peak_p':>10} | {'min_p':>10} | {'peak_hw':>14}"
            )
        else:
            print(
                f"{'idx':>3} | {'z':>3} | {'area':>5} | {'peak_score':>10} | {'peak_lag':>8} | "
                f"{'peak_p':>10} | {'min_p':>10} | {'peak_zhw':>18}"
            )
        print("-" * 150)

        for i, comp in enumerate(components):
            if used_z is None:
                peak_str = f"({comp['peak_h']},{comp['peak_w']})"
                print(
                    f"{i+1:3d} | "
                    f"{comp['area']:5d} | "
                    f"{comp['peak_score']:10.3f} | "
                    f"{comp['peak_lag']:8d} | "
                    f"{comp['peak_p']:10.2e} | "
                    f"{comp['min_p']:10.2e} | "
                    f"{peak_str:>14}"
                )
            else:
                peak_str = f"({used_z},{comp['peak_h']},{comp['peak_w']})"
                print(
                    f"{i+1:3d} | "
                    f"{used_z:3d} | "
                    f"{comp['area']:5d} | "
                    f"{comp['peak_score']:10.3f} | "
                    f"{comp['peak_lag']:8d} | "
                    f"{comp['peak_p']:10.2e} | "
                    f"{comp['min_p']:10.2e} | "
                    f"{peak_str:>18}"
                )

    return components
