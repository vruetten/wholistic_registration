"""Array-to-PNG rendering for the QC dashboard. Pure functions, no file IO, no HTTP.

Imports: numpy, PIL (plus stdlib). See dashboard/SPEC.md section 2 for the
function contracts this module must satisfy.
"""
import io
import logging
import time

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Colormap control points: (position in [0, 1], (r, g, b) uint8). Linearly
# interpolated into a 256-entry lookup table at import time. "diverging" is
# built symmetric about its 0.5 node so a zero-centred signed field maps
# exactly to the achromatic node at u01 == 0.5.
_CONTROL_POINTS = {
    "gray": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "magma": [
        (0.0, (0, 0, 4)),
        (0.25, (81, 18, 124)),
        (0.5, (183, 55, 121)),
        (0.75, (252, 137, 97)),
        (1.0, (252, 253, 191)),
    ],
    "diverging": [
        (0.0, (5, 48, 97)),
        (0.25, (67, 147, 195)),
        # u01 == 0.5 rounds (numpy round-half-to-even) to LUT index 128, not
        # 127.5, so the neutral node sits at 128/255 to land exactly there.
        (128.0 / 255.0, (247, 247, 247)),
        (0.75, (214, 96, 77)),
        (1.0, (103, 0, 31)),
    ],
}


def _build_lut(points):
    xs = np.array([p[0] for p in points], dtype=np.float64) * 255.0
    rgb = np.array([p[1] for p in points], dtype=np.float64)
    idx = np.arange(256, dtype=np.float64)
    channels = [np.interp(idx, xs, rgb[:, c]) for c in range(3)]
    return np.clip(np.round(np.stack(channels, axis=1)), 0, 255).astype(np.uint8)


_LUTS = {name: _build_lut(pts) for name, pts in _CONTROL_POINTS.items()}


def contrast_limits(
    a,
    mask=None,
    lo_pct=1.0,
    hi_pct=99.0,
    exclude_values=(-200.0,),
    subsample=200000,
):
    """Percentile contrast limits, ignoring non-finite values and exclude_values.

    mask, if given, is a boolean array the same shape as `a`; True marks pixels
    to include. If every pixel is excluded (all fill/NaN/masked-out), the
    defined fallback (0.0, 1.0) is returned rather than raising or returning NaN.
    """
    valid = np.isfinite(a)
    if mask is not None:
        assert mask.shape == a.shape, "mask shape must match a.shape"
        valid &= mask
    for v in exclude_values:
        valid &= a != v
    values = a[valid]
    if values.size == 0:
        return (0.0, 1.0)
    if values.size > subsample:
        # Deterministic seed: repeated calls on the same array (e.g. cache
        # rebuilds) return the same limits, which matters for a UI that must
        # not flicker contrast between identical requests.
        rng = np.random.default_rng(0)
        values = values[rng.choice(values.size, size=subsample, replace=False)]
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    return (float(lo), float(hi))


def diverging_limits(
    a,
    mask=None,
    pct=99.0,
    exclude_values=(-200.0,),
    subsample=200000,
):
    """Symmetric-about-zero contrast limits for a signed quantity (dx, dy, dz).

    Returns (-m, m) so the diverging colormap's achromatic centre always lands
    on value 0, regardless of any asymmetry in the data's positive/negative
    extent.
    """
    lo, hi = contrast_limits(
        a, mask=mask, lo_pct=100.0 - pct, hi_pct=pct,
        exclude_values=exclude_values, subsample=subsample,
    )
    m = max(abs(lo), abs(hi))
    if m <= 0:
        m = 1e-6
    return (-m, m)


def apply_colormap(u01, cmap):
    """Map values in [0, 1] through a named colormap LUT. NaN maps to index 0;
    callers that need NaN to render distinctly (the dashboard convention) do
    so via render_scalar's nonfinite_rgb, not here."""
    assert cmap in _LUTS, "unknown cmap %r, have %s" % (cmap, list(_LUTS))
    u = np.nan_to_num(np.clip(u01, 0.0, 1.0), nan=0.0)
    idx = np.round(u * 255).astype(np.uint8)
    return _LUTS[cmap][idx]


def render_scalar(a, lo, hi, cmap, mask=None, nonfinite_rgb=(255, 0, 255)):
    """Normalise a to [lo, hi], colormap it, and paint non-finite (and masked-out)
    pixels with nonfinite_rgb so a NaN never renders indistinguishably from a
    real zero."""
    assert a.ndim == 2, "render_scalar takes a single (H, W) plane"
    finite = np.isfinite(a)
    if mask is not None:
        assert mask.shape == a.shape, "mask shape must match a.shape"
        finite = finite & mask
    denom = hi - lo
    if denom == 0:
        denom = 1.0
    u01 = np.clip((a - lo) / denom, 0.0, 1.0)
    rgb = apply_colormap(u01, cmap)
    rgb[~finite] = nonfinite_rgb
    return rgb


def composite(base_rgb, over_rgb, alpha, over_mask=None):
    """Alpha-blend over_rgb onto base_rgb. alpha == 0 leaves base unchanged;
    alpha == 1 replaces base with over_rgb wherever over_mask is set (or
    everywhere, if over_mask is None)."""
    assert base_rgb.shape == over_rgb.shape, "base_rgb and over_rgb shape mismatch"
    if over_mask is None:
        m = np.ones(base_rgb.shape[:2], dtype=bool)
    else:
        assert over_mask.shape == base_rgb.shape[:2], "over_mask shape mismatch"
        m = over_mask
    blended = base_rgb.astype(np.float64) * (1.0 - alpha) + over_rgb.astype(np.float64) * alpha
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    out = base_rgb.copy()
    out[m] = blended[m]
    return out


def quiver_overlay(rgb, dx, dy, dz=None, stride=11, scale=1.0, dz_cmap="diverging"):
    """Draw one arrow per control point (default stride 11, matching the
    solver's control-point spacing) on a copy of rgb. Arrows are coloured by
    dz through dz_cmap when dz is given, white otherwise."""
    assert rgb.shape[:2] == dx.shape == dy.shape, "rgb/dx/dy shape mismatch"
    h, w = dx.shape
    img = Image.fromarray(rgb.copy(), "RGB")
    draw = ImageDraw.Draw(img)

    if dz is not None:
        assert dz.shape == dx.shape, "dz shape mismatch"
        lo, hi = diverging_limits(dz)

    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)
    for y in ys:
        for x in xs:
            vx, vy = dx[y, x], dy[y, x]
            if not (np.isfinite(vx) and np.isfinite(vy)):
                continue
            color = (255, 255, 255)
            if dz is not None and np.isfinite(dz[y, x]):
                u = (dz[y, x] - lo) / (hi - lo)
                color = tuple(int(c) for c in apply_colormap(np.array([[u]]), dz_cmap)[0, 0])
            draw.line([(x, y), (x + vx * scale, y + vy * scale)], fill=color, width=1)
    return np.array(img)


def montage(planes, cols=None, labels=None, max_edge_px=8192):
    """Tile equal-shaped (H, W, 3) uint8 planes into a grid. If the tiled
    canvas exceeds max_edge_px on either side, downscale to fit and log the
    applied scale factor (the signature has no second return value to carry
    it, so a silent downsample would read as full resolution to the caller)."""
    assert len(planes) > 0, "montage needs at least one plane"
    shapes = {p.shape for p in planes}
    assert len(shapes) == 1, "all planes must share one shape, got %s" % shapes
    h, w = planes[0].shape[:2]
    n = len(planes)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, p in enumerate(planes):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = p

    img = Image.fromarray(canvas, "RGB")
    if labels is not None:
        assert len(labels) == n, "labels must have one entry per plane"
        draw = ImageDraw.Draw(img)
        for i, lab in enumerate(labels):
            r, c = divmod(i, cols)
            draw.text((c * w + 4, r * h + 4), str(lab), fill=(255, 255, 255))

    edge = max(img.size)
    if edge > max_edge_px:
        factor = max_edge_px / edge
        new_size = (max(1, int(round(img.size[0] * factor))), max(1, int(round(img.size[1] * factor))))
        logger.warning(
            "montage downsampled by factor %.4f: %dx%d -> %dx%d (max_edge_px=%d)",
            factor, img.size[0], img.size[1], new_size[0], new_size[1], max_edge_px,
        )
        img = img.resize(new_size, Image.BILINEAR)
    return np.array(img)


def to_png(rgb):
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "to_png takes (H, W, 3) uint8"
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _selftest():
    import os
    import tempfile

    rng = np.random.default_rng(0)

    # --- 1. contrast_limits ignores injected NaN and -200.0 fill -----------
    # Junk values are appended, not overwritten in place, so the surviving
    # valid subset of `dirty` is exactly the untouched `clean` array (same
    # multiset, order-independent for a percentile) -- a true independent
    # oracle rather than a percentile recomputed on a shrunken sample.
    clean = np.linspace(-5.0, 5.0, 100 * 100)
    lo_oracle, hi_oracle = np.percentile(clean, [1.0, 99.0])  # independent of contrast_limits
    junk = np.concatenate([np.full(500, np.nan), np.full(500, -200.0)])
    dirty = np.concatenate([clean, junk])
    rng.shuffle(dirty)
    lo, hi = contrast_limits(dirty)
    assert abs(lo - lo_oracle) < 1e-9, (lo, lo_oracle)
    assert abs(hi - hi_oracle) < 1e-9, (hi, hi_oracle)
    print("PASS contrast_limits ignores NaN/-200 fill: (%.6f, %.6f) == oracle (%.6f, %.6f)" % (lo, hi, lo_oracle, hi_oracle))

    # --- 1b. all-excluded fallback ------------------------------------------
    all_fill = np.full((10, 10), -200.0)
    lo_fb, hi_fb = contrast_limits(all_fill)
    assert (lo_fb, hi_fb) == (0.0, 1.0), (lo_fb, hi_fb)
    print("PASS contrast_limits all-excluded fallback: (%.1f, %.1f)" % (lo_fb, hi_fb))

    # --- 2. diverging path: value 0 maps to the exact neutral midpoint -----
    field = 8.0 * np.sin(2.0 * np.pi * np.arange(300).reshape(300, 1) / 300.0) * np.ones((1, 200))
    dlo, dhi = diverging_limits(field)
    assert dlo == -dhi, (dlo, dhi)
    mid_rgb = render_scalar(np.array([[0.0]]), dlo, dhi, "diverging")[0, 0]
    assert tuple(int(c) for c in mid_rgb) == (247, 247, 247), tuple(mid_rgb)  # exact "diverging" 0.5 control point
    print("PASS diverging_limits + render_scalar(0) == neutral midpoint (247, 247, 247)")

    # --- 3. non-finite pixels render magenta --------------------------------
    a = np.array([[0.0, np.nan], [5.0, -200.0]])
    lo3, hi3 = contrast_limits(a)
    out = render_scalar(a, lo3, hi3, "gray")
    assert tuple(int(c) for c in out[0, 1]) == (255, 0, 255), tuple(out[0, 1])
    print("PASS non-finite pixel renders magenta (255, 0, 255)")

    # --- 4. composite: alpha 0 == base unchanged, alpha 1 == overlay where masked
    base = np.zeros((20, 20, 3), dtype=np.uint8)
    base[:] = (10, 20, 30)
    over = np.zeros((20, 20, 3), dtype=np.uint8)
    over[:] = (200, 210, 220)
    m = np.zeros((20, 20), dtype=bool)
    m[5:10, 5:10] = True
    c0 = composite(base, over, 0.0, over_mask=m)
    assert np.array_equal(c0, base), "alpha=0 must leave base unchanged"
    c1 = composite(base, over, 1.0, over_mask=m)
    assert np.array_equal(c1[m], over[m]), "alpha=1 must equal overlay where masked"
    assert np.array_equal(c1[~m], base[~m]), "alpha=1 must leave base outside the mask"
    print("PASS composite alpha=0 (unchanged) and alpha=1 (overlay under mask)")

    # --- 5. montage respects max_edge_px and reports its scale factor ------
    planes = [np.full((300, 300, 3), i * 40, dtype=np.uint8) for i in range(4)]
    log_records = []
    handler = logging.Handler()
    handler.emit = lambda rec: log_records.append(rec.getMessage())
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        mont = montage(planes, cols=4, max_edge_px=1000)
    finally:
        logger.removeHandler(handler)
    full_edge = max(300 * 4, 300 * 1)  # 4 cols x 1 row of 300x300 -> 1200x300
    assert max(mont.shape[:2]) <= 1000, mont.shape
    assert any("downsampled" in r for r in log_records), log_records
    print("PASS montage max_edge_px=1000 caps output %s (unclamped edge was %d), logged: %s" % (mont.shape[:2], full_edge, log_records[-1]))

    mont_small = montage(planes, cols=4, max_edge_px=8192)
    assert mont_small.shape[:2] == (300, 1200), mont_small.shape
    print("PASS montage below max_edge_px is untouched: %s" % (mont_small.shape[:2],))

    # --- 5b. quiver_overlay draws arrows and colours them by dz ------------
    base_rgb = np.zeros((110, 110, 3), dtype=np.uint8)
    qy, qx = np.meshgrid(np.arange(110), np.arange(110), indexing="ij")
    qdx = 3.0 * np.sin(2 * np.pi * qy / 110.0)
    qdy = 3.0 * np.cos(2 * np.pi * qx / 110.0)
    qdz = 3.0 * np.sin(2 * np.pi * (qy + qx) / 220.0)
    quiver_rgb = quiver_overlay(base_rgb, qdx, qdy, dz=qdz, stride=11)
    assert quiver_rgb.shape == base_rgb.shape and quiver_rgb.dtype == np.uint8
    n_drawn = int(np.any(quiver_rgb != 0, axis=2).sum())
    assert n_drawn > 0, "quiver_overlay drew no non-background pixels"
    print("PASS quiver_overlay drew %d non-background pixels over a %s canvas (dz-coloured)" % (n_drawn, base_rgb.shape[:2]))

    # --- 6. sample PNGs to a temp dir ---------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="render_selftest_")
    clean_2d = clean.reshape(100, 100)
    for name, arr in [
        ("gray.png", render_scalar(clean_2d, lo_oracle, hi_oracle, "gray")),
        ("magma.png", render_scalar(clean_2d, lo_oracle, hi_oracle, "magma")),
        ("diverging.png", render_scalar(field, dlo, dhi, "diverging")),
    ]:
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as f:
            f.write(to_png(arr))
        print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))

    # --- 7. timing: render one (1500, 630) plane through render_scalar + to_png
    h, w = 1500, 630
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dx = 10.0 * np.sin(2 * np.pi * yy / h)
    dy = 9.0 * np.cos(2 * np.pi * xx / w)
    dz = 8.0 * np.sin(2 * np.pi * (yy + xx) / (h + w))
    norm = np.sqrt(dx**2 + dy**2 + dz**2).astype(np.float32)
    n_reps = 5
    t0 = time.perf_counter()
    for _ in range(n_reps):
        lo_t, hi_t = contrast_limits(norm)  # subsample=200000 default; norm has 945000 px
        rgb_t = render_scalar(norm, lo_t, hi_t, "magma")
        png_t = to_png(rgb_t)
    elapsed = (time.perf_counter() - t0) / n_reps
    print(
        "TIMING render_scalar+to_png on (%d, %d) plane: %.4f s/call (mean of %d), "
        "contrast_limits subsample=200000 (plane has %d px, subsampled)"
        % (h, w, elapsed, n_reps, norm.size)
    )
    print("selftest OK, %d bytes PNG" % len(png_t))


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python render.py --selftest")
