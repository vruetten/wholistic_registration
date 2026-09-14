"""
Thin runner: compare the three reference-plane projectors (max-splat B,
weighted-avg B, inverse-gather C) on saved data, or on a synthetic phantom.

Usage
-----
Synthetic (default, runnable with no data -- good for a sanity check):

    python -m wholistic_registration.tests.compare_projectors --out-dir /tmp/proj_cmp

Real saved data:

    python -m wholistic_registration.tests.compare_projectors \
        --phase /path/phase_new.npy \
        --moving /path/mov_mem.npy \
        --ref /path/ref_mem.npy \
        --target-z /path/fixed_target_z.npy \
        --frames 5 160 \
        --out-dir /path/out

Expected array shapes
---------------------
    phase   : (Xmov, Ymov, K, 3)            single frame, or
              (T, Xmov, Ymov, K, 3)         stack (index with --frames)
    moving  : (K, Ymov, Xmov)               single frame, or
              (T, K, Ymov, Xmov)            stack -- transposed to (Xmov,Ymov,K)
    ref     : (Xref, Yref, Zref)            with --ref-order xyz (default), or
              (Zref, Yref, Xref)            with --ref-order zyx (shape only)
    target-z: (Nplanes,) z planes; if omitted, inferred from phase z-range.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wholistic_registration.utils import projector_compare as pc


# --------------------------------------------------------------------------
# Synthetic phantom
# --------------------------------------------------------------------------
def make_synthetic_frame(drift=0.0, stretch=0.0, seed=0):
    """
    Build one synthetic frame: a textured moving stack and a smooth forward
    map into a larger reference grid.

    drift   : rigid XY offset added to the map (large -> pushes signal off the
              reference grid -> low coverage).
    stretch : local magnification (>0 spreads moving pixels apart in reference).
    """
    rng = np.random.default_rng(seed)
    Xmov, Ymov, K = 128, 140, 5
    Xref, Yref, Zref = 180, 200, 60

    ii, jj = np.meshgrid(np.arange(Xmov), np.arange(Ymov), indexing="ij")
    ii = ii.astype(np.float32)
    jj = jj.astype(np.float32)

    # High-frequency texture + a few sharp blobs, so max-vs-average and
    # gather-vs-scatter sharpness differences are visible.
    texture = (
        80.0
        + 40.0 * np.sin(ii / 3.0) * np.cos(jj / 4.0)
        + 25.0 * np.sin((ii + jj) / 2.5)
    ).astype(np.float32)
    blobs = np.zeros_like(texture)
    for _ in range(40):
        cx, cy = rng.uniform(0, Xmov), rng.uniform(0, Ymov)
        rr = (ii - cx) ** 2 + (jj - cy) ** 2
        blobs += 200.0 * np.exp(-rr / (2 * (1.5 ** 2)))
    moving2d = texture + blobs + rng.normal(0, 3, texture.shape).astype(np.float32)

    moving = np.stack([moving2d * (1.0 + 0.1 * k) for k in range(K)], axis=-1)
    moving = moving.astype(np.float32)  # (Xmov, Ymov, K)

    # Forward map: smooth in-plane warp + rigid drift, planes spaced in z.
    z_centers = np.linspace(10, Zref - 10, K).astype(np.float32)
    warp_x = 2.0 * np.sin(jj / 20.0)
    warp_y = 2.0 * np.cos(ii / 22.0)
    scale = 1.0 + stretch
    coords = np.zeros((Xmov, Ymov, K, 3), np.float32)
    for k in range(K):
        coords[:, :, k, 0] = scale * ii + warp_x + 10.0 + drift
        coords[:, :, k, 1] = scale * jj + warp_y + 10.0 + drift
        coords[:, :, k, 2] = z_centers[k]

    ref_shape = (Xref, Yref, Zref)
    return coords, moving, z_centers, ref_shape


# --------------------------------------------------------------------------
# Loading real data
# --------------------------------------------------------------------------
def _load_frame(phase, moving, frame_idx):
    """Slice a single frame out of phase/moving that may be stacked over T."""
    if phase.ndim == 5:
        ph = phase[frame_idx]
    elif phase.ndim == 4:
        ph = phase
    else:
        raise ValueError(f"phase must be 4D or 5D, got {phase.shape}")

    if moving.ndim == 4:
        mv = moving[frame_idx]
    elif moving.ndim == 3:
        mv = moving
    else:
        raise ValueError(f"moving must be 3D or 4D, got {moving.shape}")

    # moving comes in as (K, Y, X) -> (X, Y, K)
    mv = np.transpose(mv, (2, 1, 0)).astype(np.float32)
    return np.asarray(ph, np.float32), mv


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def save_panels(results, out_dir, label, plane_idx=None):
    methods = results["methods"]
    names = list(methods.keys())
    nplanes = methods[names[0]]["out"].shape[0]
    if plane_idx is None:
        plane_idx = nplanes // 2

    # Shared intensity scale from the weighted-B output (most faithful).
    ref_vol = methods.get("weighted_B", methods[names[0]])["out"][plane_idx]
    finite = ref_vol[np.isfinite(ref_vol) & (ref_vol != 0)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmin = 0.0

    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 9))
    for col, name in enumerate(names):
        m = methods[name]
        ax = axes[0, col]
        ax.imshow(m["out"][plane_idx], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{name}\nhole={m['hole_fraction']:.3f}  {m['runtime_s']*1e3:.0f} ms"
        )
        ax.axis("off")

        axc = axes[1, col]
        axc.imshow(m["covered"][plane_idx], cmap="magma", vmin=0, vmax=1)
        axc.set_title(f"{name} coverage")
        axc.axis("off")

    fig.suptitle(f"{label}  (plane {plane_idx}/{nplanes})", fontsize=14)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}_plane{plane_idx}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _warmup(args):
    """Trigger CUDA kernel JIT once so reported runtimes are not warmup-bound."""
    coords, moving, tz, ref_shape = make_synthetic_frame(drift=0.0, stretch=0.0)
    pc.compare_projectors(
        coords_ref_xyk_xyz=coords,
        ref_volume=np.zeros(ref_shape, np.float32),
        target_z_planes=tz,
        values_xyk=moving,
        z_window=args.z_window,
        z_weight_mode=args.z_weight_mode,
        downsample_xy=args.downsample_xy,
        xy_extra_radius=args.xy_extra_radius,
    )


def run_one(label, coords, moving, target_z, ref_shape, args, ref_order):
    ref_volume = np.zeros(ref_shape, np.float32)  # shape carrier only
    results = pc.compare_projectors(
        coords_ref_xyk_xyz=coords,
        ref_volume=ref_volume,
        target_z_planes=target_z,
        values_xyk=moving,
        ref_volume_order=ref_order,
        z_window=args.z_window,
        z_weight_mode=args.z_weight_mode,
        downsample_xy=args.downsample_xy,
        xy_extra_radius=args.xy_extra_radius,
    )
    print(f"\n===== {label} =====")
    print(pc.format_metrics_table(results))
    png = save_panels(results, args.out_dir, label)
    print(f"[saved] {png}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", help="dir of make_phase_new.py output "
                   "(phase_new_f{i}.npy, mov_mem_f{i}.npy, ref_shape.npy, "
                   "fixed_target_z.npy); use with --frames")
    p.add_argument("--phase", help="path to phase_new .npy")
    p.add_argument("--moving", help="path to moving stack .npy (K,Y,X) or (T,K,Y,X)")
    p.add_argument("--ref", help="path to reference volume .npy (shape only)")
    p.add_argument("--ref-shape", help="path to a (3,) .npy holding (Xref,Yref,Zref); "
                   "use instead of --ref to avoid loading a large volume")
    p.add_argument("--target-z", help="path to fixed_target_z .npy")
    p.add_argument("--ref-order", default="xyz", choices=["xyz", "zyx"])
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="frame indices to test (for stacked phase/moving)")
    p.add_argument("--z-window", type=float, default=1.5)
    p.add_argument("--z-weight-mode", default="gaussian",
                   choices=["hard", "triangular", "gaussian"])
    p.add_argument("--downsample-xy", type=int, default=1)
    p.add_argument("--xy-extra-radius", type=int, default=0)
    p.add_argument("--out-dir", default="/tmp/projector_compare")
    args = p.parse_args()

    _warmup(args)

    if args.data_dir is not None:
        d = args.data_dir
        ref_shape = tuple(int(v) for v in np.load(os.path.join(d, "ref_shape.npy")))
        target_z = np.load(os.path.join(d, "fixed_target_z.npy")).astype(np.float32)
        frames = args.frames if args.frames is not None else [5]
        for fi in frames:
            phase = np.load(os.path.join(d, f"phase_new_f{fi}.npy")).astype(np.float32)
            mv = np.load(os.path.join(d, f"mov_mem_f{fi}.npy"))  # (K,Y,X)
            mv = np.transpose(mv, (2, 1, 0)).astype(np.float32)  # (X,Y,K)
            run_one(f"frame_{fi}", phase, mv, target_z, ref_shape, args, args.ref_order)
        return

    if args.phase is None:
        print("[mode] no --phase given: running SYNTHETIC phantom "
              "(easy frame + low-coverage frame).")
        # easy: small drift, no stretch; hard: large drift + stretch -> low coverage
        coords, moving, tz, ref_shape = make_synthetic_frame(drift=0.0, stretch=0.0)
        run_one("synthetic_easy", coords, moving, tz, ref_shape, args, "xyz")
        coords, moving, tz, ref_shape = make_synthetic_frame(
            drift=55.0, stretch=0.25, seed=1
        )
        run_one("synthetic_lowcov", coords, moving, tz, ref_shape, args, "xyz")
        return

    phase = np.load(args.phase, allow_pickle=False)
    moving = np.load(args.moving, allow_pickle=False)
    if args.ref_shape is not None:
        ref_shape = tuple(int(v) for v in np.load(args.ref_shape))
    elif args.ref is not None:
        ref_shape = np.load(args.ref, mmap_mode="r").shape
    else:
        raise SystemExit("--ref or --ref-shape is required with --phase")

    if args.target_z is not None:
        target_z = np.load(args.target_z, allow_pickle=False).astype(np.float32)
    else:
        zmid = float(np.nanmedian(phase[..., 2]))
        target_z = np.array([zmid], np.float32)
        print(f"[target-z] none given; using median phase z = {zmid:.2f}")

    frames = args.frames if args.frames is not None else [0]
    for fi in frames:
        coords, mv = _load_frame(phase, moving, fi)
        run_one(f"frame_{fi}", coords, mv, target_z, ref_shape, args, args.ref_order)


if __name__ == "__main__":
    main()
