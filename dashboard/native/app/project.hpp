// Scatter of a frame's moving intensity into reference space.
//
// This reproduces `_scatter_trilinear_to_refspace` and its two callers in
// src/wholistic_registration/tests/run_F260517_0625_qc.py, on the CPU, so the
// dashboard can project from source data instead of displaying volumes some
// past job wrote. Reproducing it exactly is what makes the stored
// refspace_movie_*.ome.tif usable as an oracle: a scatter that is merely
// similar draws a plausible picture, and a plausible picture is indistinguish-
// able from a correct one by eye.
//
// The pipeline's semantics, which this file matches point for point:
//
//   * The moving grid is supersampled in XY onto linspace(0, X-1, X*factor) —
//     the endpoints are included, so the step is (X-1)/(X*factor-1) and NOT
//     1/factor. Getting this wrong shifts every sample by up to half a moving
//     pixel, which survives as a plausible-looking registration error.
//   * The displacement field is sampled there bilinearly.
//   * Moving intensity is sampled bilinearly for membrane and by NEAREST
//     neighbour for the sparse-cell channel: order=1 against order=0 in the
//     pipeline's two upsample_values_xy_for_supersurface calls. Bilinear on
//     the sparse channel would invent intermediate brightnesses between
//     isolated cells.
//   * A sample counts when its reference coordinate is finite and inside
//     [0, N-1] on all three axes, INCLUSIVE. The upper corner index is then
//     clamped to N-1 rather than the sample being dropped.
//   * Each sample splats trilinearly over its 2x2x2 neighbourhood into a value
//     accumulator and a weight accumulator.
//   * The result is the weighted mean where total weight exceeds eps=1e-6,
//     and NaN elsewhere. NaN, not zero: an unvisited voxel has no intensity,
//     and zero is a measurement.
//
// No z-window gate. This is the "every finite in-bounds sample contributes"
// output, the one SAVE_REF_SPACE_VOLUME exists for, as distinct from the
// z-window-gated projected_* volumes.
#pragma once

#include <cstdint>
#include <vector>

namespace wrdash {

// Which interpolation the moving intensity gets. The displacement field is
// always bilinear; only the values differ between channels.
enum class ValueInterp {
  kBilinear,  // order=1, the membrane channel
  kNearest,   // order=0, the sparse-cell channel
};

struct ProjectionSettings {
  int64_t upsample_factor = 2;
  double eps = 1e-6;
  // 0 uses every available core. The scatter is memory-bound on the two
  // accumulators, so returns flatten well before the core count.
  int threads = 0;
};

// A reference-space volume, (Z, Y, X) row-major, NaN where nothing landed.
struct Volume {
  int64_t z = 0, y = 0, x = 0;
  std::vector<float> data;

  int64_t numel() const { return z * y * x; }
  float at(int64_t zi, int64_t yi, int64_t xi) const {
    return data[std::size_t((zi * y + yi) * x + xi)];
  }
};

// phase is (X, Y, K, 3), last axis (x_ref, y_ref, z_ref); mov is (K, Y, X).
// Both are the frame's full-resolution arrays — supersampling happens inside,
// so the caller never materialises the (X*f, Y*f, K, 3) intermediate, which is
// 181 MB at f=2.
Volume scatter_to_refspace(const float* phase, int64_t X, int64_t Y, int64_t K, const float* mov,
                           int64_t ref_z, int64_t ref_y, int64_t ref_x, ValueInterp interp,
                           const ProjectionSettings& settings);

// Crop to z in [z0, z1) and reduce Y and X by `factor`, each output voxel the
// mean over the finite entries of its factor x factor block, NaN where the
// whole block is NaN. This is the pipeline's _block_nanmean_2x2, generalised
// off its literal 2x2 reshape.
//
// A block mean rather than subsampling because the scattered volume is mostly
// NaN — about 10% of voxels are occupied — so taking every factor-th sample
// would discard most of the samples that exist.
//
// Throws if the cropped extent is not divisible by `factor`, rather than
// reshaping across block boundaries.
Volume reduce_blockwise(const Volume& volume, int64_t z0, int64_t z1, int64_t factor);

}  // namespace wrdash
