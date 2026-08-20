#include "app/project.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace wrdash {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// The pipeline's _make_xy_sampling_grid: linspace(0, N-1, N*factor), endpoints
// included. For factor == 1 this is the identity on integer positions.
inline double sample_position(int64_t index, int64_t n, int64_t n_up) {
  if (n_up <= 1) return 0.0;
  return double(index) * (double(n) - 1.0) / (double(n_up) - 1.0);
}

// map_coordinates(order=1, mode='nearest') on a 2-D slice of a strided array.
// `stride_x` and `stride_y` step one element along each axis; `base` points at
// element (0, 0) of the slice.
inline double bilinear(const float* base, int64_t stride_x, int64_t stride_y, int64_t X, int64_t Y,
                       double xs, double ys) {
  int64_t x0 = int64_t(std::floor(xs));
  int64_t y0 = int64_t(std::floor(ys));
  // mode='nearest' clamps; xs/ys never leave [0, N-1] here, so the clamp only
  // bites exactly at the far edge, where x0 == N-1 and the weight is zero.
  x0 = std::clamp<int64_t>(x0, 0, X - 1);
  y0 = std::clamp<int64_t>(y0, 0, Y - 1);
  const int64_t x1 = std::min<int64_t>(x0 + 1, X - 1);
  const int64_t y1 = std::min<int64_t>(y0 + 1, Y - 1);
  const double fx = xs - double(x0);
  const double fy = ys - double(y0);

  const double v00 = double(base[x0 * stride_x + y0 * stride_y]);
  const double v10 = double(base[x1 * stride_x + y0 * stride_y]);
  const double v01 = double(base[x0 * stride_x + y1 * stride_y]);
  const double v11 = double(base[x1 * stride_x + y1 * stride_y]);

  return v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy +
         v11 * fx * fy;
}

// map_coordinates(order=0): round to the nearest sample.
inline double nearest(const float* base, int64_t stride_x, int64_t stride_y, int64_t X, int64_t Y,
                      double xs, double ys) {
  const int64_t xi = std::clamp<int64_t>(int64_t(std::llround(xs)), 0, X - 1);
  const int64_t yi = std::clamp<int64_t>(int64_t(std::llround(ys)), 0, Y - 1);
  return double(base[xi * stride_x + yi * stride_y]);
}

// Scatters the supersampled rows [yu_begin, yu_end) of slice k.
void scatter_rows(const float* phase, int64_t X, int64_t Y, int64_t K, const float* mov,
                  int64_t ref_z, int64_t ref_y, int64_t ref_x, ValueInterp interp, int64_t factor,
                  int64_t k, int64_t yu_begin, int64_t yu_end, float* sum_val, float* sum_w) {
  const int64_t Xup = X * factor;
  const int64_t Yup = Y * factor;

  // phase is (X, Y, K, 3): stepping x moves Y*K*3, stepping y moves K*3.
  const int64_t phase_stride_x = Y * K * 3;
  const int64_t phase_stride_y = K * 3;
  const float* phase_k = phase + k * 3;

  // mov is (K, Y, X): stepping x moves 1, stepping y moves X.
  const float* mov_k = mov + k * Y * X;

  for (int64_t yu = yu_begin; yu < yu_end; ++yu) {
    const double ys = sample_position(yu, Y, Yup);
    for (int64_t xu = 0; xu < Xup; ++xu) {
      const double xs = sample_position(xu, X, Xup);

      const double cx = bilinear(phase_k + 0, phase_stride_x, phase_stride_y, X, Y, xs, ys);
      const double cy = bilinear(phase_k + 1, phase_stride_x, phase_stride_y, X, Y, xs, ys);
      const double cz = bilinear(phase_k + 2, phase_stride_x, phase_stride_y, X, Y, xs, ys);
      const double value = interp == ValueInterp::kBilinear
                               ? bilinear(mov_k, 1, X, X, Y, xs, ys)
                               : nearest(mov_k, 1, X, X, Y, xs, ys);

      // The pipeline's `valid`: finite, and inside the grid inclusive of the
      // far edge. A sample exactly at N-1 contributes with its upper corner
      // clamped onto itself.
      if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz) ||
          !std::isfinite(value)) {
        continue;
      }
      if (cx < 0.0 || cx > double(ref_x - 1) || cy < 0.0 || cy > double(ref_y - 1) || cz < 0.0 ||
          cz > double(ref_z - 1)) {
        continue;
      }

      const int64_t x0 = int64_t(std::floor(cx));
      const int64_t y0 = int64_t(std::floor(cy));
      const int64_t z0 = int64_t(std::floor(cz));
      const int64_t x1 = std::min<int64_t>(x0 + 1, ref_x - 1);
      const int64_t y1 = std::min<int64_t>(y0 + 1, ref_y - 1);
      const int64_t z1 = std::min<int64_t>(z0 + 1, ref_z - 1);
      const double fx = cx - double(x0);
      const double fy = cy - double(y0);
      const double fz = cz - double(z0);

      const int64_t zs[2] = {z0, z1};
      const int64_t ys_i[2] = {y0, y1};
      const int64_t xs_i[2] = {x0, x1};
      const double wz[2] = {1.0 - fz, fz};
      const double wy[2] = {1.0 - fy, fy};
      const double wx[2] = {1.0 - fx, fx};

      for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const double w = wz[dz] * wy[dy] * wx[dx];
            const int64_t idx = (zs[dz] * ref_y + ys_i[dy]) * ref_x + xs_i[dx];
            sum_val[idx] += float(w * value);
            sum_w[idx] += float(w);
          }
        }
      }
    }
  }
}

}  // namespace

Volume scatter_to_refspace(const float* phase, int64_t X, int64_t Y, int64_t K, const float* mov,
                           int64_t ref_z, int64_t ref_y, int64_t ref_x, ValueInterp interp,
                           const ProjectionSettings& settings) {
  const int64_t factor = settings.upsample_factor;
  if (factor < 1) {
    throw std::runtime_error("upsample_factor must be >= 1, got " + std::to_string(factor));
  }

  const std::size_t n_vox = std::size_t(ref_z * ref_y * ref_x);
  std::vector<float> sum_val(n_vox, 0.0f);
  std::vector<float> sum_w(n_vox, 0.0f);

  // Threads partition the supersampled rows, and two threads can splat into
  // the same voxel, so each owns a private pair of accumulators that are
  // summed at the end. Sharing one pair and racing on += would lose writes;
  // the losses are small, scattered and invisible in the picture, which is
  // exactly the kind of error an oracle comparison would then be too loose to
  // catch.
  int threads = settings.threads > 0 ? settings.threads : int(std::thread::hardware_concurrency());
  threads = std::max(1, threads);
  // One private 590 MB pair per thread at full reference size is not worth it
  // past a handful, and the kernel is memory-bound anyway.
  threads = std::min(threads, 8);
  const int64_t Yup = Y * factor;
  threads = int(std::min<int64_t>(threads, std::max<int64_t>(1, Yup)));

  if (threads == 1) {
    for (int64_t k = 0; k < K; ++k) {
      scatter_rows(phase, X, Y, K, mov, ref_z, ref_y, ref_x, interp, factor, k, 0, Yup,
                   sum_val.data(), sum_w.data());
    }
  } else {
    const auto n_parts = static_cast<std::size_t>(threads);
    std::vector<std::vector<float>> part_val(n_parts);
    std::vector<std::vector<float>> part_w(n_parts);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      part_val[std::size_t(t)].assign(n_vox, 0.0f);
      part_w[std::size_t(t)].assign(n_vox, 0.0f);
      const int64_t begin = Yup * t / threads;
      const int64_t end = Yup * (t + 1) / threads;
      pool.emplace_back([&, t, begin, end] {
        for (int64_t k = 0; k < K; ++k) {
          scatter_rows(phase, X, Y, K, mov, ref_z, ref_y, ref_x, interp, factor, k, begin, end,
                       part_val[std::size_t(t)].data(), part_w[std::size_t(t)].data());
        }
      });
    }
    for (auto& th : pool) th.join();
    for (int t = 0; t < threads; ++t) {
      const float* pv = part_val[std::size_t(t)].data();
      const float* pw = part_w[std::size_t(t)].data();
      for (std::size_t i = 0; i < n_vox; ++i) {
        sum_val[i] += pv[i];
        sum_w[i] += pw[i];
      }
    }
  }

  Volume out;
  out.z = ref_z;
  out.y = ref_y;
  out.x = ref_x;
  out.data.resize(n_vox);
  const float eps = float(settings.eps);
  for (std::size_t i = 0; i < n_vox; ++i) {
    out.data[i] = sum_w[i] > eps ? sum_val[i] / std::max(sum_w[i], eps) : kNaN;
  }
  return out;
}

Volume reduce_blockwise(const Volume& volume, int64_t z0, int64_t z1, int64_t factor) {
  if (factor < 1) {
    throw std::runtime_error("reduce factor must be >= 1, got " + std::to_string(factor));
  }
  if (z0 < 0 || z1 > volume.z || z0 >= z1) {
    throw std::runtime_error("z crop [" + std::to_string(z0) + ", " + std::to_string(z1) +
                             ") is not a non-empty range inside [0, " + std::to_string(volume.z) +
                             ")");
  }
  if (volume.y % factor != 0 || volume.x % factor != 0) {
    throw std::runtime_error("block reduce needs Y and X divisible by " + std::to_string(factor) +
                             "; got Y=" + std::to_string(volume.y) +
                             ", X=" + std::to_string(volume.x));
  }

  Volume out;
  out.z = z1 - z0;
  out.y = volume.y / factor;
  out.x = volume.x / factor;
  out.data.resize(std::size_t(out.numel()));

  for (int64_t zi = 0; zi < out.z; ++zi) {
    for (int64_t yi = 0; yi < out.y; ++yi) {
      for (int64_t xi = 0; xi < out.x; ++xi) {
        double total = 0.0;
        int64_t count = 0;
        for (int64_t dy = 0; dy < factor; ++dy) {
          for (int64_t dx = 0; dx < factor; ++dx) {
            const float v = volume.at(z0 + zi, yi * factor + dy, xi * factor + dx);
            if (std::isfinite(v)) {
              total += double(v);
              ++count;
            }
          }
        }
        out.data[std::size_t((zi * out.y + yi) * out.x + xi)] =
            count ? float(total / double(count)) : kNaN;
      }
    }
  }
  return out;
}

}  // namespace wrdash
