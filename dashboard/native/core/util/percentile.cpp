#include "core/util/percentile.hpp"

#include <algorithm>
#include <cmath>

namespace dashcore {

PercentileRange percentile_range(const std::vector<float>& data, float lo_pct, float hi_pct,
                                 std::size_t max_samples) {
  std::vector<float> buf;
  buf.reserve(std::min(data.size(), max_samples));
  if (data.size() <= max_samples) {
    for (float v : data) {
      if (std::isfinite(v)) buf.push_back(v);
    }
  } else {
    // Deterministic stride sample: cheap and reproducible frame-to-frame,
    // unlike a random draw that would jitter the computed range.
    const std::size_t stride = std::max<std::size_t>(1, data.size() / max_samples);
    for (std::size_t i = 0; i < data.size(); i += stride) {
      if (std::isfinite(data[i])) buf.push_back(data[i]);
    }
  }
  if (buf.empty()) return {0.0f, 1.0f};
  if (buf.size() == 1) return {buf[0], buf[0] + 1.0f};

  const std::size_t n_lo = std::size_t(std::clamp(lo_pct, 0.0f, 1.0f) * float(buf.size() - 1));
  const std::size_t n_hi =
      std::max(n_lo, std::size_t(std::clamp(hi_pct, 0.0f, 1.0f) * float(buf.size() - 1)));
  std::nth_element(buf.begin(), buf.begin() + std::ptrdiff_t(n_lo), buf.end());
  const float lo = buf[n_lo];
  float hi = lo;
  if (n_hi > n_lo) {
    std::nth_element(buf.begin() + std::ptrdiff_t(n_lo) + 1, buf.begin() + std::ptrdiff_t(n_hi),
                     buf.end());
    hi = buf[n_hi];
  }
  return (hi > lo) ? PercentileRange{lo, hi} : PercentileRange{lo, lo + 1.0f};
}

}  // namespace dashcore
