// A robust [lo, hi] range over a buffer of scalars, for autoscale contrast.
//
// The true min/max of a scientific array is often a handful of outliers (a
// hot pixel, a degenerate zero-area component), which then dominate a
// linear color scale and wash the interesting mid-range into a narrow band.
// Percentile clipping saturates the tails so the bulk of the data uses the
// full dynamic range — the convention this app follows (see callers) is
// 5th-95th.
#pragma once

#include <cstddef>
#include <vector>

namespace dashcore {

struct PercentileRange {
  float lo = 0.0f;
  float hi = 1.0f;
};

// `lo_pct`/`hi_pct` are in [0, 1]. Non-finite entries are ignored. Samples
// with a fixed stride down to `max_samples` when `data` is larger, so a
// huge buffer doesn't pay for a full sort — a few hundred thousand points
// already give a stable percentile. Returns {0, 1} if no finite values are
// present; widens hi to lo+1 if the two percentiles coincide (degenerate or
// constant data), so a caller never divides by a zero span.
PercentileRange percentile_range(const std::vector<float>& data, float lo_pct, float hi_pct,
                                 std::size_t max_samples = 200'000);

}  // namespace dashcore
