#include "core/util/percentile.hpp"

#include <doctest/doctest.h>

#include <limits>
#include <numeric>
#include <vector>

using dashcore::percentile_range;

TEST_CASE("percentile_range clips the tails of a uniform ramp") {
  std::vector<float> data(100);
  std::iota(data.begin(), data.end(), 0.0f);   // 0..99

  const auto r = percentile_range(data, 0.05f, 0.95f);
  CHECK(r.lo > 0.0f);
  CHECK(r.lo < 15.0f);
  CHECK(r.hi > 85.0f);
  CHECK(r.hi < 99.0f);
}

TEST_CASE("percentile_range(0, 1) reduces to the true min/max") {
  const std::vector<float> data = {3.0f, -1.0f, 7.0f, 2.0f};
  const auto r = percentile_range(data, 0.0f, 1.0f);
  CHECK(r.lo == doctest::Approx(-1.0f));
  CHECK(r.hi == doctest::Approx(7.0f));
}

TEST_CASE("percentile_range ignores non-finite entries") {
  const std::vector<float> data = {1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(), 3.0f};
  const auto r = percentile_range(data, 0.0f, 1.0f);
  CHECK(r.lo == doctest::Approx(1.0f));
  CHECK(r.hi == doctest::Approx(3.0f));
}

TEST_CASE("percentile_range on empty data returns the documented {0, 1} fallback") {
  const auto r = percentile_range({}, 0.05f, 0.95f);
  CHECK(r.lo == doctest::Approx(0.0f));
  CHECK(r.hi == doctest::Approx(1.0f));
}

TEST_CASE("percentile_range on constant data widens hi so span is never zero") {
  const std::vector<float> data(10, 5.0f);
  const auto r = percentile_range(data, 0.05f, 0.95f);
  CHECK(r.lo == doctest::Approx(5.0f));
  CHECK(r.hi > r.lo);
}

TEST_CASE("percentile_range samples a large buffer down to max_samples without crashing") {
  std::vector<float> data(500'000);
  std::iota(data.begin(), data.end(), 0.0f);
  const auto r = percentile_range(data, 0.05f, 0.95f, /*max_samples=*/1000);
  CHECK(r.lo > 0.0f);
  CHECK(r.hi < 500'000.0f);
  CHECK(r.hi > r.lo);
}
