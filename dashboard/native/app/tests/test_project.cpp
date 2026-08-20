// The C scatter against the pipeline's GPU scatter.
//
// refspace_movie_mem.ome.tif was written by run_F260517_0625_qc.py on a GPU
// through cupyx.scatter_add. This file reprojects the same frame from the same
// inputs on the CPU, applies the same z crop and 2x2 block mean, and compares
// voxel for voxel. Without this comparison the C kernel would be checked only
// by looking at it, and a wrong scatter renders a plausible picture rather
// than an error.
#include "app/project.hpp"
#include "app/run.hpp"
#include "core/npy.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace wrdash;

namespace {

// The moving stack the pipeline scattered, as it saved it: (K, Y, X).
std::vector<float> load_moving(const Run& run, int frame, Channel channel) {
  const Geometry& g = run.geometry();
  std::vector<float> mov(std::size_t(g.mov_z * g.plane_y * g.plane_x));
  const auto& raw = run.series(Artifact::kRawMoving, channel);
  for (int64_t k = 0; k < g.mov_z; ++k) {
    const Plane page = read_tiff_page(raw.path(frame), k);
    REQUIRE(page.height == g.plane_y);
    REQUIRE(page.width == g.plane_x);
    std::copy(page.data.begin(), page.data.end(), mov.begin() + std::size_t(k * g.plane_y * g.plane_x));
  }
  return mov;
}

struct Agreement {
  int64_t both_finite = 0;
  int64_t only_mine = 0;
  int64_t only_stored = 0;
  double max_abs_diff = 0.0;
  double sum_abs_diff = 0.0;
  // A mean alone hides its own outliers. These count how many voxels carry
  // the difference, which is what distinguishes "a few unstable voxels" from
  // "a small systematic offset everywhere".
  int64_t over_1e_2 = 0;
  int64_t over_1 = 0;
  double worst_mine = 0.0, worst_stored = 0.0;
  double sum_abs_stored = 0.0;

  // The mean difference means nothing without the scale it sits on.
  double mean_abs_stored() const {
    return both_finite ? sum_abs_stored / double(both_finite) : 0.0;
  }

  double mean_abs_diff() const {
    return both_finite ? sum_abs_diff / double(both_finite) : 0.0;
  }
};

Agreement compare(const Volume& mine, const std::vector<float>& stored) {
  REQUIRE(std::size_t(mine.numel()) == stored.size());
  Agreement a;
  for (std::size_t i = 0; i < stored.size(); ++i) {
    const bool f_mine = std::isfinite(mine.data[i]);
    const bool f_stored = std::isfinite(stored[i]);
    if (f_mine && f_stored) {
      ++a.both_finite;
      const double d = std::abs(double(mine.data[i]) - double(stored[i]));
      a.sum_abs_diff += d;
      a.max_abs_diff = std::max(a.max_abs_diff, d);
      if (d > 1e-2) ++a.over_1e_2;
      if (d > 1.0) ++a.over_1;
      if (d >= a.max_abs_diff) {
        a.worst_mine = double(mine.data[i]);
        a.worst_stored = double(stored[i]);
      }
      a.sum_abs_stored += std::abs(double(stored[i]));
    } else if (f_mine) {
      ++a.only_mine;
    } else if (f_stored) {
      ++a.only_stored;
    }
  }
  return a;
}

// One movie timepoint, (n_z, height, width), read page by page.
std::vector<float> load_movie_timepoint(const MovieDescriptor& movie, Channel channel, int64_t t) {
  std::vector<float> out(std::size_t(movie.n_z * movie.height * movie.width));
  for (int64_t z = 0; z < movie.n_z; ++z) {
    const Plane page = read_tiff_page(movie.files.at(channel), movie.page_for(t, z));
    std::copy(page.data.begin(), page.data.end(),
              out.begin() + std::size_t(z * movie.height * movie.width));
  }
  return out;
}

}  // namespace

TEST_CASE("the C scatter reproduces the pipeline's GPU scatter") {
  const Run run = Run::open(WRDASH_RUN_100T);
  const Geometry& g = run.geometry();
  const MovieDescriptor& movie = run.movie();
  REQUIRE(movie.present);

  // Frame 0 is both a projectable frame and movie timepoint 0. The movie is
  // written for every forward-loop frame, so movie index and frame number
  // coincide only where the frame number is the timepoint — true at 0.
  const int frame = 0;
  REQUIRE(std::find(run.projectable_frames().begin(), run.projectable_frames().end(), frame) !=
          run.projectable_frames().end());

  const auto phase = dashcore::npy::load<float>(run.series(Artifact::kPhaseNew).path(frame), 4);
  REQUIRE(phase.shape[0] == g.plane_x);
  REQUIRE(phase.shape[1] == g.plane_y);
  REQUIRE(phase.shape[2] == g.mov_z);

  ProjectionSettings settings;
  settings.upsample_factor = run.projection_params().upsample_factor;  // 2
  REQUIRE(settings.upsample_factor == 2);

  SUBCASE("membrane channel, bilinear values") {
    const auto mov = load_moving(run, frame, Channel::kMem);
    const Volume full =
        scatter_to_refspace(phase.data.data(), g.plane_x, g.plane_y, g.mov_z, mov.data(), g.ref_z,
                            g.ref_y, g.ref_x, ValueInterp::kBilinear, settings);
    CHECK(full.z == g.ref_z);
    CHECK(full.y == g.ref_y);
    CHECK(full.x == g.ref_x);

    const Volume tile = reduce_blockwise(full, movie.z0_cropped_frame, movie.z1_cropped_frame,
                                         movie.downsample_xy);
    CHECK(tile.z == movie.n_z);
    CHECK(tile.y == movie.height);
    CHECK(tile.x == movie.width);

    const auto stored = load_movie_timepoint(movie, Channel::kMem, 0);
    const Agreement a = compare(tile, stored);

    MESSAGE("mem: both finite ", a.both_finite, ", only mine ", a.only_mine, ", only stored ",
            a.only_stored, ", mean |diff| ", a.mean_abs_diff(), ", max |diff| ", a.max_abs_diff,
            ", voxels >1e-2 ", a.over_1e_2, ", >1 ", a.over_1,
            "; mean |stored| ", a.mean_abs_stored(), "; worst voxel mine=", a.worst_mine,
            " stored=", a.worst_stored);

    // The overlap must be real, not two mostly-NaN volumes agreeing on NaN.
    CHECK(a.both_finite > 1000000);
    // Occupancy must agree: a scatter with the wrong sampling grid or the
    // wrong bounds test lands samples in different voxels, which shows up
    // here long before the values do.
    const double occupancy_mismatch =
        double(a.only_mine + a.only_stored) / double(a.both_finite);
    CHECK(occupancy_mismatch < 0.01);
    // float32 accumulation in a different order on a different device, so
    // exact equality is not the bar. Measured: mean |diff| 0.0032 against a
    // mean |value| of 1080, i.e. 3e-6 relative. The bound is relative to the
    // data's own scale so it does not quietly loosen if the intensities change.
    CHECK(a.mean_abs_diff() < 1e-3 * a.mean_abs_stored());
    // The tail, bounded separately: a mean this small can still hide a
    // spreading disagreement. Measured 71 voxels of 2.06 M above 1.0 absolute;
    // the mechanism behind those is not established, and a regression that
    // widened the tail while keeping the mean low would show up here.
    CHECK(a.over_1 < 500);
  }

  SUBCASE("sparse channel, nearest values") {
    const auto mov = load_moving(run, frame, Channel::kSparseCell);
    const Volume full =
        scatter_to_refspace(phase.data.data(), g.plane_x, g.plane_y, g.mov_z, mov.data(), g.ref_z,
                            g.ref_y, g.ref_x, ValueInterp::kNearest, settings);
    const Volume tile = reduce_blockwise(full, movie.z0_cropped_frame, movie.z1_cropped_frame,
                                         movie.downsample_xy);
    const auto stored = load_movie_timepoint(movie, Channel::kSparseCell, 0);
    const Agreement a = compare(tile, stored);

    MESSAGE("sparse: both finite ", a.both_finite, ", only mine ", a.only_mine, ", only stored ",
            a.only_stored, ", mean |diff| ", a.mean_abs_diff(), ", max |diff| ", a.max_abs_diff,
            ", voxels >1e-2 ", a.over_1e_2, ", >1 ", a.over_1,
            "; mean |stored| ", a.mean_abs_stored(), "; worst voxel mine=", a.worst_mine,
            " stored=", a.worst_stored);

    CHECK(a.both_finite > 1000000);
    const double occupancy_mismatch =
        double(a.only_mine + a.only_stored) / double(a.both_finite);
    CHECK(occupancy_mismatch < 0.01);
    // Measured: mean |diff| 0.00042 against a mean |value| of 14.6, i.e. 2.9e-5
    // relative, with 17 voxels of 2.06 M above 1.0 absolute.
    CHECK(a.mean_abs_diff() < 1e-3 * a.mean_abs_stored());
    CHECK(a.over_1 < 500);
  }
}

TEST_CASE("reduce_blockwise refuses a crop it cannot tile") {
  Volume v;
  v.z = 4;
  v.y = 6;
  v.x = 5;  // odd: not divisible by 2
  v.data.assign(std::size_t(v.numel()), 1.0f);

  CHECK_THROWS_AS(reduce_blockwise(v, 0, 4, 2), std::runtime_error);
  CHECK_THROWS_AS(reduce_blockwise(v, 0, 5, 1), std::runtime_error);  // z1 past the end
  CHECK_THROWS_AS(reduce_blockwise(v, 2, 2, 1), std::runtime_error);  // empty range
  CHECK_NOTHROW(reduce_blockwise(v, 0, 4, 1));
}

TEST_CASE("reduce_blockwise averages the finite entries of a block") {
  Volume v;
  v.z = 1;
  v.y = 2;
  v.x = 4;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  //  block 0: 1, 3       block 1: 5, NaN
  //           2, 4                NaN, NaN
  v.data = {1.0f, 3.0f, 5.0f, nan, 2.0f, 4.0f, nan, nan};

  const Volume out = reduce_blockwise(v, 0, 1, 2);
  REQUIRE(out.y == 1);
  REQUIRE(out.x == 2);
  // Mean over the four finite entries, computed by hand: (1+3+2+4)/4.
  CHECK(out.at(0, 0, 0) == doctest::Approx(2.5));
  // One finite entry in the block: the mean is that entry, not a quarter of
  // it, and not zero.
  CHECK(out.at(0, 0, 1) == doctest::Approx(5.0));
}

TEST_CASE("an all-NaN block reduces to NaN, not zero") {
  Volume v;
  v.z = 1;
  v.y = 2;
  v.x = 2;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  v.data = {nan, nan, nan, nan};

  const Volume out = reduce_blockwise(v, 0, 1, 2);
  // Zero would read as a real measurement of no signal.
  CHECK(std::isnan(out.at(0, 0, 0)));
}
