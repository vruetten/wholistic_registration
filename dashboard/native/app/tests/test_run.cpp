// Run-loader tests against two real run directories.
//
// The pair is chosen because they disagree on nearly everything the loader
// has to get right, so an assumption baked into the loader shows up as a
// failure on one of them rather than as a plausible picture on both:
//
//   f260517_0625_qc_v4                 5 frames, every family on {0..4},
//                                      K=20, 220 reference planes, no movie,
//                                      no reference_source_path (this run
//                                      carries refspace_reference_mem.tif)
//   f260517_0625_qc_4slice_100t_movie  three different frame sets, K=4,
//                                      78 reference planes, a movie, a
//                                      reference recorded as path + crop
//
// WRDASH_RUN_V4 and WRDASH_RUN_100T are compiled in by tests/CMakeLists.txt,
// which registers this test only when both directories exist. A machine
// without /nrs does not run a version of this file that passes vacuously —
// it does not run it at all, and says so at configure time.
#include "app/run.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace wrdash;

namespace {

std::vector<int> range(int begin, int end_exclusive) {
  std::vector<int> out(std::size_t(end_exclusive - begin));
  std::iota(out.begin(), out.end(), begin);
  return out;
}

// {0, 10, 20, ..., 90, 99}: the stride families' frame set on the 100t run.
// Written out rather than generated, so the test states the shape it expects
// instead of re-deriving it with the same rule the loader might have used.
const std::vector<int> kStrideFrames = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99};

}  // namespace

TEST_CASE("v4: every family shares one frame set") {
  const Run run = Run::open(WRDASH_RUN_V4);

  const std::vector<int> expected = range(0, 5);
  CHECK(run.frames_for(Artifact::kPhaseNew) == expected);
  CHECK(run.frames_for(Artifact::kMotionCurrent) == expected);
  CHECK(run.frames_for(Artifact::kMovMem) == expected);
  CHECK(run.frames_for(Artifact::kMaskMov) == expected);
  CHECK(run.frames_for(Artifact::kCoverage) == expected);
  CHECK(run.frames_for(Artifact::kRefspace, Channel::kMem) == expected);
  CHECK(run.frames_for(Artifact::kRefspace, Channel::kSparseCell) == expected);
  CHECK(run.frames_for(Artifact::kProjected, Channel::kMem) == expected);
  CHECK(run.frames_for(Artifact::kRawMoving, Channel::kMem) == expected);
  CHECK(run.frames_for(Artifact::kRawMoving, Channel::kSparseCell) == expected);
}

TEST_CASE("v4: geometry comes from ref_shape.npy and the field header") {
  const Run run = Run::open(WRDASH_RUN_V4);
  const Geometry& g = run.geometry();

  CHECK(g.ref_z == 220);
  CHECK(g.ref_y == 1500);
  CHECK(g.ref_x == 630);
  CHECK(g.mov_z == 20);  // K, from phase_new's (630, 1500, 20, 3)
  CHECK(g.plane_y == 1500);
  CHECK(g.plane_x == 630);
}

TEST_CASE("v4: a run predating the recorded reference has neither movie nor source") {
  const Run run = Run::open(WRDASH_RUN_V4);

  CHECK_FALSE(run.movie().present);
  CHECK_FALSE(run.reference().present);
  // Absence is reported, not thrown on: this run holds the reference as a
  // copy in the run directory instead, and is perfectly loadable.
  CHECK(run.projection_params().upsample_factor == 2);
  CHECK(run.projection_params().z_window == doctest::Approx(3.0));
  CHECK(run.projection_params().fill_value == doctest::Approx(-200.0));
}

TEST_CASE("100t: one run holds three different frame sets") {
  const Run run = Run::open(WRDASH_RUN_100T);

  // The stride families. Not arithmetic: 90 is followed by 99, because the
  // final frame is appended to the stride.
  CHECK(run.frames_for(Artifact::kPhaseNew) == kStrideFrames);
  CHECK(run.frames_for(Artifact::kMotionCurrent) == kStrideFrames);
  CHECK(run.frames_for(Artifact::kRefspace, Channel::kMem) == kStrideFrames);
  CHECK(run.frames_for(Artifact::kRefspace, Channel::kSparseCell) == kStrideFrames);

  // The every-frame families.
  const std::vector<int> all = range(0, 100);
  CHECK(run.frames_for(Artifact::kMaskMov) == all);
  CHECK(run.frames_for(Artifact::kCoverage) == all);
  CHECK(run.frames_for(Artifact::kProjected, Channel::kMem) == all);
  CHECK(run.frames_for(Artifact::kRawMoving, Channel::kMem) == all);
  CHECK(run.frames_for(Artifact::kRawMoving, Channel::kSparseCell) == all);

  // The absent family. The directory it would live in (diagnostics/) exists,
  // so "no frames" here is genuinely "this run wrote none", not "the loader
  // looked in the wrong place".
  CHECK(run.frames_for(Artifact::kMovMem).empty());
  CHECK(run.series(Artifact::kMovMem).dir_exists);

  // The three sets really are three, not one read three times.
  CHECK(run.frames_for(Artifact::kPhaseNew).size() == 11);
  CHECK(run.frames_for(Artifact::kRawMoving, Channel::kMem).size() == 100);
  CHECK(run.frames_for(Artifact::kPhaseNew) != run.frames_for(Artifact::kRawMoving));
}

TEST_CASE("100t: geometry follows the 4-slice, 78-plane configuration") {
  const Run run = Run::open(WRDASH_RUN_100T);
  const Geometry& g = run.geometry();

  CHECK(g.ref_z == 78);
  CHECK(g.ref_y == 1500);
  CHECK(g.ref_x == 630);
  CHECK(g.mov_z == 4);  // K=4: MOV_SLICES "8,9,10,11"
  CHECK(g.plane_y == 1500);
  CHECK(g.plane_x == 630);
}

TEST_CASE("100t: the reference is a path and a crop, not a copy in the run") {
  const Run run = Run::open(WRDASH_RUN_100T);
  const ReferenceSource& ref = run.reference();

  REQUIRE(ref.present);
  REQUIRE(ref.path_exists);
  CHECK(ref.z0 == 165);
  CHECK(ref.z1 == 243);
  CHECK(ref.channel_mem == 1);
  CHECK(ref.channel_sparse == 0);

  // Read off the source file: ZCYX = (361, 2, 1500, 630), so 722 pages.
  CHECK(ref.n_z == 361);
  CHECK(ref.n_channels == 2);
  CHECK(ref.height == 1500);
  CHECK(ref.width == 630);

  // page = z * n_channels + channel.
  CHECK(ref.page_for(175, 1) == 351);
  CHECK(ref.page_for(0, 0) == 0);
  CHECK(ref.channel_for(Channel::kMem) == 1);
  CHECK(ref.channel_for(Channel::kSparseCell) == 0);

  // Out of range throws rather than landing on a real plane of the wrong
  // channel, which is what an unchecked z * 2 + c would do.
  CHECK_THROWS(ref.page_for(361, 0));
  CHECK_THROWS(ref.page_for(-1, 0));
  CHECK_THROWS(ref.page_for(175, 2));
}

TEST_CASE("100t: the movie's z crop and lateral reduction both map to the reference") {
  const Run run = Run::open(WRDASH_RUN_100T);
  const MovieDescriptor& movie = run.movie();

  REQUIRE(movie.present);
  CHECK(movie.n_t == 100);
  CHECK(movie.n_z == 55);
  CHECK(movie.height == 750);  // 1500 reduced by 2
  CHECK(movie.width == 315);   // 630 reduced by 2
  CHECK(movie.downsample_xy == 2);
  CHECK(movie.crop_base == 165);
  CHECK(movie.z0_cropped_frame == 10);
  CHECK(movie.z1_cropped_frame == 65);

  // Both channels were written and both were found.
  CHECK(movie.files.count(Channel::kMem) == 1);
  CHECK(movie.files.count(Channel::kSparseCell) == 1);

  // Pages are T-major.
  CHECK(movie.page_for(0, 0) == 0);
  CHECK(movie.page_for(0, 54) == 54);
  CHECK(movie.page_for(1, 0) == 55);
  CHECK(movie.page_for(99, 54) == 5499);
  CHECK_THROWS(movie.page_for(100, 0));
  CHECK_THROWS(movie.page_for(0, 55));

  // Movie plane 0 sits on source plane 165 + 10 = 175, which is what
  // projection_params records as movie_z_crop.z0_source_file. The two are
  // derived differently — this from crop_base + z0_cropped_frame, that
  // written by the pipeline — so agreement is a real check.
  CHECK(movie.reference_plane_for(0) == 175);
  CHECK(movie.reference_plane_for(54) == 229);
  CHECK_THROWS(movie.reference_plane_for(55));

  // The movie's lateral grid is the reference's, reduced. A viewer that
  // overlays without applying the reduction would be off by a factor of 2.
  CHECK(movie.height * movie.downsample_xy == run.geometry().ref_y);
  CHECK(movie.width * movie.downsample_xy == run.geometry().ref_x);
  // And its z span sits inside the cropped reference.
  CHECK(movie.z1_cropped_frame - movie.z0_cropped_frame == movie.n_z);
  CHECK(movie.z1_cropped_frame <= run.geometry().ref_z);
}

TEST_CASE("100t: projectable frames are the field's, not the raw stack's") {
  const Run run = Run::open(WRDASH_RUN_100T);

  // Raw moving intensity exists for all 100 frames, the displacement field
  // for 11. A projector driven by the raw frame list would happily project
  // frame 7 with some other frame's field.
  CHECK(run.projectable_frames(Channel::kMem) == kStrideFrames);
  CHECK(run.projectable_frames(Channel::kSparseCell) == kStrideFrames);
  CHECK(run.projectable_frames_both() == kStrideFrames);
  CHECK(run.projectable_frames().size() <
        run.frames_for(Artifact::kRawMoving, Channel::kMem).size());
}

TEST_CASE("100t: metrics tables are discovered, not named") {
  const Run run = Run::open(WRDASH_RUN_100T);

  // This run writes a fifth table the v4 run does not, so a hard-coded list
  // of four would miss it.
  std::vector<std::string> names;
  for (const auto& table : run.metrics()) names.push_back(table.name);
  std::sort(names.begin(), names.end());

  CHECK(names == std::vector<std::string>{"errors_membrane", "errors_sparse", "hole_summary",
                                          "refspace_movie_summary", "refspace_summary"});

  const Run v4 = Run::open(WRDASH_RUN_V4);
  CHECK(v4.metrics().size() == 4);
}

TEST_CASE("a frame outside the set throws, naming what is present") {
  const Run run = Run::open(WRDASH_RUN_100T);
  const Series& phase = run.series(Artifact::kPhaseNew);

  CHECK(phase.has(90));
  CHECK_FALSE(phase.has(91));
  // 91 lies between two frames that exist, so a loader that interpolated or
  // rounded would return frame 90's field for it.
  CHECK_THROWS_AS(phase.path(91), std::runtime_error);
}

TEST_CASE("asking a channel-free family for a channel throws") {
  const Run run = Run::open(WRDASH_RUN_100T);

  // phase_new is written once, not once per channel. Silently returning the
  // one file for a sparse-cell request would make an overlay of "the sparse
  // field" show the membrane field.
  CHECK_THROWS_AS(run.series(Artifact::kPhaseNew, Channel::kSparseCell), std::runtime_error);
  CHECK_NOTHROW(run.series(Artifact::kPhaseNew, Channel::kMem));
  CHECK_NOTHROW(run.series(Artifact::kRefspace, Channel::kSparseCell));
}

TEST_CASE("a directory that is not a run throws at open, not at first access") {
  CHECK_THROWS_AS(Run::open("/tmp"), std::runtime_error);
}
