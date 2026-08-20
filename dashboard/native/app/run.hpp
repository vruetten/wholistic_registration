// A QC run directory, as the dashboard sees it.
//
// The one structural claim this file makes: a run has no single frame set.
// Measured on f260517_0625_qc_4slice_100t_movie, one directory holds three:
//
//   phase_new, motion_current, refspace_*   {0, 10, 20, ..., 90, 99}
//   masks_mov, coverage, projected_*, raw_moving_*   {0, 1, ..., 99}
//   mov_mem                                  absent entirely
//
// The stride families are neither contiguous nor arithmetic — the final frame
// is appended to the stride, so 90 is followed by 99. Anything that assumes a
// run-wide frame list, or reconstructs frame numbers from a stride, is wrong
// on this run. Each family is therefore discovered independently by listing
// its directory, and an absent family is a legitimate empty Series rather than
// an error at open time: which families a viewer needs is the viewer's
// business, not the loader's.
//
// Frames are also discovered rather than reconstructed from a filename
// template. The Python loader builds "vol_F260517_{token}_{channel}_{n:06d}"
// with the dataset name baked in; that name is a property of one experiment,
// so here the trailing digit group of whatever files exist is parsed instead.
#pragma once

#include "app/csv_table.hpp"
#include "app/tiff_volume.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace wrdash {

enum class Artifact {
  kPhaseNew,       // diagnostics/phase_new_f{n}.npy       (X, Y, K, 3)
  kMotionCurrent,  // diagnostics/motion_current_f{n}.npy  (X, Y, K, 3)
  kMovMem,         // diagnostics/mov_mem_f{n}.npy         (K, Y, X)
  kMaskMov,        // diagnostics/masks_mov/mask_mov_{n:06d}.npz
  kCoverage,       // diagnostics/coverage/no_coverage_{n:06d}.npz
  kRefspace,       // refspace_{channel}/*_{n:06d}.tif
  kProjected,      // projected_{channel}/*_{n:06d}.tif
  kRawMoving,      // raw_moving_{channel}/*_{n:06d}.tif
};

enum class Channel { kMem, kSparseCell };

const char* name_of(Artifact artifact);
const char* name_of(Channel channel);

// True for the families that exist once per channel. The channel-free
// families ignore any channel passed with them, and Run::series rejects a
// non-default channel for those rather than silently returning the mem one.
bool is_channelled(Artifact artifact);

// One family's files. `dir_exists` separates "the run never wrote this" from
// "the directory is there and empty", which are different failures.
struct Series {
  std::filesystem::path dir;
  bool dir_exists = false;
  std::vector<int> frames;  // sorted ascending, may be empty
  std::map<int, std::filesystem::path> paths;

  bool has(int frame) const { return paths.count(frame) > 0; }

  // Throws naming the frame and the frames that do exist.
  const std::filesystem::path& path(int frame) const;
};

// Grid sizes shared by every accessor.
//
// mov_z (K) comes from the displacement field's header rather than from
// mov_mem_f0.npy, which the Python loader probes: that file is absent from
// runs whose moving stack was read lazily, and a loader that cannot open such
// a run cannot show the run the movie was made from.
struct Geometry {
  int64_t ref_z = 0, ref_y = 0, ref_x = 0;  // diagnostics/ref_shape.npy, (Z, Y, X)
  int64_t mov_z = 0;                        // K, moving slice count
  int64_t plane_y = 0, plane_x = 0;         // shared (Y, X) of every full-res plane
};

// Where the reference anatomy is, given that the run directory need not carry
// a copy. Runs before commit ad5a677 wrote refspace_reference_mem.tif; after
// it, projection_params.json records the source path and the z crop instead,
// and a viewer applies the crop itself.
struct ReferenceSource {
  bool present = false;             // false when the run records no source path
  std::filesystem::path path;       // the source anatomy file
  bool path_exists = false;         // recorded path may point off this machine
  int64_t z0 = 0, z1 = 0;           // half-open crop, in source-file plane numbers
  int channel_mem = 1;
  int channel_sparse = 0;
  int64_t n_channels = 0;           // from the file, for the page mapping
  int64_t n_z = 0;                  // source planes, before the crop
  int64_t height = 0, width = 0;

  // The file is ZCYX, so page = z * n_channels + channel. Throws if z is
  // outside the file or channel outside its channel count: an out-of-range
  // page index would otherwise land on a real plane of the wrong channel.
  int64_t page_for(int64_t z_source, int channel) const;

  int channel_for(Channel channel) const {
    return channel == Channel::kMem ? channel_mem : channel_sparse;
  }
};

// The reference-space movie, when the run wrote one. Its z axis is a crop of
// the already-cropped reference frame and its Y/X are block-reduced, so
// putting a movie plane over a reference plane needs both mappings. Getting
// only the z one right yields a correctly-indexed plane at the wrong scale.
struct MovieDescriptor {
  bool present = false;
  std::map<Channel, std::filesystem::path> files;
  int64_t n_t = 0, n_z = 0;      // n_t inferred from page count / n_z
  int64_t height = 0, width = 0;  // already reduced by downsample_xy
  int64_t z0_cropped_frame = 0, z1_cropped_frame = 0;
  int64_t crop_base = 0;      // reference_crop.z0: the cropped frame's origin
  int64_t downsample_xy = 1;  // lateral block-mean factor

  // Pages are written T-major: page = t * n_z + z.
  int64_t page_for(int64_t t, int64_t z) const;

  // The source-file plane lying under movie plane z_movie:
  // crop_base + z0_cropped_frame + z_movie.
  int64_t reference_plane_for(int64_t z_movie) const;
};

// The projection settings the run used, read from
// diagnostics/projection_params.json. Named fields only for the ones a live
// projector must match; `raw` keeps the whole document so a panel can show
// settings this struct does not model without the struct growing a field per
// pipeline revision.
struct ProjectionParams {
  double z_window = 0.0;
  double fill_value = 0.0;
  int64_t upsample_factor = 1;
  int64_t downsample_xy = 1;
  std::string xy_splat_mode;
  std::string raw;  // the file's text, verbatim
};

class Run {
 public:
  // Discovers every family, reads the geometry and the projection parameters.
  // Throws if `run_dir` has no diagnostics/ directory, no ref_shape.npy, no
  // projection_params.json, or no displacement field at all — without those a
  // directory is not a QC run and every later accessor would fail one at a
  // time.
  static Run open(const std::filesystem::path& run_dir);

  const std::filesystem::path& run_dir() const { return run_dir_; }
  const std::filesystem::path& diagnostics_dir() const { return diagnostics_dir_; }
  const Geometry& geometry() const { return geometry_; }
  const ProjectionParams& projection_params() const { return projection_params_; }
  const ReferenceSource& reference() const { return reference_; }
  const MovieDescriptor& movie() const { return movie_; }
  const std::vector<CsvTable>& metrics() const { return metrics_; }

  // Throws for a channel passed with a channel-free family, rather than
  // ignoring it.
  const Series& series(Artifact artifact, Channel channel = Channel::kMem) const;

  const std::vector<int>& frames_for(Artifact artifact, Channel channel = Channel::kMem) const {
    return series(artifact, channel).frames;
  }

  // Frames a live projection can actually run on for `channel`: the
  // intersection of the displacement field's frames with that channel's
  // moving intensity. On the 100t run that is the 11 stride frames, not the
  // 100 raw ones — the raw stack is saved every frame but the field is not,
  // and projecting frame 7 with frame 0's field would produce a picture with
  // nothing wrong on its face.
  //
  // Per channel because the membrane and sparse-cell stacks are written by
  // separate calls and a run can end up with one and not the other; the two
  // channels are projected and overlaid together, so a frame missing from
  // either is not projectable as a pair.
  const std::vector<int>& projectable_frames(Channel channel = Channel::kMem) const;

  // Frames where both channels can be projected, for the overlay.
  const std::vector<int>& projectable_frames_both() const { return projectable_both_; }

 private:
  std::filesystem::path run_dir_;
  std::filesystem::path diagnostics_dir_;
  Geometry geometry_;
  ProjectionParams projection_params_;
  ReferenceSource reference_;
  MovieDescriptor movie_;
  std::vector<CsvTable> metrics_;
  std::map<std::pair<Artifact, Channel>, Series> series_;
  std::map<Channel, std::vector<int>> projectable_;
  std::vector<int> projectable_both_;
};

}  // namespace wrdash
