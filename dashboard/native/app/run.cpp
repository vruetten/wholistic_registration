#include "app/run.hpp"

#include "core/npy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace wrdash {

namespace {

using json = nlohmann::json;

// The trailing run of digits in a stem, or -1 when the stem ends in something
// else. Both naming conventions in a run directory end this way —
// "phase_new_f90" and "vol_F260517_refspace_mem_000090" — so one rule covers
// both without either filename template appearing in this file.
int trailing_index(const std::string& stem) {
  std::size_t end = stem.size();
  std::size_t begin = end;
  while (begin > 0 && std::isdigit(static_cast<unsigned char>(stem[begin - 1]))) --begin;
  if (begin == end) return -1;
  // A stem that is all digits still names a frame; one with a huge digit run
  // is not a frame index this pipeline writes.
  if (end - begin > 9) return -1;
  return std::stoi(stem.substr(begin, end - begin));
}

// Files directly under `dir` whose name starts with `prefix` and whose
// extension is `ext`, keyed by their trailing index.
Series discover(const std::filesystem::path& dir, const std::string& prefix,
                const std::string& ext) {
  Series series;
  series.dir = dir;
  series.dir_exists = std::filesystem::is_directory(dir);
  if (!series.dir_exists) return series;

  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const auto& path = entry.path();
    if (path.extension() != ext) continue;
    const std::string stem = path.stem().string();
    if (stem.rfind(prefix, 0) != 0) continue;
    const int index = trailing_index(stem);
    if (index < 0) continue;
    // Two files claiming one frame means the directory holds more than one
    // naming convention; picking either silently would make the choice
    // invisible.
    if (series.paths.count(index)) {
      throw std::runtime_error("two files claim frame " + std::to_string(index) + " in " +
                               dir.string() + ": " + series.paths[index].filename().string() +
                               " and " + path.filename().string());
    }
    series.paths[index] = path;
  }
  series.frames.reserve(series.paths.size());
  for (const auto& [frame, _] : series.paths) series.frames.push_back(frame);
  return series;
}

// An OME .tif written with an .ome suffix has stem "name.ome"; extension()
// returns ".tif" and stem() keeps ".ome". Matching on the whole filename
// avoids depending on that.
std::filesystem::path find_file(const std::filesystem::path& dir, const std::string& filename) {
  const auto candidate = dir / filename;
  return std::filesystem::is_regular_file(candidate) ? candidate : std::filesystem::path{};
}

std::string channel_dir_suffix(Channel channel) {
  return channel == Channel::kMem ? "mem" : "sparseCell";
}

std::vector<int> intersect_sorted(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> out;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

}  // namespace

const char* name_of(Artifact artifact) {
  switch (artifact) {
    case Artifact::kPhaseNew: return "phase_new";
    case Artifact::kMotionCurrent: return "motion_current";
    case Artifact::kMovMem: return "mov_mem";
    case Artifact::kMaskMov: return "mask_mov";
    case Artifact::kCoverage: return "coverage";
    case Artifact::kRefspace: return "refspace";
    case Artifact::kProjected: return "projected";
    case Artifact::kRawMoving: return "raw_moving";
  }
  return "<unknown artifact>";
}

const char* name_of(Channel channel) {
  return channel == Channel::kMem ? "mem" : "sparseCell";
}

bool is_channelled(Artifact artifact) {
  switch (artifact) {
    case Artifact::kRefspace:
    case Artifact::kProjected:
    case Artifact::kRawMoving:
      return true;
    default:
      return false;
  }
}

const std::filesystem::path& Series::path(int frame) const {
  const auto it = paths.find(frame);
  if (it != paths.end()) return it->second;

  std::ostringstream msg;
  msg << "no file for frame " << frame << " in " << dir.string();
  if (!dir_exists) {
    msg << " (directory does not exist)";
  } else if (frames.empty()) {
    msg << " (directory exists but holds no frames)";
  } else {
    msg << "; frames present: " << frames.front() << " .. " << frames.back() << " ("
        << frames.size() << " of them)";
  }
  throw std::runtime_error(msg.str());
}

int64_t ReferenceSource::page_for(int64_t z_source, int channel) const {
  if (z_source < 0 || z_source >= n_z) {
    throw std::runtime_error("reference source plane " + std::to_string(z_source) +
                             " is outside the file's " + std::to_string(n_z) + " planes");
  }
  if (channel < 0 || channel >= n_channels) {
    throw std::runtime_error("reference channel " + std::to_string(channel) +
                             " is outside the file's " + std::to_string(n_channels) +
                             " channels");
  }
  return z_source * n_channels + channel;
}

int64_t MovieDescriptor::page_for(int64_t t, int64_t z) const {
  if (t < 0 || t >= n_t) {
    throw std::runtime_error("movie timepoint " + std::to_string(t) + " is outside [0, " +
                             std::to_string(n_t) + ")");
  }
  if (z < 0 || z >= n_z) {
    throw std::runtime_error("movie plane " + std::to_string(z) + " is outside [0, " +
                             std::to_string(n_z) + ")");
  }
  return t * n_z + z;
}

int64_t MovieDescriptor::reference_plane_for(int64_t z_movie) const {
  if (z_movie < 0 || z_movie >= n_z) {
    throw std::runtime_error("movie plane " + std::to_string(z_movie) + " is outside [0, " +
                             std::to_string(n_z) + ")");
  }
  return crop_base + z0_cropped_frame + z_movie;
}

const std::vector<int>& Run::projectable_frames(Channel channel) const {
  const auto it = projectable_.find(channel);
  if (it == projectable_.end()) {
    throw std::runtime_error(std::string("no projectable frame set for channel ") +
                             name_of(channel));
  }
  return it->second;
}

const Series& Run::series(Artifact artifact, Channel channel) const {
  if (!is_channelled(artifact) && channel != Channel::kMem) {
    throw std::runtime_error(std::string("artifact ") + name_of(artifact) +
                             " has no channels; asked for " + name_of(channel));
  }
  const auto it = series_.find({artifact, channel});
  if (it == series_.end()) {
    throw std::runtime_error(std::string("no series for artifact ") + name_of(artifact) +
                             " channel " + name_of(channel));
  }
  return it->second;
}

Run Run::open(const std::filesystem::path& run_dir) {
  Run run;
  run.run_dir_ = run_dir;
  run.diagnostics_dir_ = run_dir / "diagnostics";

  if (!std::filesystem::is_directory(run.diagnostics_dir_)) {
    throw std::runtime_error("not a QC run directory (no diagnostics/): " + run_dir.string());
  }

  const auto& diag = run.diagnostics_dir_;

  run.series_[{Artifact::kPhaseNew, Channel::kMem}] = discover(diag, "phase_new_f", ".npy");
  run.series_[{Artifact::kMotionCurrent, Channel::kMem}] =
      discover(diag, "motion_current_f", ".npy");
  run.series_[{Artifact::kMovMem, Channel::kMem}] = discover(diag, "mov_mem_f", ".npy");
  run.series_[{Artifact::kMaskMov, Channel::kMem}] =
      discover(diag / "masks_mov", "mask_mov_", ".npz");
  run.series_[{Artifact::kCoverage, Channel::kMem}] =
      discover(diag / "coverage", "no_coverage_", ".npz");

  for (const Channel channel : {Channel::kMem, Channel::kSparseCell}) {
    const std::string suffix = channel_dir_suffix(channel);
    run.series_[{Artifact::kRefspace, channel}] =
        discover(run_dir / ("refspace_" + suffix), "", ".tif");
    run.series_[{Artifact::kProjected, channel}] =
        discover(run_dir / ("projected_" + suffix), "", ".tif");
    run.series_[{Artifact::kRawMoving, channel}] =
        discover(run_dir / ("raw_moving_" + suffix), "", ".tif");
  }

  // ---- geometry ----------------------------------------------------------
  const auto ref_shape_path = diag / "ref_shape.npy";
  if (!std::filesystem::is_regular_file(ref_shape_path)) {
    throw std::runtime_error("not a QC run directory (no diagnostics/ref_shape.npy): " +
                             run_dir.string());
  }
  const auto ref_shape = dashcore::npy::load<int64_t>(ref_shape_path, 1);
  if (ref_shape.numel() != 3) {
    throw std::runtime_error("ref_shape.npy should hold 3 values (Z, Y, X), holds " +
                             std::to_string(ref_shape.numel()));
  }
  run.geometry_.ref_z = ref_shape.data[0];
  run.geometry_.ref_y = ref_shape.data[1];
  run.geometry_.ref_x = ref_shape.data[2];

  const Series& phase = run.series(Artifact::kPhaseNew);
  if (phase.frames.empty()) {
    throw std::runtime_error("not a QC run directory (no diagnostics/phase_new_f*.npy): " +
                             run_dir.string());
  }
  // peek, not load: the field is (630, 1500, K, 3) float32 — 227 MB at K=20 —
  // and only its shape is wanted here.
  const auto phase_info = dashcore::npy::peek(phase.path(phase.frames.front()));
  if (phase_info.shape.size() != 4 || phase_info.shape[3] != 3) {
    std::string got = "(";
    for (std::size_t i = 0; i < phase_info.shape.size(); ++i) {
      if (i) got += ", ";
      got += std::to_string(phase_info.shape[i]);
    }
    got += ")";
    throw std::runtime_error("phase_new should be (X, Y, K, 3), is " + got + " in " +
                             phase.path(phase.frames.front()).string());
  }
  run.geometry_.plane_x = phase_info.shape[0];
  run.geometry_.plane_y = phase_info.shape[1];
  run.geometry_.mov_z = phase_info.shape[2];

  // The reference and moving grids must agree on (Y, X): every plane accessor
  // shares one plane shape, and a mismatch means the two were written from
  // different acquisitions.
  if (run.geometry_.ref_y != run.geometry_.plane_y || run.geometry_.ref_x != run.geometry_.plane_x) {
    throw std::runtime_error(
        "reference (Y, X) = (" + std::to_string(run.geometry_.ref_y) + ", " +
        std::to_string(run.geometry_.ref_x) + ") does not match the field's (" +
        std::to_string(run.geometry_.plane_y) + ", " + std::to_string(run.geometry_.plane_x) + ")");
  }

  // ---- projection parameters --------------------------------------------
  const auto params_path = diag / "projection_params.json";
  if (!std::filesystem::is_regular_file(params_path)) {
    throw std::runtime_error("not a QC run directory (no diagnostics/projection_params.json): " +
                             run_dir.string());
  }
  std::ifstream params_file(params_path);
  std::ostringstream params_text;
  params_text << params_file.rdbuf();
  run.projection_params_.raw = params_text.str();

  const json params = json::parse(run.projection_params_.raw);
  run.projection_params_.z_window = params.value("z_window", 0.0);
  run.projection_params_.fill_value = params.value("fill_value", 0.0);
  run.projection_params_.upsample_factor = params.value("upsample_factor", int64_t{1});
  run.projection_params_.downsample_xy = params.value("downsample_xy", int64_t{1});
  run.projection_params_.xy_splat_mode = params.value("xy_splat_mode", std::string{});

  // ---- reference source --------------------------------------------------
  if (params.contains("reference_source_path") && params.contains("reference_crop")) {
    auto& ref = run.reference_;
    ref.present = true;
    ref.path = params["reference_source_path"].get<std::string>();
    const auto& crop = params["reference_crop"];
    ref.z0 = crop.value("z0", int64_t{0});
    ref.z1 = crop.value("z1", int64_t{0});
    ref.channel_mem = crop.value("channel_mem", 1);
    ref.channel_sparse = crop.value("channel_sparse", 0);

    ref.path_exists = std::filesystem::is_regular_file(ref.path);
    if (ref.path_exists) {
      // The file is ZCYX with one page per (z, channel), so the channel count
      // follows from the page count and the crop's plane span rather than
      // being assumed to be 2.
      const TiffInfo info = tiff_info(ref.path);
      ref.height = info.height;
      ref.width = info.width;
      if (ref.height != run.geometry_.ref_y || ref.width != run.geometry_.ref_x) {
        throw std::runtime_error("reference source " + ref.path.string() + " is " +
                                 std::to_string(ref.height) + "x" + std::to_string(ref.width) +
                                 ", run geometry says " + std::to_string(run.geometry_.ref_y) +
                                 "x" + std::to_string(run.geometry_.ref_x));
      }
      const int64_t n_channels = std::max<int64_t>(ref.channel_mem, ref.channel_sparse) + 1;
      if (info.n_pages % n_channels != 0) {
        throw std::runtime_error("reference source " + ref.path.string() + " has " +
                                 std::to_string(info.n_pages) + " pages, not divisible by " +
                                 std::to_string(n_channels) + " channels");
      }
      ref.n_channels = n_channels;
      ref.n_z = info.n_pages / n_channels;
      if (ref.z1 > ref.n_z) {
        throw std::runtime_error("reference crop z1=" + std::to_string(ref.z1) +
                                 " exceeds the source file's " + std::to_string(ref.n_z) +
                                 " planes");
      }
    }
  }

  // ---- movie -------------------------------------------------------------
  if (params.value("save_refspace_movie", false) && params.contains("movie_z_crop")) {
    auto& movie = run.movie_;
    const auto& crop = params["movie_z_crop"];
    movie.z0_cropped_frame = crop.value("z0_cropped_frame", int64_t{0});
    movie.z1_cropped_frame = crop.value("z1_cropped_frame", int64_t{0});
    movie.crop_base = crop.value("crop_base", int64_t{0});
    movie.n_z = crop.value("n_planes", movie.z1_cropped_frame - movie.z0_cropped_frame);
    movie.downsample_xy = params.value("movie_downsample_xy", int64_t{1});

    // movie_files names them; fall back to the conventional names when an
    // older run wrote the crop but not the list.
    std::vector<std::string> filenames;
    if (params.contains("movie_files")) {
      filenames = params["movie_files"].get<std::vector<std::string>>();
    } else {
      filenames = {"refspace_movie_mem.ome.tif", "refspace_movie_sparseCell.ome.tif"};
    }
    for (const auto& filename : filenames) {
      const auto path = find_file(run_dir, filename);
      if (path.empty()) continue;
      const Channel channel = filename.find("sparse") != std::string::npos ? Channel::kSparseCell
                                                                          : Channel::kMem;
      movie.files[channel] = path;
    }

    if (!movie.files.empty()) {
      movie.present = true;
      const TiffInfo info = tiff_info(movie.files.begin()->second);
      movie.height = info.height;
      movie.width = info.width;
      if (movie.n_z <= 0 || info.n_pages % movie.n_z != 0) {
        throw std::runtime_error("movie " + movie.files.begin()->second.string() + " has " +
                                 std::to_string(info.n_pages) + " pages, not divisible by n_z=" +
                                 std::to_string(movie.n_z));
      }
      movie.n_t = info.n_pages / movie.n_z;
    }
  }

  // ---- metrics -----------------------------------------------------------
  run.metrics_ = read_csv_dir(diag);

  // ---- projectable frames ------------------------------------------------
  // Raw moving intensity is the projector's input, not mov_mem: mov_mem is
  // absent from runs that read the moving stack lazily, while raw_moving_*
  // is written for every forward-loop frame.
  const auto& field_frames = run.series(Artifact::kPhaseNew).frames;
  for (const Channel channel : {Channel::kMem, Channel::kSparseCell}) {
    run.projectable_[channel] =
        intersect_sorted(field_frames, run.series(Artifact::kRawMoving, channel).frames);
  }
  run.projectable_both_ =
      intersect_sorted(run.projectable_[Channel::kMem], run.projectable_[Channel::kSparseCell]);

  return run;
}

}  // namespace wrdash
