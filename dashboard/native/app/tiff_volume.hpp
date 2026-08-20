// TIFF page reader for the volumes a QC run writes, and for the reference
// anatomy it points at.
//
// This lives in app/ rather than in dashcore because dashcore stays
// format-agnostic beyond .npy: the plan admits libtiff to the application
// layer only, until a second application needs it.
//
// Every read returns float, whatever the file's on-disk type. The run's own
// volumes are float32 but the reference anatomy is int16, and a display path
// that branches on dtype ends up with two versions of every contrast
// calculation. Converting once, here, is the same discipline the canonical
// orientation rule applies to axis order: normalise at the loader, and no
// module below it repeats the decision.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace wrdash {

// One 2-D plane, row-major, (Y, X) — the canonical orientation from
// dashboard/SPEC.md. No consumer transposes.
struct Plane {
  int64_t height = 0;  // Y
  int64_t width = 0;   // X
  std::vector<float> data;

  int64_t numel() const { return height * width; }
  float at(int64_t y, int64_t x) const { return data[std::size_t(y * width + x)]; }
};

struct TiffInfo {
  int64_t n_pages = 0;
  int64_t height = 0;  // page 0's; read_tiff_page checks each page against it
  int64_t width = 0;
  uint16_t bits_per_sample = 0;
  uint16_t sample_format = 0;  // 1 uint, 2 int, 3 IEEE float
};

// Walks the directory chain to count pages, so the cost is linear in page
// count — 5500 for a 100-timepoint movie. Call once per file and keep the
// result rather than asking per plane.
TiffInfo tiff_info(const std::filesystem::path& path);

// Reads one page and converts it to float.
//
// Throws, naming path and page, on: a page index past the end, a tiled file,
// more than one sample per pixel, or a dtype outside {float32, int16, uint16}.
// Refusing an unexpected dtype matters more than it looks: reinterpreting
// int16 as float32 produces a picture rather than an error, and a picture is
// what a viewer trusts.
Plane read_tiff_page(const std::filesystem::path& path, int64_t page);

}  // namespace wrdash
