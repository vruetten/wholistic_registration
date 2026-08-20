#include "app/tiff_volume.hpp"

#include <tiffio.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace wrdash {

namespace {

struct TiffCloser {
  void operator()(TIFF* t) const {
    if (t) TIFFClose(t);
  }
};
using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

TiffHandle open_or_throw(const std::filesystem::path& path) {
  // libtiff writes its own diagnostics to stderr by default; silence them so
  // the exception message is the single report of what went wrong.
  TIFFSetErrorHandler(nullptr);
  TIFFSetWarningHandler(nullptr);
  TiffHandle t(TIFFOpen(path.string().c_str(), "r"));
  if (!t) throw std::runtime_error("cannot open TIFF " + path.string());
  return t;
}

// TIFFGetField leaves its output untouched when the tag is absent, so a
// caller that does not check the return value silently reads whatever the
// variable held. Every tag this file needs is required, so absence throws.
template <typename T>
T require_field(TIFF* t, uint32_t tag, const char* tag_name,
                const std::filesystem::path& path, int64_t page) {
  T value{};
  if (!TIFFGetField(t, tag, &value)) {
    throw std::runtime_error("TIFF " + path.string() + " page " + std::to_string(page) +
                             " has no " + tag_name + " tag");
  }
  return value;
}

}  // namespace

TiffInfo tiff_info(const std::filesystem::path& path) {
  TiffHandle t = open_or_throw(path);

  TiffInfo info;
  info.width = require_field<uint32_t>(t.get(), TIFFTAG_IMAGEWIDTH, "ImageWidth", path, 0);
  info.height = require_field<uint32_t>(t.get(), TIFFTAG_IMAGELENGTH, "ImageLength", path, 0);
  info.bits_per_sample =
      require_field<uint16_t>(t.get(), TIFFTAG_BITSPERSAMPLE, "BitsPerSample", path, 0);

  // SampleFormat is optional in the TIFF spec and defaults to unsigned
  // integer. tifffile writes it, but defaulting explicitly beats inheriting
  // whatever was on the stack.
  uint16_t fmt = SAMPLEFORMAT_UINT;
  TIFFGetField(t.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
  info.sample_format = fmt;

  int64_t pages = 0;
  do {
    ++pages;
  } while (TIFFReadDirectory(t.get()));
  info.n_pages = pages;

  return info;
}

Plane read_tiff_page(const std::filesystem::path& path, int64_t page) {
  if (page < 0) {
    throw std::runtime_error("negative TIFF page " + std::to_string(page) + " for " +
                             path.string());
  }
  TiffHandle t = open_or_throw(path);

  if (!TIFFSetDirectory(t.get(), tdir_t(page))) {
    throw std::runtime_error("TIFF " + path.string() + " has no page " + std::to_string(page));
  }

  const auto width = require_field<uint32_t>(t.get(), TIFFTAG_IMAGEWIDTH, "ImageWidth", path, page);
  const auto height =
      require_field<uint32_t>(t.get(), TIFFTAG_IMAGELENGTH, "ImageLength", path, page);
  const auto bits =
      require_field<uint16_t>(t.get(), TIFFTAG_BITSPERSAMPLE, "BitsPerSample", path, page);

  uint16_t samples = 1;
  TIFFGetField(t.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
  if (samples != 1) {
    throw std::runtime_error("TIFF " + path.string() + " page " + std::to_string(page) + " has " +
                             std::to_string(samples) +
                             " samples per pixel; this reader handles single-sample planes only");
  }

  uint16_t fmt = SAMPLEFORMAT_UINT;
  TIFFGetField(t.get(), TIFFTAG_SAMPLEFORMAT, &fmt);

  if (TIFFIsTiled(t.get())) {
    throw std::runtime_error("TIFF " + path.string() + " page " + std::to_string(page) +
                             " is tiled; this reader handles strip images only");
  }

  Plane out;
  out.height = int64_t(height);
  out.width = int64_t(width);
  out.data.resize(std::size_t(out.height * out.width));

  const tmsize_t scanline_bytes = TIFFScanlineSize(t.get());
  const tmsize_t expected = tmsize_t(width) * tmsize_t(bits / 8);
  if (scanline_bytes != expected) {
    throw std::runtime_error("TIFF " + path.string() + " page " + std::to_string(page) +
                             ": scanline is " + std::to_string(scanline_bytes) +
                             " bytes, expected " + std::to_string(expected) + " for " +
                             std::to_string(width) + " samples of " + std::to_string(bits) +
                             " bits");
  }

  const auto row_bytes = static_cast<std::size_t>(scanline_bytes);
  std::vector<uint8_t> row(row_bytes);
  for (uint32_t y = 0; y < height; ++y) {
    if (TIFFReadScanline(t.get(), row.data(), y) < 0) {
      throw std::runtime_error("TIFF " + path.string() + " page " + std::to_string(page) +
                               ": failed reading scanline " + std::to_string(y));
    }
    float* dst = out.data.data() + std::size_t(int64_t(y) * out.width);
    if (fmt == SAMPLEFORMAT_IEEEFP && bits == 32) {
      std::memcpy(dst, row.data(), std::size_t(scanline_bytes));
    } else if (fmt == SAMPLEFORMAT_INT && bits == 16) {
      const auto* src = reinterpret_cast<const int16_t*>(row.data());
      for (uint32_t x = 0; x < width; ++x) dst[x] = float(src[x]);
    } else if (fmt == SAMPLEFORMAT_UINT && bits == 16) {
      const auto* src = reinterpret_cast<const uint16_t*>(row.data());
      for (uint32_t x = 0; x < width; ++x) dst[x] = float(src[x]);
    } else {
      throw std::runtime_error("TIFF " + path.string() + " page " + std::to_string(page) +
                               ": unsupported sample_format=" + std::to_string(fmt) +
                               " bits_per_sample=" + std::to_string(bits) +
                               "; this reader handles float32, int16 and uint16");
    }
  }

  return out;
}

}  // namespace wrdash
