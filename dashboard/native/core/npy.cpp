#include "core/npy.hpp"

#include "core/detail/npy_header.hpp"

#include <cnpy.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

static_assert(std::endian::native == std::endian::little,
              "dashcore::npy assumes a little-endian host");

namespace dashcore::npy {

namespace {

std::string shape_str(const std::vector<std::size_t>& s) {
  std::string out = "(";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + ")";
}

template <typename T> const char* type_name();
template <> const char* type_name<float>()   { return "float32"; }
template <> const char* type_name<double>()  { return "float64"; }
template <> const char* type_name<int32_t>() { return "int32"; }
template <> const char* type_name<int64_t>() { return "int64"; }
template <> const char* type_name<uint8_t>() { return "uint8"; }

// numpy dtype "kind" character, e.g. 'f' for float32, 'i' for int32/int64,
// 'u' for uint8. Distinguishes same-width dtypes that load<T>'s size-only
// check cannot: an <f4 file and an <i4 file both have word_size == 4.
template <typename T> char expected_kind();
template <> char expected_kind<float>()   { return 'f'; }
template <> char expected_kind<double>()  { return 'f'; }
template <> char expected_kind<int32_t>() { return 'i'; }
template <> char expected_kind<int64_t>() { return 'i'; }
template <> char expected_kind<uint8_t>() { return 'u'; }

struct Descr {
  char order;   // '<', '|', or '='
  char kind;    // 'f', 'i', 'u', ...
};

// Parses the raw .npy magic + header dict directly, ahead of handing the
// file to cnpy, so truncation, non-little-endian dtypes, dtype-kind
// confusion and fortran-order arrays get a message naming the actual
// problem instead of cnpy's generic "failed to load" (or, worse, a silent
// misread: cnpy stores raw bytes without transposing, so a fortran-order
// file loaded by a C-order-assuming caller reads rows as columns and
// produces plausible-looking garbage, and cnpy::NpyArray also discards the
// dtype-kind character entirely — only word_size survives past
// cnpy::npy_load — so kind confusion must be caught here or nowhere).
// Returns the parsed order and kind characters; throws if the header dict
// cannot be understood well enough to extract them.
//
// The dict scan itself lives in detail/npy_header.cpp so that .npz members,
// which arrive as decompressed buffers rather than files, pass through the
// identical parse instead of a second copy of it.
Descr require_c_order_little_endian(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());

  // Read the 12-byte preamble first to learn the declared header length,
  // then read exactly that much more. Reading a fixed-size prefix instead
  // would report "header exceeds available bytes" for a valid file with an
  // unusually long header — a true statement about the prefix, but a
  // misleading one about the file.
  char preamble[12] = {};
  f.read(preamble, 12);
  if (f.gcount() != 12) {
    throw std::runtime_error("truncated npy header (preamble) in " + p.string());
  }
  if (std::memcmp(preamble, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not a .npy file (bad magic) in " + p.string());
  }
  const std::uint8_t major = std::uint8_t(preamble[6]);
  std::size_t header_len = 0, dict_start = 0;
  if (major == 1) {
    std::uint16_t hl16 = 0;
    std::memcpy(&hl16, preamble + 8, 2);
    header_len = hl16;
    dict_start = 10;
  } else {
    std::uint32_t hl32 = 0;
    std::memcpy(&hl32, preamble + 8, 4);
    header_len = hl32;
    dict_start = 12;
  }

  // Bound header_len against the file's actual size before allocating for
  // it: this field is read straight off disk with no upper limit of its own
  // (65535 for a v1 file, up to ~4 GiB for v2/v3), so a corrupted or
  // bit-flipped file can otherwise request a multi-GB allocation before the
  // truncation check below ever runs. A real .npy header never approaches
  // the size of its own file.
  const auto file_size = std::filesystem::file_size(p);
  if (std::uint64_t(dict_start + header_len) > file_size) {
    throw std::runtime_error(
        "npy header length " + std::to_string(header_len) + " exceeds file size (" +
        std::to_string(file_size) + " bytes) in " + p.string());
  }

  std::string buf(dict_start + header_len, '\0');
  std::memcpy(buf.data(), preamble, std::min<std::size_t>(12, buf.size()));
  if (buf.size() > 12) {
    f.seekg(12);
    f.read(buf.data() + 12, std::streamsize(buf.size() - 12));
    if (f.gcount() != std::streamsize(buf.size() - 12)) {
      throw std::runtime_error("truncated npy header (dict) in " + p.string());
    }
  }

  const auto h = detail::parse(buf.data(), buf.size(), p.string());
  if (h.fortran_order) {
    throw std::runtime_error(
        "fortran_order=True npy not supported in " + p.string() +
        "; write with np.ascontiguousarray(...) before np.save()");
  }
  return {h.order, h.kind};
}

}  // namespace

template <typename T>
Array<T> load(const std::filesystem::path& path, int expect_ndim) {
  const Descr descr = require_c_order_little_endian(path);

  cnpy::NpyArray arr = cnpy::npy_load(path.string());

  if (arr.word_size != sizeof(T)) {
    throw std::runtime_error(
        "dtype size mismatch in " + path.string() + ": expected " +
        type_name<T>() + " (" + std::to_string(sizeof(T)) +
        " byte(s)), got word_size=" + std::to_string(arr.word_size));
  }
  if (descr.kind != expected_kind<T>()) {
    throw std::runtime_error(
        "dtype kind mismatch in " + path.string() + ": expected " +
        type_name<T>() + " (kind '" + expected_kind<T>() +
        "'), got kind '" + descr.kind + "' — same byte width, different "
        "interpretation (e.g. int32 vs float32)");
  }
  if (expect_ndim >= 0 && int(arr.shape.size()) != expect_ndim) {
    throw std::runtime_error(
        "expected " + std::to_string(expect_ndim) + "D array in " +
        path.string() + ", got shape " + shape_str(arr.shape));
  }

  Array<T> out;
  out.shape.assign(arr.shape.begin(), arr.shape.end());
  const T* src = arr.data<T>();
  out.data.assign(src, src + arr.num_vals);
  return out;
}

template Array<float>   load<float>(const std::filesystem::path&, int);
template Array<double>  load<double>(const std::filesystem::path&, int);
template Array<int32_t> load<int32_t>(const std::filesystem::path&, int);
template Array<int64_t> load<int64_t>(const std::filesystem::path&, int);
template Array<uint8_t> load<uint8_t>(const std::filesystem::path&, int);

}  // namespace dashcore::npy
