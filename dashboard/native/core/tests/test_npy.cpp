#include "core/npy.hpp"

#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

std::string format_shape(const std::vector<int64_t>& shape) {
  std::string s = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";
  s += ")";
  return s;
}

// Hand-assembles a minimal npy v1.0 file: magic + version + header dict +
// raw payload. Lets the malformed-file tests exercise exact byte layouts
// without depending on numpy being installed anywhere.
std::filesystem::path write_npy(const std::string& name, const std::string& descr,
                                bool fortran_order, const std::vector<int64_t>& shape,
                                const void* payload, std::size_t payload_bytes) {
  std::string dict = "{'descr': '" + descr + "', 'fortran_order': " +
                     (fortran_order ? "True" : "False") +
                     ", 'shape': " + format_shape(shape) + ", }";
  const std::size_t base = 10 + dict.size() + 1;   // 6 magic + 2 version + 2 length + dict + '\n'
  const std::size_t pad = (64 - (base % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');

  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY", 6);
  const std::uint8_t version[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(version), 2);
  const std::uint16_t header_len = std::uint16_t(dict.size());
  f.write(reinterpret_cast<const char*>(&header_len), 2);
  f.write(dict.data(), std::streamsize(dict.size()));
  if (payload) f.write(reinterpret_cast<const char*>(payload), std::streamsize(payload_bytes));
  return path;
}

// Writes a syntactically valid npy file whose header dict has no 'descr'
// key at all (a form the substring scan cannot recover a byte order or
// kind from), so the absent-key guard can be exercised directly.
std::filesystem::path write_npy_no_descr_key(const std::string& name) {
  std::string dict = "{'shape': (4,), 'fortran_order': False, }";
  const std::size_t base = 10 + dict.size() + 1;
  const std::size_t pad = (64 - (base % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');

  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY", 6);
  const std::uint8_t version[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(version), 2);
  const std::uint16_t header_len = std::uint16_t(dict.size());
  f.write(reinterpret_cast<const char*>(&header_len), 2);
  f.write(dict.data(), std::streamsize(dict.size()));
  const std::vector<float> payload = {1, 2, 3, 4};
  f.write(reinterpret_cast<const char*>(payload.data()),
          std::streamsize(payload.size() * sizeof(float)));
  return path;
}

// Writes a syntactically valid npy file whose header dict has no
// 'fortran_order' key at all, so the absent-key guard (guard 4) can be
// exercised directly, symmetric with write_npy_no_descr_key above.
std::filesystem::path write_npy_no_fortran_key(const std::string& name) {
  std::string dict = "{'descr': '<f4', 'shape': (4,), }";
  const std::size_t base = 10 + dict.size() + 1;
  const std::size_t pad = (64 - (base % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');

  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY", 6);
  const std::uint8_t version[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(version), 2);
  const std::uint16_t header_len = std::uint16_t(dict.size());
  f.write(reinterpret_cast<const char*>(&header_len), 2);
  f.write(dict.data(), std::streamsize(dict.size()));
  const std::vector<float> payload = {1, 2, 3, 4};
  f.write(reinterpret_cast<const char*>(payload.data()),
          std::streamsize(payload.size() * sizeof(float)));
  return path;
}

// A v2 npy file (32-bit header_len field) whose claimed header length (~3
// GiB) is far larger than the file actually written (12 bytes: magic +
// version + the length field itself, no header dict, no payload) — probes
// the header-length-vs-file-size guard that must fire before `std::string
// hdr(header_len, '\0')` would otherwise attempt a multi-GB allocation.
std::filesystem::path write_npy_huge_header_len(const std::string& name) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY", 6);
  const std::uint8_t version[2] = {2, 0};
  f.write(reinterpret_cast<const char*>(version), 2);
  const std::uint32_t claimed_len = 3u * 1024u * 1024u * 1024u;   // ~3 GiB
  f.write(reinterpret_cast<const char*>(&claimed_len), 4);
  return path;
}

// Same as write_npy but the declared header length exceeds what's actually
// written, so the header read hits EOF mid-dict. The over-claim (+5) is
// deliberately smaller than the 10-byte preamble: large enough to overshoot
// EOF while reading the dict, but small enough that claimed_len still stays
// under the on-disk file size, so this exercises the mid-dict EOF guard
// specifically rather than the (coarser, file-size-level) header-length
// guard that write_npy_huge_header_len targets above.
std::filesystem::path write_npy_truncated_header(const std::string& name) {
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }";
  const std::size_t base = 10 + dict.size() + 1;
  const std::size_t pad = (64 - (base % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');

  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY", 6);
  const std::uint8_t version[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(version), 2);
  const std::uint16_t claimed_len = std::uint16_t(dict.size() + 5);   // lies, but only a little
  f.write(reinterpret_cast<const char*>(&claimed_len), 2);
  f.write(dict.data(), std::streamsize(dict.size()));   // writes fewer bytes than claimed
  return path;
}

// doctest's CHECK_THROWS_WITH(_AS) matches the exception message against an
// exact (optionally wildcarded) pattern, which is pickier than this file
// needs: these tests only care that the readable reason is IN the message
// somewhere, not that it's the whole message. Substring check by hand.
void expect_runtime_error_containing(const std::function<void()>& fn,
                                     const std::string& needle) {
  try {
    fn();
  } catch (const std::runtime_error& e) {
    INFO("exception message: ", e.what());
    CHECK(std::string(e.what()).find(needle) != std::string::npos);
    return;
  } catch (...) {
    FAIL("threw something other than std::runtime_error");
    return;
  }
  FAIL("did not throw");
}

}  // namespace

TEST_CASE("valid float32 2D array round-trips with correct shape and values") {
  const std::vector<float> values = {1, 2, 3, 4, 5, 6};
  const auto path = write_npy("valid_f32_2d.npy", "<f4", false, {2, 3},
                              values.data(), values.size() * sizeof(float));
  const auto arr = dashcore::npy::load<float>(path, /*expect_ndim=*/2);
  REQUIRE(arr.ndim() == 2);
  CHECK(arr.size(0) == 2);
  CHECK(arr.size(1) == 3);
  CHECK(arr.numel() == 6);
  for (std::size_t i = 0; i < values.size(); ++i) CHECK(arr.data[i] == values[i]);
}

TEST_CASE("valid float64 2D array round-trips") {
  const std::vector<double> values = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
  const auto path = write_npy("valid_f64_2d.npy", "<f8", false, {2, 3},
                              values.data(), values.size() * sizeof(double));
  const auto arr = dashcore::npy::load<double>(path, /*expect_ndim=*/2);
  REQUIRE(arr.ndim() == 2);
  CHECK(arr.size(0) == 2);
  CHECK(arr.size(1) == 3);
  for (std::size_t i = 0; i < values.size(); ++i) CHECK(arr.data[i] == values[i]);
}

TEST_CASE("valid int64 1D array round-trips") {
  const std::vector<int64_t> values = {10, -20, 30, 40};
  const auto path = write_npy("valid_i64_1d.npy", "<i8", false, {4},
                              values.data(), values.size() * sizeof(int64_t));
  const auto arr = dashcore::npy::load<int64_t>(path, /*expect_ndim=*/1);
  REQUIRE(arr.ndim() == 1);
  CHECK(arr.size(0) == 4);
  for (std::size_t i = 0; i < values.size(); ++i) CHECK(arr.data[i] == values[i]);
}

TEST_CASE("fortran_order:True is refused") {
  const std::vector<float> values = {1, 2, 3, 4};
  const auto path = write_npy("fortran.npy", "<f4", true, {2, 2},
                              values.data(), values.size() * sizeof(float));
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "fortran_order");
}

TEST_CASE("big-endian descr is refused") {
  const std::vector<float> values = {1, 2, 3, 4};
  const auto path = write_npy("bigendian.npy", ">f4", false, {4},
                              values.data(), values.size() * sizeof(float));
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "endian");
}

TEST_CASE("wrong ndim is refused") {
  const std::vector<float> values = {1, 2, 3, 4, 5, 6};
  const auto path = write_npy("wrong_ndim.npy", "<f4", false, {2, 3},
                              values.data(), values.size() * sizeof(float));
  CHECK_THROWS_AS(dashcore::npy::load<float>(path, /*expect_ndim=*/3), std::runtime_error);
}

TEST_CASE("truncated header is refused") {
  const auto path = write_npy_truncated_header("truncated_header.npy");
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "truncated");
}

TEST_CASE("dtype size mismatch is refused") {
  const std::vector<float> values = {1, 2, 3, 4};
  const auto path = write_npy("wrong_dtype.npy", "<f4", false, {4},
                              values.data(), values.size() * sizeof(float));
  CHECK_THROWS_AS(dashcore::npy::load<int64_t>(path), std::runtime_error);
}

TEST_CASE("same-width dtype kind mismatch (float32 loaded as int32) is refused") {
  const std::vector<float> values = {1.5f, 2.5f, 3.5f, 4.5f};
  const auto path = write_npy("f32_as_i32.npy", "<f4", false, {4},
                              values.data(), values.size() * sizeof(float));
  expect_runtime_error_containing([&] { dashcore::npy::load<int32_t>(path); }, "kind");
}

TEST_CASE("same-width dtype kind mismatch (int32 loaded as float32) is refused") {
  const std::vector<int32_t> values = {1, 2, 3, 4};
  const auto path = write_npy("i32_as_f32.npy", "<i4", false, {4},
                              values.data(), values.size() * sizeof(int32_t));
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); }, "kind");
}

TEST_CASE("uint8 dtype kind is checked too") {
  const std::vector<std::uint8_t> values = {1, 2, 3, 4};
  const auto path = write_npy("u8_as_u8.npy", "|u1", false, {4},
                              values.data(), values.size() * sizeof(std::uint8_t));
  const auto arr = dashcore::npy::load<std::uint8_t>(path);
  CHECK(arr.data == values);
}

TEST_CASE("absent fortran_order key is refused rather than silently treated as C-order") {
  const auto path = write_npy_no_fortran_key("no_fortran_key.npy");
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "fortran_order");
}

TEST_CASE("header length exceeding the file's actual size is refused before allocating") {
  const auto path = write_npy_huge_header_len("huge_header_len.npy");
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "header length");
}

TEST_CASE("absent descr key is refused, quoting the offending header") {
  const auto path = write_npy_no_descr_key("no_descr.npy");
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "cannot parse dtype descriptor");
  // The message should quote the actual header text, not just say "bad file".
  expect_runtime_error_containing([&] { dashcore::npy::load<float>(path); },
                                  "shape");
}

TEST_CASE("native byte order '=' descr is accepted") {
  const std::vector<float> values = {1, 2, 3, 4};
  const auto path = write_npy("native_order.npy", "=f4", false, {4},
                              values.data(), values.size() * sizeof(float));
  const auto arr = dashcore::npy::load<float>(path);
  CHECK(arr.data == values);
}
