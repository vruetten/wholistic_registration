// A .npz reader: the zip container numpy writes for np.savez /
// np.savez_compressed, holding one .npy member per named array.
//
// Members are decompressed into memory at load and then read through the
// same header parse the standalone .npy loader uses, so an archive member
// gets the identical guards — byte order, dtype kind, fortran order, ndim —
// rather than a looser second path. That matters more here than for .npy:
// the archives this reads hold masks and coverage, where a misread produces
// a plausible picture instead of an error.
//
// Whole-archive-into-memory is deliberate. A .npz is a zip, so a member
// cannot be memory-mapped or read in isolation without first inflating it;
// callers wanting one small member out of a large archive pay for the
// archive. The archives in this repo are tens of MB.
#pragma once

#include "core/npy.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace dashcore::npz {

class Archive {
 public:
  // Throws std::runtime_error naming the path if the file is not a readable
  // zip, or if any member fails to decompress.
  static Archive load(const std::filesystem::path& path);

  // Member names with the ".npy" suffix stripped, in archive order —
  // "x", "y", "z", "shape", "dense", "axis_order" for this repo's masks.
  const std::vector<std::string>& names() const { return names_; }
  bool has(const std::string& name) const { return members_.count(name) > 0; }

  // Same five guards as npy::load, plus the optional ndim check. Throws if
  // `name` is not present — an absent member is a caller error, not an empty
  // array.
  template <typename T>
  npy::Array<T> get(const std::string& name, int expect_ndim = -1) const;

  // numpy writes `np.savez(axis_order="xyz")` as a 0-d '<U3' array: three
  // UCS-4 code points, little-endian. Decodes to std::string, throwing if any
  // code point is outside ASCII rather than emitting replacement bytes.
  // Accepts 'S' (bytes) too. Trailing NULs, which numpy pads short strings
  // with, are stripped.
  std::string get_string(const std::string& name) const;

  // numpy's '|b1': one byte, 0 or 1. Any other value throws.
  bool get_bool(const std::string& name) const;

 private:
  std::filesystem::path path_;
  std::vector<std::string> names_;
  std::map<std::string, std::vector<char>> members_;  // decompressed, header included

  // The member's decompressed bytes plus its parsed header, or a throw
  // naming the archive and member.
  const std::vector<char>& raw(const std::string& name) const;
};

extern template npy::Array<float>   Archive::get<float>(const std::string&, int) const;
extern template npy::Array<double>  Archive::get<double>(const std::string&, int) const;
extern template npy::Array<int32_t> Archive::get<int32_t>(const std::string&, int) const;
extern template npy::Array<int64_t> Archive::get<int64_t>(const std::string&, int) const;
extern template npy::Array<uint8_t> Archive::get<uint8_t>(const std::string&, int) const;

}  // namespace dashcore::npz
