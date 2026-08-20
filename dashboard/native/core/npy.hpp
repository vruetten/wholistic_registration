// A typed .npy loader with no return type tied to any particular tensor or
// dataset abstraction — just shape + a flat, row-major buffer. Whatever a
// caller wants to call the array (a matrix, a table column, a lookup array)
// is a decision made above this file, not inside it.
//
// Every load<T> applies five guards, in this order, before returning:
//   1. magic + header parse without truncation
//   2. header dict has a parseable 'descr' key (unparseable => throw, quoting
//      the offending header text — never a silent "guard skipped")
//   3. little-endian (or byte-order-agnostic) descr
//   4. fortran_order == False
//   5. on-disk dtype matches T exactly: both word size AND kind character
//      (f/i/u/...), so a same-width dtype mix-up (e.g. int32 saved where
//      float32 was expected) is refused rather than reinterpreted
// and, if `expect_ndim >= 0`, a sixth: shape.size() == expect_ndim.
// Any failure throws std::runtime_error with a message naming the path and
// the guard that failed — a bad file is a loud stop, not a quietly
// misread array.
//
// This file assumes a little-endian host (static_assert'd in npy.cpp);
// descr's '=' byte-order marker ("native") is accepted on that basis rather
// than resolved at runtime.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace dashcore::npy {

template <typename T>
struct Array {
  std::vector<int64_t> shape;   // e.g. {rows, cols} for a 2-D array
  std::vector<T> data;          // row-major (C order), size == product(shape)

  int64_t ndim() const { return int64_t(shape.size()); }
  int64_t size(int dim) const { return shape.at(std::size_t(dim)); }
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

// What the header says, without reading the array body. A caller that only
// needs a shape — to size a panel, to learn how many planes a field has —
// would otherwise pay a full read for it, and these files reach 227 MB.
struct Info {
  std::vector<int64_t> shape;
  char kind = 'f';            // 'f', 'i', 'u', ... as the descr spells it
  std::size_t word_size = 0;  // bytes per element for the numeric kinds
};

// Applies the same magic, truncation, byte-order and fortran-order guards
// load() applies, and reports the header rather than checking it against an
// expected type: peek does not know what the caller wants.
Info peek(const std::filesystem::path& path);

template <typename T>
Array<T> load(const std::filesystem::path& path, int expect_ndim = -1);

extern template Array<float>    load<float>(const std::filesystem::path&, int);
extern template Array<double>   load<double>(const std::filesystem::path&, int);
extern template Array<int32_t>  load<int32_t>(const std::filesystem::path&, int);
extern template Array<int64_t>  load<int64_t>(const std::filesystem::path&, int);
extern template Array<uint8_t>  load<uint8_t>(const std::filesystem::path&, int);

}  // namespace dashcore::npy
