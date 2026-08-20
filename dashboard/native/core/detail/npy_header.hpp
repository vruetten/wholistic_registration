// The .npy header parser, over a byte buffer rather than a path, so that the
// same guards apply whether the header came from a standalone .npy file or
// from one member of a .npz archive. A .npz is a zip of .npy members, so
// without this seam the archive reader would either duplicate the guards or,
// worse, skip them — and the arrays it reads (masks, coverage) are exactly
// the ones where a silent misread looks like real data.
//
// Parsing only. Deciding whether a header is acceptable belongs to the
// caller, which knows what type it asked for; this file reports what the
// header says, and throws only when the header cannot be understood at all.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dashcore::npy::detail {

struct Header {
  char order = '<';              // '<', '|' or '='
  char kind = 'f';               // 'f', 'i', 'u', 'b', 'U', ...
  // The digits after the descr's kind character, verbatim. For the numeric
  // kinds that is bytes per element ('<f4' -> 4). For 'U' it is a count of
  // UCS-4 characters, so an element occupies 4 * word_size bytes ('<U3' -> 3,
  // 12 bytes); for 'S' it is bytes again. Callers that accept 'U' must apply
  // that factor themselves — this struct reports the header, it does not
  // reinterpret it.
  std::size_t word_size = 0;
  bool fortran_order = false;
  std::vector<int64_t> shape;    // empty for a 0-d (scalar) array
  std::size_t data_offset = 0;   // first byte of array data within the buffer

  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

// `where` names the source in every error message: a path for a standalone
// file, "<archive>::<member>" for a .npz member.
//
// Throws std::runtime_error on bad magic, a truncated header, a header dict
// whose 'descr', 'fortran_order' or 'shape' key cannot be parsed, a header
// length exceeding `size`, or a non-little-endian byte-order character.
// fortran_order=True parses successfully and is reported, not rejected —
// rejecting is the caller's call, and load() does reject it.
Header parse(const char* buf, std::size_t size, const std::string& where);

}  // namespace dashcore::npy::detail
