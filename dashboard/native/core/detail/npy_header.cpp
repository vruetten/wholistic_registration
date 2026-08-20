#include "core/detail/npy_header.hpp"

#include <cstring>
#include <stdexcept>

namespace dashcore::npy::detail {

namespace {

[[noreturn]] void fail(const std::string& msg, const std::string& where) {
  throw std::runtime_error(msg + " in " + where);
}

}  // namespace

Header parse(const char* buf, std::size_t size, const std::string& where) {
  if (size < 10) fail("truncated npy header (shorter than the magic + version + length)", where);
  if (std::memcmp(buf, "\x93NUMPY", 6) != 0) fail("not a .npy stream (bad magic)", where);

  const std::uint8_t major = std::uint8_t(buf[6]);
  std::size_t header_len = 0;
  std::size_t dict_start = 0;
  if (major == 1) {
    std::uint16_t hl16 = 0;
    std::memcpy(&hl16, buf + 8, 2);
    header_len = hl16;
    dict_start = 10;
  } else {
    if (size < 12) fail("truncated npy header (version 2+ length field)", where);
    std::uint32_t hl32 = 0;
    std::memcpy(&hl32, buf + 8, 4);
    header_len = hl32;
    dict_start = 12;
  }

  // Bound the declared header length against what the buffer actually holds
  // before reading it: the field comes straight off disk with no limit of its
  // own, so a corrupted file can otherwise claim a header far larger than the
  // whole array.
  if (dict_start + header_len > size) {
    fail("npy header length " + std::to_string(header_len) + " exceeds the available " +
             std::to_string(size) + " bytes",
         where);
  }

  const std::string hdr(buf + dict_start, header_len);
  Header out;
  out.data_offset = dict_start + header_len;

  // descr: "{'descr': '<f4', ...}". The byte-order and kind characters are the
  // two bytes after the opening quote, the word size the digits after them. A
  // header this scan cannot parse is refused outright — the byte-order and
  // dtype-kind guards both depend on having actually read these characters, so
  // a quiet skip here would be the silent misread the loader exists to prevent.
  const auto key = hdr.find("descr");
  const auto colon = (key != std::string::npos) ? hdr.find(':', key) : std::string::npos;
  const auto q_open = (colon != std::string::npos) ? hdr.find('\'', colon) : std::string::npos;
  if (q_open == std::string::npos || q_open + 2 >= hdr.size()) {
    fail("cannot parse dtype descriptor: header was \"" + hdr + "\"", where);
  }
  out.order = hdr[q_open + 1];
  out.kind = hdr[q_open + 2];
  if (out.order != '<' && out.order != '|' && out.order != '=') {
    fail(std::string("non-little-endian npy (descr byte-order '") + out.order + "')", where);
  }
  const auto q_close = hdr.find('\'', q_open + 1);
  if (q_close == std::string::npos) {
    fail("unterminated dtype descriptor: header was \"" + hdr + "\"", where);
  }
  const std::string digits = hdr.substr(q_open + 3, q_close - (q_open + 3));
  if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) {
    fail("cannot parse dtype width from descr '" + hdr.substr(q_open + 1, q_close - q_open - 1) +
             "'",
         where);
  }
  out.word_size = std::size_t(std::stoul(digits));

  // fortran_order. Absent or unparseable is refused rather than defaulted to
  // C-order: a fortran-order array read as C-order returns transposed data
  // with no error at all.
  const auto fkey = hdr.find("fortran_order");
  if (fkey == std::string::npos) {
    fail("cannot find 'fortran_order': header was \"" + hdr + "\"", where);
  }
  const auto fcol = hdr.find(':', fkey);
  if (fcol == std::string::npos) {
    fail("cannot parse 'fortran_order': header was \"" + hdr + "\"", where);
  }
  const auto tpos = hdr.find("True", fcol);
  const auto fpos = hdr.find("False", fcol);
  if (tpos == std::string::npos && fpos == std::string::npos) {
    fail("cannot parse 'fortran_order' value: header was \"" + hdr + "\"", where);
  }
  out.fortran_order = (tpos != std::string::npos && (fpos == std::string::npos || tpos < fpos));

  // shape: "'shape': (630, 1500, 20, 3), }" — or "()" for a 0-d array, which
  // numpy writes for the scalar members this repo's .npz masks carry
  // ('dense', 'axis_order'). An empty shape is legitimate, not a parse failure.
  const auto skey = hdr.find("shape");
  const auto s_open = (skey != std::string::npos) ? hdr.find('(', skey) : std::string::npos;
  const auto s_close = (s_open != std::string::npos) ? hdr.find(')', s_open) : std::string::npos;
  if (s_close == std::string::npos) {
    fail("cannot parse 'shape': header was \"" + hdr + "\"", where);
  }
  const std::string dims = hdr.substr(s_open + 1, s_close - s_open - 1);
  std::string num;
  for (char c : dims) {
    if (c >= '0' && c <= '9') {
      num += c;
    } else if (!num.empty()) {
      out.shape.push_back(int64_t(std::stoll(num)));
      num.clear();
    }
  }
  if (!num.empty()) out.shape.push_back(int64_t(std::stoll(num)));

  return out;
}

}  // namespace dashcore::npy::detail
