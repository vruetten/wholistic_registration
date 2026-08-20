#include "core/npz.hpp"

#include "core/detail/npy_header.hpp"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace dashcore::npz {

namespace {

constexpr std::uint32_t kEocdSig = 0x06054b50;
constexpr std::uint32_t kCentralSig = 0x02014b50;
constexpr std::uint32_t kLocalSig = 0x04034b50;

std::uint16_t rd16(const char* p) {
  std::uint16_t v = 0;
  std::memcpy(&v, p, 2);
  return v;
}

std::uint32_t rd32(const char* p) {
  std::uint32_t v = 0;
  std::memcpy(&v, p, 4);
  return v;
}

template <typename T>
const char* type_name();
template <> const char* type_name<float>()   { return "float32"; }
template <> const char* type_name<double>()  { return "float64"; }
template <> const char* type_name<int32_t>() { return "int32"; }
template <> const char* type_name<int64_t>() { return "int64"; }
template <> const char* type_name<uint8_t>() { return "uint8"; }

template <typename T> char expected_kind();
template <> char expected_kind<float>()   { return 'f'; }
template <> char expected_kind<double>()  { return 'f'; }
template <> char expected_kind<int32_t>() { return 'i'; }
template <> char expected_kind<int64_t>() { return 'i'; }
template <> char expected_kind<uint8_t>() { return 'u'; }

std::string shape_str(const std::vector<int64_t>& s) {
  std::string out = "(";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + ")";
}

// Raw-deflate inflate (windowBits negative: zip members carry no zlib
// wrapper). `expected` comes from the zip's own uncompressed-size field, so
// a member that inflates to a different length is a corrupt archive and
// throws rather than being silently truncated or padded.
std::vector<char> inflate_raw(const char* src, std::size_t n, std::size_t expected,
                              const std::string& where) {
  std::vector<char> out(expected);
  if (expected == 0) return out;

  z_stream zs{};
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
    throw std::runtime_error("inflateInit2 failed for " + where);
  }
  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(src));
  zs.avail_in = uInt(n);
  zs.next_out = reinterpret_cast<Bytef*>(out.data());
  zs.avail_out = uInt(expected);

  const int rc = inflate(&zs, Z_FINISH);
  const uLong produced = zs.total_out;
  inflateEnd(&zs);

  if (rc != Z_STREAM_END) {
    throw std::runtime_error("inflate failed (zlib code " + std::to_string(rc) + ") for " + where);
  }
  if (produced != expected) {
    throw std::runtime_error("inflated " + std::to_string(produced) + " bytes but the archive "
                             "declares " + std::to_string(expected) + " for " + where);
  }
  return out;
}

}  // namespace

Archive Archive::load(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path.string());

  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < 22) throw std::runtime_error("too short to be a .npz (zip) file: " + path.string());

  // Locate the end-of-central-directory record by scanning back from the end.
  // The trailing comment is empty in every archive numpy writes, so the record
  // is normally the last 22 bytes; scanning covers the general case.
  std::size_t eocd = std::string::npos;
  for (std::size_t i = buf.size() - 22 + 1; i-- > 0;) {
    if (rd32(buf.data() + i) == kEocdSig) {
      eocd = i;
      break;
    }
  }
  if (eocd == std::string::npos) {
    throw std::runtime_error("no zip end-of-central-directory record in " + path.string() +
                             " (not a .npz?)");
  }

  const std::uint16_t nrecs = rd16(buf.data() + eocd + 10);
  std::size_t cd = rd32(buf.data() + eocd + 16);

  Archive out;
  out.path_ = path;

  for (std::uint16_t rec = 0; rec < nrecs; ++rec) {
    if (cd + 46 > buf.size() || rd32(buf.data() + cd) != kCentralSig) {
      throw std::runtime_error("corrupt zip central directory in " + path.string());
    }
    const std::uint16_t method = rd16(buf.data() + cd + 10);
    const std::uint32_t comp_size = rd32(buf.data() + cd + 20);
    const std::uint32_t uncomp_size = rd32(buf.data() + cd + 24);
    const std::uint16_t name_len = rd16(buf.data() + cd + 28);
    const std::uint16_t extra_len = rd16(buf.data() + cd + 30);
    const std::uint16_t comment_len = rd16(buf.data() + cd + 32);
    const std::uint32_t local_off = rd32(buf.data() + cd + 42);
    std::string name(buf.data() + cd + 46, name_len);
    cd += 46 + name_len + extra_len + comment_len;

    // The local header repeats the name and extra-field lengths, and the two
    // extra-field lengths legitimately differ between the two headers, so the
    // data offset must come from the local header rather than the central one.
    if (local_off + 30 > buf.size() || rd32(buf.data() + local_off) != kLocalSig) {
      throw std::runtime_error("corrupt zip local header for member '" + name + "' in " +
                               path.string());
    }
    const std::uint16_t l_name_len = rd16(buf.data() + local_off + 26);
    const std::uint16_t l_extra_len = rd16(buf.data() + local_off + 28);
    const std::size_t data_off = local_off + 30 + l_name_len + l_extra_len;
    if (data_off + comp_size > buf.size()) {
      throw std::runtime_error("member '" + name + "' runs past the end of " + path.string());
    }

    const std::string where = path.string() + "::" + name;
    std::vector<char> member;
    if (method == 0) {
      member.assign(buf.data() + data_off, buf.data() + data_off + comp_size);
      if (member.size() != uncomp_size) {
        throw std::runtime_error("stored member '" + name + "' has size " +
                                 std::to_string(member.size()) + " but the archive declares " +
                                 std::to_string(uncomp_size) + " in " + path.string());
      }
    } else if (method == 8) {
      member = inflate_raw(buf.data() + data_off, comp_size, uncomp_size, where);
    } else {
      throw std::runtime_error("unsupported zip compression method " + std::to_string(method) +
                               " for member '" + name + "' in " + path.string());
    }

    // numpy names members "<key>.npy"; callers ask for the key.
    std::string key = name;
    if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0) {
      key.resize(key.size() - 4);
    }
    out.names_.push_back(key);
    out.members_.emplace(std::move(key), std::move(member));
  }

  return out;
}

const std::vector<char>& Archive::raw(const std::string& name) const {
  const auto it = members_.find(name);
  if (it == members_.end()) {
    std::string have;
    for (const auto& n : names_) {
      if (!have.empty()) have += ", ";
      have += n;
    }
    throw std::runtime_error("no member '" + name + "' in " + path_.string() + "; has: " + have);
  }
  return it->second;
}

template <typename T>
npy::Array<T> Archive::get(const std::string& name, int expect_ndim) const {
  const std::vector<char>& m = raw(name);
  const std::string where = path_.string() + "::" + name;
  const auto h = npy::detail::parse(m.data(), m.size(), where);

  if (h.fortran_order) {
    throw std::runtime_error("fortran_order=True member in " + where +
                             "; write with np.ascontiguousarray(...) before np.savez()");
  }
  if (h.word_size != sizeof(T)) {
    throw std::runtime_error("dtype size mismatch in " + where + ": expected " + type_name<T>() +
                             " (" + std::to_string(sizeof(T)) + " byte(s)), got word_size=" +
                             std::to_string(h.word_size));
  }
  if (h.kind != expected_kind<T>()) {
    throw std::runtime_error(std::string("dtype kind mismatch in ") + where + ": expected " +
                             type_name<T>() + " (kind '" + expected_kind<T>() + "'), got kind '" +
                             h.kind + "' — same byte width, different interpretation");
  }
  if (expect_ndim >= 0 && int(h.shape.size()) != expect_ndim) {
    throw std::runtime_error("expected " + std::to_string(expect_ndim) + "D array in " + where +
                             ", got shape " + shape_str(h.shape));
  }

  npy::Array<T> out;
  out.shape = h.shape;
  const int64_t n = h.numel();
  const std::size_t bytes = std::size_t(n) * sizeof(T);
  if (h.data_offset + bytes != m.size()) {
    throw std::runtime_error("member " + where + " declares shape " + shape_str(h.shape) +
                             " (" + std::to_string(bytes) + " bytes) but carries " +
                             std::to_string(m.size() - h.data_offset));
  }
  out.data.resize(std::size_t(n));
  std::memcpy(out.data.data(), m.data() + h.data_offset, bytes);
  return out;
}

std::string Archive::get_string(const std::string& name) const {
  const std::vector<char>& m = raw(name);
  const std::string where = path_.string() + "::" + name;
  const auto h = npy::detail::parse(m.data(), m.size(), where);

  if (h.kind != 'U' && h.kind != 'S') {
    throw std::runtime_error(std::string("expected a string member in ") + where + ", got kind '" +
                             h.kind + "'");
  }
  if (!h.shape.empty()) {
    throw std::runtime_error("expected a scalar string in " + where + ", got shape " +
                             shape_str(h.shape));
  }

  std::string out;
  if (h.kind == 'S') {
    out.assign(m.data() + h.data_offset, h.word_size);
  } else {
    // '<U<n>': n UCS-4 code points, little-endian. Anything outside ASCII is
    // refused rather than truncated into a replacement byte — no caller here
    // has a use for one, and a quiet mangling would misname an axis order.
    const char* p = m.data() + h.data_offset;
    for (std::size_t i = 0; i < h.word_size; ++i) {
      std::uint32_t cp = rd32(p + 4 * i);
      if (cp > 0x7f) {
        throw std::runtime_error("non-ASCII code point U+" + std::to_string(cp) + " in " + where);
      }
      out.push_back(char(cp));
    }
  }
  // numpy pads a short string to the dtype width with NULs.
  while (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

bool Archive::get_bool(const std::string& name) const {
  const std::vector<char>& m = raw(name);
  const std::string where = path_.string() + "::" + name;
  const auto h = npy::detail::parse(m.data(), m.size(), where);

  if (h.kind != 'b') {
    throw std::runtime_error(std::string("expected a bool member in ") + where + ", got kind '" +
                             h.kind + "'");
  }
  if (!h.shape.empty()) {
    throw std::runtime_error("expected a scalar bool in " + where + ", got shape " +
                             shape_str(h.shape));
  }
  const auto v = static_cast<unsigned char>(m[h.data_offset]);
  if (v > 1) {
    throw std::runtime_error("bool member " + where + " holds byte " + std::to_string(v));
  }
  return v == 1;
}

template npy::Array<float>   Archive::get<float>(const std::string&, int) const;
template npy::Array<double>  Archive::get<double>(const std::string&, int) const;
template npy::Array<int32_t> Archive::get<int32_t>(const std::string&, int) const;
template npy::Array<int64_t> Archive::get<int64_t>(const std::string&, int) const;
template npy::Array<uint8_t> Archive::get<uint8_t>(const std::string&, int) const;

}  // namespace dashcore::npz
