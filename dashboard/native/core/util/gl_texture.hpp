// Tiny RAII wrapper around an OpenGL 2-D texture. Owns the ID and frees on
// destruction; upload() re-uploads pixel data in place so a panel can update
// its texture per frame (e.g. a changed slice or contrast range) without
// churn on the GL side.
//
// Format is fixed to GL_RGBA / GL_UNSIGNED_BYTE so a single upload path
// covers grayscale, colormapped and overlay images alike — the caller
// composes RGBA in a std::vector<uint8_t> before handing it in.
#pragma once

#include <cstdint>
#include <vector>

#include <imgui.h>   // for ImTextureID

namespace dashcore {

class GLTexture {
 public:
  GLTexture() = default;
  ~GLTexture();
  GLTexture(const GLTexture&) = delete;
  GLTexture& operator=(const GLTexture&) = delete;
  GLTexture(GLTexture&& other) noexcept;
  GLTexture& operator=(GLTexture&& other) noexcept;

  // (Re)upload RGBA pixels. Creates the texture on first call. Requires a
  // current OpenGL context.
  void upload(int width, int height, const std::vector<std::uint8_t>& rgba);

  // Opaque texture handle in whatever integer type the ImGui build uses
  // (void* in classic builds, ImU64 with modern texture backends).
  ImTextureID imgui_id() const;

  int width()  const { return width_; }
  int height() const { return height_; }
  bool valid() const { return id_ != 0; }

  // Live driver limit (GL_MAX_TEXTURE_SIZE). 0 if no context / query failed.
  static int max_size();

 private:
  unsigned int id_ = 0;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace dashcore
