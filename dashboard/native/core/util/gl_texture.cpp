#include "core/util/gl_texture.hpp"

#include <GLFW/glfw3.h>  // brings in <GL/gl.h> on the platforms we build for

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace dashcore {

namespace {

// Queries the live driver limit rather than trusting a hard-coded constant
// (kChunkRows in heatmap_panel.hpp): software rasterizers (Mesa llvmpipe,
// common under headless/remote rendering) have historically reported limits
// well below the 16k-32k typical of a real GPU.
GLint max_texture_size() {
  if (glfwGetCurrentContext() == nullptr) return 0;
  GLint max_size = 0;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_size);
  return max_size;
}

void throw_on_gl_error(const char* what) {
  const GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    throw std::runtime_error("GLTexture::upload: " + std::string(what) +
                              " failed with glGetError()=" + std::to_string(err));
  }
}

// Deleting a texture with no current context is a SIGSEGV on this driver;
// leak the id instead. Shared by the destructor and move-assignment, since
// both destroy a (possibly non-zero) `id_` and closing the window can tear
// down the context before either one runs.
void safe_delete(unsigned int id) {
  if (id && glfwGetCurrentContext() != nullptr) glDeleteTextures(1, &id);
}

}  // namespace

GLTexture::~GLTexture() {
  safe_delete(id_);
  id_ = 0;
}

GLTexture::GLTexture(GLTexture&& other) noexcept
    : id_(other.id_), width_(other.width_), height_(other.height_) {
  other.id_ = 0;
  other.width_ = 0;
  other.height_ = 0;
}

GLTexture& GLTexture::operator=(GLTexture&& other) noexcept {
  if (this != &other) {
    safe_delete(id_);
    id_ = other.id_;
    width_ = other.width_;
    height_ = other.height_;
    other.id_ = 0;
    other.width_ = 0;
    other.height_ = 0;
  }
  return *this;
}

void GLTexture::upload(int width, int height, const std::vector<std::uint8_t>& rgba) {
  const GLint max_size = max_texture_size();
  if (max_size > 0 && (width > max_size || height > max_size)) {
    throw std::runtime_error(
        "GLTexture::upload: " + std::to_string(width) + "x" + std::to_string(height) +
        " exceeds this driver's GL_MAX_TEXTURE_SIZE=" + std::to_string(max_size));
  }
  const std::size_t needed = std::size_t(width) * std::size_t(height) * 4;
  if (rgba.size() < needed) {
    throw std::runtime_error("GLTexture::upload: buffer has " + std::to_string(rgba.size()) +
                              " bytes, need " + std::to_string(needed) + " for " +
                              std::to_string(width) + "x" + std::to_string(height));
  }

  if (id_ == 0) {
    glGenTextures(1, &id_);
    glBindTexture(GL_TEXTURE_2D, id_);
    // Nearest filtering: displaying scientific data with linear interp would
    // blur adjacent samples. A discrete sample is what the caller wants to
    // see. ImGui does the on-screen scaling to display size regardless.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glBindTexture(GL_TEXTURE_2D, id_);
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (width == width_ && height == height_) {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    throw_on_gl_error("glTexSubImage2D");
  } else {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    throw_on_gl_error("glTexImage2D");
    width_ = width;
    height_ = height;
  }
}

ImTextureID GLTexture::imgui_id() const {
  return static_cast<ImTextureID>(id_);
}

int GLTexture::max_size() { return int(max_texture_size()); }

}  // namespace dashcore
