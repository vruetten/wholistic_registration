#include "core/util/gl_texture.hpp"
#include "test_support.hpp"

#include <GLFW/glfw3.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using dashcore::GLTexture;

TEST_CASE("GLTexture::upload creates then re-uploads without error") {
  if (!dashcore_test::has_usable_gl_context()) {
    MESSAGE("skipping: no usable GL_MAX_TEXTURE_SIZE under this renderer");
    return;
  }
  GLTexture tex;
  CHECK_FALSE(tex.valid());

  std::vector<std::uint8_t> rgba(4 * 4 * 4, 128);
  tex.upload(4, 4, rgba);
  CHECK(tex.valid());
  CHECK(tex.width() == 4);
  CHECK(tex.height() == 4);

  // Same dims: TexSubImage2D path.
  tex.upload(4, 4, rgba);
  CHECK(tex.width() == 4);

  // Different dims: TexImage2D reallocation path.
  std::vector<std::uint8_t> rgba2(8 * 2 * 4, 200);
  tex.upload(8, 2, rgba2);
  CHECK(tex.width() == 8);
  CHECK(tex.height() == 2);
}

TEST_CASE("GLTexture::upload refuses a size exceeding GL_MAX_TEXTURE_SIZE") {
  if (!dashcore_test::has_usable_gl_context()) {
    MESSAGE("skipping: no usable GL_MAX_TEXTURE_SIZE under this renderer");
    return;
  }
  GLint max_size = 0;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_size);
  REQUIRE(max_size > 0);

  GLTexture tex;
  const int oversized = max_size + 1;
  std::vector<std::uint8_t> rgba(4, 0);   // deliberately undersized: must throw before reading it
  CHECK_THROWS_AS(tex.upload(oversized, 1, rgba), std::runtime_error);
  CHECK_FALSE(tex.valid());   // refused before glGenTextures
}

TEST_CASE("GLTexture::upload refuses a buffer smaller than width*height*4") {
  if (!dashcore_test::has_usable_gl_context()) {
    MESSAGE("skipping: no usable GL_MAX_TEXTURE_SIZE under this renderer");
    return;
  }
  GLTexture tex;
  // 4x4 RGBA needs 64 bytes; one byte short must be refused before it is
  // ever handed to glTexImage2D, not read out of bounds by the GL driver.
  std::vector<std::uint8_t> rgba(4 * 4 * 4 - 1, 0);
  CHECK_THROWS_AS(tex.upload(4, 4, rgba), std::runtime_error);
  CHECK_FALSE(tex.valid());
}
