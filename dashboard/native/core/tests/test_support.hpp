// Shared fixtures for dashcore's tests: a headless ImGui context (no GPU
// needed) and a lazily-created hidden GLFW/GL window (needed only by tests
// that upload a real texture). Kept out of the seam by living under
// tests/ itself, never under the library's own include root.
#pragma once

#include "core/util/gl_texture.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>

#include <stdexcept>

namespace dashcore_test {

// RAII ImGui + ImPlot context sized for headless logic testing: Begin/End,
// tables, docking and ImPlot's colormap widgets all work without a renderer
// backend, as long as DisplaySize is a real value (NewFrame() asserts
// otherwise) and every NewFrame() this frame is matched by an EndFrame()
// before the next one. ImPlot's own calls assert on a null current context,
// hence creating both together here.
struct ImGuiScope {
  ImGuiScope() {
    ctx_ = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1280.0f, 800.0f);
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    unsigned char* pixels = nullptr;
    int w = 0, h = 0;
    io.Fonts->GetTexDataAsAlpha8(&pixels, &w, &h);   // builds the atlas CPU-side
    plot_ctx_ = ImPlot::CreateContext();
  }
  ~ImGuiScope() {
    ImPlot::DestroyContext(plot_ctx_);
    ImGui::DestroyContext(ctx_);
  }
  ImGuiScope(const ImGuiScope&) = delete;
  ImGuiScope& operator=(const ImGuiScope&) = delete;

  void new_frame() { ImGui::NewFrame(); }
  void end_frame() { ImGui::EndFrame(); }

  ImGuiContext* ctx_ = nullptr;
  ImPlotContext* plot_ctx_ = nullptr;
};

// One hidden window + current GL context, shared by every test in this
// binary that needs to actually upload a texture. Throws readably if no
// display is reachable — export DISPLAY before running dashcore_tests.
inline GLFWwindow* gl_test_window() {
  static GLFWwindow* window = [] {
    if (!glfwInit()) {
      throw std::runtime_error(
          "glfwInit() failed: dashcore_tests needs a reachable X/Wayland "
          "display for its GL-texture tests. Set DISPLAY before running it.");
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* w = glfwCreateWindow(64, 64, "dashcore_tests", nullptr, nullptr);
    if (!w) throw std::runtime_error("glfwCreateWindow() failed");
    glfwMakeContextCurrent(w);
    return w;
  }();
  return window;
}

// True if the live GL context reports a usable GL_MAX_TEXTURE_SIZE. Some
// headless software rasterizers (llvmpipe invoked via xvfb, no GLX in this
// environment) report 0 for GL_MAX_TEXTURE_SIZE and glGetError() returns
// nonstandard values, so every GLTexture::upload() call throws or leaves
// stale rebuild bookkeeping behind for reasons that are a renderer
// limitation, not a defect in the code under test (confirmed: the real
// display, DISPLAY=:1 with NVIDIA direct rendering, passes the same tests
// cleanly). A GL-dependent test calls this after gl_test_window() and
// returns early when it is false, rather than let the renderer limitation
// surface as an unrelated assertion failure.
inline bool has_usable_gl_context() {
  gl_test_window();
  return dashcore::GLTexture::max_size() > 0;
}

}  // namespace dashcore_test
