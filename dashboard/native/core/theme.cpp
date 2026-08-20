#include "core/theme.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <array>
#include <filesystem>

namespace dashcore::theme {

namespace {

// One process-wide scale, mirroring the way ImGui itself already exposes a
// single current context via ImGui::GetIO() — a second "current" global on
// top of that isn't adding a new kind of state, just a convenience reader
// for a value every panel needs. Set once at startup by `apply()`.
float g_scale = 1.0f;

const char* const kCandidateFonts[] = {
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
};

// 32-stop Matplotlib "magma" sample, linearly interpolated by ImPlot between
// stops. Lifted from icampsnfr:src/util/colormap.cpp's kMagma table (same
// values) — ImPlot's own built-in set has no perceptually-uniform "magma"
// entry (Viridis and Plasma are its closest built-ins), so this app supplies
// its own rather than substituting a different map under the same name.
constexpr std::array<ImVec4, 32> kMagmaStops = {{
    {0.001f, 0.000f, 0.014f, 1.0f}, {0.017f, 0.011f, 0.061f, 1.0f},
    {0.049f, 0.030f, 0.126f, 1.0f}, {0.089f, 0.049f, 0.198f, 1.0f},
    {0.136f, 0.062f, 0.271f, 1.0f}, {0.190f, 0.070f, 0.324f, 1.0f},
    {0.243f, 0.078f, 0.354f, 1.0f}, {0.294f, 0.089f, 0.372f, 1.0f},
    {0.343f, 0.101f, 0.383f, 1.0f}, {0.391f, 0.113f, 0.389f, 1.0f},
    {0.441f, 0.121f, 0.391f, 1.0f}, {0.489f, 0.133f, 0.389f, 1.0f},
    {0.539f, 0.145f, 0.383f, 1.0f}, {0.588f, 0.156f, 0.373f, 1.0f},
    {0.638f, 0.169f, 0.359f, 1.0f}, {0.687f, 0.184f, 0.340f, 1.0f},
    {0.735f, 0.200f, 0.318f, 1.0f}, {0.780f, 0.221f, 0.294f, 1.0f},
    {0.822f, 0.246f, 0.268f, 1.0f}, {0.859f, 0.276f, 0.243f, 1.0f},
    {0.891f, 0.310f, 0.220f, 1.0f}, {0.918f, 0.345f, 0.198f, 1.0f},
    {0.941f, 0.383f, 0.179f, 1.0f}, {0.958f, 0.424f, 0.164f, 1.0f},
    {0.972f, 0.466f, 0.154f, 1.0f}, {0.982f, 0.509f, 0.147f, 1.0f},
    {0.988f, 0.553f, 0.144f, 1.0f}, {0.994f, 0.596f, 0.144f, 1.0f},
    {0.996f, 0.641f, 0.148f, 1.0f}, {0.996f, 0.686f, 0.161f, 1.0f},
    {0.995f, 0.783f, 0.208f, 1.0f}, {0.987f, 0.991f, 0.749f, 1.0f},
}};

}  // namespace

float content_scale(GLFWwindow* window) {
  float x = 1.0f, y = 1.0f;
  if (window) glfwGetWindowContentScale(window, &x, &y);
  return std::max(1.0f, std::min(x, y));
}

void apply(float scale) {
  g_scale = (scale > 0.0f) ? scale : 1.0f;

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  // ~6 overrides on stock Dark (Tracy main.cpp:180-191 pattern): a touch of
  // rounding so controls don't read as 2008-era square, and an accent color
  // on the active tab/title so the focused panel is unambiguous at a glance.
  style.WindowRounding   = 4.0f;
  style.FrameRounding    = 3.0f;
  style.ScrollbarRounding = 3.0f;
  style.Colors[ImGuiCol_WindowBg].w    = 0.95f;
  style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.41f, 0.68f, 1.00f);
  style.Colors[ImGuiCol_TabActive]     = ImVec4(0.20f, 0.41f, 0.68f, 1.00f);
  style.ScaleAllSizes(g_scale);

  ImGuiIO& io = ImGui::GetIO();
  const float px = 15.0f * g_scale;
  for (const char* path : kCandidateFonts) {
    if (std::filesystem::exists(path)) {
      io.Fonts->AddFontFromFileTTF(path, px);
      return;
    }
  }
  // No system TTF found: still honor the scaled pixel size on the built-in
  // atlas rather than silently staying at the unscaled default.
  ImFontConfig cfg;
  cfg.SizePixels = px;
  io.Fonts->AddFontDefault(&cfg);
}

float Scaled(float px) { return px * g_scale; }

int magma_colormap() {
  const int existing = ImPlot::GetColormapIndex("Magma");
  if (existing != -1) return existing;
  return ImPlot::AddColormap("Magma", kMagmaStops.data(), int(kMagmaStops.size()),
                             /*qual=*/false);
}

int greys_colormap() {
  const int existing = ImPlot::GetColormapIndex("Greys (dark-low)");
  if (existing != -1) return existing;
  static constexpr std::array<ImU32, 2> kStops = {IM_COL32_BLACK, IM_COL32_WHITE};
  return ImPlot::AddColormap("Greys (dark-low)", kStops.data(), int(kStops.size()),
                             /*qual=*/false);
}

std::vector<int> sequential_colormaps() {
  return {magma_colormap(), ImPlotColormap_Viridis, greys_colormap(), ImPlotColormap_RdBu};
}

int next_colormap(int current, const std::vector<int>& options) {
  if (options.empty()) return current;
  const auto it = std::find(options.begin(), options.end(), current);
  if (it == options.end()) return options.front();
  const std::size_t idx = std::size_t(it - options.begin());
  return options[(idx + 1) % options.size()];
}

}  // namespace dashcore::theme
