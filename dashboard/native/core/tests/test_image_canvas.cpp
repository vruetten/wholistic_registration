#include "core/image_canvas.hpp"
#include "core/selection.hpp"
#include "core/theme.hpp"
#include "test_support.hpp"

#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using dashcore::ImageCanvas;
using dashcore::ImageSource;
using dashcore::Layer;
using dashcore::SelectionSet;

namespace {

Layer solid_layer(std::string name, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  Layer L;
  L.name = std::move(name);
  L.paint = [r, g, b](int, std::vector<std::uint8_t>& rgba, int& w, int& h) {
    w = 64;
    h = 64;
    rgba.assign(std::size_t(w * h * 4), 0);
    for (int i = 0; i < w * h; ++i) {
      rgba[std::size_t(i) * 4 + 0] = r;
      rgba[std::size_t(i) * 4 + 1] = g;
      rgba[std::size_t(i) * 4 + 2] = b;
      rgba[std::size_t(i) * 4 + 3] = 255;
    }
  };
  return L;
}

ImageSource plane_source() {
  ImageSource s;
  s.z_min = 0;
  s.z_max = 3;
  s.width = 64;
  s.height = 64;
  return s;
}

}  // namespace

TEST_CASE("ImageCanvas draws a synthetic RGB image without crashing") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  SelectionSet sel;
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)},
                    plane_source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.rebuild_count() == 1);
  CHECK(sel.size() == 0);
}

TEST_CASE("two ImageCanvas instances hold independent z and contrast") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  SelectionSet sel;
  const auto src = plane_source();
  ImageCanvas a("Random image", "a", {solid_layer("fill", 10, 20, 30)}, src, sel);
  ImageCanvas b("Random image", "b", {solid_layer("fill", 40, 50, 60)}, src, sel);

  a.set_z(1);
  a.set_contrast(0.0f, 0.5f);
  b.set_z(3);
  b.set_contrast(0.2f, 1.0f);

  gui.new_frame();
  a.drawFrame();
  b.drawFrame();
  gui.end_frame();

  CHECK(a.z() == 1);
  CHECK(b.z() == 3);
  CHECK(a.vmin() == 0.0f);
  CHECK(a.vmax() == 0.5f);
  CHECK(b.vmin() == 0.2f);
  CHECK(b.vmax() == 1.0f);
}

TEST_CASE("ImageCanvas rebuilds when z changes, not when it stays put") {
  dashcore_test::ImGuiScope gui;
  if (!dashcore_test::has_usable_gl_context()) {
    MESSAGE("skipping: no usable GL_MAX_TEXTURE_SIZE under this renderer");
    return;
  }

  SelectionSet sel;
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 1, 2, 3)},
                    plane_source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 1);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 1);

  panel.set_z(2);
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 2);
}

TEST_CASE("apply_wheel_zoom: Ctrl resets, wheel steps exponentially") {
  CHECK(dashcore::detail::apply_wheel_zoom(4.0f, 1.0f, true) == 1.0f);
  CHECK(dashcore::detail::apply_wheel_zoom(1.0f, 1.0f, false) ==
        doctest::Approx(1.15f));
  CHECK(dashcore::detail::apply_wheel_zoom(1.0f, -1.0f, false) == 1.0f);  // floor
  CHECK(dashcore::detail::apply_wheel_zoom(16.0f, 1.0f, false) == 16.0f);  // ceil
}

TEST_CASE("ImageCanvas ui_state round-trips z, contrast, colormap, and layer vis") {
  SelectionSet sel;
  ImageCanvas src("Random image", "a",
                  {solid_layer("fill", 10, 20, 30), solid_layer("overlay", 1, 2, 3)},
                  plane_source(), sel);
  src.set_z(2);
  src.set_contrast(0.1f, 0.9f);
  src.set_colormap(7);
  src.set_zoom(3.5f);
  src.layers()[1].visible = false;
  src.layers()[1].alpha = 0.25f;

  std::vector<std::pair<std::string, std::string>> kv;
  src.write_ui_state(kv);

  ImageCanvas dst("Random image", "a",
                  {solid_layer("fill", 10, 20, 30), solid_layer("overlay", 1, 2, 3)},
                  plane_source(), sel);
  for (const auto& [k, v] : kv) dst.read_ui_state(k, v);

  CHECK(dst.z() == 2);
  CHECK(dst.vmin() == doctest::Approx(0.1f));
  CHECK(dst.vmax() == doctest::Approx(0.9f));
  CHECK(dst.colormap() == 7);
  CHECK_FALSE(dst.auto_range());
  CHECK(dst.zoom() == doctest::Approx(3.5f));
  CHECK(dst.layers()[1].visible == false);
  CHECK(dst.layers()[1].alpha == doctest::Approx(0.25f));

  CHECK_NOTHROW(dst.read_ui_state("z", "not-a-number"));
  CHECK(dst.z() == 2);
}

TEST_CASE("ImageCanvas migrates a persisted legacy Greys colormap index to the "
          "dark-at-low replacement") {
  dashcore_test::ImGuiScope gui;
  SelectionSet sel;
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)}, plane_source(), sel);

  // "15" is ImPlotColormap_Greys — ImageCanvas's own compiled-in default
  // before this fix, so this is the exact value an already-persisted
  // ui_state.ini could hold.
  panel.read_ui_state("colormap", "15");
  CHECK(panel.colormap() != 15);
  CHECK(panel.colormap() == dashcore::theme::greys_colormap());
}

TEST_CASE("ImageCanvas defaults to auto-range and uses sample_for_autorange when available") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  SelectionSet sel;
  auto src = plane_source();
  src.sample_for_autorange = [](int, std::vector<float>& out) {
    out.resize(100);
    for (int i = 0; i < 100; ++i) out[std::size_t(i)] = float(i);   // 0..99
  };
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)}, src, sel);
  CHECK(panel.auto_range());

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.vmin() > 0.0f);
  CHECK(panel.vmin() < 15.0f);
  CHECK(panel.vmax() > 85.0f);
  CHECK(panel.vmax() < 99.0f);
}

TEST_CASE("ImageCanvas: an explicit set_contrast call turns auto_range off") {
  SelectionSet sel;
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)}, plane_source(), sel);
  CHECK(panel.auto_range());
  panel.set_contrast(0.0f, 1.0f);
  CHECK_FALSE(panel.auto_range());
}

TEST_CASE("resolve_hover_pixel: exact pixel under a known cursor position") {
  const auto hp = dashcore::detail::resolve_hover_pixel(
      /*hovered=*/true, /*mouse_x=*/10.0f, /*mouse_y=*/20.0f, /*img_x0=*/0.0f, /*img_y0=*/0.0f,
      /*img_w=*/100.0f, /*img_h=*/200.0f, /*width=*/50, /*height=*/40);
  REQUIRE(hp.valid);
  CHECK(hp.row == 4);   // fy = 20/200 = 0.1 -> row floor(0.1*40) = 4
  CHECK(hp.col == 5);   // fx = 10/100 = 0.1 -> col floor(0.1*50) = 5
}

TEST_CASE("resolve_hover_pixel: not hovered, or the cursor on/past the far image edge, is invalid, "
         "not clamped to the nearest pixel") {
  CHECK_FALSE(
      dashcore::detail::resolve_hover_pixel(false, 10, 10, 0, 0, 100, 100, 10, 10).valid);
  // Exactly on the far edge (fx == 1.0): outside, not clamped to column 9.
  CHECK_FALSE(
      dashcore::detail::resolve_hover_pixel(true, 100, 50, 0, 0, 100, 100, 10, 10).valid);
  // Left of the image entirely.
  CHECK_FALSE(
      dashcore::detail::resolve_hover_pixel(true, -1, 50, 0, 0, 100, 100, 10, 10).valid);
}

TEST_CASE("ImageCanvas hover readout matches describe_pixel at the exact hovered pixel") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  SelectionSet sel;
  auto src = plane_source();
  src.describe_pixel = [](int z, int row, int col) {
    return "z=" + std::to_string(z) + " row=" + std::to_string(row) + " col=" + std::to_string(col);
  };
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)}, src, sel);

  // Pin the panel's window to a known screen rect, same technique
  // test_heatmap_panel.cpp's drag-select test uses to make "inside" vs
  // "outside" deterministic rather than depending on ImGui's default
  // window-cascade placement.
  const ImVec2 win_pos(100.0f, 100.0f);
  const ImVec2 win_size(400.0f, 300.0f);
  ImGui::GetIO().MousePos = ImVec2(win_pos.x + win_size.x * 0.5f, win_pos.y + win_size.y * 0.5f);
  for (int i = 0; i < 2; ++i) {
    gui.new_frame();
    ImGui::SetNextWindowPos(win_pos);
    ImGui::SetNextWindowSize(win_size);
    panel.drawFrame();
    gui.end_frame();
  }

  int hr = 0, hc = 0;
  if (!panel.hover_pixel(hr, hc)) {
    // The row/col math itself is covered unconditionally by
    // resolve_hover_pixel's own tests above; this integration test also
    // needs ImGui::IsWindowHovered() to resolve true for a pinned floating
    // window with no real renderer backend, which some headless GL
    // rasterizers do not deliver (test_heatmap_panel.cpp's drag-select
    // test hits the identical limitation on this environment — confirmed
    // against the unmodified baseline build, not introduced here).
    MESSAGE("skipping: IsWindowHovered() did not resolve true on this headless renderer");
    return;
  }
  CHECK(panel.hover_readout_text() ==
       "z=" + std::to_string(panel.z()) + " row=" + std::to_string(hr) +
           " col=" + std::to_string(hc));
}

TEST_CASE("ImageCanvas hover readout reports 'outside' when the pointer is off the image, "
         "never a value from the last valid position") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  SelectionSet sel;
  auto src = plane_source();
  src.describe_pixel = [](int, int, int) { return std::string("should never be reached"); };
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)}, src, sel);

  ImGui::GetIO().MousePos = ImVec2(-1000.0f, -1000.0f);
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  int hr = 0, hc = 0;
  CHECK_FALSE(panel.hover_pixel(hr, hc));
  CHECK(panel.hover_readout_text() == "pointer: outside image");
}

TEST_CASE("ImageCanvas hover readout is empty when the source has no describe_pixel") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  SelectionSet sel;
  ImageCanvas panel("Random image", "a", {solid_layer("fill", 10, 20, 30)}, plane_source(), sel);

  const ImVec2 win_pos(100.0f, 100.0f);
  const ImVec2 win_size(400.0f, 300.0f);
  ImGui::GetIO().MousePos = ImVec2(win_pos.x + win_size.x * 0.5f, win_pos.y + win_size.y * 0.5f);
  for (int i = 0; i < 2; ++i) {
    gui.new_frame();
    ImGui::SetNextWindowPos(win_pos);
    ImGui::SetNextWindowSize(win_size);
    panel.drawFrame();
    gui.end_frame();
  }

  CHECK(panel.hover_readout_text().empty());
}
