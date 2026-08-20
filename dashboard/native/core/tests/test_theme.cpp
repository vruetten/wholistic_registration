#include "core/colormap.hpp"
#include "core/theme.hpp"
#include "test_support.hpp"

#include <doctest/doctest.h>

#include <vector>

TEST_CASE("Scaled() is the identity before apply() has executed") {
  // A fresh test binary process may exercise this before any apply() call;
  // documented behavior is "unscaled," not zero or undefined.
  CHECK(dashcore::theme::Scaled(10.0f) == doctest::Approx(10.0f));
}

TEST_CASE("apply() sets the scale Scaled() subsequently reports") {
  dashcore_test::ImGuiScope gui;
  dashcore::theme::apply(2.0f);
  CHECK(dashcore::theme::Scaled(10.0f) == doctest::Approx(20.0f));

  dashcore::theme::apply(1.0f);
  CHECK(dashcore::theme::Scaled(10.0f) == doctest::Approx(10.0f));
}

TEST_CASE("apply() rejects a non-positive scale by falling back to 1.0") {
  dashcore_test::ImGuiScope gui;
  dashcore::theme::apply(0.0f);
  CHECK(dashcore::theme::Scaled(10.0f) == doctest::Approx(10.0f));
  dashcore::theme::apply(-3.0f);
  CHECK(dashcore::theme::Scaled(10.0f) == doctest::Approx(10.0f));
}

TEST_CASE("content_scale() reads a real value from a live window") {
  GLFWwindow* w = dashcore_test::gl_test_window();
  const float scale = dashcore::theme::content_scale(w);
  CHECK(scale >= 1.0f);
}

TEST_CASE("magma_colormap() registers once per context and is idempotent within it") {
  dashcore_test::ImGuiScope gui;
  const int first = dashcore::theme::magma_colormap();
  const int second = dashcore::theme::magma_colormap();
  CHECK(first == second);
  CHECK(first >= 0);
}

TEST_CASE("sequential_colormaps() offers magma first, plus viridis/greys/RdBu") {
  dashcore_test::ImGuiScope gui;
  const auto maps = dashcore::theme::sequential_colormaps();
  REQUIRE(maps.size() == 4);
  CHECK(maps[0] == dashcore::theme::magma_colormap());
}

TEST_CASE("next_colormap cycles through the given options, wrapping") {
  const std::vector<int> options = {10, 20, 30};
  CHECK(dashcore::theme::next_colormap(10, options) == 20);
  CHECK(dashcore::theme::next_colormap(20, options) == 30);
  CHECK(dashcore::theme::next_colormap(30, options) == 10);
}

TEST_CASE("next_colormap falls back to the first option for an unrecognized current value") {
  const std::vector<int> options = {10, 20, 30};
  CHECK(dashcore::theme::next_colormap(999, options) == 10);
}

TEST_CASE("greys_colormap() registers once per context and runs dark-at-low") {
  dashcore_test::ImGuiScope gui;
  const int first = dashcore::theme::greys_colormap();
  const int second = dashcore::theme::greys_colormap();
  CHECK(first == second);
  CHECK(first >= 0);

  // The regression this app shipped: ImPlot's own built-in Greys runs
  // white-at-low/black-at-high, the opposite polarity from every other
  // sequential map here. greys_colormap() must not reproduce that.
  const auto lo = dashcore::colormap_rgba(0.0f, 0.0f, 1.0f, first);
  const auto hi = dashcore::colormap_rgba(1.0f, 0.0f, 1.0f, first);
  CHECK(lo == std::array<std::uint8_t, 4>{0, 0, 0, 255});
  CHECK(hi == std::array<std::uint8_t, 4>{255, 255, 255, 255});
}

TEST_CASE("migrate_legacy_colormap remaps the built-in Greys index to the "
          "dark-at-low replacement") {
  dashcore_test::ImGuiScope gui;
  const int legacy = int(ImPlotColormap_Greys);
  const int migrated = dashcore::migrate_legacy_colormap(legacy);
  CHECK(migrated == dashcore::theme::greys_colormap());
  CHECK(migrated != legacy);

  const auto lo = dashcore::colormap_rgba(0.0f, 0.0f, 1.0f, migrated);
  const auto hi = dashcore::colormap_rgba(1.0f, 0.0f, 1.0f, migrated);
  CHECK(lo == std::array<std::uint8_t, 4>{0, 0, 0, 255});
  CHECK(hi == std::array<std::uint8_t, 4>{255, 255, 255, 255});
}

TEST_CASE("migrate_legacy_colormap leaves every other index unchanged") {
  dashcore_test::ImGuiScope gui;
  CHECK(dashcore::migrate_legacy_colormap(int(ImPlotColormap_Viridis)) ==
       int(ImPlotColormap_Viridis));
  CHECK(dashcore::migrate_legacy_colormap(dashcore::theme::magma_colormap()) ==
       dashcore::theme::magma_colormap());
}

TEST_CASE("every colormap sequential_colormaps() offers is dark-at-low, bright-at-high") {
  // Enumerates the registry itself, not a hardcoded list of names — a
  // colormap added to sequential_colormaps() in the future is covered here
  // automatically. This is the exact check that would have failed against
  // the shipped default (ImPlotColormap_Greys, white-at-low/black-at-high)
  // before greys_colormap() replaced it.
  dashcore_test::ImGuiScope gui;
  for (int cmap : dashcore::theme::sequential_colormaps()) {
    const auto lo = dashcore::colormap_rgba(0.0f, 0.0f, 1.0f, cmap);
    const auto hi = dashcore::colormap_rgba(1.0f, 0.0f, 1.0f, cmap);
    const int lum_lo = int(lo[0]) + int(lo[1]) + int(lo[2]);
    const int lum_hi = int(hi[0]) + int(hi[1]) + int(hi[2]);
    const char* name = ImPlot::GetColormapName(ImPlotColormap(cmap));
    CAPTURE(name);
    CAPTURE(lum_lo);
    CAPTURE(lum_hi);
    CHECK(lum_hi > lum_lo);
  }
}
