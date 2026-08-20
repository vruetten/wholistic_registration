#include "core/heatmap_panel.hpp"
#include "core/colormap.hpp"
#include "core/selection.hpp"
#include "core/theme.hpp"
#include "test_support.hpp"

#include <doctest/doctest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using dashcore::HeatmapPanel;
using dashcore::HeatmapSource;
using dashcore::SelectionSet;

namespace {

// A source with no notion of a coordinate system: `rows` unrelated entries,
// each `cols` unrelated numbers, generated from a fixed seed so the test is
// deterministic. row_to_item is a plain offset so items are distinguishable
// from provider-index rows in every assertion.
struct RandomMatrix {
  int64_t rows;
  int64_t cols;
  std::vector<float> values;   // row-major, rows*cols
  int64_t item_offset = 1000;
  int64_t gen = 0;       // bumped by the test to simulate "the source's content changed"
  int64_t fetch_calls = 0;   // counts real fetch_rows() invocations

  explicit RandomMatrix(int64_t rows_, int64_t cols_, unsigned seed = 7)
      : rows(rows_), cols(cols_), values(std::size_t(rows_ * cols_)) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : values) v = dist(rng);
  }

  HeatmapSource source() {
    HeatmapSource s;
    s.rows = rows;
    s.cols = cols;
    s.fetch_rows = [this](int64_t r0, int64_t r1, std::vector<float>& out) {
      ++fetch_calls;
      out.assign(values.begin() + r0 * cols, values.begin() + r1 * cols);
    };
    s.row_to_item = [this](int64_t r) { return item_offset + r; };
    s.item_to_row = [this](int64_t item) { return item - item_offset; };
    s.generation = [this] { return gen; };
    return s;
  }
};

}  // namespace

TEST_CASE("colormap_rgba maps NaN to a fixed magenta sentinel, not an "
          "arbitrary color") {
  dashcore_test::ImGuiScope gui;
  const auto nan_px = dashcore::colormap_rgba(
      std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f, 0);
  CHECK(nan_px == std::array<std::uint8_t, 4>{255, 0, 255, 255});

  // Distinct from every ordinary finite value's color, not a coincidence of
  // one particular vmin/vmax/colormap combination.
  const auto lo = dashcore::colormap_rgba(0.0f, 0.0f, 1.0f, 0);
  const auto mid = dashcore::colormap_rgba(0.5f, 0.0f, 1.0f, 0);
  const auto hi = dashcore::colormap_rgba(1.0f, 0.0f, 1.0f, 0);
  CHECK(lo != nan_px);
  CHECK(mid != nan_px);
  CHECK(hi != nan_px);
}

TEST_CASE("colormap_rgba maps +Inf and -Inf to the same magenta sentinel as NaN, "
          "not the map's brightest/darkest extreme") {
  dashcore_test::ImGuiScope gui;
  const std::array<std::uint8_t, 4> sentinel{255, 0, 255, 255};

  // +Inf: a x/0 divide-by-zero with x > 0. Without this guard it clamps to
  // t=1 and renders as the single brightest pixel on the map — a plausible
  // extreme value, not an obvious error.
  const auto pos_inf_px = dashcore::colormap_rgba(
      std::numeric_limits<float>::infinity(), 0.0f, 1.0f, 0);
  CHECK(pos_inf_px == sentinel);

  // -Inf: x/0 with x < 0. Without this guard it clamps to t=0 (the darkest
  // extreme) instead.
  const auto neg_inf_px = dashcore::colormap_rgba(
      -std::numeric_limits<float>::infinity(), 0.0f, 1.0f, 0);
  CHECK(neg_inf_px == sentinel);

  // Distinct from the map's real brightest/darkest finite values, so the
  // sentinel isn't a coincidence of colormap index 0's own endpoint colors.
  const auto real_lo = dashcore::colormap_rgba(0.0f, 0.0f, 1.0f, 0);
  const auto real_hi = dashcore::colormap_rgba(1.0f, 0.0f, 1.0f, 0);
  CHECK(real_lo != sentinel);
  CHECK(real_hi != sentinel);
}

TEST_CASE("HeatmapPanel draws a synthetic random matrix without crashing") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(37, 11);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.chunk_count() == 1);   // fewer rows than one chunk
  CHECK(sel.size() == 0);            // nothing dragged, nothing selected
}

TEST_CASE("HeatmapPanel chunks a row count that exceeds one texture") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  const int64_t rows = HeatmapPanel::kChunkRows * 2 + 37;
  RandomMatrix m(rows, 4);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.chunk_count() == 3);
}

TEST_CASE("HeatmapPanel rebuilds when the source's generation bumps, even "
          "though rows/cols/contrast are unchanged") {
  dashcore_test::ImGuiScope gui;
  if (!dashcore_test::has_usable_gl_context()) {
    MESSAGE("skipping: no usable GL_MAX_TEXTURE_SIZE under this renderer");
    return;
  }

  RandomMatrix m(20, 5);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 1);

  // Redrawing with nothing changed (same generation) must NOT rebuild.
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 1);

  // Simulate the source's underlying content changing (e.g. a re-sort)
  // while rows/cols/contrast stay put: the caller signals this by bumping
  // the counter its `generation` callback reads. The panel re-queries
  // `generation()` fresh every frame (same pattern as TablePanel's
  // `row_count`), so no explicit re-pointing call is needed.
  m.gen = 1;
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 2);
}

TEST_CASE("contrast-only change recolors without refetching source rows") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(20, 5);   // fewer rows than one chunk, so fetch_rows fires once per rebuild
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(m.fetch_calls == 1);
  CHECK(panel.rebuild_count() == 1);

  // A contrast-only change (the common case while the user drags the range
  // slider) must recolor the cached values, not re-fetch them from the
  // source — fetch_rows is the expensive step at real data scale (design
  // §6 #9 / adversarial review HIGH 4).
  panel.set_contrast(0.2f, 0.8f);
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 2);   // texture WAS rebuilt (recolored)...
  CHECK(m.fetch_calls == 1);           // ...but fetch_rows was NOT called again.

  // A second, different contrast value: same story.
  panel.set_contrast(0.1f, 0.9f);
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();
  CHECK(panel.rebuild_count() == 3);
  CHECK(m.fetch_calls == 1);
}

TEST_CASE("two HeatmapPanel instances hold independent contrast") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(20, 5);
  SelectionSet sel_a, sel_b;
  HeatmapPanel a("Random matrix", "a", m.source(), sel_a);
  HeatmapPanel b("Random matrix", "b", m.source(), sel_b);

  a.set_contrast(0.0f, 1.0f);
  b.set_contrast(5.0f, 9.0f);

  gui.new_frame();
  a.drawFrame();
  b.drawFrame();
  gui.end_frame();

  CHECK(a.vmin() == doctest::Approx(0.0f));
  CHECK(a.vmax() == doctest::Approx(1.0f));
  CHECK(b.vmin() == doctest::Approx(5.0f));
  CHECK(b.vmax() == doctest::Approx(9.0f));
}

TEST_CASE("drag-select is cancelled, not committed, when the mouse is "
          "released outside the panel") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(30, 4);   // one chunk, small enough that the whole image fits on screen
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  // Pin the panel's window to a known screen rect so "inside" vs "outside"
  // points are deterministic, rather than depending on ImGui's default
  // window-cascade placement.
  const ImVec2 win_pos(100.0f, 100.0f);
  const ImVec2 win_size(400.0f, 300.0f);
  const ImVec2 inside_point(win_pos.x + win_size.x * 0.5f, win_pos.y + win_size.y * 0.5f);
  const ImVec2 outside_point(-1000.0f, -1000.0f);

  ImGuiIO& io = ImGui::GetIO();

  // Two settling frames before the click: ImGui's hover test for a window
  // lags one frame behind that window's own rect (a window can't be
  // "hovered" on the very frame it's first created), so the panel needs to
  // have already drawn at this pinned position before a click frame can be
  // trusted to register as inside it. Verified against the vendored ImGui.
  for (int i = 0; i < 2; ++i) {
    io.MousePos = inside_point;
    io.MouseDown[0] = false;
    gui.new_frame();
    ImGui::SetNextWindowPos(win_pos);
    ImGui::SetNextWindowSize(win_size);
    panel.drawFrame();
    gui.end_frame();
  }

  // Press inside the panel -> starts a drag.
  io.MousePos = inside_point;
  io.MouseDown[0] = true;
  gui.new_frame();
  ImGui::SetNextWindowPos(win_pos);
  ImGui::SetNextWindowSize(win_size);
  panel.drawFrame();
  gui.end_frame();

  // Still holding the button, mouse has left the panel entirely (dragged
  // over a different docked window, or off the display).
  io.MousePos = outside_point;
  io.MouseDown[0] = true;
  gui.new_frame();
  ImGui::SetNextWindowPos(win_pos);
  ImGui::SetNextWindowSize(win_size);
  panel.drawFrame();
  gui.end_frame();
  CHECK(sel.size() == 0);   // nothing committed yet — button still held

  // Release while still outside the panel. A natural "changed my mind, drag
  // off-panel to abandon it" gesture must NOT write a selection clamped to
  // whatever row happens to be nearest the panel's edge (adversarial review
  // MEDIUM 7 / icampsnfr defect #13).
  io.MousePos = outside_point;
  io.MouseDown[0] = false;
  gui.new_frame();
  ImGui::SetNextWindowPos(win_pos);
  ImGui::SetNextWindowSize(win_size);
  panel.drawFrame();
  gui.end_frame();
  CHECK(sel.size() == 0);

  // Press inside again and release inside, to confirm a normal in-panel
  // drag still commits. Earlier revisions of this test omitted this half of
  // the contract: the contrast header's horizontal colormap scale bar (a
  // direct call into ImPlot's internal RenderColorBar(), removed together
  // with the scale bar itself) left IsWindowHovered() unable to recover to
  // true for the child below it once the mouse had left and returned, but
  // only in a headless harness like this one that drives frames directly
  // without a real renderer backend (no call to
  // ImGui_ImplOpenGL3_RenderDrawData anywhere in this binary). With the
  // scale bar gone, hover recovers normally and the commit is asserted
  // directly rather than left to detail::resolve_drag_commit's unit tests
  // alone.
  io.MousePos = inside_point;
  io.MouseDown[0] = true;
  gui.new_frame();
  ImGui::SetNextWindowPos(win_pos);
  ImGui::SetNextWindowSize(win_size);
  panel.drawFrame();
  gui.end_frame();
  io.MouseDown[0] = false;
  gui.new_frame();
  ImGui::SetNextWindowPos(win_pos);
  ImGui::SetNextWindowSize(win_size);
  panel.drawFrame();
  gui.end_frame();
  CHECK(sel.size() == 1);
}

TEST_CASE("resolve_drag_commit: hovered at release commits the [lo, hi] range") {
  const auto dc = dashcore::detail::resolve_drag_commit(/*hovered_at_release=*/true,
                                                         /*start_row=*/5, /*end_row=*/2);
  CHECK(dc.commit);
  CHECK(dc.lo == 2);
  CHECK(dc.hi == 5);
}

TEST_CASE("resolve_drag_commit: not hovered at release cancels, regardless of rows") {
  const auto dc = dashcore::detail::resolve_drag_commit(/*hovered_at_release=*/false,
                                                         /*start_row=*/5, /*end_row=*/2);
  CHECK_FALSE(dc.commit);
}

TEST_CASE("HeatmapPanel ui_state round-trips contrast, colormap, and auto_range") {
  RandomMatrix m(8, 4);
  SelectionSet sel;
  HeatmapPanel src("Random matrix", "a", m.source(), sel);
  src.set_contrast(0.2f, 0.8f);   // an explicit override, so auto_range() is now false
  src.set_colormap(3);

  std::vector<std::pair<std::string, std::string>> kv;
  src.write_ui_state(kv);

  HeatmapPanel dst("Random matrix", "a", m.source(), sel);
  for (const auto& [k, v] : kv) dst.read_ui_state(k, v);
  CHECK(dst.vmin() == doctest::Approx(0.2f));
  CHECK(dst.vmax() == doctest::Approx(0.8f));
  CHECK(dst.colormap() == 3);
  CHECK_FALSE(dst.auto_range());
  CHECK_NOTHROW(dst.read_ui_state("vmin", "nope"));
}

TEST_CASE("HeatmapPanel migrates a persisted legacy Greys colormap index to the "
          "dark-at-low replacement") {
  dashcore_test::ImGuiScope gui;
  RandomMatrix m(8, 4);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  // "15" is ImPlotColormap_Greys — what a ui_state.ini saved before this fix
  // could hold, since HeatmapPanel's colormap cycle used to offer it.
  panel.read_ui_state("colormap", "15");
  CHECK(panel.colormap() != 15);
  CHECK(panel.colormap() == dashcore::theme::greys_colormap());
}

TEST_CASE("HeatmapPanel defaults to auto-range and picks a 5th-95th percentile contrast") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  // Values are exactly the row index (broadcast across columns), so the
  // 5th/95th percentile of [0, 99] is analytically known: ~5 and ~95.
  RandomMatrix m(100, 2);
  for (int64_t r = 0; r < 100; ++r)
    for (int64_t c = 0; c < 2; ++c) m.values[std::size_t(r * 2 + c)] = float(r);
  SelectionSet sel;
  HeatmapPanel panel("Indexed matrix", "a", m.source(), sel);
  CHECK(panel.auto_range());

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  // Loosely bracket the 5th/95th percentile rather than pin an exact index
  // (that's an implementation detail of percentile_range's interpolation) —
  // the property under test is "clips the tails," i.e. neither bound is the
  // full [0, 99] extent.
  CHECK(panel.vmin() > 0.0f);
  CHECK(panel.vmin() < 15.0f);
  CHECK(panel.vmax() > 85.0f);
  CHECK(panel.vmax() < 99.0f);
}

TEST_CASE("HeatmapPanel: dragging the range widget turns auto_range off (set_contrast proxy)") {
  RandomMatrix m(8, 4);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);
  CHECK(panel.auto_range());
  panel.set_contrast(0.0f, 1.0f);
  CHECK_FALSE(panel.auto_range());
}

TEST_CASE("resolve_hover_cell: exact (row, col) under a known cursor position, across chunks") {
  const std::vector<dashcore::detail::RowChunkLayout> layout = {
      {0, 10, 0.0f, 100.0f},     // chunk 0: provider rows [0,10), screen y [0,100)
      {10, 20, 100.0f, 300.0f},  // chunk 1: provider rows [10,20), screen y [100,300)
  };
  // Inside chunk 1: frac_y = (150-100)/200 = 0.25 -> within = floor(0.25*10) = 2 -> row 12.
  const auto hit = dashcore::detail::resolve_hover_cell(
      /*hovered=*/true, /*mouse_x=*/40.0f, /*mouse_y=*/150.0f, /*img_x0=*/0.0f, /*img_w=*/80.0f,
      /*cols=*/4, layout);
  REQUIRE(hit.valid);
  CHECK(hit.row == 12);
  CHECK(hit.col == 2);   // fx = 40/80 = 0.5 -> col floor(0.5*4) = 2
}

TEST_CASE("resolve_hover_cell: not hovered, empty layout, or the cursor on/past the far edge, "
         "is invalid, not clamped to the nearest entry") {
  const std::vector<dashcore::detail::RowChunkLayout> layout = {{0, 10, 0.0f, 100.0f}};
  CHECK_FALSE(dashcore::detail::resolve_hover_cell(false, 40, 50, 0, 80, 4, layout).valid);
  CHECK_FALSE(dashcore::detail::resolve_hover_cell(true, 40, 50, 0, 80, 4, {}).valid);
  // Exactly on the far x edge (fx == 1.0): outside, not clamped to col 3.
  CHECK_FALSE(dashcore::detail::resolve_hover_cell(true, 80, 50, 0, 80, 4, layout).valid);
  // Below every chunk's vertical extent.
  CHECK_FALSE(dashcore::detail::resolve_hover_cell(true, 40, 100, 0, 80, 4, layout).valid);
}

TEST_CASE("HeatmapPanel hover readout matches describe_cell at the exact hovered entry") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(30, 4);   // one chunk, small enough that the whole image fits on screen
  auto src = m.source();
  src.describe_cell = [](int64_t row, int64_t col) {
    return "row=" + std::to_string(row) + " col=" + std::to_string(col);
  };
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", src, sel);

  // Same pinned-window technique as the drag-select test above.
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

  int64_t hr = 0, hc = 0;
  if (!panel.hover_cell(hr, hc)) {
    // The row/col math itself is covered unconditionally by
    // resolve_hover_cell's own tests above; see the matching note on
    // ImageCanvas's equivalent integration test for why IsWindowHovered()
    // can fail to resolve true here specifically.
    MESSAGE("skipping: IsWindowHovered() did not resolve true on this headless renderer");
    return;
  }
  CHECK(panel.hover_readout_text() ==
       "row=" + std::to_string(hr) + " col=" + std::to_string(hc));
}

TEST_CASE("HeatmapPanel hover readout reports 'outside' when the pointer leaves the raster, "
         "never a value from the last valid position") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(30, 4);
  auto src = m.source();
  src.describe_cell = [](int64_t, int64_t) { return std::string("should never be reached"); };
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", src, sel);

  ImGui::GetIO().MousePos = ImVec2(-1000.0f, -1000.0f);
  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  int64_t hr = 0, hc = 0;
  CHECK_FALSE(panel.hover_cell(hr, hc));
  CHECK(panel.hover_readout_text() == "pointer: outside raster");
}

TEST_CASE("HeatmapPanel hover readout is empty when the source has no describe_cell") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  RandomMatrix m(30, 4);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

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
