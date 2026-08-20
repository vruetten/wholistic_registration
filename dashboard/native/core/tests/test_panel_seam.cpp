// The seam this whole build is meant to prove: dashcore compiles and links
// with no other directory on its include path — there is no app/ anywhere
// in this repository yet, and this test target (see tests/CMakeLists.txt)
// adds nothing but its own tests/ folder beyond what the `dashcore` target
// itself exposes — and every panel base class here can be driven end to end
// by data that has nothing to do with any particular pipeline: a table of
// grocery items and a matrix of uniform random numbers.
//
// A version of Panel that took an application-context reference in its
// abstract interface would still pass a compile check that merely built
// dashcore in isolation; it would not pass this one, because there would be
// no way to construct a TablePanel or HeatmapPanel here without also
// constructing whatever application type that reference named.
#include "core/heatmap_panel.hpp"
#include "core/image_canvas.hpp"
#include "core/selection.hpp"
#include "core/table_panel.hpp"
#include "test_support.hpp"

#include <doctest/doctest.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

using dashcore::ColumnType;
using dashcore::HeatmapPanel;
using dashcore::HeatmapSource;
using dashcore::ImageCanvas;
using dashcore::ImageSource;
using dashcore::Layer;
using dashcore::SelectionSet;
using dashcore::SortSpec;
using dashcore::TableColumn;
using dashcore::TablePanel;

namespace {

struct GroceryList {
  std::vector<std::string> item = {"eggs", "flour", "milk", "sugar", "yeast", "butter"};
  std::vector<int64_t> quantity = {12, 2, 1, 5, 3, 8};
};

std::vector<TableColumn> grocery_columns(GroceryList& g) {
  return {
      TableColumn{"item", ColumnType::String,
                 [&g](int64_t r) { return g.item[std::size_t(r)]; }, nullptr},
      TableColumn{"quantity", ColumnType::Int64,
                 [&g](int64_t r) { return std::to_string(g.quantity[std::size_t(r)]); },
                 [&g](int64_t r) { return double(g.quantity[std::size_t(r)]); }},
  };
}

struct UniformRandomMatrix {
  int64_t rows, cols;
  std::vector<float> values;

  UniformRandomMatrix(int64_t rows_, int64_t cols_, unsigned seed)
      : rows(rows_), cols(cols_), values(std::size_t(rows_ * cols_)) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : values) v = dist(rng);
  }

  HeatmapSource source() {
    HeatmapSource s;
    s.rows = rows;
    s.cols = cols;
    s.fetch_rows = [this](int64_t r0, int64_t r1, std::vector<float>& out) {
      out.assign(values.begin() + r0 * cols, values.begin() + r1 * cols);
    };
    s.row_to_item = [](int64_t r) { return r; };
    s.item_to_row = [](int64_t item) { return item; };
    return s;
  }
};

}  // namespace

TEST_CASE("seam: TablePanel over a grocery list, no pipeline type in sight") {
  dashcore_test::ImGuiScope gui;
  GroceryList g;
  TablePanel panel("Groceries", "a", grocery_columns(g), [&g] { return int64_t(g.item.size()); });

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.display_order().size() == g.item.size());
}

TEST_CASE("seam: HeatmapPanel over a uniform random matrix, no pipeline type in sight") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  UniformRandomMatrix m(64, 16, /*seed=*/123);
  SelectionSet sel;
  HeatmapPanel panel("Random matrix", "a", m.source(), sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.chunk_count() == 1);
}

TEST_CASE("seam: ImageCanvas over a random RGB image, no pipeline type in sight") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  Layer fill;
  fill.name = "fill";
  fill.paint = [](int, std::vector<std::uint8_t>& rgba, int& w, int& h) {
    w = 64;
    h = 64;
    rgba.assign(std::size_t(w * h * 4), 80);
    for (int i = 0; i < w * h; ++i) rgba[std::size_t(i) * 4 + 3] = 255;
  };
  ImageSource src;
  src.z_min = 0;
  src.z_max = 1;
  src.width = 64;
  src.height = 64;
  SelectionSet sel;
  ImageCanvas panel("Random image", "a", {std::move(fill)}, src, sel);

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.rebuild_count() == 1);
}

TEST_CASE("seam: two-instance swap, one of each class, independent state") {
  dashcore_test::ImGuiScope gui;
  dashcore_test::gl_test_window();

  GroceryList g;
  TablePanel table_a("Groceries", "a", grocery_columns(g), [&g] { return int64_t(g.item.size()); });
  TablePanel table_b("Groceries", "b", grocery_columns(g), [&g] { return int64_t(g.item.size()); });
  table_a.set_sort({{1, true}});    // by quantity ascending
  table_b.set_sort({{0, true}});    // by item name ascending

  UniformRandomMatrix m(20, 5, /*seed=*/9);
  SelectionSet sel_a, sel_b;
  HeatmapPanel heatmap_a("Random matrix", "a", m.source(), sel_a);
  HeatmapPanel heatmap_b("Random matrix", "b", m.source(), sel_b);
  heatmap_a.set_contrast(-1.0f, 0.0f);
  heatmap_b.set_contrast(0.0f, 1.0f);

  gui.new_frame();
  table_a.drawFrame();
  table_b.drawFrame();
  heatmap_a.drawFrame();
  heatmap_b.drawFrame();
  gui.end_frame();

  CHECK(table_a.display_order() != table_b.display_order());
  CHECK(heatmap_a.vmin() != heatmap_b.vmin());
  CHECK(heatmap_a.vmax() != heatmap_b.vmax());
}
