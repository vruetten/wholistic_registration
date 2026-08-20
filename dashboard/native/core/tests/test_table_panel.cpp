#include "core/table_panel.hpp"
#include "test_support.hpp"

#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

using dashcore::ColumnType;
using dashcore::SortSpec;
using dashcore::TableColumn;
using dashcore::TablePanel;

namespace {

// Two columns over five unrelated fixed-point entries — nothing spatial or
// pipeline-shaped, just enough structure to exercise numeric and lexical
// sorting.
struct Entries {
  std::vector<std::string> name = {"pear", "apple", "fig", "date", "banana"};
  std::vector<double> price = {2.5, 1.0, 4.0, 3.0, 1.0};
};

std::vector<TableColumn> columns_for(const Entries& e) {
  return {
      TableColumn{"name", ColumnType::String,
                 [&e](int64_t r) { return e.name[std::size_t(r)]; }, nullptr},
      TableColumn{"price", ColumnType::Double,
                 [&e](int64_t r) { return std::to_string(e.price[std::size_t(r)]); },
                 [&e](int64_t r) { return e.price[std::size_t(r)]; }},
  };
}

}  // namespace

TEST_CASE("sort_rows refuses a numeric column with a null sort_key") {
  // A Double column that forgot to set sort_key: without a guard, this
  // silently sorts on to_string() lexicographically ("10" < "2" < "9"),
  // which is exactly wrong for the panel's actual purpose (sort cells by a
  // numeric stat to find outliers). It must throw instead of degrading.
  std::vector<double> price = {2.5, 1.0, 4.0};
  std::vector<TableColumn> cols = {
      TableColumn{"price", ColumnType::Double,
                 [&price](int64_t r) { return std::to_string(price[std::size_t(r)]); },
                 nullptr},   // BUG: forgot sort_key
  };
  CHECK_THROWS_AS(dashcore::sort_rows(cols, int64_t(price.size()), {{0, true}}),
                  std::invalid_argument);
}

TEST_CASE("TablePanel refuses construction with a numeric column missing sort_key") {
  std::vector<int64_t> quantity = {1, 2, 3};
  std::vector<TableColumn> cols = {
      TableColumn{"quantity", ColumnType::Int64,
                 [&quantity](int64_t r) { return std::to_string(quantity[std::size_t(r)]); },
                 nullptr},   // BUG: forgot sort_key
  };
  CHECK_THROWS_AS(
      TablePanel("Broken", "a", cols, [&quantity] { return int64_t(quantity.size()); }),
      std::invalid_argument);
}

TEST_CASE("sort_rows: empty specs is the identity permutation") {
  Entries e;
  const auto cols = columns_for(e);
  const auto order = dashcore::sort_rows(cols, int64_t(e.name.size()), {});
  CHECK(order == std::vector<int64_t>{0, 1, 2, 3, 4});
}

TEST_CASE("sort_rows: ascending numeric sort with a stable tie-break") {
  Entries e;
  const auto cols = columns_for(e);
  const auto order = dashcore::sort_rows(cols, int64_t(e.name.size()), {{1, true}});
  // price: apple=1.0, banana=1.0, pear=2.5, date=3.0, fig=4.0 — apple before
  // banana because input order (1, 4) is preserved on a tie.
  CHECK(order == std::vector<int64_t>{1, 4, 0, 3, 2});
}

TEST_CASE("sort_rows: descending lexical sort on a String column") {
  Entries e;
  const auto cols = columns_for(e);
  const auto order = dashcore::sort_rows(cols, int64_t(e.name.size()), {{0, false}});
  std::vector<std::string> names_in_order;
  for (auto r : order) names_in_order.push_back(e.name[std::size_t(r)]);
  CHECK(names_in_order == std::vector<std::string>{"pear", "fig", "date", "banana", "apple"});
}

TEST_CASE("TablePanel draws synthetic non-spatial data without crashing") {
  dashcore_test::ImGuiScope gui;
  Entries e;
  TablePanel panel("Fruit prices", "a", columns_for(e),
                   [&e] { return int64_t(e.name.size()); });

  gui.new_frame();
  panel.drawFrame();
  gui.end_frame();

  CHECK(panel.display_order().size() == e.name.size());
}

TEST_CASE("two TablePanel instances hold independent sort state") {
  dashcore_test::ImGuiScope gui;
  Entries e;
  TablePanel a("Fruit prices", "a", columns_for(e), [&e] { return int64_t(e.name.size()); });
  TablePanel b("Fruit prices", "b", columns_for(e), [&e] { return int64_t(e.name.size()); });

  a.set_sort({{1, true}});    // by price ascending
  b.set_sort({{0, false}});   // by name descending

  gui.new_frame();
  a.drawFrame();
  b.drawFrame();
  gui.end_frame();

  CHECK(a.display_order() == std::vector<int64_t>{1, 4, 0, 3, 2});
  CHECK(b.display_order() == std::vector<int64_t>{0, 2, 3, 4, 1});
  CHECK(a.display_order() != b.display_order());
}

TEST_CASE("TablePanel ui_state round-trips sort") {
  Entries e;
  TablePanel src("Fruit prices", "a", columns_for(e), [&e] { return int64_t(e.name.size()); });
  src.set_sort({{1, false}});
  std::vector<std::pair<std::string, std::string>> kv;
  src.write_ui_state(kv);

  TablePanel dst("Fruit prices", "a", columns_for(e), [&e] { return int64_t(e.name.size()); });
  for (const auto& [k, v] : kv) dst.read_ui_state(k, v);
  CHECK(dst.display_order() == src.display_order());
}
