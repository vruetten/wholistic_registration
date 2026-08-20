// TablePanel: a sortable ImGui table over columns and rows supplied as data.
//
// A column is a display name, a type (used to pick a numeric-vs-lexical
// comparator) and a renderer; the set of columns and what backs them is
// entirely the constructor caller's business. Rows are not stored here
// either — `row_count` is asked fresh every frame, so a caller whose data
// changes underneath it (a filter, a live reload) doesn't need to tell this
// class anything.
#pragma once

#include "core/panel.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dashcore {

enum class ColumnType { Int64, Double, String };

struct TableColumn {
  std::string name;
  ColumnType type = ColumnType::String;

  // Renders the display text for the row at provider-index `row`
  // (0 <= row < row_count(), NOT a domain id). Always required.
  std::function<std::string(int64_t row)> to_string;

  // Numeric sort key. Required when type != String; ignored otherwise, since
  // a String column sorts lexicographically on to_string() instead.
  std::function<double(int64_t row)> sort_key;
};

struct SortSpec {
  int column = 0;
  bool ascending = true;
};

// Pure: returns a permutation of [0, row_count) ordered by `specs` (first
// entry primary, rest are tie-breakers within it), using each column's
// declared comparator. An empty `specs` returns the identity permutation.
// Exposed separately from the ImGui glue in TablePanel so sort correctness
// is testable without simulating a header click.
std::vector<int64_t> sort_rows(const std::vector<TableColumn>& columns,
                                int64_t row_count,
                                const std::vector<SortSpec>& specs);

class TablePanel : public Panel {
 public:
  TablePanel(std::string title, std::string instance_id,
             std::vector<TableColumn> columns,
             std::function<int64_t()> row_count);

  // Applies `specs` immediately, bypassing the need for a real header click,
  // and — since ImGui re-establishes its own sort state the first time a
  // table with no saved settings draws — remembers the first entry of
  // `specs` as that table's default-sort column so the choice survives its
  // own first frame. Only the first entry is remembered this way: ImGui's
  // table settings support exactly one default-sort column, not a full
  // multi-key spec. `display_order()` reflects the complete multi-key
  // request immediately, before any frame; after the first `drawFrame()`,
  // only the primary key persists.
  void set_sort(std::vector<SortSpec> specs);

  // The provider-index order currently on screen, i.e. what `set_sort` (or
  // the last header click) produced. Useful for a caller that wants to
  // export "what the user is looking at" in that order.
  const std::vector<int64_t>& display_order() const { return order_; }

  void write_ui_state(std::vector<std::pair<std::string, std::string>>& out) const override;
  void read_ui_state(const std::string& key, const std::string& value) override;

 protected:
  void drawContent() override;

 private:
  void resort(const std::vector<SortSpec>& specs, int64_t row_count);

  std::vector<TableColumn> columns_;
  std::function<int64_t()> row_count_;
  std::vector<int64_t> order_;

  int default_sort_column_ = -1;   // -1: none requested via set_sort
  bool default_sort_ascending_ = true;
};

}  // namespace dashcore
