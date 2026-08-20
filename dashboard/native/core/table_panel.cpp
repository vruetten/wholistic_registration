#include "core/table_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace dashcore {

namespace {

// A numeric column with no sort_key would otherwise fall back to sorting on
// to_string() lexicographically ("10" < "2" < "9") while still looking and
// behaving like a normal sortable numeric column — a fail-fast violation
// that produces a plausible-looking, silently wrong order. Checked at both
// construction (TablePanel) and every sort_rows() call, since sort_rows is
// itself directly callable without going through TablePanel.
void validate_columns(const std::vector<TableColumn>& columns) {
  for (const auto& c : columns) {
    if (c.type != ColumnType::String && !c.sort_key) {
      throw std::invalid_argument(
          "TableColumn '" + c.name + "': type is numeric but sort_key is null");
    }
  }
}

}  // namespace

std::vector<int64_t> sort_rows(const std::vector<TableColumn>& columns,
                               int64_t row_count,
                               const std::vector<SortSpec>& specs) {
  validate_columns(columns);
  std::vector<int64_t> order(static_cast<std::size_t>(row_count), int64_t{0});
  std::iota(order.begin(), order.end(), int64_t{0});
  if (specs.empty()) return order;

  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    for (const auto& s : specs) {
      if (s.column < 0 || s.column >= int(columns.size())) continue;
      const TableColumn& col = columns[std::size_t(s.column)];
      int cmp = 0;
      if (col.sort_key) {
        const double va = col.sort_key(a);
        const double vb = col.sort_key(b);
        cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
      } else {
        cmp = col.to_string(a).compare(col.to_string(b));
        cmp = (cmp < 0) ? -1 : (cmp > 0) ? 1 : 0;
      }
      if (cmp != 0) return s.ascending ? (cmp < 0) : (cmp > 0);
    }
    return false;   // equal under every spec: stable_sort preserves input order
  });
  return order;
}

TablePanel::TablePanel(std::string title, std::string instance_id,
                       std::vector<TableColumn> columns,
                       std::function<int64_t()> row_count)
    : Panel(std::move(title), std::move(instance_id)),
      columns_(std::move(columns)),
      row_count_(std::move(row_count)) {
  validate_columns(columns_);
}

void TablePanel::resort(const std::vector<SortSpec>& specs, int64_t row_count) {
  order_ = sort_rows(columns_, row_count, specs);
}

void TablePanel::write_ui_state(std::vector<std::pair<std::string, std::string>>& out) const {
  out.push_back({"sort_col", std::to_string(default_sort_column_)});
  out.push_back({"sort_asc", default_sort_ascending_ ? "1" : "0"});
}

void TablePanel::read_ui_state(const std::string& key, const std::string& value) {
  try {
    if (key == "sort_col") default_sort_column_ = std::stoi(value);
    else if (key == "sort_asc") default_sort_ascending_ = (value != "0");
    else return;
    if (default_sort_column_ >= 0) {
      set_sort({{default_sort_column_, default_sort_ascending_}});
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "TablePanel[%s]: failed to parse ui_state %s=%s (%s); keeping prior sort\n",
                 instance_id().c_str(), key.c_str(), value.c_str(), e.what());
  }
}

void TablePanel::set_sort(std::vector<SortSpec> specs) {
  if (!specs.empty() && specs.front().column >= 0 &&
      specs.front().column < int(columns_.size())) {
    default_sort_column_ = specs.front().column;
    default_sort_ascending_ = specs.front().ascending;
  } else {
    default_sort_column_ = -1;
  }
  resort(specs, row_count_ ? row_count_() : 0);
}

void TablePanel::drawContent() {
  const int64_t n = row_count_ ? row_count_() : 0;

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable |
                                ImGuiTableFlags_Resizable;
  // Include instance_id() explicitly rather than trusting the enclosing
  // window's ID to disambiguate two same-class panels — two TablePanel
  // instances sharing a literal table id observably shared sort-spec
  // state in testing, regardless of which window each was drawn in.
  const std::string table_id = "table##" + instance_id();
  if (!ImGui::BeginTable(table_id.c_str(), int(columns_.size()), flags)) return;

  for (int i = 0; i < int(columns_.size()); ++i) {
    const TableColumn& c = columns_[std::size_t(i)];
    ImGuiTableColumnFlags cflags =
        (c.type == ColumnType::String && !c.sort_key) ? ImGuiTableColumnFlags_NoSort : 0;
    // Only takes effect the first time this table id draws with no saved
    // sort settings; a real header click (or a previously saved layout)
    // always wins afterward.
    if (i == default_sort_column_) {
      cflags |= ImGuiTableColumnFlags_DefaultSort;
      cflags |= default_sort_ascending_ ? ImGuiTableColumnFlags_PreferSortAscending
                                        : ImGuiTableColumnFlags_PreferSortDescending;
    }
    ImGui::TableSetupColumn(c.name.c_str(), cflags);
  }
  ImGui::TableHeadersRow();

  ImGuiTableSortSpecs* imgui_specs = ImGui::TableGetSortSpecs();
  const bool need_resort = (int64_t(order_.size()) != n) ||
                           (imgui_specs && imgui_specs->SpecsDirty);
  if (need_resort) {
    std::vector<SortSpec> specs;
    if (imgui_specs) {
      specs.reserve(std::size_t(imgui_specs->SpecsCount));
      for (int i = 0; i < imgui_specs->SpecsCount; ++i) {
        const auto& s = imgui_specs->Specs[i];
        specs.push_back({s.ColumnIndex, s.SortDirection == ImGuiSortDirection_Ascending});
      }
    }
    resort(specs, n);
    if (!specs.empty()) {
      default_sort_column_ = specs.front().column;
      default_sort_ascending_ = specs.front().ascending;
    }
    if (imgui_specs) imgui_specs->SpecsDirty = false;
  }

  ImGuiListClipper clipper;
  clipper.Begin(int(order_.size()));
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const int64_t row = order_[std::size_t(i)];
      ImGui::TableNextRow();
      for (std::size_t c = 0; c < columns_.size(); ++c) {
        ImGui::TableSetColumnIndex(int(c));
        ImGui::TextUnformatted(columns_[c].to_string(row).c_str());
      }
    }
  }
  ImGui::EndTable();
}

}  // namespace dashcore
