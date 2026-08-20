#include "core/heatmap_panel.hpp"

#include "core/colormap.hpp"
#include "core/theme.hpp"
#include "core/util/percentile.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace dashcore {

namespace detail {

DragCommit resolve_drag_commit(bool hovered_at_release, int64_t start_row, int64_t end_row) {
  if (!hovered_at_release) return {false, 0, 0};
  return {true, std::min(start_row, end_row), std::max(start_row, end_row)};
}

HoverCell resolve_hover_cell(bool hovered, float mouse_x, float mouse_y, float img_x0, float img_w,
                             int64_t cols, const std::vector<RowChunkLayout>& layout) {
  if (!hovered || layout.empty() || img_w <= 0.0f || cols <= 0) return {};
  const float frac_x = (mouse_x - img_x0) / img_w;
  if (frac_x < 0.0f || frac_x >= 1.0f) return {};
  if (mouse_y < layout.front().y0 || mouse_y >= layout.back().y1) return {};

  for (const auto& c : layout) {
    if (mouse_y < c.y0 || mouse_y >= c.y1) continue;
    const float frac_y = (mouse_y - c.y0) / (c.y1 - c.y0);
    const int64_t within = std::clamp<int64_t>(int64_t(frac_y * float(c.r1 - c.r0)), 0,
                                               c.r1 - c.r0 - 1);
    HoverCell result;
    result.valid = true;
    result.row = c.r0 + within;
    result.col = std::clamp<int64_t>(int64_t(frac_x * float(cols)), 0, cols - 1);
    return result;
  }
  return {};   // between-chunk gap: should not occur, but never a stale entry
}

}  // namespace detail

namespace {

void recolor_chunk(const std::vector<float>& values, float vmin, float vmax,
                   int colormap, std::vector<std::uint8_t>& rgba) {
  rgba.resize(values.size() * 4);
  for (std::size_t k = 0; k < values.size(); ++k) {
    const auto px = colormap_rgba(values[k], vmin, vmax, colormap);
    std::copy(px.begin(), px.end(), rgba.begin() + std::ptrdiff_t(4 * k));
  }
}

// 5th-95th percentile over every currently-fetched chunk — the auto-range
// convention this app follows throughout (see core/util/percentile.hpp).
// Samples across chunks with the same stride cap percentile_range applies
// within one chunk, so a many-chunk source doesn't pay for a full
// concatenation before sampling.
PercentileRange auto_contrast(const std::vector<std::vector<float>>& chunk_values) {
  constexpr std::size_t kMaxSample = 200'000;
  std::size_t total = 0;
  for (const auto& c : chunk_values) total += c.size();
  if (total == 0) return {0.0f, 1.0f};

  std::vector<float> sample;
  sample.reserve(std::min(total, kMaxSample));
  const std::size_t stride = std::max<std::size_t>(1, total / kMaxSample);
  std::size_t seen = 0;
  for (const auto& c : chunk_values) {
    for (float v : c) {
      if (seen % stride == 0) sample.push_back(v);
      ++seen;
    }
  }
  return percentile_range(sample, 0.05f, 0.95f, kMaxSample);
}

}  // namespace

HeatmapPanel::HeatmapPanel(std::string title, std::string instance_id,
                           HeatmapSource source, SelectionSet& selection)
    : Panel(std::move(title), std::move(instance_id)),
      source_(std::move(source)),
      selection_(selection),
      colormap_(int(ImPlotColormap_Viridis)) {}

void HeatmapPanel::set_contrast(float vmin, float vmax) {
  vmin_ = vmin;
  vmax_ = vmax;
  auto_range_ = false;
}

void HeatmapPanel::set_colormap(int colormap) { colormap_ = colormap; }

void HeatmapPanel::write_ui_state(std::vector<std::pair<std::string, std::string>>& out) const {
  out.push_back({"vmin", std::to_string(vmin_)});
  out.push_back({"vmax", std::to_string(vmax_)});
  out.push_back({"colormap", std::to_string(colormap_)});
  out.push_back({"auto_range", auto_range_ ? "1" : "0"});
}

void HeatmapPanel::read_ui_state(const std::string& key, const std::string& value) {
  try {
    if (key == "vmin") vmin_ = std::stof(value);
    else if (key == "vmax") vmax_ = std::stof(value);
    else if (key == "colormap") colormap_ = migrate_legacy_colormap(std::stoi(value));
    else if (key == "auto_range") auto_range_ = (value != "0");
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "HeatmapPanel[%s]: failed to parse ui_state %s=%s (%s); keeping prior value\n",
                 instance_id().c_str(), key.c_str(), value.c_str(), e.what());
  }
}

bool HeatmapPanel::isReady() const {
  return source_.rows > 0 && source_.cols > 0 && bool(source_.fetch_rows);
}

void HeatmapPanel::drawContent() {
  // Scope every child id (the contrast controls, the scroll region) under
  // instance_id() explicitly, rather than relying solely on the enclosing
  // window to disambiguate two same-class panels — see the matching note
  // in table_panel.cpp for why that trust turned out to be misplaced.
  ImGui::PushID(instance_id().c_str());
  draw_contrast_row();
  ImGui::Separator();
  rebuild_if_needed();
  draw_image_and_handle_drag();
  draw_hover_readout();
  ImGui::PopID();
}

bool HeatmapPanel::hover_cell(int64_t& row, int64_t& col) const {
  if (!hover_valid_) return false;
  row = hover_row_;
  col = hover_col_;
  return true;
}

std::string HeatmapPanel::hover_readout_text() const {
  if (!source_.describe_cell) return {};
  if (!hover_valid_) return "pointer: outside raster";
  return source_.describe_cell(hover_row_, hover_col_);
}

void HeatmapPanel::rebuild_if_needed() {
  const int64_t generation = source_.generation ? source_.generation() : 0;
  const bool data_changed = (source_.rows != built_rows_) || (source_.cols != built_cols_) ||
                            (generation != built_generation_) || chunks_.empty();
  // Toggling auto-range back on (with the data unchanged) must recompute the
  // range from the already-fetched cache, not wait for the next data change.
  const bool auto_range_turned_on = auto_range_ && !built_auto_range_;
  built_auto_range_ = auto_range_;

  if (data_changed) {
    const int max_tex = GLTexture::max_size();
    chunk_rows_ = kChunkRows;
    if (max_tex > 0) {
      chunk_rows_ = std::min(chunk_rows_, int64_t(max_tex));
      if (source_.cols > max_tex) {
        throw std::runtime_error("HeatmapPanel: " + std::to_string(source_.cols) +
                                 " columns exceed GL_MAX_TEXTURE_SIZE=" +
                                 std::to_string(max_tex));
      }
    }
    const int64_t nchunks = (source_.rows + chunk_rows_ - 1) / chunk_rows_;
    chunks_.resize(std::size_t(std::max<int64_t>(nchunks, 0)));
    chunk_values_.resize(chunks_.size());

    for (int64_t i = 0; i < nchunks; ++i) {
      const int64_t r0 = i * chunk_rows_;
      const int64_t r1 = std::min(source_.rows, r0 + chunk_rows_);
      const int64_t n_rows = r1 - r0;

      std::vector<float>& values = chunk_values_[std::size_t(i)];
      values.assign(std::size_t(n_rows * source_.cols), 0.0f);
      source_.fetch_rows(r0, r1, values);
    }
    built_rows_ = source_.rows;
    built_cols_ = source_.cols;
    built_generation_ = generation;
  }

  if (auto_range_ && (data_changed || auto_range_turned_on)) {
    const auto range = auto_contrast(chunk_values_);
    vmin_ = range.lo;
    vmax_ = range.hi;
  }

  const bool contrast_changed = (vmin_ != built_vmin_) || (vmax_ != built_vmax_) ||
                                (colormap_ != built_colormap_);
  if (!data_changed && !contrast_changed) return;
  ++rebuild_count_;

  // Recolor from chunk_values_ (freshly fetched above, or the untouched
  // cache from before) — never re-runs fetch_rows for a contrast-only
  // change (adversarial review HIGH 4: fetch_rows, not the colormap lookup
  // or the upload, is the expensive step at real data scale).
  std::vector<std::uint8_t> rgba;
  for (std::size_t i = 0; i < chunks_.size(); ++i) {
    const int64_t r0 = int64_t(i) * chunk_rows_;
    const int64_t r1 = std::min(source_.rows, r0 + chunk_rows_);
    const int64_t n_rows = r1 - r0;
    recolor_chunk(chunk_values_[i], vmin_, vmax_, colormap_, rgba);
    chunks_[i].upload(int(source_.cols), int(n_rows), rgba);
  }

  built_vmin_ = vmin_;
  built_vmax_ = vmax_;
  built_colormap_ = colormap_;
}

void HeatmapPanel::draw_contrast_row() {
  ImGui::PushID("contrast");
  dashcore::draw_contrast_header(vmin_, vmax_, auto_range_, colormap_,
                                 theme::sequential_colormaps(), theme::Scaled(180.0f),
                                 theme::Scaled(60.0f));
  ImGui::PopID();
}

void HeatmapPanel::draw_image_and_handle_drag() {
  const float row_px = std::max(1.0f, theme::Scaled(1.0f));

  // NoMove: ImGui::Image() below is not an interactive item, so without this
  // flag a click-drag anywhere over the raster is indistinguishable (to
  // ImGui) from "drag empty window space to move the window" — the default
  // for ConfigWindowsMoveFromTitleBarOnly == false — which relocates this
  // panel's ENTIRE floating window under the cursor instead of performing a
  // row-range drag-select, and also locks IsWindowHovered() to this child
  // for the rest of the gesture regardless of where the mouse actually goes
  // (verified directly against the vendored ImGui: FindHoveredWindowEx forces
  // g.HoveredWindow to g.MovingWindow while a body-drag is in flight). That
  // second effect is what silently defeated a naive `if (!hovered) cancel`
  // guard at release — hovered stayed true even with the mouse far outside.
  //
  // A negative height leaves that many pixels at the BOTTOM of the content
  // region for the hover readout drawn after this child returns — see the
  // matching comment on ImageCanvas::draw_image_and_handle_drag for why an
  // unreserved ImVec2(0, 0) child pushes that text past the window's own
  // bottom edge.
  const float readout_h =
      source_.describe_cell ? ImGui::GetTextLineHeightWithSpacing() * 3.0f : 0.0f;
  ImGui::BeginChild("##heatmap_scroll", ImVec2(0, -readout_h), ImGuiChildFlags_None,
                    ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove);
  const float avail_w = ImGui::GetContentRegionAvail().x;

  std::vector<detail::RowChunkLayout> layout;
  layout.reserve(chunks_.size());

  int64_t r0 = 0;
  float img_x0 = 0.0f;
  bool have_img_x0 = false;
  for (auto& tex : chunks_) {
    const int64_t rows_in_chunk = std::min(chunk_rows_, source_.rows - r0);
    const float h = float(rows_in_chunk) * row_px;
    const ImVec2 top_left = ImGui::GetCursorScreenPos();
    if (!have_img_x0) {
      img_x0 = top_left.x;
      have_img_x0 = true;
    }
    ImGui::Image(tex.imgui_id(), ImVec2(avail_w, h));
    layout.push_back({r0, r0 + rows_in_chunk, top_left.y, top_left.y + h});
    r0 += rows_in_chunk;
  }

  const bool hovered = ImGui::IsWindowHovered();
  const float mouse_y = ImGui::GetIO().MousePos.y;
  const float mouse_x = ImGui::GetIO().MousePos.x;

  auto row_at = [&](float y) -> int64_t {
    if (layout.empty() || source_.rows == 0) return 0;
    if (y <= layout.front().y0) return 0;
    if (y >= layout.back().y1) return source_.rows - 1;
    for (const auto& c : layout) {
      if (y >= c.y0 && y < c.y1) {
        const float frac = (y - c.y0) / (c.y1 - c.y0);
        const int64_t within = std::clamp<int64_t>(
            int64_t(frac * float(c.r1 - c.r0)), 0, c.r1 - c.r0 - 1);
        return c.r0 + within;
      }
    }
    return source_.rows - 1;
  };
  auto y_at = [&](int64_t row) -> float {
    for (const auto& c : layout) {
      if (row >= c.r0 && row < c.r1) {
        return c.y0 + (float(row - c.r0) / float(c.r1 - c.r0)) * (c.y1 - c.y0);
      }
    }
    return layout.empty() ? 0.0f : layout.back().y1;
  };

  // detail::resolve_hover_cell uses strict bounds (not row_at's
  // clamped-to-nearest-edge semantics used for drag-select below): a
  // hovered entry must be genuinely under the cursor, never a clamped edge
  // entry presented as if the pointer were there.
  const auto hc = detail::resolve_hover_cell(hovered, mouse_x, mouse_y, img_x0, avail_w,
                                             source_.cols, layout);
  hover_valid_ = hc.valid;
  hover_row_ = hc.row;
  hover_col_ = hc.col;

  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    drag_start_row_ = row_at(mouse_y);
    dragging_ = true;
  }
  if (dragging_) {
    const int64_t end_row = row_at(mouse_y);
    const int64_t lo = std::min(drag_start_row_, end_row);
    const int64_t hi = std::max(drag_start_row_, end_row);

    const float y_lo = y_at(lo);
    const float y_hi = y_at(hi) + row_px;
    const ImVec2 win_pos = ImGui::GetWindowPos();
    ImGui::GetWindowDrawList()->AddRect(
        ImVec2(win_pos.x, y_lo), ImVec2(win_pos.x + avail_w, y_hi),
        IM_COL32(255, 255, 255, 200));

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      dragging_ = false;
      // `hovered` reflects the real mouse position at release only because
      // the child is NoMove (see the BeginChild comment above); without that
      // flag ImGui's own window-move hover capture would defeat this check.
      const detail::DragCommit dc = detail::resolve_drag_commit(hovered, drag_start_row_, end_row);
      if (dc.commit) {
        std::vector<int64_t> items;
        items.reserve(std::size_t(dc.hi - dc.lo + 1));
        for (int64_t r = dc.lo; r <= dc.hi; ++r) {
          items.push_back(source_.row_to_item ? source_.row_to_item(r) : r);
        }
        if (ImGui::GetIO().KeyCtrl) selection_.add_many(items);
        else                        selection_.set(std::move(items));
      }
    }
  }

  if (source_.item_to_row && !layout.empty()) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float x0 = ImGui::GetWindowPos().x;
    for (int64_t id : selection_.ids()) {
      const int64_t row = source_.item_to_row(id);
      if (row < 0 || row >= source_.rows) continue;
      const float y0 = y_at(row);
      dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + avail_w, y0 + row_px),
                        IM_COL32(255, 220, 40, 70));
    }
  }

  ImGui::EndChild();
}

// A fixed line below the scrollable raster, not a cursor tooltip — see the
// matching note on ImageCanvas::draw_hover_readout for why a tooltip would
// occlude the entry the user is trying to read.
void HeatmapPanel::draw_hover_readout() {
  const std::string text = hover_readout_text();
  if (text.empty()) return;
  ImGui::TextUnformatted(text.c_str());
}

}  // namespace dashcore
