#include "core/image_canvas.hpp"

#include "core/colormap.hpp"
#include "core/theme.hpp"
#include "core/ui/widgets.hpp"
#include "core/util/percentile.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace dashcore {

namespace {

void blend_over(std::vector<std::uint8_t>& dst, const std::vector<std::uint8_t>& src,
                float alpha, int n_px) {
  const float a = std::clamp(alpha, 0.0f, 1.0f);
  for (int i = 0; i < n_px; ++i) {
    const std::size_t o = std::size_t(i) * 4;
    const float sa = (src[o + 3] / 255.0f) * a;
    if (sa <= 0.0f) continue;
    const float da = 1.0f - sa;
    dst[o + 0] = std::uint8_t(src[o + 0] * sa + dst[o + 0] * da + 0.5f);
    dst[o + 1] = std::uint8_t(src[o + 1] * sa + dst[o + 1] * da + 0.5f);
    dst[o + 2] = std::uint8_t(src[o + 2] * sa + dst[o + 2] * da + 0.5f);
    dst[o + 3] = 255;
  }
}

}  // namespace

ImageCanvas::ImageCanvas(std::string title, std::string instance_id,
                         std::vector<Layer> layers, ImageSource source,
                         SelectionSet& selection)
    : Panel(std::move(title), std::move(instance_id)),
      layers_(std::move(layers)),
      source_(std::move(source)),
      selection_(selection),
      z_(source_.z_min) {}

void ImageCanvas::set_z(int z) {
  z = std::clamp(z, source_.z_min, source_.z_max);
  if (z == z_) return;
  z_ = z;
  dirty_ = true;
}

void ImageCanvas::set_contrast(float vmin, float vmax) {
  auto_range_ = false;
  if (vmin == vmin_ && vmax == vmax_) return;
  vmin_ = vmin;
  vmax_ = vmax;
  dirty_ = true;
}

void ImageCanvas::set_colormap(int colormap) {
  if (colormap == colormap_) return;
  colormap_ = colormap;
  dirty_ = true;
}

void ImageCanvas::set_zoom(float zoom) {
  zoom_ = std::clamp(zoom, 1.0f, 16.0f);
}

bool ImageCanvas::hover_pixel(int& row, int& col) const {
  if (!hover_valid_) return false;
  row = hover_row_;
  col = hover_col_;
  return true;
}

std::string ImageCanvas::hover_readout_text() const {
  if (!source_.describe_pixel) return {};
  if (!hover_valid_) return "pointer: outside image";
  return source_.describe_pixel(z_, hover_row_, hover_col_);
}

namespace detail {

float apply_wheel_zoom(float zoom, float wheel, bool ctrl_reset, float lo, float hi) {
  if (ctrl_reset) return 1.0f;
  if (wheel == 0.0f) return std::clamp(zoom, lo, hi);
  return std::clamp(zoom * std::pow(1.15f, wheel), lo, hi);
}

HoverPixel resolve_hover_pixel(bool hovered, float mouse_x, float mouse_y, float img_x0,
                               float img_y0, float img_w, float img_h, int width, int height) {
  if (!hovered || img_w <= 0.0f || img_h <= 0.0f || width <= 0 || height <= 0) return {};
  const float fx = (mouse_x - img_x0) / img_w;
  const float fy = (mouse_y - img_y0) / img_h;
  if (fx < 0.0f || fx >= 1.0f || fy < 0.0f || fy >= 1.0f) return {};
  HoverPixel result;
  result.valid = true;
  result.row = std::clamp(int(fy * float(height)), 0, height - 1);
  result.col = std::clamp(int(fx * float(width)), 0, width - 1);
  return result;
}

}  // namespace detail

void ImageCanvas::write_ui_state(std::vector<std::pair<std::string, std::string>>& out) const {
  out.push_back({"z", std::to_string(z_)});
  out.push_back({"vmin", std::to_string(vmin_)});
  out.push_back({"vmax", std::to_string(vmax_)});
  out.push_back({"auto_range", auto_range_ ? "1" : "0"});
  out.push_back({"colormap", std::to_string(colormap_)});
  out.push_back({"zoom", std::to_string(zoom_)});
  for (const auto& layer : layers_) {
    out.push_back({"layer." + layer.name + ".visible", layer.visible ? "1" : "0"});
    out.push_back({"layer." + layer.name + ".alpha", std::to_string(layer.alpha)});
  }
}

void ImageCanvas::read_ui_state(const std::string& key, const std::string& value) {
  try {
    if (key == "z") {
      set_z(std::stoi(value));
      return;
    }
    if (key == "vmin") {
      set_contrast(std::stof(value), vmax_);
      return;
    }
    if (key == "vmax") {
      set_contrast(vmin_, std::stof(value));
      return;
    }
    if (key == "auto_range") {
      // Read directly (not via set_contrast, which would force this false):
      // a saved "auto_range=1" must stay on across a restart, re-deriving
      // its range from the data rather than being pinned at a stale number.
      auto_range_ = (value != "0");
      dirty_ = true;
      return;
    }
    if (key == "colormap") {
      set_colormap(migrate_legacy_colormap(std::stoi(value)));
      return;
    }
    if (key == "zoom") {
      set_zoom(std::stof(value));
      return;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "ImageCanvas[%s]: failed to parse ui_state %s=%s (%s); keeping prior value\n",
                 instance_id().c_str(), key.c_str(), value.c_str(), e.what());
    return;
  }
  if (key.rfind("layer.", 0) == 0) {
    const auto rest = key.substr(6);
    const auto dot = rest.rfind('.');
    if (dot == std::string::npos) return;
    const std::string name = rest.substr(0, dot);
    const std::string field = rest.substr(dot + 1);
    for (auto& layer : layers_) {
      if (layer.name != name) continue;
      if (field == "visible") layer.visible = (value != "0");
      else if (field == "alpha") {
        try {
          layer.alpha = std::stof(value);
        } catch (const std::exception& e) {
          std::fprintf(stderr,
                       "ImageCanvas[%s]: failed to parse ui_state %s=%s (%s); keeping prior "
                       "value\n",
                       instance_id().c_str(), key.c_str(), value.c_str(), e.what());
          break;
        }
      }
      dirty_ = true;
      break;
    }
  }
}

bool ImageCanvas::isReady() const {
  return source_.width > 0 && source_.height > 0 && !layers_.empty();
}

int64_t ImageCanvas::item_at(int row, int col) const {
  if (source_.pixel_to_item) return source_.pixel_to_item(z_, row, col);
  return (int64_t(z_) * source_.height + row) * source_.width + col;
}

void ImageCanvas::drawHeader() {
  // Preserves this class's prior (and still common) default of showing an
  // intensity image in plain grayscale; Magma/Viridis/RdBu are one click
  // away via the contrast header's colormap picker. Resolved here, not in
  // the constructor, because registering "Greys (dark-low)" needs a live
  // ImPlot context (theme::greys_colormap()) that a subclass constructor
  // cannot guarantee exists yet; drawHeader() runs before drawContent()
  // every frame (Panel::drawFrame()), so this always resolves before the
  // value is first used to color a pixel. A restored colormap (read_ui_state
  // -> set_colormap) already left colormap_ >= 0 by the time this runs, so
  // it is left untouched.
  if (colormap_ < 0) colormap_ = theme::greys_colormap();

  ImGui::PushID(instance_id().c_str());
  ImGui::PushID("contrast");
  if (dashcore::draw_contrast_header(vmin_, vmax_, auto_range_, colormap_,
                                     theme::sequential_colormaps(), theme::Scaled(180.0f),
                                     theme::Scaled(60.0f))) {
    dirty_ = true;
  }
  ImGui::PopID();

  if (source_.z_max > source_.z_min) {
    if (ui::SliderIntBox("plane", &z_, source_.z_min, source_.z_max)) dirty_ = true;
    ImGui::SameLine();
  }
  ui::SliderFloatBox("zoom", &zoom_, 1.0f, 16.0f, "%.2fx");
  ImGui::SameLine();
  if (ImGui::SmallButton("reset view")) zoom_ = 1.0f;
  ImGui::SameLine();
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    ImGui::PushID(int(i));
    if (ImGui::Checkbox(layers_[i].name.c_str(), &layers_[i].visible)) dirty_ = true;
    ImGui::SameLine();
    if (ui::SliderFloatBox("a", &layers_[i].alpha, 0.0f, 1.0f, "%.2f", 60.0f, 50.0f)) {
      dirty_ = true;
    }
    if (i + 1 < layers_.size()) ImGui::SameLine();
    ImGui::PopID();
  }
  ImGui::PopID();
}

void ImageCanvas::drawContent() {
  ImGui::PushID(instance_id().c_str());
  rebuild_if_needed();
  draw_image_and_handle_drag();
  draw_hover_readout();
  ImGui::PopID();
}

void ImageCanvas::rebuild_if_needed() {
  // Auto-range recompute: on the first rebuild ever, or whenever the user
  // has just turned it back on — not every frame, so it doesn't fight a
  // manual override that hasn't been re-enabled. z is passed through in
  // case a future source wants a per-plane range; the current sources all
  // sample the whole volume once (see anatomy_panel.cpp) and ignore it.
  if (auto_range_ && !autorange_computed_ && source_.sample_for_autorange) {
    std::vector<float> sample;
    source_.sample_for_autorange(z_, sample);
    const auto range = percentile_range(sample, 0.05f, 0.95f);
    vmin_ = range.lo;
    vmax_ = range.hi;
    autorange_computed_ = true;
    dirty_ = true;
  } else if (!auto_range_) {
    autorange_computed_ = false;   // next time it's re-enabled, recompute
  }

  if (!dirty_ && z_ == built_z_ && vmin_ == built_vmin_ && vmax_ == built_vmax_ &&
      colormap_ == built_colormap_ && tex_.valid()) {
    return;
  }
  ++rebuild_count_;

  const int w = source_.width;
  const int h = source_.height;
  const int n_px = w * h;
  std::vector<std::uint8_t> composed(std::size_t(n_px) * 4, 0);
  std::vector<std::uint8_t> layer_rgba;

  for (const auto& layer : layers_) {
    if (!layer.visible || !layer.paint) continue;
    int lw = 0, lh = 0;
    layer.paint(z_, layer_rgba, lw, lh);
    if (lw != w || lh != h || int(layer_rgba.size()) < n_px * 4) continue;
    blend_over(composed, layer_rgba, layer.alpha, n_px);
  }

  tex_.upload(w, h, composed);
  tex_w_ = w;
  tex_h_ = h;
  built_z_ = z_;
  built_vmin_ = vmin_;
  built_vmax_ = vmax_;
  built_colormap_ = colormap_;
  dirty_ = false;
}

void ImageCanvas::draw_image_and_handle_drag() {
  if (!tex_.valid()) return;

  // Apply cursor-centered scroll on the frame after a wheel zoom, once the
  // child's content size has caught up (SetScrollX the same frame is clamped
  // to last frame's smaller max).
  if (have_pending_scroll_) {
    ImGui::SetNextWindowScroll(ImVec2(pending_scroll_x_, pending_scroll_y_));
    have_pending_scroll_ = false;
  }

  // A negative height leaves that many pixels at the BOTTOM of the content
  // region for whatever is drawn after this child (ImGui's own idiom for
  // "reserve space below"). Without this, BeginChild's default ImVec2(0, 0)
  // claims the full remaining region, and the hover readout drawn after
  // draw_image_and_handle_drag() returns lands past the window's own
  // bottom edge — invisible, not merely unscrolled-to. Reserved
  // unconditionally at a fixed 3 lines whenever a readout could appear (so
  // the image area doesn't resize every time the cursor crosses the edge),
  // 0 when this source has no describe_pixel at all (nothing will ever be
  // drawn there).
  const float readout_h =
      source_.describe_pixel ? ImGui::GetTextLineHeightWithSpacing() * 3.0f : 0.0f;

  // NoScrollWithMouse: the wheel is zoom, not the child's scrollbar
  // (icampsnfr:volume_view.cpp). Middle-drag pans.
  ImGui::BeginChild("##image_scroll", ImVec2(0, -readout_h), ImGuiChildFlags_None,
                    ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove |
                        ImGuiWindowFlags_NoScrollWithMouse);

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float sx = avail.x / float(std::max(tex_w_, 1));
  const float sy = avail.y / float(std::max(tex_h_, 1));
  const float fit = std::min(sx, sy);
  const float old_scale = std::max(fit, 1e-6f) * zoom_;

  if (ImGui::IsWindowHovered() && ImGui::GetIO().MouseWheel != 0.0f) {
    const float old_zoom = zoom_;
    zoom_ = detail::apply_wheel_zoom(zoom_, ImGui::GetIO().MouseWheel,
                                     ImGui::GetIO().KeyCtrl);
    if (zoom_ != old_zoom && old_scale > 0.0f) {
      const ImVec2 win = ImGui::GetWindowPos();
      const ImVec2 mouse = ImGui::GetIO().MousePos;
      const float content_x = mouse.x - win.x + ImGui::GetScrollX();
      const float content_y = mouse.y - win.y + ImGui::GetScrollY();
      const float new_scale = std::max(fit, 1e-6f) * zoom_;
      const float k = new_scale / old_scale;
      pending_scroll_x_ = content_x * k - (mouse.x - win.x);
      pending_scroll_y_ = content_y * k - (mouse.y - win.y);
      have_pending_scroll_ = true;
    }
  }

  if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
    const ImVec2 d = ImGui::GetIO().MouseDelta;
    ImGui::SetScrollX(ImGui::GetScrollX() - d.x);
    ImGui::SetScrollY(ImGui::GetScrollY() - d.y);
  }

  const float scale = std::max(fit, 1e-6f) * zoom_;
  view_scale_ = scale;
  const ImVec2 img_size(float(tex_w_) * scale, float(tex_h_) * scale);
  const ImVec2 top_left = ImGui::GetCursorScreenPos();
  ImGui::Image(tex_.imgui_id(), img_size);
  ImGui::SetCursorScreenPos(top_left);
  ImGui::InvisibleButton("image_area", img_size);

  // Window hover, not IsItemHovered: the child is NoMove and the button can
  // lose item-hover mid-click when ImGui treats the gesture as a window
  // move (heatmap_panel.cpp has the same note). AllowWhenBlockedByActiveItem
  // keeps hover true while our own button is the active item.
  const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
  const ImVec2 mouse = ImGui::GetIO().MousePos;

  auto pixel_at = [&](ImVec2 p) -> std::pair<int, int> {
    const float fx = (p.x - top_left.x) / img_size.x;
    const float fy = (p.y - top_left.y) / img_size.y;
    const int col = std::clamp(int(fx * float(source_.width)), 0, source_.width - 1);
    const int row = std::clamp(int(fy * float(source_.height)), 0, source_.height - 1);
    return {row, col};
  };

  const auto hp = detail::resolve_hover_pixel(hovered, mouse.x, mouse.y, top_left.x, top_left.y,
                                              img_size.x, img_size.y, source_.width,
                                              source_.height);
  hover_valid_ = hp.valid;
  hover_row_ = hp.row;
  hover_col_ = hp.col;

  if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    const auto [r, c] = pixel_at(mouse);
    drag_r0_ = drag_r1_ = r;
    drag_c0_ = drag_c1_ = c;
    dragging_ = true;
    // Commit the click immediately so a press (no drag) selects this frame
    // instead of waiting for a release that window-move can swallow.
    const int64_t id = item_at(r, c);
    if (ImGui::GetIO().KeyCtrl) selection_.add(id);
    else                        selection_.set({id});
  }
  if (dragging_) {
    const auto [r, c] = pixel_at(mouse);
    drag_r1_ = r;
    drag_c1_ = c;
    const float x0 = top_left.x + float(std::min(drag_c0_, drag_c1_)) * scale;
    const float y0 = top_left.y + float(std::min(drag_r0_, drag_r1_)) * scale;
    const float x1 = top_left.x + float(std::max(drag_c0_, drag_c1_) + 1) * scale;
    const float y1 = top_left.y + float(std::max(drag_r0_, drag_r1_) + 1) * scale;
    ImGui::GetWindowDrawList()->AddRect(ImVec2(x0, y0), ImVec2(x1, y1),
                                        IM_COL32(255, 255, 255, 200));

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      dragging_ = false;
      if (!hovered) {
        // abandon — same rule as HeatmapPanel: release outside cancels
      } else {
        const int r0 = std::min(drag_r0_, drag_r1_);
        const int r1 = std::max(drag_r0_, drag_r1_);
        const int c0 = std::min(drag_c0_, drag_c1_);
        const int c1 = std::max(drag_c0_, drag_c1_);
        const int64_t n_px = int64_t(r1 - r0 + 1) * int64_t(c1 - c0 + 1);
        // A drag across a 2k×1k plane is millions of ids; committing that
        // every release OOMs or hangs the frame. A click (or a small box)
        // is the interaction this panel is for.
        constexpr int64_t kMaxDragPx = 16384;
        std::vector<int64_t> items;
        if (n_px > kMaxDragPx) {
          items.push_back(item_at(drag_r0_, drag_c0_));
        } else {
          items.reserve(std::size_t(n_px));
          for (int rr = r0; rr <= r1; ++rr) {
            for (int cc = c0; cc <= c1; ++cc) items.push_back(item_at(rr, cc));
          }
        }
        if (ImGui::GetIO().KeyCtrl) selection_.add_many(items);
        else                        selection_.set(std::move(items));
      }
    }
  }

  ImGui::EndChild();
}

// A fixed line below the scrollable image, not a cursor tooltip: a tooltip
// sits exactly on top of the pixel the user is trying to read, which
// defeats the point of a value readout. Drawn outside the scroll child
// (draw_image_and_handle_drag already returned) so it never scrolls out of
// view and never occludes the hovered pixel regardless of zoom or pan.
void ImageCanvas::draw_hover_readout() {
  const std::string text = hover_readout_text();
  if (text.empty()) return;
  ImGui::TextUnformatted(text.c_str());
}

}  // namespace dashcore
