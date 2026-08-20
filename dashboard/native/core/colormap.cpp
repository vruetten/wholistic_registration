#include "core/colormap.hpp"

#include "core/theme.hpp"
#include "core/ui/widgets.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>

namespace dashcore {

std::array<std::uint8_t, 4> colormap_rgba(float value, float vmin, float vmax, int colormap) {
  if (!std::isfinite(value)) return {255, 0, 255, 255};
  const float denom = (vmax > vmin) ? (vmax - vmin) : 1.0f;
  const float t = std::clamp((value - vmin) / denom, 0.0f, 1.0f);
  const ImVec4 c = ImPlot::SampleColormap(t, ImPlotColormap(colormap));
  return {std::uint8_t(c.x * 255.0f + 0.5f), std::uint8_t(c.y * 255.0f + 0.5f),
          std::uint8_t(c.z * 255.0f + 0.5f), 255};
}

int migrate_legacy_colormap(int stored) {
  if (stored == int(ImPlotColormap_Greys)) return theme::greys_colormap();
  return stored;
}

bool draw_contrast_header(float& vmin, float& vmax, bool& auto_range, int& colormap,
                          const std::vector<int>& colormap_options, float range_w,
                          float button_w) {
  bool changed = false;

  // Named dropdown, not a blind-cycling button: every registered map in
  // `colormap_options` (Magma, Viridis, Greys, RdBu) is listed by name and
  // directly selectable, with a small swatch preview next to each entry.
  ImGui::SetNextItemWidth(button_w * 3.0f);
  const char* current_name = ImPlot::GetColormapName(ImPlotColormap(colormap));
  if (ImGui::BeginCombo("##colormap", current_name ? current_name : "?")) {
    for (int cmap : colormap_options) {
      ImGui::PushID(cmap);
      ImPlot::ColormapIcon(ImPlotColormap(cmap));
      ImGui::SameLine();
      const char* name = ImPlot::GetColormapName(ImPlotColormap(cmap));
      const bool selected = (cmap == colormap);
      if (ImGui::Selectable(name ? name : "?", selected)) {
        colormap = cmap;
        changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("auto", &auto_range)) changed = true;
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Auto-range to the 5th-95th percentile of the current data. "
                      "Dragging the range below turns this off.");
  }
  ImGui::SameLine();
  // RangeFloatBox, not a bare DragFloatRange2: each handle carries its own
  // type-in box (dashcore::ui::SliderFloatBox), so the range is settable by
  // number as well as by drag. The current [vmin, vmax] doubles as the
  // nominal slider span for both handles — each handle can be dragged
  // anywhere between the other's value and its own, and a typed value
  // outside that span widens the handle rather than clamping it.
  if (ui::RangeFloatBox("##range", &vmin, &vmax, vmin, vmax, "%.3f", range_w)) {
    auto_range = false;
    changed = true;
  }
  return changed;
}

}  // namespace dashcore
