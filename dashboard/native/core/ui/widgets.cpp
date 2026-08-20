#include "core/ui/widgets.hpp"

#include "core/theme.hpp"

#include <algorithm>
#include <cstring>

namespace dashcore::ui {

namespace {

// Default control widths, in logical pixels. Deliberately shared so every
// panel's rows line up when docked side by side.
//
// The slider is narrow because it doesn't carry the number: printing the
// value both inside the grab and in the adjacent box is redundant and costs
// the horizontal room a header row needs. The box is the readout, and it
// updates live while the grab is dragged.
constexpr float kSliderW = 96.0f;
constexpr float kBoxW    = 62.0f;

// ImGui labels carry an optional "##id" suffix that must not be displayed.
// Returns the visible portion; may be empty, in which case no label is drawn.
const char* visible_label_end(const char* label) {
  const char* hash = std::strstr(label, "##");
  return hash ? hash : label + std::strlen(label);
}

void draw_trailing_label(const char* label) {
  const char* end = visible_label_end(label);
  if (end == label) return;
  ImGui::SameLine();
  ImGui::TextUnformatted(label, end);
}

}  // namespace

bool SliderFloatBox(const char* label, float* v, float v_min, float v_max,
                    const char* fmt, float slider_w, float box_w,
                    ImGuiSliderFlags flags, const char* tooltip) {
  ImGui::PushID(label);
  bool changed = false;
  bool hovered = false;

  // Widen the slider to cover a value that was typed outside its nominal
  // range, so the handle keeps tracking rather than pinning at an end.
  const float lo = std::min(v_min, *v);
  const float hi = std::max(v_max, *v);

  ImGui::SetNextItemWidth(theme::Scaled(slider_w > 0.0f ? slider_w : kSliderW));
  changed |= ImGui::SliderFloat("##slider", v, lo, hi, "", flags);
  hovered |= ImGui::IsItemHovered();

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::SetNextItemWidth(theme::Scaled(box_w > 0.0f ? box_w : kBoxW));
  // EnterReturnsTrue so a half-typed number ("0.0" on the way to "0.05")
  // doesn't repaint on every keystroke.
  changed |= ImGui::InputFloat("##box", v, 0.0f, 0.0f, fmt,
                               ImGuiInputTextFlags_EnterReturnsTrue |
                               ImGuiInputTextFlags_CharsScientific);
  hovered |= ImGui::IsItemHovered();

  draw_trailing_label(label);
  if (tooltip && (hovered || ImGui::IsItemHovered())) ImGui::SetTooltip("%s", tooltip);
  ImGui::PopID();
  return changed;
}

bool SliderIntBox(const char* label, int* v, int v_min, int v_max,
                  float slider_w, float box_w, const char* tooltip) {
  ImGui::PushID(label);
  bool changed = false;
  bool hovered = false;

  const int lo = std::min(v_min, *v);
  const int hi = std::max(v_max, *v);

  ImGui::SetNextItemWidth(theme::Scaled(slider_w > 0.0f ? slider_w : kSliderW));
  changed |= ImGui::SliderInt("##slider", v, lo, hi, "");
  hovered |= ImGui::IsItemHovered();

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::SetNextItemWidth(theme::Scaled(box_w > 0.0f ? box_w : kBoxW));
  changed |= ImGui::InputInt("##box", v, 0, 0,
                             ImGuiInputTextFlags_EnterReturnsTrue);
  hovered |= ImGui::IsItemHovered();

  draw_trailing_label(label);
  if (tooltip && (hovered || ImGui::IsItemHovered())) ImGui::SetTooltip("%s", tooltip);
  ImGui::PopID();
  return changed;
}

bool SteppedIntBox(const char* label, int* v, int v_min, int v_max,
                   float slider_w, float box_w, const char* tooltip, int step) {
  ImGui::PushID(label);
  bool changed = false;

  // ImGuiItemFlags_ButtonRepeat so holding - or + walks the range, which is
  // the point of having them.
  ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
  if (ImGui::Button("-")) { *v = std::max(v_min, *v - step); changed = true; }
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);

  if (SliderIntBox("##v", v, v_min, v_max, slider_w, box_w, tooltip)) {
    // Snap a dragged or typed value onto the step grid, so the pairing
    // invariant holds however the value was set.
    if (step > 1) *v -= (*v - v_min) % step;
    changed = true;
  }

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  if (ImGui::Button("+")) { *v = std::min(v_max, *v + step); changed = true; }
  ImGui::PopItemFlag();

  const char* end = visible_label_end(label);
  if (end != label) {
    ImGui::SameLine();
    ImGui::TextUnformatted(label, end);
  }
  ImGui::PopID();
  return changed;
}

bool RangeFloatBox(const char* label, float* lo, float* hi,
                   float v_min, float v_max, const char* fmt,
                   float slider_w, float box_w) {
  ImGui::PushID(label);
  const float half = (slider_w > 0.0f ? slider_w : kSliderW) * 0.5f;
  bool changed = false;

  changed |= SliderFloatBox("##lo", lo, v_min, v_max, fmt, half, box_w);
  ImGui::SameLine();
  changed |= SliderFloatBox("##hi", hi, v_min, v_max, fmt, half, box_w);

  // Keep the interval non-empty. Whichever handle the user just moved wins, so
  // dragging lo past hi pushes hi rather than snapping lo back.
  if (*lo > *hi) {
    if (ImGui::IsItemActive() || ImGui::IsItemDeactivatedAfterEdit()) *lo = *hi;
    else                                                             *hi = *lo;
  }

  draw_trailing_label(label);
  ImGui::PopID();
  return changed;
}

}  // namespace dashcore::ui
