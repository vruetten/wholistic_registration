#include "core/ui/plot_autoscale.hpp"

#include <imgui.h>
#include <implot.h>

namespace dashcore::ui {

bool PlotAutoscale::draw(const char* label) {
  bool changed = ImGui::Checkbox(label, &autoscale_on_change);
  ImGui::SameLine();
  if (ImGui::SmallButton("fit now")) {
    fit_now_pending = true;
    changed = true;
  }
  return changed;
}

int PlotAutoscale::consume_fit_flags(bool inputs_changed) {
  const bool fit = (autoscale_on_change && inputs_changed) || fit_now_pending;
  fit_now_pending = false;
  return fit ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
}

}  // namespace dashcore::ui
