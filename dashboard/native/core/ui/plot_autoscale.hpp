// The shared "autoscale" affordance for an ImPlot axis whose data changes
// out from under it (a trace panel, a penalty ladder, a QC histogram).
//
// Naively OR-ing ImPlotAxisFlags_AutoFit into every SetupAxes call re-fits
// EVERY frame, which fights any pan/zoom the user just did — see
// core/ui/axis_link.hpp for the sharper version of this trap on a linked
// axis. The fix used throughout this app: only request AutoFit on the frame
// the plotted data actually changed, or when the user explicitly asks via
// "fit now". A checkbox lets the user pin a range for comparison instead.
#pragma once

namespace dashcore::ui {

struct PlotAutoscale {
  bool autoscale_on_change = true;
  bool fit_now_pending = false;

  // Draws the "autoscale" checkbox and a "fit now" button on the current
  // ImGui line. Returns true if either was just used this frame.
  bool draw(const char* label = "autoscale");

  // The ImPlotAxisFlags_AutoFit bit to OR into an axis's flags this frame,
  // as a plain int so this header doesn't need to include implot.h.
  // `inputs_changed` is the caller's report of whether the data behind the
  // plot changed since the previous frame (e.g. a selection version bump).
  // Consumes (clears) fit_now_pending — call once per frame, after draw().
  int consume_fit_flags(bool inputs_changed);
};

}  // namespace dashcore::ui
