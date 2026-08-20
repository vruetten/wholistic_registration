// A shared axis range, so several plots scroll and zoom as one.
//
// ImPlot links axes by pointer: every plot that calls SetupAxisLinks with the
// same pair of doubles reads and writes the same range, so panning one pans
// all of them, with no callback graph and no update ordering to get wrong.
//
// The catch is that ImPlotAxisFlags_AutoFit on a linked axis overwrites the
// linked value every frame, which silently defeats the user's zoom. So a
// linked axis is never auto-fitted: it is initialised once to the data extent
// and thereafter belongs to the user, with `reset` as the way back. Y axes are
// free to auto-fit, and should — that is what makes zooming into a quiet
// stretch of a signal actually show it.
#pragma once

namespace dashcore::ui {

struct AxisLink {
  double min = 0.0;
  double max = 1.0;
  bool   initialized = false;

  // Adopt `lo`..`hi` the first time a real data extent is available. Later
  // calls are ignored, so a panel appearing after the user has already zoomed
  // joins at the current range instead of yanking everyone back to full view.
  void ensure(double lo, double hi) {
    if (initialized || !(hi > lo)) return;
    min = lo;
    max = hi;
    initialized = true;
  }

  void reset(double lo, double hi) {
    if (!(hi > lo)) return;
    min = lo;
    max = hi;
    initialized = true;
  }
};

}  // namespace dashcore::ui
