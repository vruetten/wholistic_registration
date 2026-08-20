// Look and DPI scale, applied once at startup.
//
// There is no hardcoded scale constant anywhere in this file. A fixed value
// (e.g. 1.25) looks right on exactly the monitor it was tuned on and wrong
// everywhere else; `content_scale` reads the real value the platform
// reports for the window actually being shown.
#pragma once

#include <vector>

struct GLFWwindow;   // avoid forcing every includer to pull in glfw3.h

namespace dashcore::theme {

// The content scale GLFW reports for `window` (1.0 on a standard 96/101 DPI
// monitor, >1.0 on HiDPI). Clamped to >= 1.0 and to the smaller of the two
// axes, so a misreported display never shrinks the UI below its logical
// size or stretches it unevenly.
float content_scale(GLFWwindow* window);

// Applies StyleColorsDark plus a handful of overrides (rounded controls, a
// visible active-tab/title accent), then ImGui::GetStyle().ScaleAllSizes so
// every built-in metric — padding, rounding, scrollbar width — tracks
// `scale`. Also stages a system TTF into the current font atlas at
// `15 * scale` px if one of a few common fonts is found, falling back to the
// built-in atlas at the same pixel size otherwise.
//
// Call only after ImGui::CreateContext(). Sets the value `Scaled()` returns
// until the next call.
void apply(float scale);

// Multiplies a logical-pixel constant — a fixed column width, an icon size —
// by the scale last passed to `apply()`. Every hardcoded pixel size a panel
// authors should route through this rather than a bare literal, so it tracks
// the display actually in use. Returns `px` unscaled if `apply()` has not
// executed yet (so a widget drawn very early, or in a test, still gets a
// sane size instead of zero).
float Scaled(float px);

// Registers "Magma" (Matplotlib's, 32-stop) as an ImPlot colormap in the
// CURRENT ImPlot context if it isn't there yet, and returns its index either
// way. Colormaps live on the ImPlot context, not on this library's own
// state, so a fresh context (a new test, a freshly launched app process)
// needs this called again — callers ask for the index every time they need
// it rather than caching it once at startup. Requires a current ImPlot
// context.
int magma_colormap();

// Registers "Greys (dark-low)" (a two-stop black-to-white grayscale) as an
// ImPlot colormap in the CURRENT ImPlot context if it isn't there yet, and
// returns its index either way. ImPlot's own built-in Greys entry runs
// white-at-low/black-at-high (implot.cpp: `Greys[] = {IM_COL32_WHITE,
// IM_COL32_BLACK}`), the opposite polarity from every other sequential map
// this app offers (Magma runs black-to-pale-yellow, Viridis runs
// dark-purple-to-yellow); this function supplies a correctly-ordered
// grayscale rather than reusing the built-in one under a different meaning.
// Same registration pattern as magma_colormap(); requires a current ImPlot
// context.
int greys_colormap();

// The curated colormaps offered for a scalar 2-D panel: Magma (the default,
// perceptually uniform), Viridis, Greys (dark-low, see greys_colormap()),
// and RdBu — a diverging map, for the rare source whose sign carries
// meaning, where a sequential map would hide it. Order is the listing order
// of the dropdown draw_contrast_header draws. Registers Magma and Greys
// (dark-low) as a side effect if needed. Requires a current ImPlot context.
std::vector<int> sequential_colormaps();

// Advances `current` to the next entry of `options` (wrapping), or to
// options.front() if `current` isn't in the list at all. Pure and
// ImPlot-free so it's directly testable independent of any widget; not
// currently called by draw_contrast_header, which now offers every
// colormap through a named dropdown (see colormap.cpp) rather than
// click-cycling through them one at a time.
int next_colormap(int current, const std::vector<int>& options);

}  // namespace dashcore::theme
