// Shared value->color mapping and the one contrast/colormap header widget
// every 2-D panel (HeatmapPanel, ImageCanvas) draws through. Previously each
// panel reimplemented its own contrast row; this is the "ONE implementation
// instead of five" the ui/widgets.hpp directory rule asks for, extended to a
// widget that needs ImPlot (so it can't live in ui/widgets.hpp, which stays
// free of that dependency).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace dashcore {

// Maps one scalar to an RGBA pixel under `colormap` (an ImPlotColormap
// value), given the current [vmin, vmax] contrast range. NaN and +/-Inf (all
// plausible real values — a stat from a 0/0 or a divide-by-zero, a
// degenerate empty region) always map to a fixed magenta sentinel rather
// than resolving to an arbitrary color: NaN via unspecified clamp-of-NaN
// comparison behavior, +/-Inf via clamping to the map's brightest/darkest
// end and reading as an ordinary extreme value. For a triage tool, a
// non-finite value should stand out, not blend in as plausible data.
std::array<std::uint8_t, 4> colormap_rgba(float value, float vmin, float vmax, int colormap);

// Maps a persisted legacy colormap index to its corrected replacement, or
// returns `stored` unchanged when no migration applies. Exists for exactly
// one case: a ui_state.ini saved before this app registered its own
// dark-at-low grayscale map may hold ImPlot's built-in ImPlotColormap_Greys
// (a fixed index, white-at-low/black-at-high — the polarity bug a user
// reported directly), and reading that index back verbatim would keep
// showing the wrong polarity indefinitely even after the running code is
// fixed. Requires a current ImPlot context when a migration actually
// applies (registers theme::greys_colormap() in that case).
int migrate_legacy_colormap(int stored);

// Draws the shared contrast+colormap header: a named dropdown that lists
// every entry of `options` by its registered ImPlot name (so Magma, Viridis,
// etc. are directly selectable rather than reachable only by cycling), an
// "auto" checkbox, and a [vmin, vmax] range control with a type-in box on
// each handle (dashcore::ui::RangeFloatBox — every slider in this app
// carries a type-in box, per the widgets-directory rule), all on one row.
//
// Dragging the range widget is a manual override: it clears `auto_range`
// the moment the user moves it, same as icampsnfr's SliderFloatBox / the
// design plan's "turning a slider means autoscale switches off." Toggling
// the checkbox back on does not recompute the range itself — the caller
// (which owns the data) is responsible for noticing `auto_range` turned on
// and recomputing vmin/vmax before the next draw.
//
// `range_w` and `button_w` are logical pixels, already scaled by the caller
// (e.g. via theme::Scaled) — this function has no DPI concept of its own.
// Returns true if vmin, vmax, colormap, or auto_range changed this frame.
bool draw_contrast_header(float& vmin, float& vmax, bool& auto_range, int& colormap,
                          const std::vector<int>& colormap_options, float range_w,
                          float button_w);

}  // namespace dashcore
