// Reusable ImGui controls shared by every panel.
//
// The rule for this directory: nothing here knows about any particular
// panel, data source, or SelectionSet. A widget takes plain values and
// returns whether the user changed them — that's what lets "a slider with a
// type-in box" or "a stepped integer control" be one implementation instead
// of five slightly different ones.
#pragma once

#include <imgui.h>

namespace dashcore::ui {

// A slider paired with a type-in box.
//
// ImGui sliders already support ctrl+click to type, but that is invisible to
// anyone who doesn't already know it exists, and these values are routinely
// set from a number read off a plot rather than by feel — so typing is the
// primary interaction, not the fallback.
//
// The typed value is NOT clamped to [v_min, v_max]. The slider's bounds are a
// convenient default range, not a constraint on the quantity. When the value
// falls outside, the slider silently widens to include it so the handle
// stays meaningful.
//
// Layout is [slider][box] label. `slider_w` / `box_w` are logical pixels;
// 0 means "use the default". Returns true on any change this frame.
//
// `tooltip` is handled here rather than by the caller: the widget's last
// ImGui item is the trailing label, so a caller's own IsItemHovered() would
// test the wrong thing.
bool SliderFloatBox(const char* label, float* v, float v_min, float v_max,
                    const char* fmt = "%.3f",
                    float slider_w = 0.0f, float box_w = 0.0f,
                    ImGuiSliderFlags flags = 0,
                    const char* tooltip = nullptr);

bool SliderIntBox(const char* label, int* v, int v_min, int v_max,
                  float slider_w = 0.0f, float box_w = 0.0f,
                  const char* tooltip = nullptr);

// SliderIntBox with -/+ buttons either side of the box.
//
// Stepping one at a time is the dominant way an index into a short list gets
// used — you walk it looking for the interesting entry — and neither
// dragging a slider nor typing a number does that well. The buttons clamp
// and hold-to-repeat. `step` is how far -/+ and the slider move per click;
// pass 2 (etc.) when adjacent indices are paired and landing on an odd one
// would split a pair that has no meaning apart.
bool SteppedIntBox(const char* label, int* v, int v_min, int v_max,
                   float slider_w = 0.0f, float box_w = 0.0f,
                   const char* tooltip = nullptr, int step = 1);

// Two handles on one row for a [lo, hi] range, each with its own box, with
// lo <= hi maintained. Useful anywhere a display range needs setting by
// number as well as by drag.
bool RangeFloatBox(const char* label, float* lo, float* hi,
                   float v_min, float v_max, const char* fmt = "%.3f",
                   float slider_w = 0.0f, float box_w = 0.0f);

}  // namespace dashcore::ui
