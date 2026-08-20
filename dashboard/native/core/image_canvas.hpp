// ImageCanvas: a scrollable, layered 2-D image with a plane slider and
// drag-select in item-id space. The base owns the texture, compositing,
// fit-to-pane zoom (wheel / slider), and the gesture; a subclass (or the
// constructor caller) supplies an ordered list of layers. There is no mode
// enum — visibility and alpha are per-layer fields.
//
// A layer paints one plane into an RGBA buffer. The canvas alpha-blends
// visible layers in list order and uploads a single texture. Hit-testing
// works in pixel-index space, not texel space, so selection stays exact
// regardless of how the image is scaled on screen.
#pragma once

#include "core/panel.hpp"
#include "core/selection.hpp"
#include "core/util/gl_texture.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dashcore {

struct Layer {
  std::string name;
  float alpha = 1.0f;
  bool visible = true;
  // Fills `rgba` (row-major RGBA8, size w*h*4) for plane `z`. Sets w,h to
  // the plane size. A no-op (w=h=0) skips the layer this frame.
  std::function<void(int z, std::vector<std::uint8_t>& rgba, int& w, int& h)> paint;
};

struct ImageSource {
  int z_min = 0;
  int z_max = 0;   // inclusive
  int width = 0;
  int height = 0;
  // Pixel (z, row, col) -> the id this panel's SelectionSet should hold.
  // Unset: the id is the flat C-order index (z * H + row) * W + col.
  std::function<int64_t(int z, int row, int col)> pixel_to_item;

  // Optional: fills `out` with a representative sample of this source's
  // underlying scalar values, for auto-range to compute a percentile
  // contrast from. The canvas itself only ever sees composited RGBA (each
  // Layer already paints its own colors), so it cannot derive a numeric
  // range without this — unlike HeatmapPanel, which owns the raw floats.
  // `z` is passed through for a source whose range should track the current
  // plane; a source with one global range for every plane is free to ignore
  // it. Unset means "no percentile auto-range is available" — auto_range
  // then simply leaves the contrast at whatever it was last set to.
  std::function<void(int z, std::vector<float>& out)> sample_for_autorange;

  // Optional: formats one line of human-readable text describing the pixel
  // at (z, row, col) — the quantity it holds, its units, and its value —
  // for this source's own hover readout. Called once per frame, only for
  // whichever pixel is currently hovered: a caller with raw floats behind
  // this source indexes directly (as app/panels/anatomy_panel.cpp and
  // residual_panel.cpp do), so this is never a per-frame scan of the image.
  // Unset means no readout line is drawn — same "no affordance that does
  // nothing" rule as sample_for_autorange leaving auto-range inert until a
  // source supplies one.
  std::function<std::string(int z, int row, int col)> describe_pixel;
};

namespace detail {

// Wheel zoom: Ctrl resets to 1 (fit-to-pane); otherwise exponential step
// clamped to [lo, hi]. Exposed so the mapping is testable without a live
// scroll region — same pattern as resolve_dock_target / colormap_rgba.
float apply_wheel_zoom(float zoom, float wheel, bool ctrl_reset,
                       float lo = 1.0f, float hi = 16.0f);

struct HoverPixel {
  bool valid = false;
  int row = 0;
  int col = 0;
};

// Pure geometry behind the hovered pixel each frame: given the cursor's
// screen position, the image's on-screen top-left and size, and the
// image's pixel dimensions, returns whether the cursor sits strictly
// inside the image and, if so, which pixel. `hovered` is the caller's own
// window-hover test (ImGui::IsWindowHovered), passed in rather than
// queried here so this stays testable without a live ImGui frame — same
// pattern as apply_wheel_zoom above. A cursor exactly on the image's far
// edge (fx == 1.0 or fy == 1.0) is OUTSIDE, not clamped to the nearest
// pixel: never present a clamped edge pixel as if the pointer were there.
HoverPixel resolve_hover_pixel(bool hovered, float mouse_x, float mouse_y, float img_x0,
                               float img_y0, float img_w, float img_h, int width, int height);

}  // namespace detail

class ImageCanvas : public Panel {
 public:
  ImageCanvas(std::string title, std::string instance_id,
              std::vector<Layer> layers, ImageSource source,
              SelectionSet& selection);

  int z() const { return z_; }
  void set_z(int z);

  float vmin() const { return vmin_; }
  float vmax() const { return vmax_; }
  // Sets an explicit contrast range and turns auto-range off (same "you now
  // own this" semantics as HeatmapPanel::set_contrast).
  void set_contrast(float vmin, float vmax);
  bool auto_range() const { return auto_range_; }

  int colormap() const { return colormap_; }
  void set_colormap(int colormap);

  float zoom() const { return zoom_; }
  void set_zoom(float zoom);

  // Last-frame hover in pixel-index space. False when the cursor is not
  // over the image (letterbox, another window).
  bool hover_pixel(int& row, int& col) const;
  float view_scale() const { return view_scale_; }

  // This frame's hover readout: `source_.describe_pixel`'s text for the
  // hovered pixel, the fixed string "pointer: outside image" when the
  // cursor is not over the image, or empty when this source has no
  // describe_pixel at all (nothing to show). Exposed so a test can assert
  // on the text directly rather than reading back a screenshot — same
  // reason hover_pixel() is public.
  std::string hover_readout_text() const;

  const std::vector<Layer>& layers() const { return layers_; }
  std::vector<Layer>& layers() { return layers_; }

  void mark_dirty() { dirty_ = true; }
  int64_t rebuild_count() const { return rebuild_count_; }

  void write_ui_state(std::vector<std::pair<std::string, std::string>>& out) const override;
  void read_ui_state(const std::string& key, const std::string& value) override;

 protected:
  void drawContent() override;
  void drawHeader() override;
  bool isReady() const override;

 private:
  void rebuild_if_needed();
  void draw_image_and_handle_drag();
  void draw_hover_readout();
  int64_t item_at(int row, int col) const;

  std::vector<Layer> layers_;
  ImageSource source_;
  SelectionSet& selection_;

  int z_ = 0;
  float vmin_ = 0.0f;
  float vmax_ = 1.0f;
  // On by default, same convention as HeatmapPanel — see
  // ImageSource::sample_for_autorange for how the range is actually
  // computed (this class holds no raw scalars of its own).
  bool auto_range_ = true;
  bool autorange_computed_ = false;
  // ImPlotColormap value, or the app-registered "Greys (dark-low)" index
  // once resolved; kept as int (set in the .cpp) so this header needn't
  // include implot.h — same pattern as HeatmapPanel::colormap_. Starts at -1
  // ("unresolved default") rather than calling theme::greys_colormap() here:
  // that call registers a colormap on the current ImPlot context, and a
  // Panel subclass may legitimately be constructed before any ImPlot context
  // exists. drawHeader() resolves it on the first frame, by which point a
  // context is required anyway (draw_contrast_header uses ImPlot).
  int colormap_ = -1;
  float zoom_ = 1.0f;   // 1 = fit pane; >1 magnifies, scrollbars pan
  bool have_pending_scroll_ = false;
  float pending_scroll_x_ = 0.0f;
  float pending_scroll_y_ = 0.0f;
  bool dirty_ = true;
  int64_t rebuild_count_ = 0;
  int built_z_ = -1;
  float built_vmin_ = 0.0f;
  float built_vmax_ = 0.0f;
  int built_colormap_ = -1;

  GLTexture tex_;
  int tex_w_ = 0;
  int tex_h_ = 0;

  float view_scale_ = 1.0f;
  bool hover_valid_ = false;
  int hover_row_ = 0, hover_col_ = 0;

  bool dragging_ = false;
  int drag_r0_ = 0, drag_c0_ = 0;
  int drag_r1_ = 0, drag_c1_ = 0;
};

}  // namespace dashcore
