// HeatmapPanel: base for a "rows x cols, one scalar per entry" panel — a
// scrollable, false-colored image with a contrast control and row-range
// drag-select. The base owns the GL texture(s), the contrast header and the
// drag gesture; a subclass supplies only the data, through HeatmapSource.
//
// Row-chunked textures. Most GL implementations refuse a texture taller
// than 16384 (some far less), so a source with more rows than one texture
// allows is split into chunks of up to kChunkRows rows each, stacked
// vertically with no gap. Hit-testing during drag-select works in row-index
// space (a fraction of each chunk's on-screen height), not texel space, so
// selection stays exact regardless of how the image is scaled on screen.
//
// Known limit: each chunk is fetched from source_.fetch_rows in full
// whenever the row count, column count, or source_.generation() changes —
// not lazily re-fetched as the visible window scrolls, and not driven by
// scroll position. A contrast-only change (e.g. dragging the range slider)
// does NOT re-fetch: it recolors the last-fetched values from a per-chunk
// cache and re-uploads, which is the one case a drag gesture actually hits
// every frame. Fine at the row counts this was built and tested against
// (tens of thousands); revisit before pointing it at millions of rows.
#pragma once

#include "core/panel.hpp"
#include "core/selection.hpp"
#include "core/util/gl_texture.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ImPlotContext;   // colormap type is declared in implot.h, included by the .cpp

namespace dashcore {

namespace detail {

// Pure decision behind a completed drag-select gesture: whether it commits a
// row range at all, and which range. A release outside the panel (the user
// dragged off it to abandon the gesture) must cancel rather than commit a
// selection clamped to whatever row is nearest the panel's edge (adversarial
// review MEDIUM 7 / icampsnfr defect #13) — `hovered_at_release` is the
// caller's report of whether the mouse was actually over the panel at the
// moment of release. Exposed for direct testing, same pattern as
// Panel::detail::resolve_dock_target and colormap_rgba above.
struct DragCommit {
  bool commit;
  int64_t lo, hi;   // valid only when commit is true
};
DragCommit resolve_drag_commit(bool hovered_at_release, int64_t start_row, int64_t end_row);

// One row-chunk's vertical extent on screen this frame, mirroring
// HeatmapPanel's own per-texture layout: provider-index rows [r0, r1) are
// drawn from screen y0 to y1.
struct RowChunkLayout {
  int64_t r0, r1;
  float y0, y1;
};

struct HoverCell {
  bool valid = false;
  int64_t row = 0;
  int64_t col = 0;
};

// Pure geometry behind the hovered (display_row, col) each frame, given
// this frame's per-chunk vertical layout. `hovered` is the caller's own
// window-hover test, passed in rather than queried here so this stays
// testable without a live ImGui frame — same pattern as
// resolve_drag_commit above. A cursor exactly on the raster's far edge, or
// outside every chunk's vertical extent (an empty `layout`, or a gap
// between chunks that should not occur), is OUTSIDE, not clamped to the
// nearest row/column: never present a clamped edge entry as if the
// pointer were there.
HoverCell resolve_hover_cell(bool hovered, float mouse_x, float mouse_y, float img_x0, float img_w,
                             int64_t cols, const std::vector<RowChunkLayout>& layout);

}  // namespace detail

struct HeatmapSource {
  int64_t rows = 0;
  int64_t cols = 0;

  // Fills `out` (row-major, size (r1 - r0) * cols) with the values for
  // provider-index rows [r0, r1).
  std::function<void(int64_t r0, int64_t r1, std::vector<float>& out)> fetch_rows;

  // display_row (0-based position on screen, after whatever ordering the
  // source applies) -> the id this panel's SelectionSet should hold for it.
  std::function<int64_t(int64_t display_row)> row_to_item;

  // Inverse of row_to_item, e.g. so a caller can scroll to a given id.
  std::function<int64_t(int64_t item)> item_to_row;

  // Opaque counter the owner bumps whenever `fetch_rows` would return
  // different values for the same (r0, r1) — e.g. after a companion
  // TablePanel re-sorts the display order. `rows`/`cols`/contrast staying
  // fixed is the *common* case for this ("re-sorted, not resized"), so the
  // panel's dirty check cannot infer a content change from shape alone; the
  // owner must say so explicitly. Queried fresh every frame (same pattern
  // as TablePanel's `row_count`), not cached at construction. Unset (null)
  // means "this source's content never changes independent of rows/cols" —
  // the panel then treats every generation as 0.
  std::function<int64_t()> generation;

  // Optional: formats one line of human-readable text describing the
  // hovered entry at (display_row, col) — its quantity, units, and value —
  // for this source's own hover readout. Called once per frame, only for
  // the currently hovered entry, so a caller with a raw matrix behind this
  // source indexes it directly (O(1), as app/panels/cell_raster_panel.cpp
  // does) rather than scanning the image. Unset means no readout line is
  // drawn.
  std::function<std::string(int64_t display_row, int64_t col)> describe_cell;
};

class HeatmapPanel : public Panel {
 public:
  static constexpr int64_t kChunkRows = 8192;

  // `selection` is written by this panel's own drag-select gesture. Which
  // SelectionSet that is — and therefore what an id written into it means —
  // is entirely the caller's choice; this class only ever calls `set` and
  // `add_many` on whatever reference it was given.
  HeatmapPanel(std::string title, std::string instance_id,
              HeatmapSource source, SelectionSet& selection);

  float vmin() const { return vmin_; }
  float vmax() const { return vmax_; }
  // Sets an explicit contrast range and turns auto-range off — an explicit
  // override is the same kind of action as the user dragging the range
  // widget, so it gets the same "you now own this" semantics.
  void set_contrast(float vmin, float vmax);
  bool auto_range() const { return auto_range_; }
  int colormap() const { return colormap_; }
  void set_colormap(int colormap);

  void write_ui_state(std::vector<std::pair<std::string, std::string>>& out) const override;
  void read_ui_state(const std::string& key, const std::string& value) override;

  int64_t chunk_count() const { return int64_t(chunks_.size()); }

  // Last-frame hover in (display_row, col) space. False when the cursor is
  // not over the raster — same convention as ImageCanvas::hover_pixel.
  bool hover_cell(int64_t& row, int64_t& col) const;

  // This frame's hover readout: `source_.describe_cell`'s text for the
  // hovered entry, the fixed string "pointer: outside raster" when the
  // cursor is not over the raster, or empty when this source has no
  // describe_cell at all. Exposed so a test can assert on the text
  // directly — same reason ImageCanvas::hover_readout_text() is public.
  std::string hover_readout_text() const;

  // How many times rebuild_if_needed() has actually rebuilt the chunks
  // (fetched + recolored + re-uploaded), since construction. Exposed mainly
  // so a test can assert a rebuild did or didn't happen without reading GPU
  // texture contents back.
  int64_t rebuild_count() const { return rebuild_count_; }

 protected:
  void drawContent() override;
  bool isReady() const override;

 private:
  void rebuild_if_needed();
  void draw_contrast_row();
  void draw_image_and_handle_drag();
  void draw_hover_readout();

  HeatmapSource source_;
  SelectionSet& selection_;

  float vmin_ = 0.0f;
  float vmax_ = 1.0f;
  // On by default: a fresh panel should show something legible without the
  // user having to guess a range first. Computed as the 5th-95th percentile
  // of the currently-fetched rows (see rebuild_if_needed) — the same
  // convention icampsnfr:phase_matrix_view.cpp uses for its auto-range.
  bool auto_range_ = true;
  bool built_auto_range_ = false;
  int colormap_ = 0;   // ImPlotColormap value; kept as int so this header needn't include implot.h

  std::vector<GLTexture> chunks_;
  // Cached per-chunk fetch results (parallel to chunks_), kept so a
  // contrast-only change can recolor from cache instead of re-fetching the
  // whole dataset from source_ (adversarial review HIGH 4).
  std::vector<std::vector<float>> chunk_values_;
  int64_t chunk_rows_ = kChunkRows;
  int64_t built_rows_ = -1;
  int64_t built_cols_ = -1;
  int64_t built_generation_ = -1;
  float built_vmin_ = 0.0f;
  float built_vmax_ = 0.0f;
  int built_colormap_ = -1;
  int64_t rebuild_count_ = 0;

  bool dragging_ = false;
  int64_t drag_start_row_ = 0;

  bool hover_valid_ = false;
  int64_t hover_row_ = 0;
  int64_t hover_col_ = 0;
};

}  // namespace dashcore
