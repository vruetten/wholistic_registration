// Panel: base class for a single dockable ImGui window.
//
// Deliberately carries no application type. A version of this class once
// threaded an app-context reference through every virtual, which meant one
// of two things: either the library didn't compile without that type, or
// the type moved into this library and it quietly owned the whole app's
// world. Neither is a real seam. Instead, a subclass captures whatever it
// needs — data sources, callbacks, references — as ordinary constructor
// arguments, exactly like any other C++ object.
#pragma once

#include "core/selection.hpp"

#include <string>
#include <utility>
#include <vector>

namespace dashcore {

// A single well-known dockspace that every Panel docks into on first
// appearance. The application creates it once (typically via
// `ImGui::DockSpaceOverViewport(ImGui::GetID(kDefaultDockspaceLabel), ...)`)
// before drawing any panels; a Panel that draws before the dockspace exists
// simply floats undocked for that frame instead of failing.
inline constexpr const char* kDefaultDockspaceLabel = "dashcore_dockspace";

namespace detail {

// Pure decision behind Panel::drawFrame()'s first-appearance docking:
// which of the two candidate node ids (the dockspace's central node, or the
// dockspace itself as a fallback) is safe to hand to SetNextWindowDockID.
// Docking into a node that has since been split (i.e. is no longer a leaf)
// is a silent no-op in ImGui — icampsnfr:app.cpp:475-489's whole reason for
// checking IsLeafNode() before trusting either candidate. Exposed here,
// separately from the ImGui glue in Panel::drawFrame(), so this decision is
// testable without simulating a live docking layout at all — same pattern
// as TablePanel's sort_rows.
//
// Returns 0 ("don't call SetNextWindowDockID this frame") if neither
// candidate is a confirmed leaf.
unsigned int resolve_dock_target(bool dockspace_exists, unsigned int dockspace_id,
                                 bool dockspace_is_leaf, bool central_exists,
                                 unsigned int central_id, bool central_is_leaf);

}  // namespace detail

class Panel {
 public:
  Panel(std::string title, std::string instance_id);
  virtual ~Panel() = default;

  Panel(const Panel&) = delete;
  Panel& operator=(const Panel&) = delete;

  // Begin/End the window, focus tracking, first-appearance docking into the
  // default dockspace's central node (falling back to the dockspace itself
  // when there is no central node yet, e.g. before it has been split), and
  // the isReady()/drawEmptyState() branch. Call once per frame.
  void drawFrame();

  // "Title##instance_id" — ImGui's window identity IS this string, so two
  // panels sharing a title would merge into one window without a distinct
  // instance_id per construction.
  std::string window_name() const { return window_name_; }
  const std::string& instance_id() const { return instance_id_; }

  bool isFocused() const { return focused_; }

  // Per-instance UI (contrast, plane, sort, …). Keys are unprefixed; the
  // owner namespaces them by instance_id so two panels of the same class
  // cannot clobber each other. Empty defaults: nothing to persist.
  virtual void write_ui_state(std::vector<std::pair<std::string, std::string>>& /*out*/) const {}
  virtual void read_ui_state(const std::string& /*key*/, const std::string& /*value*/) {}

 protected:
  const std::string& title() const { return title_; }

  // The panel's own controls, drawn every frame regardless of isReady().
  virtual void drawContent() = 0;
  virtual void drawHeader() {}
  virtual bool isReady() const { return true; }
  virtual void drawEmptyState();

  // Called by the owner after some SelectionSet this panel cares about has
  // changed. The base class never calls this itself — nothing here owns a
  // SelectionSet — so a Panel that has no use for it is free to leave the
  // no-op default in place.
  virtual void onSelectionChanged(const SelectionSet&) {}

 private:
  std::string title_;
  std::string instance_id_;
  std::string window_name_;
  bool focused_ = false;
};

}  // namespace dashcore
