#include "core/panel.hpp"

#include <imgui.h>
#include <imgui_internal.h>   // FindWindowSettingsByID, ImHashStr, DockBuilder*

#include <exception>

namespace dashcore {

namespace detail {

unsigned int resolve_dock_target(bool dockspace_exists, unsigned int dockspace_id,
                                 bool dockspace_is_leaf, bool central_exists,
                                 unsigned int central_id, bool central_is_leaf) {
  if (!dockspace_exists) return 0;
  if (central_exists && central_is_leaf) return central_id;
  if (dockspace_is_leaf) return dockspace_id;
  return 0;
}

}  // namespace detail

Panel::Panel(std::string title, std::string instance_id)
    : title_(std::move(title)),
      instance_id_(std::move(instance_id)),
      window_name_(title_ + "##" + instance_id_) {}

void Panel::drawFrame() {
  // A window ImGui has no saved settings for is new to this layout — first
  // launch, or a panel added since the saved layout was written. Dock it
  // once into a sensible node instead of letting it come up as a free
  // float over everything else. SetNextWindowDockID applies to the very
  // next Begin only, so it must sit immediately before this one.
  //
  // ImGuiCond_FirstUseEver means a saved position always wins, so this never
  // fights a placement the user already made.
  if (ImGui::FindWindowSettingsByID(ImHashStr(window_name_.c_str())) == nullptr) {
    const ImGuiID dockspace_id = ImGui::GetID(kDefaultDockspaceLabel);
    // The dockspace may not exist yet (nothing has called DockSpaceOverViewport
    // this session), may exist but not yet be split into a central node, or
    // may have been split by a saved/previous layout that has since lost its
    // central-node flag. Docking into a node that doesn't exist, or that is
    // no longer a leaf (has since been split into children), is a silent
    // no-op in ImGui — so both candidates are confirmed leaves before either
    // is trusted (mirrors icampsnfr:app.cpp:475-489).
    const ImGuiDockNode* dockspace_node = ImGui::DockBuilderGetNode(dockspace_id);
    const ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace_id);
    const ImGuiID target = detail::resolve_dock_target(
        dockspace_node != nullptr, dockspace_id,
        dockspace_node && dockspace_node->IsLeafNode(),
        central != nullptr, central ? central->ID : 0,
        central && central->IsLeafNode());
    if (target != 0) ImGui::SetNextWindowDockID(target, ImGuiCond_FirstUseEver);
  }

  if (ImGui::Begin(window_name_.c_str())) {
    focused_ = ImGui::IsWindowFocused();
    try {
      drawHeader();
      if (isReady()) drawContent();
      else            drawEmptyState();
    } catch (const std::exception& e) {
      ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "error: %s", e.what());
    }
  } else {
    focused_ = false;
  }
  ImGui::End();
}

void Panel::drawEmptyState() {
  ImGui::TextDisabled("No data.");
}

}  // namespace dashcore
