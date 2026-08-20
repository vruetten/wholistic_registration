#include "core/panel.hpp"

#include <doctest/doctest.h>

using dashcore::detail::resolve_dock_target;

TEST_CASE("resolve_dock_target: no dockspace yet -> no target") {
  CHECK(resolve_dock_target(/*dockspace_exists=*/false, /*dockspace_id=*/1,
                            /*dockspace_is_leaf=*/false, /*central_exists=*/false,
                            /*central_id=*/0, /*central_is_leaf=*/false) == 0);
}

TEST_CASE("resolve_dock_target: central node exists and is a leaf -> central wins") {
  CHECK(resolve_dock_target(/*dockspace_exists=*/true, /*dockspace_id=*/1,
                            /*dockspace_is_leaf=*/false, /*central_exists=*/true,
                            /*central_id=*/2, /*central_is_leaf=*/true) == 2);
}

TEST_CASE("resolve_dock_target: no central, dockspace itself is a leaf -> dockspace wins") {
  CHECK(resolve_dock_target(/*dockspace_exists=*/true, /*dockspace_id=*/1,
                            /*dockspace_is_leaf=*/true, /*central_exists=*/false,
                            /*central_id=*/0, /*central_is_leaf=*/false) == 1);
}

TEST_CASE("resolve_dock_target: dockspace has been split into a non-leaf root "
          "with no central node -> refuses rather than docking into a parent") {
  // This is the exact scenario HIGH 6 catalogues: a saved/previous layout
  // split the dockspace, the central-node flag was lost (e.g. the user
  // closed the last window in the central region), and neither candidate is
  // a leaf. Docking into dockspace_id here would be ImGui's documented
  // silent no-op — the window would float loose with no error.
  CHECK(resolve_dock_target(/*dockspace_exists=*/true, /*dockspace_id=*/1,
                            /*dockspace_is_leaf=*/false, /*central_exists=*/false,
                            /*central_id=*/0, /*central_is_leaf=*/false) == 0);
}

TEST_CASE("resolve_dock_target: central node exists but is NOT a leaf, and "
          "dockspace itself is also not a leaf -> refuses") {
  CHECK(resolve_dock_target(/*dockspace_exists=*/true, /*dockspace_id=*/1,
                            /*dockspace_is_leaf=*/false, /*central_exists=*/true,
                            /*central_id=*/2, /*central_is_leaf=*/false) == 0);
}

TEST_CASE("resolve_dock_target: central node exists but is NOT a leaf; "
          "dockspace itself IS a leaf -> falls back to the dockspace") {
  CHECK(resolve_dock_target(/*dockspace_exists=*/true, /*dockspace_id=*/1,
                            /*dockspace_is_leaf=*/true, /*central_exists=*/true,
                            /*central_id=*/2, /*central_is_leaf=*/false) == 1);
}
