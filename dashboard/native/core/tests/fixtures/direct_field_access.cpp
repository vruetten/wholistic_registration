// Negative-compile fixture for the SelectionSet accessor-only API (adversarial
// review finding HIGH 1): if this file ever compiles, SelectionSet::ids and
// ::version have regressed back to public fields, and the whole
// version-counter invalidation scheme is once again enforced only by
// convention. See core/tests/CMakeLists.txt's try_compile() of this file,
// which asserts the *opposite* — that this must fail to build.
#include "core/selection.hpp"

int main() {
  dashcore::SelectionSet s;
  s.ids.insert(1);
  s.version = 5;
  return 0;
}
