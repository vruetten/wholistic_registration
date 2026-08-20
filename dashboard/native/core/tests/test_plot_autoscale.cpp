#include "core/ui/plot_autoscale.hpp"

#include <doctest/doctest.h>

using dashcore::ui::PlotAutoscale;

TEST_CASE("PlotAutoscale: default state fits only when inputs changed") {
  PlotAutoscale pa;
  CHECK(pa.autoscale_on_change);
  CHECK(pa.consume_fit_flags(/*inputs_changed=*/true) != 0);
  CHECK(pa.consume_fit_flags(/*inputs_changed=*/false) == 0);
}

TEST_CASE("PlotAutoscale: turning autoscale off stops fitting even when inputs change") {
  PlotAutoscale pa;
  pa.autoscale_on_change = false;
  CHECK(pa.consume_fit_flags(/*inputs_changed=*/true) == 0);
}

TEST_CASE("PlotAutoscale: fit_now_pending forces one fit regardless of autoscale, then clears") {
  PlotAutoscale pa;
  pa.autoscale_on_change = false;
  pa.fit_now_pending = true;
  CHECK(pa.consume_fit_flags(/*inputs_changed=*/false) != 0);
  CHECK_FALSE(pa.fit_now_pending);
  // The one-shot request was consumed: the next frame doesn't keep fitting.
  CHECK(pa.consume_fit_flags(/*inputs_changed=*/false) == 0);
}
