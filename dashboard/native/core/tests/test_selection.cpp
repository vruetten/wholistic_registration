#include "core/selection.hpp"

#include <doctest/doctest.h>

using dashcore::SelectionSet;

TEST_CASE("SelectionSet starts empty") {
  SelectionSet s;
  CHECK(s.size() == 0);
  CHECK(s.version() == 0);
}

TEST_CASE("add bumps version only on real change") {
  SelectionSet s;
  s.add(5);
  CHECK(s.contains(5));
  CHECK(s.version() == 1);

  s.add(5);   // duplicate
  CHECK(s.version() == 1);

  s.add(6);
  CHECK(s.version() == 2);
  CHECK(s.size() == 2);
}

TEST_CASE("add_many bumps version once per batch, not per id") {
  SelectionSet s;
  s.add_many({1, 2, 3});
  CHECK(s.size() == 3);
  CHECK(s.version() == 1);

  s.add_many({3, 4});   // one new (4), one duplicate (3)
  CHECK(s.size() == 4);
  CHECK(s.version() == 2);

  s.add_many({});
  CHECK(s.version() == 2);
}

TEST_CASE("set replaces wholesale and bumps version only on real change") {
  SelectionSet s;
  s.set({1, 2});
  CHECK(s.size() == 2);
  CHECK(s.version() == 1);

  s.set({1, 2});   // identical contents — no-op
  CHECK(s.version() == 1);

  s.set({2, 1});   // same contents, different order — still no-op
  CHECK(s.version() == 1);

  s.set({});
  CHECK(s.size() == 0);
  CHECK(s.version() == 2);

  s.set({});   // already empty — no-op
  CHECK(s.version() == 2);
}

TEST_CASE("clear is a no-op on an already-empty set") {
  SelectionSet s;
  s.clear();
  CHECK(s.version() == 0);

  s.add(1);
  s.clear();
  CHECK(s.size() == 0);
  CHECK(s.version() == 2);
}

TEST_CASE("an id is opaque: two unrelated selections may reuse the same integer") {
  SelectionSet a, b;
  a.add(42);
  b.add(42);
  CHECK(a.contains(42));
  CHECK(b.contains(42));
  // No shared state between instances.
  a.clear();
  CHECK_FALSE(a.contains(42));
  CHECK(b.contains(42));
}
