// SelectionSet: the linking abstraction between panels.
//
// Any panel can both read the current selection and mutate it. Whichever
// panel causes a change bumps `version`; other panels compare their own
// cached version and refresh only when it moved. That's what lets several
// independent panels stay in sync with no callback graph between them.
//
// An id is an opaque int64_t. This type does not know, and must never be
// made to know, what an id denotes — a table row, a matrix row, a location
// in some coordinate system, or anything else. That interpretation belongs
// entirely to whichever owner constructed the ids and handed them out; two
// different owners may even use the same integer to mean two different
// things, and this type has no way to notice and no business trying.
//
// `toggle` and `remove` are deliberately not provided. Neither is exercised
// by anything in this library, and a method that exists but is never wired
// to an interaction is worse than no method: it invites a caller to assume
// modifier-key semantics ("ctrl-click toggles") that were never implemented
// anywhere. Add one back only alongside the interaction that calls it.
#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace dashcore {

using ItemId = int64_t;

// `ids`/`version` are private: the version-bump invariant this whole file's
// header comment describes ("whichever panel causes a change bumps version")
// is meaningless as documentation if any caller can write `selection.ids
// .erase(x)` directly and desync every panel's cache with no error. Every
// mutation goes through a method below that also bumps `version`; there is
// no way to change membership without it.
struct SelectionSet {
  void clear() {
    if (!ids_.empty()) { ids_.clear(); ++version_; }
  }

  // Replaces the set wholesale. Bumps `version` only if the resulting
  // contents actually differ from the current ones — consistent with
  // add()/add_many() below, both of which are already change-aware. A
  // re-drag that lands on the exact same rows (adjust, overshoot, redo) is
  // not a fresh interaction from a downstream cache's point of view: nothing
  // it would recompute from actually changed.
  void set(std::vector<ItemId> new_ids) {
    std::unordered_set<ItemId> next(new_ids.begin(), new_ids.end());
    if (next == ids_) return;
    ids_ = std::move(next);
    ++version_;
  }

  void add(ItemId id) {
    if (ids_.insert(id).second) ++version_;
  }

  void add_many(const std::vector<ItemId>& new_ids) {
    bool changed = false;
    for (auto id : new_ids) if (ids_.insert(id).second) changed = true;
    if (changed) ++version_;
  }

  bool contains(ItemId id) const { return ids_.count(id) != 0; }
  std::size_t size() const { return ids_.size(); }
  std::uint64_t version() const { return version_; }

  // Read-only traversal for a panel that needs to enumerate the current
  // selection (e.g. to draw all selected rows). Direct mutation through this
  // reference is not possible — it's a const&.
  const std::unordered_set<ItemId>& ids() const { return ids_; }

 private:
  std::unordered_set<ItemId> ids_;
  std::uint64_t version_ = 0;   // bumped on every mutating call that changes membership
};

}  // namespace dashcore
