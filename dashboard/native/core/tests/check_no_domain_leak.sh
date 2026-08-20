#!/usr/bin/env bash
# Fails if anything under core/ mentions a domain concept as a whole word:
# a pipeline "run", a "cell", a "voxel", "NMF" itself, or the registration
# domain's "motion", "refspace" and "wholistic". Word-boundary matching means
# "runs", "running" and "runtime" are untouched — only the literal token "run"
# (as a noun referring to a pipeline execution) trips it, which is what an
# interface leak actually looks like.
#
# "frame" and "plane" are deliberately absent. Both are registration-domain
# words, but both are also dashcore's own vocabulary — the per-frame draw loop
# and ImageCanvas's plane slider — so listing them would fail on 57 existing
# lines of GUI code and say nothing about a leak.
set -euo pipefail

core_dir="${1:?usage: check_no_domain_leak.sh <core-dir>}"

matches="$(grep -rniE '\b(cell|voxel|nmf|run|motion|refspace|wholistic)\b' "$core_dir" \
  --include='*.hpp' --include='*.cpp' \
  --exclude-dir=build 2>/dev/null || true)"

if [[ -n "$matches" ]]; then
  echo "ERROR: core/ contains domain-specific vocabulary:" >&2
  echo "$matches" >&2
  exit 1
fi

echo "OK: core/ is free of cell/run/voxel/NMF/motion/refspace/wholistic vocabulary."
