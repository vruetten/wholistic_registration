# Per-pass checklists

Paste the relevant pass's list verbatim into each reviewer brief — reviewers
have no other access to it.

## Pass 1 — bugs & errors

General:
- Wrong-variable / copy-paste typos (x where y intended, row copied and half-edited)
- Shadowed names; dead assignments that mask the intended value
- Off-by-one in loop bounds, window/patch indexing, slicing
- Swallowed exceptions (bare `except`, caught-and-ignored) hiding failures
- Broken or stale imports; circular imports; module-level side effects
- Mutable default arguments; state leaking between calls
- Resource leaks (unclosed files, handles, pools)
- Boundary handling: empty inputs, single-element, NaN/Inf, zero-size dims

Numerical / array code:
- Axis-order and layout confusion (zyx vs xyz, transposes, C vs F order)
- dtype overflow and silent truncation (uint8/uint16 arithmetic, float32
  accumulation over large sums, int division)
- Aliasing and unintended in-place mutation (views vs copies)
- GPU/CPU shim divergence (cupy vs numpy behavior differences; implicit
  host↔device transfers changing types)
- Unit/scale mismatches (pixel vs micron, downsample factors, normalization
  applied twice or not at all)
- Broadcasting that silently does the wrong thing instead of erroring

## Pass 2 — math & numerics

Verification strategies, strongest first:
- Identity cases: identity transform ⇒ identity output (zero flow ⇒ unchanged
  image; resize factor 1 ⇒ input)
- Reference implementations: compare against scipy/skimage equivalents on
  small arrays
- Invariants and properties: energy/mass conservation, symmetry, composition
  (coarse-to-fine flow composition matches direct), inverse round-trips
- Textbook check: transcribe the implemented formula and diff it against the
  cited/standard definition (ZNCC, Gaussian kernels, SVD conventions — sign,
  ordering, normalization)
- Boundary behavior: interpolation at edges, padding modes, odd vs even sizes

## Pass 3 — performance

- Device↔host transfer churn inside loops
- Redundant full-array copies; temporaries in hot loops
- Unvectorized per-pixel/per-patch Python loops over large arrays
- Allocation inside hot loops that could hoist
- Quadratic scans where a sort/dict/index does it in n·log n
- Chunking sanity for out-of-core arrays (dask/zarr chunk shape vs access
  pattern)
- Repeated recomputation of invariants (pyramids, kernels, masks)

Rule: profile before claiming, wherever the code path runs in this
environment; otherwise the finding is read-level and says so.

## Pass 4 — architecture

Fowler smell baseline (each a judgement call, named and quoted, never a hard
violation; a documented repo convention overrides it):
- Mysterious Name · Duplicated Code · Feature Envy · Data Clumps · Primitive
  Obsession · Repeated Switches · Shotgun Surgery · Divergent Change ·
  Speculative Generality · Message Chains · Middle Man · Refused Bequest

Package-level additions:
- Module boundaries: does each module change for one reason; are layers
  (io / algorithm / orchestration) separable
- Public-API surface: what `__init__` exports vs what callers actually import
- Parallel lineages (v1 vs v2, `_new`/`_v3` suffixes): convergence or
  deprecation path
- Config flow: how parameters travel from file to inner loop; hidden globals
