# Corresponding files
## motion_correlation_pattern.py
the main file to analyse and visualize motion. **As of 2026-08-03 (V2)**, the mode decomposition has been refactored in-place with a new sparse-compact objective — there is no separate `_v2.py` file.
## motion_stage_cache.py
the file to store the temp results

---

## Update Summary — 2026-08-03 (V2)

This document has been updated to reflect the V2 refactoring. See the **V2: Sparse-Compact Decomposition** section at the end for full details. Key changes:

### Mode Decomposition (V1 → V2) (Most important)
- **New objective**: sparse-compact loss replaces L1 group-Lasso + mode-column penalty + temporal smoothing. A single penalty term `λ_sc` encodes both sparsity and spatial compactness via distance-weighted group-Lasso, with automatic centroid learning.
- **No temporal smoothing**: the second-difference penalty on `h_k(t)` is deliberately removed. Activation shape is now purely data-driven, avoiding over-constraint on short/transient episodes.
- **H constraint**: unit-sphere Riemannian gradient (`||H[k]||_2 = 1`) replaces post-hoc row normalization with scale absorption into B.
- **Simplified post-processing**: a single hard-threshold step replaces the old merge → prune → refine pipeline.
- **Backtracking line search**: replaces fixed-step gradient descent. B uses sufficient-decrease backtracking; H uses Armijo backtracking (c=1e-4).
- **Current canonical velocity configuration**: V2 is applied to frame-to-frame velocity (`use_velocity=True`) with `Kmax=8`, `K_selection_method="svd"`, `svd_target_r2=0.90`, `lambda_sc=0.05`, `rho=1.0`, `kappa=4.0`, `support_rel_thresh=0.08`, and `max_iter=200`.

### Motion Unit Extraction
- **Current canonical preprocessing**: registered displacement fields are averaged into non-overlapping 7 x 7 patches and median-filtered with a 1 x 3 x 3 x 1 window. Motion-unit detection uses frame-to-frame velocity, `use_abs_dev=True`, and a MAD threshold with `window_size_t=21`, `window_size_xy=3`. The runner does not pass `median_local` to `getMotionUnit`, so its internal local-median calculation uses API defaults (`median_window_t=21`, `median_window_xy=5`); record this for exact reproduction.
- `estimate_rest_state_motion`: new **`return_median`** parameter — when True, also returns the local median motion for reuse in `getMotionUnit`.
- `getMotionUnit`: new **`use_abs_dev`** mode (default `True`) — a patch is active when its **deviation from the local median baseline** exceeds the MAD threshold (`|motionMag - median_local| > restMotion`), instead of comparing raw magnitude against the threshold. New parameters: `median_local`, `median_window_t`, `median_window_xy`.

### Clustering & Pattern Building
- New **`getMotionPattern()`** function with **`unit_type="region"|"mode"`** — supports clustering `MotionMode` objects directly without first splitting into `MotionRegion`s.
- New **`b_distance="correlation"`** option for response-field distance: uses `1 - |Pearson_r(B1, sign·B2)|` on the spatial overlap, as an alternative to the existing normalized L2 distance.
- **`getMotionRegionPattern()`** is now a backward-compatible wrapper that delegates to `getMotionPattern(unit_type="region")`.
- `filter_regions_for_patterns` now handles both `MotionRegion` and `MotionMode` objects, auto-computing missing attributes (`strength`, `area_effective`, `duration`, `mean_response_vector`) from `response_strength` / `response_field` / `activation` as needed.
- New helpers: `collect_modes_from_episodes`, `collect_units_from_episodes`, `_unit_centroid`, `_response_field_correlation_on_overlap`.
- **Current canonical direct-mode configuration**: `getMotionPattern(..., unit_type="mode")` clusters modes directly, so no Stage 05 region cache is written. It uses `min_iou=0.08`, `omega=mu=0.5`, `b_distance="correlation"`, `spatial_rule="iou"`, complete linkage, `cluster_dist_thresh=0.45`, `compute_unified=True`, `unified_mask_mode="best_cc"`, `unified_sign_method="correlation"`, `min_pattern_members=2`, and `min_unified_area=50`. `n_members >= 5` is a downstream analysis/visualisation filter, not an extraction threshold. The separate threshold comparison changes only `cluster_dist_thresh` to 0.55 while reusing the same mode cache.

### File Organization
- V2 code lives directly in `motion_correlation_pattern.py`. V1 code has been replaced in-place; there is no separate `motion_correlation_pattern_v2.py` file.

---

# Methods
## Abstract
Essentially, our goal is to identify regions, timestamps and specific patterns of genuine significant motion (excluding artifacts) from the displacement fields obtained via image registration, and we split this workflow into four sequential steps.
## Pipeline
### Extract motion units
#### Assumptions
For any given region, motion is either absent or consists of small random fluctuations for most of the time, with pronounced movement occurring only sporadically at a small subset of time points. Our target is to extract these prominent motion events.
#### 1.compute patch-wise motion
In fact, we do not need to perform motion analysis down to the single-pixel level. Even if meaningful motion exists within such tiny regions, reliable identification of these movements cannot be guaranteed. Accordingly, we simply partition the full image into a grid of non-overlapping n×n patches, where the motion value of each patch is defined as the average motion across all constituent pixels inside the patch.

#### 2.estimate the resting-state motion amplitude for each patch
To this end, we apply median filtering with relatively large temporal and spatial window sizes for estimation. Median rather than mean filtering is adopted because genuine motion events generally feature large amplitude values, which can severely bias the mean calculation while exerting much weaker interference on the median. We avoid computing a single universal baseline value for the entire dataset, as resting motion intensity varies across anatomical locations and over time. This inherent assumption inevitably precludes our analysis of slow, long-timescale motions.

**V2 update (`estimate_rest_state_motion`):** Uses Median Absolute Deviation (MAD) from a local spatiotemporal median: `restMotion = scale × 1.4826 × median(|motionMag - median_local|)`. The new `return_median` parameter (default `False`) allows returning the local median alongside the resting-state estimate, so `getMotionUnit` can reuse it without recomputing another median filter.

#### 3.Extract motion units
For each patch, extract the temporal segments where the amplitude exceeds the resting amplitude for several consecutive frames, and denote these segments as motion units.

**V2 update (`getMotionUnit`):** Two detection modes are now available, controlled by `use_abs_dev` (default `True`):

- **`use_abs_dev=True` (default, new):** A patch is active when its deviation from the local baseline exceeds the MAD-estimated threshold: `|motionMag - median_local| > restMotion`. This is more robust because it measures *excess* motion relative to the local context rather than raw magnitude.
- **`use_abs_dev=False` (old behavior):** `motionMag > restMotion` — raw magnitude comparison.

When `use_abs_dev=True`, the local median can be either precomputed via `estimate_rest_state_motion(..., return_median=True)` and passed as `median_local`, or computed internally with `median_window_t` and `median_window_xy` controlling the filter size.

#### Weakness
We couldn't analyze the  slow, long-timescale motions.

### Merged into motion episodes

**Current canonical episode configuration:** `tolerant_time=1`, `min_total_area=30`, `expand_frames=1`, `min_cc_area=8`, and `global_motion_mode="median"`. After grouping, episodes are filtered with `max_fov_fraction=0.5`, `min_duration=3`, `max_global_corr=0.90`, `max_edge_fraction=0.80`, and `edge_width=3`.
#### Assumption
When numerous patches contain detected motion units at the same time point, it indicates that the organism is undergoing coordinated physiological activities associated with movement. Such activities may arise from the coordination of multiple distinct biological events. However, these fine-grained details are not considered in the current analysis; synchronous occurrence of these motion units implies that they originate from a single large-scale physiological event.
#### Implementation details
Motion units with nearly coincident start and end timestamps are grouped into a single motion episode. This grouping is implemented via graph decomposition. Each motion episode is treated as an individual vertex. An edge is created between two vertices if the differences of their onset and offset times are both smaller than the threshold $t_{\text{thre}}$. All episodes belonging to the same connected component are subsequently merged into a single episode.This strategy efficiently handles propagating sequential motions exemplified as follows: four successive episodes $\mathrm{A}\,(t\sim t+10)$, $\mathrm{B}\,(t+1\sim t+11)$, $\mathrm{C}\,(t+2\sim t+12)$, $\mathrm{D}\,(t+3\sim t+13)$. Without this graph-based merging, these four events would be treated as independent motions, whereas the proposed scheme aggregates them into one unified episode.


### Decomposite into motion modes/regions

> **⚠️ V2 note (2026-08-03):** The method described below is the **V1** decomposition (L1 group-Lasso + temporal smoothing + merge/prune/refine pipeline). The current **V2** method uses a sparse-compact objective with automatic centroid learning, Riemannian H updates, and a simplified hard-threshold-only post-processing. See the **V2: Sparse-Compact Decomposition** section at the end of this document for the updated formulation.

### Assumption
We assume that all patches affected by the same underlying physical motion source share consistent motion directions. In our framework, such a source is defined as a motion mode, which imposes a fixed, time-invariant motion vector on every patch within its effective spatial coverage. Furthermore, instantaneous response amplitudes across patches belonging to an identical motion mode are mutually proportional. Equivalently, these patches share a common temporal-only activation profile independent of spatial location.

For example, some ball is contracting and relaxing.
<img src="../../../docs/assets/motion_mode.png" alt="result" width="320" height="130">

#### Objective function

##### Notation

For one motion episode containing $N$ valid patches and $T$ time frames (after subtracting global background motion):

- $\mathbf{Y}_i(t) \in \mathbb{R}^2$ — cumulative displacement of patch $i$ at time $t$, with the global background drift removed.
- $h_k(t) \in \mathbb{R}$ — scalar temporal activation of mode $k$ at time $t$.
- $\mathbf{b}_{ik} \in \mathbb{R}^2$ — time-invariant 2D response vector of patch $i$ to mode $k$ (the direction and magnitude this mode "pulls" patch $i$).

The model assumes a bilinear decomposition:

```math
\mathbf{Y}_i(t) \approx \sum_{k=1}^{K} h_k(t) \cdot \mathbf{b}_{ik}
```

Stacking all patches and both spatial dimensions into matrix form:

```math
\underbrace{\mathbf{M}}_{2N \times T} \approx \underbrace{\mathbf{B}}_{2N \times K} \; \underbrace{\mathbf{H}}_{K \times T}
```

where:
- $\mathbf{M} = \begin{bmatrix} \mathbf{Y}_x^\top \\ \mathbf{Y}_y^\top \end{bmatrix}$ stacks the x-components (top $N$ rows) and y-components (bottom $N$ rows) of all patches.
- $\mathbf{B}$ has the same structure: rows $1..N$ are $\mathbf{b}_{ik}$ x-components; rows $N+1..2N$ are y-components.
- $\mathbf{H}_{k,t} = h_k(t)$.

##### Objective function

The diagnostic loss minimized during fitting is:

```math
\mathcal{L}(\mathbf{B}, \mathbf{H}) = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{patch}} + \mathcal{L}_{\text{mode}} + \mathcal{L}_{\text{smooth}}
```

**1. Normalized reconstruction error:**

```math
\mathcal{L}_{\text{recon}} = \frac{\|\mathbf{M} - \mathbf{B}\mathbf{H}\|_F^2}{\|\mathbf{M}\|_F^2}
```

Dividing by the total data energy $\|\mathbf{M}\|_F^2$ makes the loss scale independent of the number of patches, frame count, and overall motion magnitude. This allows a single set of hyperparameters $\lambda$ to work across diverse episodes.

**2. Patch-level group sparsity (encourages spatially localized modes):**

```math
\mathcal{L}_{\text{patch}} = \frac{\lambda_B}{N K} \sum_{i=1}^{N} \sum_{k=1}^{K} \frac{\|\mathbf{b}_{ik}\|}{B_{\text{scale}}}
```

Each $\mathbf{b}_{ik}$ is the 2D response vector of patch $i$ to mode $k$. The $\ell_2$ norm $\|\mathbf{b}_{ik}\|$ treats the two spatial components as a single group — a patch is either "in" or "out" of a mode's influence, regardless of direction. This group-lasso penalty drives irrelevant patch responses to **exactly zero**, producing spatially sparse modes where each mode only affects a compact subset of patches.

$B_{\text{scale}} = \sqrt{\operatorname{mean}(\mathbf{M}^2)}$ is the RMS motion magnitude of the episode. Dividing by it makes $\lambda_B$ dimensionless and comparable across datasets with different motion strengths.

**3. Mode-level column sparsity (encourages automatic mode selection):**

```math
\mathcal{L}_{\text{mode}} = \frac{\lambda_{\text{mode}}}{K} \sum_{k=1}^{K} \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N} \|\mathbf{b}_{ik}\|^2}}{B_{\text{scale}}}
```

While $\mathcal{L}_{\text{patch}}$ sparsifies individual patch responses, $\mathcal{L}_{\text{mode}}$ penalizes the entire column $\mathbf{B}_{:,k}$ as a group, encouraging modes with negligible total energy to be zeroed out entirely. This implements **automatic mode selection**: if the chosen $K$ is larger than needed, weak or redundant modes are suppressed by this penalty and later pruned.

**4. Temporal smoothness of activations:**

```math
\mathcal{L}_{\text{smooth}} = \frac{\lambda_H}{K \cdot (T-2)} \sum_{k=1}^{K} \sum_{t=1}^{T-2} \big(h_k[t] - 2h_k[t+1] + h_k[t+2]\big)^2
```

This penalizes the squared second difference of each activation time course. Real biological motion (peristalsis, heartbeat-driven pulsation, etc.) has smoothly varying temporal profiles with gradual onset and offset, not frame-to-frame jitter. The second-difference penalty suppresses high-frequency noise while preserving genuine smooth temporal dynamics. Dividing by $K \cdot (T-2)$ normalizes across episodes with different numbers of modes and time lengths.

In compact matrix form, using the $(T-2) \times T$ second-difference matrix $\mathbf{D}_2$:

```math
\mathcal{L}_{\text{smooth}} = \lambda_H \cdot \operatorname{mean}\!\big((\mathbf{H}\mathbf{D}_2^\top)^2\big)
```

where $\mathbf{D}_2$ has entries $D_2[t,t]=1$, $D_2[t,t+1]=-2$, $D_2[t,t+2]=1$ for $t=1..T-2$, and $\mathbf{L} = \mathbf{D}_2^\top \mathbf{D}_2$ is the $T \times T$ second-difference Gram matrix.

##### Design rationale (why these four terms together)

| Term | Purpose | What happens without it |
|---|---|---|
| $\mathcal{L}_{\text{recon}}$ | Data fidelity — the decomposition must explain the observed motion | No meaningful fit |
| $\mathcal{L}_{\text{patch}}$ | **Spatial localization** — each mode should affect only a compact set of patches | Modes become dense, affecting all patches equally — uninterpretable and unbiological |
| $\mathcal{L}_{\text{mode}}$ | **Model selection** — automatically determines the effective number of modes | Overfitting noise with spurious low-energy modes |
| $\mathcal{L}_{\text{smooth}}$ | **Temporal regularization** — activations should be smooth, not jittery | High-frequency noise in $h_k(t)$; unstable convergence |

The scale ambiguity $h_k \cdot \mathbf{b}_{ik} = (\alpha h_k) \cdot (\mathbf{b}_{ik}/\alpha)$ is resolved by normalizing each $h_k$ to unit norm after every H-update and absorbing the scale into $\mathbf{B}$. Without this, the sparsity penalties on $\mathbf{B}$ would be ill-defined (one could trivially shrink the penalty by scaling H up and B down).

#### How to solve it

The objective $\mathcal{L}(\mathbf{B}, \mathbf{H})$ is **biconvex** — convex in $\mathbf{B}$ given $\mathbf{H}$, convex in $\mathbf{H}$ given $\mathbf{B}$, but not jointly convex. We use **alternating proximal gradient descent**: each iteration performs one gradient step on $\mathbf{B}$ followed by one gradient step on $\mathbf{H}$, cycling until convergence.

##### B-subproblem (H fixed): Proximal Gradient Descent

The smooth part $\mathcal{L}_{\text{recon}}$ is differentiable in $\mathbf{B}$, while $\mathcal{L}_{\text{patch}} + \mathcal{L}_{\text{mode}}$ are convex but non-smooth (norms). We apply the **proximal gradient method**:

**Step 1 — Gradient descent on the smooth term:**

```math
\tilde{\mathbf{B}} = \mathbf{B} - \eta_B \cdot \nabla_{\mathbf{B}} \mathcal{L}_{\text{recon}}
```

```math
\nabla_{\mathbf{B}} \mathcal{L}_{\text{recon}} = \frac{2}{\|\mathbf{M}\|_F^2} (\mathbf{B}\mathbf{H} - \mathbf{M}) \mathbf{H}^\top
```

The step size is set inversely proportional to the Lipschitz constant of the gradient:

```math
\eta_B = \frac{1}{\frac{2}{\|\mathbf{M}\|_F^2} \|\mathbf{H}\mathbf{H}^\top\|_2 + \epsilon}
```

**Step 2 — Group soft-thresholding (proximal operator of $\mathcal{L}_{\text{patch}}$):**

For each patch $i$ and mode $k$, treat the 2D vector $\mathbf{b}_{ik} \in \mathbb{R}^2$ as an indivisible group:

```math
\mathbf{b}_{ik} \leftarrow \mathbf{b}_{ik} \cdot \max\!\left(0,\; 1 - \frac{\tau_{\text{patch}}}{\|\mathbf{b}_{ik}\|}\right)
```

where $\tau_{\text{patch}} = \eta_B \cdot \frac{\lambda_B}{N K \cdot B_{\text{scale}}}$.

This is the proximal operator of the $\ell_{2,1}$ mixed norm. Intuitively: if the 2D response strength $\|\mathbf{b}_{ik}\|$ is below the threshold $\tau_{\text{patch}}$, the patch response is set to exactly zero for that mode. Otherwise it is shrunk toward zero by $\tau_{\text{patch}}$.

**Step 3 — Column soft-thresholding (proximal operator of $\mathcal{L}_{\text{mode}}$):**

For each mode $k$, treat the entire column $\mathbf{B}_{:,k} \in \mathbb{R}^{2N}$ as a group:

```math
\mathbf{B}_{:,k} \leftarrow \mathbf{B}_{:,k} \cdot \max\!\left(0,\; 1 - \frac{\tau_{\text{mode}}}{\|\mathbf{B}_{:,k}\|}\right)
```

where $\tau_{\text{mode}} = \eta_B \cdot \frac{\lambda_{\text{mode}}}{K \sqrt{N} \cdot B_{\text{scale}}}$.

This shrinks the entire spatial pattern of a mode — if a mode's total energy across all patches is too small, the whole column collapses to zero, effectively removing that mode.

##### H-subproblem (B fixed): Gradient Descent

Both $\mathcal{L}_{\text{recon}}$ and $\mathcal{L}_{\text{smooth}}$ are smooth in $\mathbf{H}$, so no proximal operator is needed. A standard gradient step suffices:

```math
\mathbf{H} \leftarrow \mathbf{H} - \eta_H \cdot \nabla_{\mathbf{H}} (\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{smooth}})
```

```math
\nabla_{\mathbf{H}} \mathcal{L}_{\text{recon}} = \frac{2}{\|\mathbf{M}\|_F^2} \mathbf{B}^\top (\mathbf{B}\mathbf{H} - \mathbf{M})
```

```math
\nabla_{\mathbf{H}} \mathcal{L}_{\text{smooth}} = \frac{2\lambda_H}{K \cdot (T-2)} \cdot \mathbf{H} \mathbf{L}
```

where $\mathbf{L} = \mathbf{D}_2^\top \mathbf{D}_2$ is the second-difference Gram matrix. The step size:

```math
\eta_H = \frac{1}{\frac{2}{\|\mathbf{M}\|_F^2} \|\mathbf{B}^\top\mathbf{B}\|_2 + \frac{2\lambda_H}{K\cdot(T-2)} \|\mathbf{L}\|_2 + \epsilon}
```

**Step 3 — Resolve scale indeterminacy:**

After the gradient step, each row of $\mathbf{H}$ is normalized to unit $\ell_2$ norm, and the corresponding column of $\mathbf{B}$ absorbs the scale:

For each mode $k$: $\quad s_k = \|\mathbf{h}_k\|, \quad \mathbf{h}_k \leftarrow \mathbf{h}_k / s_k, \quad \mathbf{B}_{:,k} \leftarrow s_k \cdot \mathbf{B}_{:,k}$

This ensures the sparsity penalties on $\mathbf{B}$ are well-defined and the diagnostic loss decreases monotonically.

##### Convergence

The alternating updates repeat until:

```math
\operatorname{mean}\!\big(|\mathbf{B} - \mathbf{B}_{\text{old}}|\big) + \operatorname{mean}\!\big(|\mathbf{H} - \mathbf{H}_{\text{old}}|\big) < \text{tol} = 10^{-4}
```

or a maximum of `max_iter` = 100 iterations is reached. In practice, the algorithm typically converges within 20–50 iterations for most episodes.

##### Initialization

$\mathbf{B}$ and $\mathbf{H}$ are initialized via **spatial seeding**:

1. Rank patches by motion energy $\sum_t \|\mathbf{Y}_i(t)\|^2$.
2. Select the top $K$ high-energy patches as seeds, enforcing a minimum spatial distance (`min_seed_dist = 3`) between seeds to ensure diverse spatial coverage.
3. For each seed patch $i$, initialize $h_k(t)$ as the projection of the patch's trajectory $\mathbf{Y}_i(t)$ onto its mean direction vector $\bar{\mathbf{b}}_i = \frac{1}{T}\sum_t \mathbf{Y}_i(t)$:
   

```math
h_k(t) = \frac{\mathbf{Y}_i(t)^\top \bar{\mathbf{b}}_i}{\|\bar{\mathbf{b}}_i\|}
```

4. Initialize $\mathbf{B}$ by solving $\min_{\mathbf{B}} \|\mathbf{M} - \mathbf{B}\mathbf{H}\|_F^2$ in closed form:
   

```math
\mathbf{B} = \mathbf{M} \mathbf{H}^\top (\mathbf{H}\mathbf{H}^\top + 10^{-6}\mathbf{I})^{-1}
```

##### Post-optimization pipeline

After the alternating optimization converges, three additional steps refine the decomposition:

1. **Reconstruction-preserving merge** — modes whose activations are highly correlated ($|\cos(h_i, h_j)| > 0.98$) are candidates for merging. Each candidate group is approximated by a single rank-1 mode via SVD, and the merge is **accepted only if** the global $R^2$ drops by less than `max_r2_drop = 0.03`. This prevents over-merging distinct modes that happen to have similar activation shapes.

2. **Pruning** — modes are removed if they fail any of: (a) total response mass below `min_mode_mass`, (b) incremental explained energy below `min_incremental_energy = 0.005`, (c) spatial support area below `min_support_area = 3`, or (d) spatial density above `max_mode_density = 1.0` (catches degenerate modes affecting all patches).

3. **Final refinement** — a short re-optimization (10–30 iterations) from the pruned B, H to allow the remaining modes to absorb the energy previously explained by the pruned ones.

##### From modes to regions: spatial splitting

A single motion mode may have a response support that spans several spatially disconnected regions. This often indicates that multiple distinct biological sources happen to share a similar activation profile and were merged into one mode by the bilinear decomposition. We therefore split each mode into spatially coherent **motion regions** by applying gap-tolerant connected-component labeling on the mode's response strength map. Concretely: we use morphological closing and dilation to bridge small gaps (tolerant connectivity), extract connected components on the tolerant mask, then restrict each component to the original (un-dilated) support pixels. Small isolated fragments below a minimum area threshold are discarded directly. This produces a flat list of `MotionRegion` objects, each inheriting its parent mode's activation $h_k(t)$ but carrying a spatially localized response pattern and mask. These regions serve as the basic units for downstream cross-episode pattern discovery.

### Clustering into motion patterns

> **⚠️ V2 note (2026-08-03):** The clustering API has been updated. The primary entry point is now **`getMotionPattern()`** (with `unit_type="region"|"mode"`), which supports clustering both `MotionRegion` and `MotionMode` objects. The old `getMotionRegionPattern()` is retained as a backward-compatible wrapper. See the **V2: Clustering Updates** subsection at the end of this document.

#### Assumption

The same type of motion event — characterized by a consistent spatial location, a consistent motion direction, and a consistent temporal activation profile — **recurs across different episodes**. For instance, a peristaltic wave passing through a given gastrointestinal segment will generate a similar activation time course and a similar spatial pattern of patch-level response vectors each time it occurs. Our goal is to discover these recurring patterns by clustering motion regions across episodes.

#### Method overview

We treat each `MotionRegion` from all episodes as a node and perform hierarchical clustering. A pair of regions is considered for clustering only if they satisfy two levels of similarity:

**1. Hard spatial gate.** Two regions must occupy overlapping anatomical locations to belong to the same pattern. This is enforced by requiring a minimum IoU (Intersection over Union) between their spatial masks (default: `min_iou = 0.10`). Region pairs with no spatial overlap are assigned infinite distance and never clustered together.

**2. Soft similarity distance.** For region pairs that pass the spatial gate, we compute a combined distance:

```math
D(r_i, r_j) = \omega \cdot D_h(r_i, r_j) + \mu \cdot D_b(r_i, r_j)
```

where:
- $D_h$: **sign-aware DTW distance** between the two regions' temporal activations $h_i(t)$ and $h_j(t)$. Because the sign of $h_k(t)$ is arbitrary ($h_k \cdot \mathbf{b}_{ik} = (-h_k) \cdot (-\mathbf{b}_{ik})$), we compute DTW against both $h_j$ and $-h_j$ and take the smaller distance.
- $D_b$: **response field distance** computed only on the spatial overlap of the two regions — i.e., how similar are the 2D response vectors $\mathbf{b}_i$ and $\mathbf{b}_j$ (after sign-alignment) in the area where both regions are active.
- $\omega$ and $\mu$ are weights balancing temporal similarity against spatial response similarity (default equally weighted).

**3. Complete-linkage hierarchical clustering.** We use complete (maximum) linkage to avoid the chain effect where a series of marginally similar intermediate regions bridges two fundamentally dissimilar ones. Clusters are cut at a distance threshold (`cluster_dist_thresh`), and each resulting cluster forms one `MotionPattern`.

**4. Prototype computation.** For each pattern, we compute:
- **Prototype activation:** medoid of all member activations (chosen via weighted DTW-based centrality, preserving variable-length time courses).
- **Prototype response vector:** weighted average of member mean response vectors, with sign-alignment to the medoid activation.
- **Prototype spatial map:** weighted average of member response strength maps.
- **Spatial center and covariance:** weighted statistics of member region centroids.

#### Rationale

This design follows a simple principle: **same biological event → same place, same direction, same temporal profile**. The spatial IoU gate ensures we only compare regions that could plausibly be the same anatomical structure. The DTW distance accommodates variable-length activations (different episodes have different durations). The sign-aware comparison handles the inherent sign ambiguity from the bilinear mode decomposition. The response-field distance on overlapping support prevents regions with similar activations but opposite motion directions from being merged.

The complete-linkage hierarchical clustering is deliberately conservative — it prefers splitting over merging when in doubt. This is appropriate because over-merging distinct motion patterns would conflate different biological events, while under-merging (splitting one true pattern into two) is a less severe failure mode that can be addressed in downstream analysis.

---

# V2: Sparse-Compact Decomposition

## Motivation

V1 uses L1 group-Lasso + mode-column penalty + temporal smoothing, but has several limitations:

1. **Complex post-processing**: three steps (merge → prune → refine) are needed to clean up modes after optimization.
2. **Temporal smoothness assumption**: the second-difference penalty assumes slow-varying activations, which is unsuitable for fast transient motion (e.g. heartbeat, rapid peristalsis).
3. **Weak spatial regularization**: L1 only encourages sparsity, not spatial compactness — modes can fracture into multiple disconnected fragments that require downstream region-splitting to separate.

V2 adopts a unified **sparse-compact** objective that encodes both spatial sparsity and compactness in a single penalty term, and automatically learns the spatial centroid of each mode.

## Objective Function

### Notation

Same as V1: $\mathbf{M} \in \mathbb{R}^{2N \times T}$, $\mathbf{B} \in \mathbb{R}^{2N \times K}$, $\mathbf{H} \in \mathbb{R}^{K \times T}$.

Additional definitions:
- $\mathbf{r}_i \in [0,1]^2$ — normalized spatial coordinates of patch $i$.
- $\boldsymbol{\mu}_k \in [0,1]^2$ — amplitude-weighted spatial centroid of mode $k$.

### Full Objective

```math
\min_{\mathbf{B},\mathbf{H},\boldsymbol{\mu}} \quad
\frac{\|\mathbf{M} - \mathbf{B}\mathbf{H}\|_F^2}{\|\mathbf{M}\|_F^2}
+ \frac{\lambda_{sc}}{N K s_B} \sum_{i=1}^{N} \sum_{k=1}^{K}
\big(\rho + \kappa \cdot \|\mathbf{r}_i - \boldsymbol{\mu}_k\|_2^2\big) \cdot \|\mathbf{b}_{ik}\|_2
```

```math
\text{subject to} \quad \|\mathbf{H}[k]\|_2 = 1 \quad \text{for every } k
```

where $s_B = \sqrt{T} \cdot \sqrt{\operatorname{mean}(\mathbf{M}^2)}$ is the motion magnitude scale.

### Penalty Semantics

| Parameter | Meaning | Effect |
|---|---|---|
| $\lambda_{sc}$ | Total regularization strength | Larger → sparser and more compact modes |
| $\rho$ (rho) | Base sparsity weight | Applied uniformly to all patches; drives irrelevant $b_{ik} \to 0$ |
| $\kappa$ (kappa) | Compactness weight | Patches far from centroid $\boldsymbol{\mu}_k$ receive stronger sparsity penalty, encouraging modes to be spatially clustered |

Intuitively:
- The effective penalty weight for patch $i$ under mode $k$ is $\rho + \kappa \cdot \|\mathbf{r}_i - \boldsymbol{\mu}_k\|^2$.
- Close to the centroid → low penalty → strong $b_{ik}$ allowed.
- Far from the centroid → high penalty → $b_{ik}$ compressed to zero.
- The centroid $\boldsymbol{\mu}_k$ is automatically learned as the amplitude-weighted mean of participating patches.

### Key Differences from V1

| Aspect | V1 | V2 |
|---|---|---|
| B regularization | L1 group-Lasso ($\|\mathbf{b}_{ik}\|$) + mode-column penalty ($\|\mathbf{B}_{:,k}\|$) | **Weighted** L1 group-Lasso, weights depend on distance to learned centroid |
| Spatial modeling | None (relies on post-hoc region splitting) | **Automatic centroid learning**; compactness encoded in the penalty |
| Temporal modeling | Second-difference smoothing ($\lambda_H$) | **No temporal smoothing** — activation shape is purely data-driven |
| H constraint | Normalize then absorb scale into B | **Unit-sphere constraint** $\|\mathbf{H}[k]\|_2 = 1$ (Riemannian gradient) |
| Post-processing | merge + prune + refine (3 steps) | **Hard threshold only** (1 step) |
| Optimization | Fixed-step gradient descent + proximal operators | Proximal gradient + **backtracking line search** |
| Parameter count | 4 ($\lambda_B$, $\lambda_{\text{mode}}$, $\lambda_H$, tol) | 3 ($\lambda_{sc}$, $\rho$, $\kappa$) |

### Why Temporal Smoothing Was Removed

The second-difference penalty assumes $h_k(t)$ varies slowly. In practice:
- Many biological motions are fast and transient (heartbeats, rapid peristaltic waves).
- Episodes are typically short (5–15 frames).
- A second-difference penalty on such short sequences over-constrains the shape of $h_k(t)$.

V2 deliberately imposes **no shape prior** on $h_k(t)$, letting the data determine the activation curve.

## Optimization Algorithm

### B Update: Weighted Group-Lasso Proximal Gradient

1. Compute gradient of the smooth (reconstruction) part:
   ```math
   \nabla_{\mathbf{B}} \mathcal{L}_{\text{recon}} = \frac{2}{\|\mathbf{M}\|_F^2} (\mathbf{B}\mathbf{H} - \mathbf{M}) \mathbf{H}^\top
   ```

2. Step size (inverse of gradient Lipschitz constant):
   ```math
   \eta_B = 1 \;\big/\; \left(\frac{2}{\|\mathbf{M}\|_F^2} \|\mathbf{H}\mathbf{H}^\top\|_2\right)
   ```

3. Compute spatial weight matrix $\mathbf{W} \in \mathbb{R}^{N \times K}$:
   ```math
   W[i,k] = \rho + \kappa \cdot \|\mathbf{r}_i - \boldsymbol{\mu}_k\|_2^2
   ```

4. Weighted group soft-thresholding (for each patch $i$, mode $k$):
   ```math
   \tau_{ik} = \eta_B \cdot \lambda_{sc} \cdot W[i,k] \;\big/\; (N K s_B)
   ```
   ```math
   \mathbf{b}_{ik} \leftarrow \mathbf{b}_{ik} \cdot \max\!\left(0,\; 1 - \frac{\tau_{ik}}{\|\mathbf{b}_{ik}\|_2}\right)
   ```

5. **Backtracking line search**: if the trial B increases the objective (with centroid held fixed), the step size is halved and the proximal step retried, up to `max_backtracking = 30` times. The acceptance condition is **sufficient decrease** (not Armijo):
   ```math
   \mathcal{L}(\mathbf{B}_{\text{trial}}, \mathbf{H}, \boldsymbol{\mu}) \leq \mathcal{L}(\mathbf{B}_{\text{old}}, \mathbf{H}, \boldsymbol{\mu}) + 10^{-12}
   ```

### μ Update: Exact Minimizer

For fixed B, the centroid $\boldsymbol{\mu}_k$ has a closed-form solution — the amplitude-weighted spatial mean:

```math
\boldsymbol{\mu}_k = \frac{\sum_i \|\mathbf{b}_{ik}\|_2 \cdot \mathbf{r}_i}{\sum_i \|\mathbf{b}_{ik}\|_2}
```

### H Update: Riemannian Gradient on the Unit Sphere

1. Compute Euclidean gradient:
   ```math
   \nabla_{\mathbf{H}} = \frac{2}{\|\mathbf{M}\|_F^2} \mathbf{B}^\top (\mathbf{B}\mathbf{H} - \mathbf{M})
   ```

2. Project onto the tangent space of the product of unit spheres:
   ```math
   \nabla_{\mathbf{H}}^{\text{tangent}} = \nabla_{\mathbf{H}} - (\nabla_{\mathbf{H}} \odot \mathbf{H}) \odot \mathbf{H}
   ```
   where $\odot$ is element-wise multiplication with row-wise summation over the inner product.

3. Step along the tangent direction, then retract back to the sphere:
   ```math
   \mathbf{H}_{\text{trial}} = \text{normalize\_rows}\!\left(\mathbf{H} - \eta_H \cdot \nabla_{\mathbf{H}}^{\text{tangent}}\right)
   ```
   The step size is $\eta_H = 1 \;\big/\; \left(\frac{2}{\|\mathbf{M}\|_F^2} \|\mathbf{B}^\top\mathbf{B}\|_2\right)$.

4. **Armijo backtracking** with $c = 10^{-4}$:
   ```math
   \mathcal{L}(\mathbf{B}, \mathbf{H}_{\text{trial}}, \boldsymbol{\mu}) \leq \mathcal{L}(\mathbf{B}, \mathbf{H}_{\text{old}}, \boldsymbol{\mu}) - 10^{-4} \cdot \eta_H \cdot \|\nabla_{\mathbf{H}}^{\text{tangent}}\|_F^2
   ```
   If the condition fails, $\eta_H$ is halved and the step retried.

### Global Safety

After both B and H blocks, if the full objective increased relative to the previous iteration, **both B and H are reverted** to their previous values and the step sizes are set to zero for that iteration.

### Convergence Check

The algorithm stops when **both** conditions are satisfied:

```math
\frac{|\mathcal{L}_{\text{old}} - \mathcal{L}|}{\max(|\mathcal{L}_{\text{old}}|, \epsilon)} \leq \text{tol}
\quad\text{AND}\quad
\operatorname{mean}(|\Delta\mathbf{B}|) + \operatorname{mean}(|\Delta\mathbf{H}|) \leq \sqrt{\text{tol}}
```

Default: `tol = 1e-4`, `max_iter = 200`.

## Post-Processing

Only **one step** — hard thresholding applied after optimization converges:

```math
a_{ik} = \|\mathbf{b}_{ik}\|_2, \qquad
\tau_k = \text{support\_rel\_thresh} \cdot \max_i a_{ik}
```

```math
\mathbf{b}_{ik} \leftarrow \mathbf{0} \quad \text{if} \quad a_{ik} \leq \tau_k
```

Both spatial components (x, y) are zeroed together. Default: `support_rel_thresh = 0.08`.

## Parameter Recommendations

| Parameter | Recommended Range | Notes |
|---|---|---|
| `lambda_sc` | 0.01–0.2 | Total regularization; start at 0.05 |
| `rho` | 0.5–2.0 | Base sparsity; default 1.0 |
| `kappa` | 2.0–8.0 | Compactness; default 4.0; increase for tighter spatial clustering |
| `support_rel_thresh` | 0.05–0.15 | Hard threshold relative cutoff |
| `svd_target_r2` | 0.80–0.95 | Target cumulative R² for SVD-based K selection |

### V1 → V2 Parameter Migration

**Do not reuse V1's $\lambda_B$ as V2's $\lambda_{sc}$.** The two objectives have fundamentally different penalty forms:
- V1: $\lambda_B \cdot \operatorname{mean}(\|\mathbf{b}_{ik}\| / B_{\text{scale}})$ — uniform-weight L1
- V2: $\lambda_{sc} \cdot \operatorname{mean}((\rho + \kappa \cdot d^2) \cdot \|\mathbf{b}_{ik}\| / s_B)$ — spatially-weighted L1

A small grid search over V2 parameters is recommended rather than attempting to convert from V1 values.

## API Usage

The examples below show generic API values. The current canonical velocity and direct-mode configuration is recorded in the V2 update summary above.

```python
from wholistic_registration.utils.motion_correlation_pattern import (
    decompose_episode_motion_modes,
    getMotionModes,
)

# Decompose a single episode
modes = decompose_episode_motion_modes(
    episode,
    Kmax=4,                # or higher, e.g. 8
    lambda_sc=0.05,        # total regularization
    rho=1.0,               # base sparsity weight
    kappa=4.0,             # spatial compactness
    max_iter=200,
    support_rel_thresh=0.08,
    K_selection_method='svd',   # 'svd' or 'fixed'
    svd_target_r2=0.85,
    use_velocity=False,    # False = cumulative displacement; True = frame-to-frame velocity
    verbose=True,
)

# Or decompose all episodes at once
all_modes = getMotionModes(motion_episodes, Kmax=4, lambda_sc=0.05, ...)
```

## `episode.mode_model` Metadata (V2)

After optimization, `episode.mode_model` is a dict with the following keys:

| Key | Description |
|---|---|
| `B` | Final B after hard threshold |
| `B_before_hard_threshold` | Raw B at optimization convergence |
| `H` | Activation matrix ($K \times T$) |
| `loss_history` | List of per-iteration dicts with `loss`, `recon`, `sparse_compact`, `sparse_compact_raw`, `step_B`, `step_H`, `delta`, `relative_decrease`, `H_norm_error` |
| `mode_centers_normalized` | Learned mode centroids in normalized coordinates $[0,1]^2$ |
| `mode_centers_patch` | Centroids in patch-index coordinates |
| `lambda_sc`, `rho`, `kappa` | Regularization hyperparameters |
| `Kmax`, `K_init`, `K_final`, `K_modes`, `seeds` | K selection and initialization info |
| `total_energy` | $\|\mathbf{M}\|_F^2$ |
| `optimized_recon`, `optimized_r2` | Reconstruction loss / R² **before** hard threshold |
| `final_recon`, `final_r2` | Reconstruction loss / R² **after** hard threshold |
| `hard_threshold_support_mask` | Boolean mask of surviving (patch, mode) entries |
| `hard_threshold_removed_count` | Number of (patch, mode) entries zeroed by hard threshold |
| `hard_threshold_removed_energy` | Sum of squared B entries removed |
| `K_selection_method`, `K_selected`, `K_select_info`, `svd_target_r2` | K selection diagnostics |
| `B_scale`, `use_velocity`, `motion_field_used` | Motion field metadata |
| `H_constraint` | `"row_l2_norm_equals_1"` |
| `objective` | `"reconstruction_plus_sparse_compact"` |
| `postprocessing` | `"hard_threshold_only"` |

---

# V2: Clustering Updates

## New Entry Point: `getMotionPattern()`

The primary clustering function is now **`getMotionPattern()`**, which supersedes `getMotionRegionPattern()`:

```python
from wholistic_registration.utils.motion_correlation_pattern import getMotionPattern

patterns, kept_units, groups, labels, info = getMotionPattern(
    motion_episodes,
    unit_type="region",       # "region" (default) or "mode"
    min_strength=0.0,
    min_area=5,
    min_duration=1,
    min_iou=0.10,
    omega=1.0,                # weight for activation DTW distance
    mu=1.0,                   # weight for response field distance
    b_distance="l2",          # "l2" or "correlation"
    cluster_dist_thresh=0.8,
    linkage_method="complete",
    incompatible_dist=1e6,
    verbose=True,
)
```

### New Parameters

**`unit_type`** — controls what is clustered:
- `"region"` (default): collect `MotionRegion`s from all episodes (same as old `getMotionRegionPattern`). Requires running `split_episode_modes_to_regions` first.
- `"mode"`: cluster `MotionMode` objects directly, **skipping the spatial region-splitting step**. This is useful when modes are already spatially coherent and don't need further fragmentation.

**`b_distance`** — controls how response-field similarity is measured:
- `"l2"` (default): normalized L2 distance on the spatial overlap: $\|\mathbf{B}_1 - \text{sign} \cdot \mathbf{B}_2\| \;/\; (\|\mathbf{B}_1\| + \|\mathbf{B}_2\|)$.
- `"correlation"`: sign-insensitive Pearson correlation distance on the overlap: $1 - |\text{Pearson}_r(\mathbf{B}_1, \text{sign} \cdot \mathbf{B}_2)|$.

### Backward Compatibility

`getMotionRegionPattern()` is retained as a wrapper that delegates to `getMotionPattern(unit_type="region")`. Existing code using `getMotionRegionPattern` will continue to work without modification.

### `filter_regions_for_patterns` — Now Handles Both Types

The filtering function has been extended to work with both `MotionRegion` and `MotionMode` objects. When an attribute is missing on a `MotionMode` (e.g., `strength`, `area_effective`, `duration`, `mean_response_vector`), it is automatically computed from `response_strength` / `response_field` / `activation`.

### New Helper Functions

| Function | Description |
|---|---|
| `collect_regions_from_episodes(episodes)` | Collect all `MotionRegion`s from all episodes |
| `collect_modes_from_episodes(episodes)` | Collect all `MotionMode`s from all episodes |
| `collect_units_from_episodes(episodes, unit_type)` | Unified collector; dispatches to the above based on `unit_type` |
| `_unit_centroid(unit)` | Compute spatial centroid from `response_strength` (fallback if `center_xy` is missing) |
| `_response_field_correlation_on_overlap(r1, r2, sign2)` | Pearson correlation distance between response fields on spatial overlap |
