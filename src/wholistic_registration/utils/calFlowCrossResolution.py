import numpy as np

from . import calculate, cp, cupy_ndimage, interp
from .imresize import imresize


####################################################################################################
## The same as before
def correctMotionGrid(data_raw, coords_new):
    """
    Correct the motion using 3D interpolation for GPU arrays.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array), shape (H, W, D).
        coords_new (cupy.ndarray): New grid coordinates for interpolation (GPU array), shape (H, W, D, 3).

    Returns:
        data_corrected (cupy.ndarray): The corrected 3D data after interpolation (GPU array), shape (H, W, D).
    """
    # Extract dimensions from input data
    x, y, z = data_raw.shape  # data_raw shape: (H, W, D)

    # Ensure the data is on GPU and convert to float32 for precision
    data_raw = cp.asarray(data_raw, dtype=cp.float32)  # Convert to GPU array, shape: (H, W, D)
    coords_new = cp.asarray(coords_new)  # Convert to GPU array, shape: (H, W, D, 3)

    # Transpose coordinates from (H, W, D, 3) to (3, H, W, D) for interpolation function
    # This reorders dimensions to match the expected input format for interp3Grid
    coords_new = cp.transpose(coords_new, (3, 0, 1, 2))  # Shape: (3, H, W, D)

    # Perform 3D interpolation using the deformed coordinates
    # This warps the original data according to the motion field
    data1_tran = interp.interp3Grid(data_raw, coords_new, method="linear")  # Shape: (H, W, D)

    # Reshape the interpolated data back to original dimensions
    data_corrected = cp.reshape(data1_tran, (x, y, z))  # Shape: (H, W, D)

    return data_corrected


def getNeiDiff(phi_current, r):
    """
    Calculate the neighbor difference using a filter to enforce smoothness.

    Args:
        phi_current (cupy.ndarray): The 3D motion field data (GPU array), shape (H, W, D, 3).
        r (int): The radius of the filter (size = 2*r+1).

    Returns:
        neiDiff (cupy.ndarray): The filtered 3D data after applying the neighbor difference filter, shape (H, W, D, 3).
    """
    # Create the neighbor filter with size (2*r+1) x (2*r+1) x 1 x 1
    # This filter will be used to compute the difference between each point and its neighbors
    NeiFltr = cp.ones((r * 2 + 1, r * 2 + 1, 1, 1), dtype=cp.float32)  # Shape: (2*r+1, 2*r+1, 1, 1)

    # Normalize the filter by dividing by the number of neighbors (excluding center)
    # This ensures the filter sums to zero, making it a difference operator
    NeiFltr = NeiFltr / ((r * 2 + 1) ** 2 - 1)  # Normalize by number of neighbors minus center

    # Set the center element to -1 to create a difference filter
    # This makes the filter compute: (sum of neighbors) - center_value
    NeiFltr[r, r] = -1  # Center element becomes negative

    # Apply the filter to compute neighbor differences
    # This enforces smoothness by penalizing large differences between neighboring motion vectors
    neiDiff = calculate.imfilter(
        phi_current, NeiFltr, boundary="replicate", output="same", functionality="corr"
    )  # Shape: (H, W, D, 3)

    return neiDiff


def calError(It, penaltyRaw, smoothPenaltySum):
    """
    Calculate the error and penalty terms for the given 3D data.

    Args:
        It (cupy.ndarray): The temporal difference (GPU array), shape (H, W, D).
        penaltyRaw (cupy.ndarray): The 4D penalty raw values (GPU array), shape (H, W, D, 3).
        smoothPenaltySum (float): The smoothing penalty sum value.

    Returns:
        tuple: diffError (float), penaltyError (float)
    """
    # Get the shape of the temporal difference image
    x, y, z = It.shape  # It shape: (H, W, D)

    # Calculate the intensity difference error (mean squared error)
    # This measures how well the warped moving image matches the reference image
    diffError = cp.mean(It**2)  # Scalar value

    # Calculate the smoothness penalty error
    # Square the penalty values and sum across the 4th dimension (x,y,z motion components)
    penaltyCorrected = cp.sum(penaltyRaw**2, axis=3) * smoothPenaltySum  # Shape: (H, W, D)

    # Normalize the penalty error by the total number of voxels
    penaltyError = cp.sum(penaltyCorrected) / (x * y * z)  # Scalar value

    # Handle both CuPy and NumPy arrays by converting to CPU if needed
    if hasattr(diffError, "get"):
        return diffError.get(), penaltyError.get()  # Convert GPU arrays to CPU
    else:
        return float(diffError), float(penaltyError)  # Already CPU arrays


def getSpatialGradientInOrgGrid(data_raw, coords_new):
    """
    Calculate the spatial gradient on deformed coordinates using 3D interpolation.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array), shape (H, W, D).
        coords_new (cupy.ndarray): Deformed coordinates, shape (H, W, D, 3)
                                   where coords_new[...,0]=x, coords_new[...,1]=y, coords_new[...,2]=z.

    Returns:
        Ix (cupy.ndarray): Gradient along x-axis, shape (H, W, D).
        Iy (cupy.ndarray): Gradient along y-axis, shape (H, W, D).
        Iz (cupy.ndarray): Gradient along z-axis, shape (H, W, D).
    """
    step = 1.0  # Step size for finite differences
    x, y, z = data_raw.shape  # data_raw shape: (H, W, D)

    # Extract deformed coordinates for each dimension
    x_coords, y_coords, z_coords = (
        coords_new[..., 0],
        coords_new[..., 1],
        coords_new[..., 2],
    )  # Each shape: (H, W, D)

    # --- Compute gradient along x direction (Ix) ---
    # Perturb x-coordinate by adding and subtracting step
    x_coords_incre = cp.clip(x_coords + step, 0, x - 1)  # Shape: (H, W, D)
    x_coords_decre = cp.clip(x_coords - step, 0, x - 1)  # Shape: (H, W, D)

    # Interpolate at (x+step, y, z) and (x-step, y, z) to get intensity values
    data_incre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords_incre, y_coords, z_coords))
    )  # Shape: (H, W, D)
    data_decre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords_decre, y_coords, z_coords))
    )  # Shape: (H, W, D)

    # Compute x-gradient using finite differences
    Ix = (data_incre - data_decre) / (2 * step)  # Shape: (H, W, D)

    # --- Compute gradient along y direction (Iy) ---
    # Perturb y-coordinate by adding and subtracting step
    y_coords_incre = cp.clip(y_coords + step, 0, y - 1)  # Shape: (H, W, D)
    y_coords_decre = cp.clip(y_coords - step, 0, y - 1)  # Shape: (H, W, D)

    # Interpolate at (x, y+step, z) and (x, y-step, z) to get intensity values
    data_incre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords_incre, z_coords))
    )  # Shape: (H, W, D)
    data_decre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords_decre, z_coords))
    )  # Shape: (H, W, D)

    # Compute y-gradient using finite differences
    Iy = (data_incre - data_decre) / (2 * step)  # Shape: (H, W, D)

    # --- Compute gradient along z direction (Iz) ---
    # Perturb z-coordinate by adding and subtracting step
    z_coords_incre = cp.clip(z_coords + step, 0, z - 1)  # Shape: (H, W, D)
    z_coords_decre = cp.clip(z_coords - step, 0, z - 1)  # Shape: (H, W, D)

    # Interpolate at (x, y, z+step) and (x, y, z-step) to get intensity values
    data_incre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords, z_coords_incre))
    )  # Shape: (H, W, D)
    data_decre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords, z_coords_decre))
    )  # Shape: (H, W, D)

    # Compute z-gradient using finite differences
    Iz = (data_incre - data_decre) / (2 * step)  # Shape: (H, W, D)

    return Ix, Iy, Iz


def getFlow3_withPenalty6(
    Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum, verbose=False
):
    """
    Compute the flow with penalty and 3x3 matrix determinant using Lucas-Kanade method.

    Args:
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt (cupy.ndarray): The components for flow calculation, each shape (H, W, D).
        smoothPenaltySum (float): The smooth penalty sum.
        neiSum (cupy.ndarray): The neighbor sum, shape (H, W, D, 3).

    Returns:
        cupy.ndarray: The computed phi gradient flow, shape (H, W, D, 3).
    """
    # Add smoothness penalty to diagonal elements of the structure tensor
    # This regularizes the solution and prevents singular matrices
    Ixx += smoothPenaltySum  # Add penalty to x-x component
    Iyy += smoothPenaltySum  # Add penalty to y-y component
    Izz += smoothPenaltySum  # Add penalty to z-z component

    # Add neighbor sum to the temporal gradient terms
    # This incorporates the smoothness constraint into the optical flow equation
    Ixt += neiSum[:, :, :, 0]  # Add x-component of neighbor sum
    Iyt += neiSum[:, :, :, 1]  # Add y-component of neighbor sum
    Izt += neiSum[:, :, :, 2]  # Add z-component of neighbor sum

    # Calculate the determinant of the 3x3 structure tensor matrix
    # This is used to check if the matrix is invertible
    DET = calculate.getDet3(Ixx, Ixy, Ixz, Iyy, Iyz, Izz)  # Shape: (H, W, D)

    # Calculate the minors (2x2 determinants) for the adjugate matrix
    # These are used to compute the inverse of the structure tensor
    M11 = calculate.getDet2(Iyy, Iyz, Iyz, Izz)  # Minor for (1,1) element
    M12 = -calculate.getDet2(Ixy, Iyz, Ixz, Izz)  # Minor for (1,2) element (with sign)
    M13 = calculate.getDet2(Ixy, Iyy, Ixz, Iyz)  # Minor for (1,3) element
    M22 = calculate.getDet2(Ixx, Ixz, Ixz, Izz)  # Minor for (2,2) element
    M23 = -calculate.getDet2(Ixx, Ixy, Ixz, Iyz)  # Minor for (2,3) element (with sign)
    M33 = calculate.getDet2(Ixx, Ixy, Ixy, Iyy)  # Minor for (3,3) element

    # Compute the optical flow using the inverse of the structure tensor
    # This is the Lucas-Kanade solution: v = -A^(-1) * b
    Vx = (M11 * Ixt + M12 * Iyt + M13 * Izt) / DET  # x-component of motion
    Vy = (M12 * Ixt + M22 * Iyt + M23 * Izt) / DET  # y-component of motion
    Vz = (M13 * Ixt + M23 * Iyt + M33 * Izt) / DET  # z-component of motion

    # Stack the motion components into a single array
    phi_gradient = cp.stack((Vx, Vy, Vz), axis=-1)  # Shape: (H, W, D, 3)

    # Replace NaN values with 0 to handle singular cases
    # This prevents numerical issues when the determinant is very small

    num_nans = cp.isnan(phi_gradient)
    if cp.sum(num_nans) > 0:
        if verbose:
            print(f"number of nans: {cp.sum(num_nans)}")
        phi_gradient[cp.isnan(phi_gradient)] = 0
    return phi_gradient


def compute_new_grid(grid, r, motion_shape):
    """
    Normalize the original grid to let the coordinates of control points be integer.

    Args:
        grid (tuple): Original grid coordinates (x_coord, y_coord, z_coord), each shape (H, W, D).
        r (int): Filter radius.
        motion_shape (tuple): Shape of the motion field (H, W, D).

    Returns:
        cupy.ndarray: Normalized grid coordinates, shape (3, H, W, D).
    """
    x_coord, y_coord, z_coord = grid  # Each shape: (H, W, D)

    # Normalize x and y coordinates to control point grid
    # The factor (2*r+1) represents the spacing between control points
    x_new = (x_coord - r) / (2 * r + 1)  # Shape: (H, W, D)
    y_new = (y_coord - r) / (2 * r + 1)  # Shape: (H, W, D)

    # Clamp the normalized coordinates to valid range
    x_new = cp.minimum(cp.maximum(x_new, 0.0), motion_shape[0])  # Clamp to [0, motion_shape[0]]
    y_new = cp.minimum(cp.maximum(y_new, 0.0), motion_shape[1])  # Clamp to [0, motion_shape[1]]

    # Keep z coordinate unchanged (no normalization needed)
    z_new = z_coord  # Shape: (H, W, D)

    # Stack the normalized coordinates
    return cp.stack([x_new, y_new, z_new], axis=0)  # Shape: (3, H, W, D)


## The same as before
####################################################################################################


def _softmax_stable(x, axis=-1, eps=1e-8):
    """
    Numerically stable softmax.
    """
    x_max = cp.max(x, axis=axis, keepdims=True)
    ex = cp.exp(x - x_max)
    return ex / (cp.sum(ex, axis=axis, keepdims=True) + eps)


def _patch_grid_regularization(
    mu_patch,
    conf_patch,
    lam=1.0,
    num_iters=40,
    eps=1e-6,
):
    """
    Solve a simple confidence-weighted spatial regularization on the patch grid:

        min_z  sum_ij conf_ij * (z_ij - mu_ij)^2
             + lam * sum_(neighbors) (z_ij - z_kl)^2

    by Jacobi-style iterations.

    Parameters
    ----------
    mu_patch : cp.ndarray, shape (Ny, Nx)
        Local patch-wise soft depth estimate.
    conf_patch : cp.ndarray, shape (Ny, Nx)
        Confidence per patch, in [0, 1].
    lam : float
        Spatial regularization strength.
    num_iters : int
        Number of Jacobi iterations.
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    z_patch : cp.ndarray, shape (Ny, Nx)
        Regularized patch-wise depth field.
    """
    z = mu_patch.astype(cp.float32, copy=True)
    conf = conf_patch.astype(cp.float32, copy=False)

    for _ in range(num_iters):
        up = cp.pad(z[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = cp.pad(z[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = cp.pad(z[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = cp.pad(z[:, 1:], ((0, 0), (0, 1)), mode="edge")

        neighbor_sum = up + down + left + right

        z = (conf * mu_patch + lam * neighbor_sum) / (conf + 4.0 * lam + eps)

    return z


def _global_zncc_scores(mov_feat, ref_feat, use_hann_weight=True, eps=1e-6):
    H, W, K = mov_feat.shape
    _, _, Z = ref_feat.shape

    mov_feat = mov_feat.astype(cp.float32, copy=False)
    ref_feat = ref_feat.astype(cp.float32, copy=False)

    if use_hann_weight:
        weight = calculate.hann2d(H, W).astype(cp.float32)
    else:
        weight = cp.ones((H, W), dtype=cp.float32)

    weight = weight / (cp.sum(weight) + eps)

    scores = cp.zeros((K, Z), dtype=cp.float32)

    ref_mean = cp.sum(weight[:, :, None] * ref_feat, axis=(0, 1))  # (Z,)
    ref_centered = ref_feat - ref_mean[None, None, :]
    ref_var = cp.sum(weight[:, :, None] * ref_centered * ref_centered, axis=(0, 1))  # (Z,)

    for k in range(K):
        mov_k = mov_feat[:, :, k]

        mov_mean = cp.sum(weight * mov_k)
        mov_centered = mov_k - mov_mean
        mov_var = cp.sum(weight * mov_centered * mov_centered)

        numerator = cp.sum(
            weight[:, :, None] * mov_centered[:, :, None] * ref_centered,
            axis=(0, 1),
        )

        scores[k, :] = numerator / (cp.sqrt(mov_var * ref_var) + eps)

    scores = cp.clip(scores, -1.0, 1.0)
    return scores


def FindInitZ_stack_global_fixed_spacing(
    data_mov,
    data_ref,
    delta_ref_idx,
    use_gradient=True,
    use_hann_weight=True,
    direction=1,
    weight_eps=1e-6,
    return_debug=False,
):
    mov = calculate.to_3d(data_mov)
    ref = calculate.to_3d(data_ref)

    H, W, K = mov.shape
    Hr, Wr, Z = ref.shape

    if (Hr, Wr) != (H, W):
        raise ValueError(f"XY size mismatch: data_mov has {(H, W)}, data_ref has {(Hr, Wr)}")

    if K < 1:
        raise ValueError("Moving input must have at least one slice.")

    if Z < 1:
        raise ValueError("Reference input must have at least one slice.")

    # ------------------------------------------------------------
    # 1. feature transform
    # ------------------------------------------------------------
    if use_gradient:
        mov_feat = cp.stack(
            [calculate.grad_mag_2d(mov[:, :, k]) for k in range(K)],
            axis=2,
        )
        ref_feat = cp.stack(
            [calculate.grad_mag_2d(ref[:, :, z]) for z in range(Z)],
            axis=2,
        )
    else:
        mov_feat = mov.astype(cp.float32, copy=False)
        ref_feat = ref.astype(cp.float32, copy=False)

    scores = _global_zncc_scores(
        mov_feat=mov_feat,
        ref_feat=ref_feat,
        use_hann_weight=use_hann_weight,
        eps=weight_eps,
    )  # (K, Z)

    offsets = cp.arange(K, dtype=cp.float32) * float(delta_ref_idx) * float(direction)

    min_offset = cp.min(offsets)
    max_offset = cp.max(offsets)

    z0_min = int(cp.ceil(-min_offset).item())
    z0_max = int(cp.floor((Z - 1) - max_offset).item())

    if z0_max < z0_min:
        raise ValueError(
            f"No valid z0 exists. Z={Z}, K={K}, delta_ref_idx={delta_ref_idx}. "
            f"Need reference z range >= {(K - 1) * abs(delta_ref_idx) + 1}."
        )

    best_score = -cp.inf
    best_z_init = None
    best_z0 = None

    for z0 in range(z0_min, z0_max + 1):
        z_init = z0 + offsets

        z_idx = cp.rint(z_init).astype(cp.int32)

        score = cp.sum(scores[cp.arange(K), z_idx])

        if score > best_score:
            best_score = score
            best_z_init = z_init.copy()
            best_z0 = z0

    z_init_np = cp.asnumpy(best_z_init).astype(np.uint16)

    if return_debug:
        debug = {
            "scores": cp.asnumpy(scores),
            "best_score": float(best_score.get()),
            "best_z0": int(best_z0),
            "delta_ref_idx": float(delta_ref_idx),
            "direction": int(direction),
        }
        return z_init_np, debug

    return z_init_np


def generate_continuous_H_gpu(stack, zRatio):
    """
    stack: cp.ndarray, shape (X,Y,Z)
    """
    stack_gpu = cp.asarray(stack)

    def H(coords_phys):
        coords = cp.asarray(coords_phys)
        coords_idx = coords.copy()
        coords_idx[..., 2] = coords[..., 2] / zRatio
        shape = coords_idx.shape
        coords_flat = coords_idx.reshape(-1, 3).T
        values = cupy_ndimage.map_coordinates(stack_gpu, coords_flat, order=3, mode="nearest")
        return values.reshape(*shape[:-1])

    return H

_PROJECT_TO_PLANES_NANMAX_SRC = r'''
extern "C" __global__
void project_to_planes_nanmax_kernel(
    const float* values,          // (Nvox,), shifted values, should be non-negative
    const float* coords,          // (Nvox, 3), xyz
    const float* target_z,        // (Nplanes,)
    float* out,                   // (Nplanes, Xout, Yout), flattened
    const long long Nvox,
    const int Nplanes,
    const int Xref,
    const int Yref,
    const int Xout,
    const int Yout,
    const float z_window,
    const int downsample_xy,
    const int use_nearest_xy,
    const int xy_extra_radius
) {
    long long tid = blockDim.x * blockIdx.x + threadIdx.x;
    long long total = Nvox * (long long)Nplanes;

    if (tid >= total) return;

    long long voxel_id = tid / Nplanes;
    int plane_id = (int)(tid - voxel_id * Nplanes);

    float val = values[voxel_id];
    if (!isfinite(val)) return;

    float x = coords[voxel_id * 3 + 0];
    float y = coords[voxel_id * 3 + 1];
    float z = coords[voxel_id * 3 + 2];

    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;

    float z0 = target_z[plane_id];

    // z slab selection:
    // the moving surface point contributes to this target plane
    // only if it lies within z0 +/- z_window.
    if (fabsf(z - z0) > z_window) return;

    int x_start, x_end;
    int y_start, y_end;

    if (use_nearest_xy) {
        // Old behavior: one point -> one output pixel.
        int xi = (int)floorf(x + 0.5f);
        int yi = (int)floorf(y + 0.5f);

        x_start = xi;
        x_end   = xi;
        y_start = yi;
        y_end   = yi;
    } else {
        // Subpixel footprint splatting:
        // A point at x=10.1 contributes to pixels around floor(x)=10 and ceil(x)=11.
        // xy_extra_radius expands this footprint.
        int xf = (int)floorf(x);
        int yf = (int)floorf(y);

        x_start = xf - xy_extra_radius;
        x_end   = xf + 1 + xy_extra_radius;
        y_start = yf - xy_extra_radius;
        y_end   = yf + 1 + xy_extra_radius;
    }

    for (int xi = x_start; xi <= x_end; ++xi) {
        if (xi < 0 || xi >= Xref) continue;

        int xo = xi / downsample_xy;
        if (xo < 0 || xo >= Xout) continue;

        for (int yi = y_start; yi <= y_end; ++yi) {
            if (yi < 0 || yi >= Yref) continue;

            int yo = yi / downsample_xy;
            if (yo < 0 || yo >= Yout) continue;

            long long out_idx =
                ((long long)plane_id * Xout * Yout)
                + ((long long)xo * Yout)
                + yo;

            // Safe because values have been shifted to be non-negative.
            atomicMax((int*)(&out[out_idx]), __float_as_int(val));
        }
    }
}
'''

_PROJECT_TO_PLANES_WEIGHTED_AVG_SRC = r'''
extern "C" __global__
void project_to_planes_weighted_avg_kernel(
    const float* values,          // (Nvox,)
    const float* coords,          // (Nvox, 3), xyz
    const float* target_z,        // (Nplanes,)
    float* sum_val,               // (Nplanes, Xout, Yout)
    float* sum_w,                 // (Nplanes, Xout, Yout)
    const long long Nvox,
    const int Nplanes,
    const int Xref,
    const int Yref,
    const int Xout,
    const int Yout,
    const float z_window,
    const float z_sigma,
    const int downsample_xy,
    const int z_weight_mode
) {
    long long tid = blockDim.x * blockIdx.x + threadIdx.x;
    long long total = Nvox * (long long)Nplanes;

    if (tid >= total) return;

    long long voxel_id = tid / Nplanes;
    int plane_id = (int)(tid - voxel_id * Nplanes);

    float val = values[voxel_id];
    if (!isfinite(val)) return;

    float x = coords[voxel_id * 3 + 0];
    float y = coords[voxel_id * 3 + 1];
    float z = coords[voxel_id * 3 + 2];

    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;

    if (x < 0.0f || x > (float)(Xref - 1) || y < 0.0f || y > (float)(Yref - 1)) {
        return;
    }

    float z0 = target_z[plane_id];
    float dz = fabsf(z - z0);

    if (dz > z_window) return;

    float wz = 1.0f;

    if (z_weight_mode == 0) {
        // hard slab
        wz = 1.0f;
    } else if (z_weight_mode == 1) {
        // triangular slab weight
        wz = 1.0f - dz / (z_window + 1e-6f);
        if (wz < 0.0f) wz = 0.0f;
    } else {
        // gaussian slab weight
        float sigma = z_sigma;
        if (sigma <= 0.0f) sigma = z_window * 0.5f;
        wz = expf(-0.5f * dz * dz / (sigma * sigma + 1e-6f));
    }

    if (wz <= 0.0f || !isfinite(wz)) return;

    // Important:
    // do bilinear splatting directly in output-resolution coordinates.
    float xo_f = x / (float)downsample_xy;
    float yo_f = y / (float)downsample_xy;

    int xo0 = (int)floorf(xo_f);
    int yo0 = (int)floorf(yo_f);

    float dx = xo_f - (float)xo0;
    float dy = yo_f - (float)yo0;

    for (int ax = 0; ax <= 1; ++ax) {
        int xo = xo0 + ax;
        if (xo < 0 || xo >= Xout) continue;

        float wx = (ax == 0) ? (1.0f - dx) : dx;

        for (int ay = 0; ay <= 1; ++ay) {
            int yo = yo0 + ay;
            if (yo < 0 || yo >= Yout) continue;

            float wy = (ay == 0) ? (1.0f - dy) : dy;

            float w = wx * wy * wz;
            if (w <= 0.0f || !isfinite(w)) continue;

            long long out_idx =
                ((long long)plane_id * Xout * Yout)
                + ((long long)xo * Yout)
                + yo;

            atomicAdd(&sum_val[out_idx], val * w);
            atomicAdd(&sum_w[out_idx], w);
        }
    }
}
'''

# CUDA kernels are compiled on first use, not at import time: `cp` is the
# shim from utils/__init__, which is plain numpy on machines without CuPy/CUDA
# — and numpy has no RawKernel, so constructing these at module scope made the
# whole module unimportable on CPU (taking every pure-CPU function with it).
_KERNEL_CACHE = {}


def _get_raw_kernel(name, source):
    """Compile (once) and return a CUDA RawKernel; GPU-only by construction."""
    if name not in _KERNEL_CACHE:
        if not hasattr(cp, "RawKernel"):
            raise RuntimeError(
                f"CUDA kernel '{name}' requires CuPy with a CUDA device; "
                "this process is running on the NumPy fallback."
            )
        _KERNEL_CACHE[name] = cp.RawKernel(source, name)
    return _KERNEL_CACHE[name]


def _nanmax_kernel():
    return _get_raw_kernel("project_to_planes_nanmax_kernel", _PROJECT_TO_PLANES_NANMAX_SRC)


def _weighted_avg_kernel():
    return _get_raw_kernel(
        "project_to_planes_weighted_avg_kernel", _PROJECT_TO_PLANES_WEIGHTED_AVG_SRC
    )

def project_coords_to_fixed_planes_weighted_gpu(
    coords_ref_xyk_xyz,
    ref_volume,
    target_z_planes,
    values_xyk=None,
    ref_volume_order="xyz",
    z_window=1.5,
    z_sigma=None,
    z_weight_mode="gaussian",
    downsample_xy=1,
    fill_value=0.0,
    return_numpy=False,
    output_order="zyx",
    eps=1e-6,
):
    """
    Smooth projection onto fixed reference-space z planes.

    Compared with max splatting, this function uses:
        - continuous reference sampling;
        - z-slab weighting;
        - XY bilinear splatting;
        - weighted average accumulation.

    This is recommended for generating stable fixed-view movies.

    Parameters
    ----------
    coords_ref_xyk_xyz:
        shape (Xmov, Ymov, K, 3)
        phase_new, mapping moving grid to reference coordinates.

    ref_volume:
        reference volume.

        ref_volume_order="xyz": shape (Xref, Yref, Zref)
        ref_volume_order="zyx": shape (Zref, Yref, Xref)

    target_z_planes:
        fixed reference z planes to observe.

    values_xyk:
        If None:
            values are sampled from ref_volume at coords_ref_xyk_xyz.
            This matches your current reference-derived output logic.

        If provided:
            directly project values_xyk into reference planes.

    z_window:
        Only samples with abs(z_ref - target_z) <= z_window are used.

    z_sigma:
        Gaussian z weighting sigma.
        If None, use z_window / 2.

    z_weight_mode:
        "hard", "triangular", or "gaussian".

    downsample_xy:
        Output XY downsample factor.

    fill_value:
        Fill value for pixels with no accumulated weight.

    output_order:
        "zyx": return (Nplanes, Yout, Xout)
        "zxy": return (Nplanes, Xout, Yout)
    """

    coords = cp.asarray(coords_ref_xyk_xyz, dtype=cp.float32)

    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(
            f"coords_ref_xyk_xyz should have shape (X,Y,K,3), got {coords.shape}"
        )

    ref = cp.asarray(ref_volume, dtype=cp.float32)

    if ref.ndim != 3:
        raise ValueError(f"ref_volume should be 3D, got {ref.shape}")

    if ref_volume_order == "xyz":
        Xref, Yref, Zref = ref.shape
        ref_xyz = ref
    elif ref_volume_order == "zyx":
        Zref, Yref, Xref = ref.shape
        ref_xyz = ref.transpose(2, 1, 0)
    else:
        raise ValueError("ref_volume_order should be 'xyz' or 'zyx'.")

    downsample_xy = int(downsample_xy)
    if downsample_xy < 1:
        raise ValueError("downsample_xy should be >= 1.")

    Xout = int((Xref + downsample_xy - 1) // downsample_xy)
    Yout = int((Yref + downsample_xy - 1) // downsample_xy)

    target_z = cp.atleast_1d(cp.asarray(target_z_planes, dtype=cp.float32))
    Nplanes = int(target_z.size)

    if values_xyk is None:
        H_ref = generate_continuous_H_gpu(ref_xyz, zRatio=1)
        values = apply_H_to_matrix_gpu(coords, H_ref).astype(cp.float32, copy=False)
    else:
        values = cp.asarray(values_xyk, dtype=cp.float32)
        if values.shape != coords.shape[:-1]:
            raise ValueError(
                f"values_xyk shape {values.shape} does not match coords shape {coords.shape[:-1]}"
            )

    coords_flat = cp.ascontiguousarray(coords.reshape(-1, 3))
    values_flat = cp.ascontiguousarray(values.reshape(-1))

    Nvox = int(values_flat.size)

    sum_val = cp.zeros((Nplanes, Xout, Yout), dtype=cp.float32)
    sum_w = cp.zeros((Nplanes, Xout, Yout), dtype=cp.float32)

    if z_sigma is None:
        z_sigma = float(z_window) * 0.5

    if z_weight_mode == "hard":
        z_weight_mode_id = 0
    elif z_weight_mode == "triangular":
        z_weight_mode_id = 1
    elif z_weight_mode == "gaussian":
        z_weight_mode_id = 2
    else:
        raise ValueError("z_weight_mode should be 'hard', 'triangular', or 'gaussian'.")

    threads = 256
    blocks = int((Nvox * Nplanes + threads - 1) // threads)

    _weighted_avg_kernel()(
        (blocks,),
        (threads,),
        (
            values_flat,
            coords_flat,
            target_z,
            sum_val,
            sum_w,
            np.int64(Nvox),
            np.int32(Nplanes),
            np.int32(Xref),
            np.int32(Yref),
            np.int32(Xout),
            np.int32(Yout),
            np.float32(z_window),
            np.float32(z_sigma),
            np.int32(downsample_xy),
            np.int32(z_weight_mode_id),
        ),
    )

    out = cp.empty_like(sum_val)

    valid = sum_w > eps
    out = cp.where(
        valid,
        sum_val / cp.maximum(sum_w, eps),
        cp.asarray(fill_value, dtype=cp.float32),
    )

    if output_order == "zyx":
        out = out.transpose(0, 2, 1)
        weight_out = sum_w.transpose(0, 2, 1)
    elif output_order == "zxy":
        weight_out = sum_w
    else:
        raise ValueError("output_order should be 'zyx' or 'zxy'.")

    if return_numpy:
        return cp.asnumpy(out), cp.asnumpy(weight_out)

    return out, weight_out

def project_coords_to_fixed_planes_gpu(
    coords_ref_xyk_xyz,
    ref_volume=None,
    target_z_planes=None,
    values_xyk=None,
    ref_volume_order="xyz",
    z_window=1.5,
    downsample_xy=1,
    fill_value=0.0,
    return_numpy=False,
    output_order="zyx",
    xy_splat_mode="subpixel_footprint",
    xy_extra_radius=0,
):
    """
    Project signal onto fixed reference-space z planes using z-slab selection
    and XY subpixel footprint max splatting.

    Main intended mode for your current task:
        values_xyk=None

    In this mode:
        1. sample ref_volume at coords_ref_xyk_xyz;
        2. select samples whose z coordinate lies within target_z +/- z_window;
        3. splat them to fixed reference XY planes with max projection.

    Parameters
    ----------
    coords_ref_xyk_xyz:
        ndarray or cp.ndarray, shape (Xmov, Ymov, K, 3)

        coords_ref_xyk_xyz[..., 0] = x_ref
        coords_ref_xyk_xyz[..., 1] = y_ref
        coords_ref_xyk_xyz[..., 2] = z_ref

        Usually this is phase_new from getMotion_v2.

    ref_volume:
        Reference volume.

        If ref_volume_order == "xyz":
            shape should be (Xref, Yref, Zref)

        If ref_volume_order == "zyx":
            shape should be (Zref, Yref, Xref)

    target_z_planes:
        Scalar or array-like.

        Fixed reference z planes that you want to observe.

    values_xyk:
        Optional, shape (Xmov, Ymov, K).

        If None:
            values are sampled from ref_volume at coords_ref_xyk_xyz.

        If provided:
            values_xyk is directly splatted into reference planes.
            This is useful if you want to project moving-frame signal back
            into reference space.

    z_window:
        Half thickness of z slab.

        A point contributes to target plane z0 if:
            abs(z_ref - z0) <= z_window

    downsample_xy:
        Output XY downsample factor.

        This is applied during splatting:
            xo = xi // downsample_xy
            yo = yi // downsample_xy

    fill_value:
        Value for output pixels with no projected samples.

    xy_splat_mode:
        "nearest":
            old behavior. One sample writes to one rounded output pixel.

        "subpixel_footprint":
            recommended. One sample writes to floor/ceil XY footprint.
            This reduces holes caused by forward mapping.

    xy_extra_radius:
        Extra radius around the floor/ceil footprint.

        If xy_splat_mode="subpixel_footprint":

            xy_extra_radius=0:
                sample affects 2x2 pixels:
                    floor(x):floor(x)+1,
                    floor(y):floor(y)+1

            xy_extra_radius=1:
                sample affects 4x4 pixels:
                    floor(x)-1 : floor(x)+2,
                    floor(y)-1 : floor(y)+2

        Start with 0. If holes remain, try 1.

    output_order:
        "zyx":
            return shape (Nplanes, Yout, Xout), image-friendly.

        "zxy":
            return shape (Nplanes, Xout, Yout), internal coordinate order.

    return_numpy:
        If True, return numpy array.
        If False, return cupy array.
    """

    if target_z_planes is None:
        raise ValueError("target_z_planes must be provided.")

    coords = cp.asarray(coords_ref_xyk_xyz, dtype=cp.float32)

    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(
            f"coords_ref_xyk_xyz should have shape (X,Y,K,3), got {coords.shape}"
        )

    if ref_volume is None:
        raise ValueError("ref_volume must be provided to infer reference shape.")

    ref = cp.asarray(ref_volume, dtype=cp.float32)

    if ref.ndim != 3:
        raise ValueError(f"ref_volume should be 3D, got {ref.shape}")

    if ref_volume_order == "xyz":
        Xref, Yref, Zref = ref.shape
        ref_xyz = ref
    elif ref_volume_order == "zyx":
        Zref, Yref, Xref = ref.shape
        ref_xyz = ref.transpose(2, 1, 0)
    else:
        raise ValueError("ref_volume_order should be 'xyz' or 'zyx'.")

    downsample_xy = int(downsample_xy)
    if downsample_xy < 1:
        raise ValueError("downsample_xy should be >= 1.")

    Xout = int((Xref + downsample_xy - 1) // downsample_xy)
    Yout = int((Yref + downsample_xy - 1) // downsample_xy)

    target_z = cp.atleast_1d(cp.asarray(target_z_planes, dtype=cp.float32))
    Nplanes = int(target_z.size)

    # ------------------------------------------------------------
    # 1. Prepare values to project.
    # ------------------------------------------------------------
    if values_xyk is None:
        # Your preferred mode:
        # sample reference volume at continuous reference coordinates.
        H_ref = generate_continuous_H_gpu(ref_xyz, zRatio=1)
        values = apply_H_to_matrix_gpu(coords, H_ref).astype(cp.float32, copy=False)
    else:
        values = cp.asarray(values_xyk, dtype=cp.float32)
        if values.shape != coords.shape[:-1]:
            raise ValueError(
                f"values_xyk shape {values.shape} does not match coords shape {coords.shape[:-1]}"
            )

    # ------------------------------------------------------------
    # 2. Shift values to non-negative range for atomicMax on float bits.
    #
    # Do NOT clamp by cp.maximum(values, 0), because your adjusted reference
    # can have negative background values, e.g. percentile 0.1 = -188.
    # ------------------------------------------------------------
    finite_values = values[cp.isfinite(values)]

    if finite_values.size > 0:
        vmin = cp.min(finite_values)
        value_shift = cp.maximum(-vmin + cp.asarray(1.0, dtype=cp.float32),
                                 cp.asarray(0.0, dtype=cp.float32))
    else:
        value_shift = cp.asarray(0.0, dtype=cp.float32)

    values_for_kernel = values + value_shift

    coords_flat = cp.ascontiguousarray(coords.reshape(-1, 3))
    values_flat = cp.ascontiguousarray(values_for_kernel.reshape(-1))

    Nvox = int(values_flat.size)

    # Use -inf as empty marker.
    out = cp.full((Nplanes, Xout, Yout), -cp.inf, dtype=cp.float32)

    if xy_splat_mode == "nearest":
        use_nearest_xy = 1
        xy_extra_radius_kernel = 0
    elif xy_splat_mode == "subpixel_footprint":
        use_nearest_xy = 0
        xy_extra_radius_kernel = int(xy_extra_radius)
        if xy_extra_radius_kernel < 0:
            raise ValueError("xy_extra_radius should be >= 0.")
    else:
        raise ValueError(
            "xy_splat_mode should be 'nearest' or 'subpixel_footprint'."
        )

    # ------------------------------------------------------------
    # 3. GPU splatting.
    # ------------------------------------------------------------
    threads = 256
    blocks = int((Nvox * Nplanes + threads - 1) // threads)

    _nanmax_kernel()(
        (blocks,),
        (threads,),
        (
            values_flat,
            coords_flat,
            target_z,
            out,
            np.int64(Nvox),
            np.int32(Nplanes),
            np.int32(Xref),
            np.int32(Yref),
            np.int32(Xout),
            np.int32(Yout),
            np.float32(z_window),
            np.int32(downsample_xy),
            np.int32(use_nearest_xy),
            np.int32(xy_extra_radius_kernel),
        ),
    )

    # ------------------------------------------------------------
    # 4. Shift values back.
    # ------------------------------------------------------------
    valid_out = cp.isfinite(out)
    out = cp.where(valid_out, out - value_shift, out)

    # ------------------------------------------------------------
    # 5. Fill empty pixels.
    # ------------------------------------------------------------
    if fill_value is not None:
        out = cp.where(
            cp.isfinite(out),
            out,
            cp.asarray(fill_value, dtype=cp.float32)
        )

    # ------------------------------------------------------------
    # 6. Output order.
    # ------------------------------------------------------------
    if output_order == "zyx":
        # internal: (Nplanes, Xout, Yout)
        # image:    (Nplanes, Yout, Xout)
        out = out.transpose(0, 2, 1)
    elif output_order == "zxy":
        pass
    else:
        raise ValueError("output_order should be 'zyx' or 'zxy'.")

    if return_numpy:
        return cp.asnumpy(out)

    return out


def apply_H_to_matrix_gpu(A, H):
    """
    A: cp.ndarray, shape (X,Y,Z,3)
    H: interpolate function
    """
    coords = cp.asarray(A)
    shape = coords.shape
    coords_flat = coords.reshape(-1, 3)
    R = H(coords_flat)
    return R.reshape(shape[:-1])


def correctMotion(data_raw, motion_field):
    """
    Apply motion correction to raw data using the computed motion field.

    Args:
        data_raw (numpy.ndarray): The raw 3D data, shape (H, W, D).
        motion_field (numpy.ndarray): The computed motion field, shape (H, W, D, 3).

    Returns:
        data_tran (numpy.ndarray): The motion-corrected data, shape (H, W, D).
    """
    # Generate coordinate grid for the data
    grid = np.meshgrid(
        *[np.arange(n, dtype=np.float32) for n in data_raw.shape],  # Create coordinate arrays
        indexing="ij",  # Use matrix indexing
        sparse=False,  # Return full grid
    )  # Returns tuple of 3 arrays, each shape: (H, W, D)

    # Compute corrected coordinates using motion field
    coords_new = interp.correctGrid(motion_field, grid)  # Shape: (H, W, D, 3)

    # Apply motion correction using 3D interpolation
    data_tran = correctMotionGrid(data_raw, coords_new)  # Shape: (H, W, D)

    # Convert to CPU if needed
    if hasattr(data_tran, "get"):
        data_tran = cp.asnumpy(data_tran)  # Convert GPU array to CPU
    else:
        data_tran = np.asarray(data_tran)  # Already CPU array

    return data_tran


# 1. Wrong-region detection helpers
def get_local_error_on_control_points(It, xG_grid, yG_grid, zG_grid, kernelsize=11, metric="mse"):
    """
    Compute local residual error on control points from a dense residual map It.

    Parameters
    ----------
    It : cp.ndarray, shape (x, y, z)
        Residual on moving grid.
    xG_grid, yG_grid, zG_grid : cp.ndarray
        Control-point meshgrid.
    kernelsize : int
        Patch size in xy plane.
    metric : str
        'mse' or 'mae'.

    Returns
    -------
    err_cp : cp.ndarray
        Local averaged error sampled on control points.
    err_dense : cp.ndarray
        Dense pixel-wise error map before averaging.
    """
    if metric.lower() == "mse":
        err_dense = It**2
    elif metric.lower() == "mae":
        err_dense = cp.abs(It)
    else:
        raise ValueError("metric must be 'mse' or 'mae'")

    kernel = cp.ones((kernelsize, kernelsize, 1), dtype=cp.float32) / (kernelsize**2)
    err_avg = calculate.imfilter(err_dense, kernel, "replicate", "same", "corr")
    err_cp = err_avg[xG_grid, yG_grid, zG_grid]
    return err_cp, err_dense


def detect_significant_mad(values, threshold=3.0):
    """
    MAD-based outlier detection on a flattened array.
    Returns indices into values.ravel().
    """
    arr = cp.asarray(values).ravel()
    if arr.size == 0:
        return cp.array([], dtype=cp.int64)

    med = cp.median(arr)
    mad = cp.median(cp.abs(arr - med))
    denom = 1.4826 * mad

    if denom == 0:
        if cp.any(arr != med):
            idx = cp.where(arr != med)[0]
        else:
            idx = cp.array([], dtype=cp.int64)
    else:
        z = cp.abs(arr - med) / denom
        idx = cp.where(z > threshold)[0]

    return idx.astype(cp.int64)


def _normalize_vec_field(vx, vy, vz, eps=1e-6):
    norm = cp.sqrt(vx**2 + vy**2 + vz**2) + eps
    return vx / norm, vy / norm, vz / norm


def get_wrong_regions(
    err2d, r, xG, yG, x_threshold, y_threshold, mad_threshold=3.0, min_component_size=2
):
    """
    Detect connected high-error regions on one z slice of control-point error map.

    Parameters
    ----------
    err2d : cp.ndarray, shape (nx_cp, ny_cp)
        Error map on control points for one z slice.
    r : int
        Control-point radius.
    xG, yG : cp.ndarray
        1D control-point coordinates in image grid.
    x_threshold, y_threshold : int
        Image size bounds.
    mad_threshold : float
        MAD threshold for abnormal control points.
    min_component_size : int
        Minimum connected component size to keep.

    Returns
    -------
    region_coor : list[cp.ndarray]
        Each element = [min_y, max_y, min_x, max_x] in moving-image coordinates.
    connected_components : list[cp.ndarray]
        Connected abnormal control-point indices.
    """
    region_coor = []
    connected_components = []

    if err2d.size == 0 or cp.all(err2d == 0):
        return region_coor, connected_components

    rows, cols = err2d.shape

    if rows * cols > 144 and rows > 2 and cols > 2:
        inner_rows = cp.arange(1, rows - 1)
        inner_cols = cp.arange(1, cols - 1)
        R, C = cp.meshgrid(inner_rows, inner_cols, indexing="ij")
        flat_idx = detect_significant_mad(err2d[R, C], threshold=mad_threshold)
        sig_pts = cp.stack([R.ravel()[flat_idx], C.ravel()[flat_idx]], axis=1)
    else:
        flat_idx = detect_significant_mad(err2d, threshold=mad_threshold)
        if flat_idx.size == 0:
            return region_coor, connected_components
        sig_pts = cp.stack(cp.unravel_index(flat_idx, err2d.shape), axis=1)

    npts = int(sig_pts.shape[0])
    if npts == 0:
        return region_coor, connected_components

    # Build 4-neighbor adjacency
    adj = [[] for _ in range(npts)]
    for i in range(npts):
        for j in range(i + 1, npts):
            p1, p2 = sig_pts[i], sig_pts[j]
            if abs(int(p1[0] - p2[0])) + abs(int(p1[1] - p2[1])) == 1:
                adj[i].append(j)
                adj[j].append(i)

    visited = np.zeros(npts, dtype=bool)

    for i in range(npts):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            cur = stack.pop()
            if visited[cur]:
                continue
            visited[cur] = True
            comp.append(sig_pts[cur])
            for nb in adj[cur]:
                if not visited[nb]:
                    stack.append(nb)

        if len(comp) >= min_component_size:
            connected_components.append(cp.asarray(comp))

    dx = int(cp.ceil(1.5 * r))

    for comp in connected_components:
        min_row, max_row = int(comp[:, 0].min()), int(comp[:, 0].max())
        min_col, max_col = int(comp[:, 1].min()), int(comp[:, 1].max())

        cp_minx = int(xG[min_row])
        cp_miny = int(yG[min_col])
        cp_maxx = int(xG[max_row])
        cp_maxy = int(yG[max_col])

        min_x = max(0, cp_minx - dx)
        min_y = max(0, cp_miny - dx)
        max_x = min(cp_maxx + dx, x_threshold - 1)
        max_y = min(cp_maxy + dx, y_threshold - 1)

        region_coor.append(cp.asarray([min_y, max_y, min_x, max_x], dtype=cp.int32))

    return region_coor, connected_components


def _make_ball(radius_xy, z_ratio):
    """
    Small anisotropic 3D structuring element, isotropic in physical space.

    Convention (module-wide, see `zRatio_hr`): z_ratio is *physical units per
    z index*, i.e. dz_phys = dz_idx * z_ratio, with one xy index = one physical
    unit. So an index offset (dx, dy, dz) is at physical distance
        d^2 = dx^2 + dy^2 + (dz * z_ratio)^2
    and the ball of physical radius radius_xy reaches radius_xy / z_ratio
    z indices. z_ratio > 1 (z coarser than xy) therefore *shrinks* the element
    in z; z_ratio < 1 (as at coarse pyramid layers, where zRatio_hr =
    zRatio_HR / 2**layer) *grows* it.
    """
    radius_xy = int(max(1, radius_xy))
    z_ratio = max(float(z_ratio), 1e-6)

    # ceil, not round/floor: the index box must contain the whole physical ball
    # even when radius_xy / z_ratio lands just under an integer numerically.
    radius_z = int(max(1, np.ceil(radius_xy / z_ratio)))

    xs = cp.arange(-radius_xy, radius_xy + 1, dtype=cp.float32)
    ys = cp.arange(-radius_xy, radius_xy + 1, dtype=cp.float32)
    zs = cp.arange(-radius_z, radius_z + 1, dtype=cp.float32)

    XX, YY, ZZ = cp.meshgrid(xs, ys, zs, indexing="ij")
    dist2 = XX**2 + YY**2 + (ZZ * z_ratio) ** 2
    return (dist2 <= radius_xy**2 + 1e-6).astype(cp.bool_)


def _connected_grow(seed_mask, allowed_mask, structure, max_steps):
    """
    Grow from seed inside allowed_mask for at most max_steps iterations.
    """
    region = seed_mask.copy()
    for _ in range(int(max_steps)):
        grown = cupy_ndimage.binary_dilation(region, structure=structure)
        grown = grown & allowed_mask
        if bool(cp.all(grown == region).item()):
            break
        region = grown
    return region


def build_reference_trap_mask_from_bad_moving_fast_roi(
    bad_mask,
    phase_new,
    data_ref_layer,
    z_ratio_ref,
    expand_radius_xy=2,
    sigma_grad=1.0,
    intensity_k=2.5,
):
    """
    ROI-first trap-mask construction.

    Compared with build_reference_trap_mask_from_bad_moving:
    - Do not create full-volume seed_mask_ref.
    - Do not run binary_closing on the full reference volume.
    - Compute seed bbox directly from seed coordinates.
    - Create seed_crop only inside local ROI.
    - Run closing / smoothing / distance transform / growth only inside ROI.
    """
    x_ref, y_ref, z_ref = data_ref_layer.shape
    trap_mask_ref = cp.zeros((x_ref, y_ref, z_ref), dtype=cp.bool_)

    if not bool(cp.any(bad_mask).item()):
        return trap_mask_ref

    # ------------------------------------------------------------
    # 1) moving bad mask -> reference seed coordinates
    # ------------------------------------------------------------
    seed_coords = phase_new[bad_mask]
    if seed_coords.shape[0] == 0:
        return trap_mask_ref

    seed_x = cp.rint(seed_coords[:, 0]).astype(cp.int32)
    seed_y = cp.rint(seed_coords[:, 1]).astype(cp.int32)
    seed_z = cp.rint(seed_coords[:, 2]).astype(cp.int32)

    keep = (
        (seed_x >= 0)
        & (seed_x < x_ref)
        & (seed_y >= 0)
        & (seed_y < y_ref)
        & (seed_z >= 0)
        & (seed_z < z_ref)
    )

    seed_x = seed_x[keep]
    seed_y = seed_y[keep]
    seed_z = seed_z[keep]

    if seed_x.size == 0:
        return trap_mask_ref

    # ------------------------------------------------------------
    # 2) compute ROI directly from seed coordinates
    # ------------------------------------------------------------
    xmin = int(seed_x.min().item())
    xmax = int(seed_x.max().item())
    ymin = int(seed_y.min().item())
    ymax = int(seed_y.max().item())
    zmin = int(seed_z.min().item())
    zmax = int(seed_z.max().item())

    margin_xy = int(max(1, expand_radius_xy + 2))
    # z_ratio_ref (= zRatio_hr) is physical units per z index, so the same
    # physical margin is (expand_radius_xy + 2) / z_ratio_ref indices in z.
    margin_z = int(max(1, np.ceil((expand_radius_xy + 2) / max(float(z_ratio_ref), 1e-6))))

    x0 = max(0, xmin - margin_xy)
    x1 = min(x_ref, xmax + margin_xy + 1)
    y0 = max(0, ymin - margin_xy)
    y1 = min(y_ref, ymax + margin_xy + 1)
    z0 = max(0, zmin - margin_z)
    z1 = min(z_ref, zmax + margin_z + 1)

    # ------------------------------------------------------------
    # 3) create seed mask only inside ROI
    # ------------------------------------------------------------
    seed_crop = cp.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=cp.bool_)

    seed_x_local = seed_x - x0
    seed_y_local = seed_y - y0
    seed_z_local = seed_z - z0

    seed_crop[seed_x_local, seed_y_local, seed_z_local] = True

    structure_small = _make_ball(1, z_ratio_ref)
    seed_crop = cupy_ndimage.binary_closing(seed_crop, structure=structure_small)

    if not bool(cp.any(seed_crop).item()):
        return trap_mask_ref

    # ------------------------------------------------------------
    # 4) crop reference only after ROI is known
    # ------------------------------------------------------------
    ref_crop = data_ref_layer[x0:x1, y0:y1, z0:z1].astype(cp.float32)

    # ------------------------------------------------------------
    # 5) smooth reference inside ROI
    # ------------------------------------------------------------
    if sigma_grad > 0:
        ref_sm = cupy_ndimage.gaussian_filter(ref_crop, sigma=(sigma_grad, sigma_grad, sigma_grad))
    else:
        ref_sm = ref_crop

    # ------------------------------------------------------------
    # 6) robust seed intensity statistics
    # ------------------------------------------------------------
    seed_vals = ref_sm[seed_crop]
    if seed_vals.size == 0:
        return trap_mask_ref

    I_med = float(cp.median(seed_vals).item())
    I_mad = float(cp.median(cp.abs(seed_vals - I_med)).item())

    tol = max(intensity_k * 1.4826 * I_mad, 1e-3)

    # ------------------------------------------------------------
    # 7) distance-to-seed constraint inside ROI
    # ------------------------------------------------------------
    near_structure = _make_ball(
        radius_xy=max(1, int(round(expand_radius_xy))),
        z_ratio=z_ratio_ref,
    )

    near_mask = cupy_ndimage.binary_dilation(
        seed_crop,
        structure=near_structure,
        iterations=1,
    )
    # ------------------------------------------------------------
    # 8) intensity similarity constraint
    # ------------------------------------------------------------
    intensity_mask = cp.abs(ref_sm - I_med) <= tol

    allowed_mask = (near_mask & intensity_mask) | seed_crop

    # ------------------------------------------------------------
    # 9) connected growth from seed inside ROI
    # ------------------------------------------------------------
    grow_structure = _make_ball(1, z_ratio_ref)
    region = _connected_grow(
        seed_mask=seed_crop,
        allowed_mask=allowed_mask,
        structure=grow_structure,
        max_steps=expand_radius_xy,
    )

    region = region | seed_crop

    # ------------------------------------------------------------
    # 10) paste local trap mask back to full reference space
    # ------------------------------------------------------------
    trap_mask_ref[x0:x1, y0:y1, z0:z1] = region

    return trap_mask_ref


def build_bad_region_mask_from_cp_error(
    err_cp,
    r,
    xG,
    yG,
    x_size,
    y_size,
    mad_threshold=3.0,
    min_component_size=2,
    exclude_border_if_large=True,
    large_cp_threshold=144,
    eps=1e-12,
):
    """
    Fast GPU version of build_bad_region_mask_from_cp_error.

    Parameters
    ----------
    err_cp : cp.ndarray, shape (nx_cp, ny_cp, z)
        Error on control points.

    r : int or float
        Control-point radius.

    xG, yG : cp.ndarray or array-like
        1D control-point coordinates in dense moving-image grid.
        xG corresponds to err_cp axis 0.
        yG corresponds to err_cp axis 1.

    x_size, y_size : int
        Dense moving-image size.

    mad_threshold : float
        MAD threshold for abnormal control points.

    min_component_size : int
        Minimum connected-component size on control-point grid.

    Returns
    -------
    bad_mask : cp.ndarray, shape (x_size, y_size, z_size), bool
        Dense moving-grid mask.

    num_regions_total : int
        Number of kept connected components.
    """
    if not isinstance(err_cp, cp.ndarray):
        err_cp = cp.asarray(err_cp)

    xG = cp.asarray(xG).astype(cp.int32)
    yG = cp.asarray(yG).astype(cp.int32)

    nx, ny, z_size = err_cp.shape

    if err_cp.size == 0:
        return cp.zeros((x_size, y_size, z_size), dtype=cp.bool_), 0

    # ------------------------------------------------------------
    # 1. Build significant control-point mask on GPU
    #    Same idea as original code:
    #    if control-point grid is large, ignore border control points.
    # ------------------------------------------------------------
    sig_mask = cp.zeros((nx, ny, z_size), dtype=cp.bool_)

    use_inner = exclude_border_if_large and nx * ny > large_cp_threshold and nx > 2 and ny > 2

    if use_inner:
        err_use = err_cp[1:-1, 1:-1, :]
        flat = err_use.reshape(-1, z_size)

        med = cp.median(flat, axis=0)
        mad = cp.median(cp.abs(flat - med[None, :]), axis=0)

        robust_z = cp.abs(flat - med[None, :]) / (1.4826 * mad[None, :] + eps)

        sig_flat = robust_z > mad_threshold
        sig_flat[:, mad < eps] = False

        sig_mask[1:-1, 1:-1, :] = sig_flat.reshape(nx - 2, ny - 2, z_size)

    else:
        flat = err_cp.reshape(-1, z_size)

        med = cp.median(flat, axis=0)
        mad = cp.median(cp.abs(flat - med[None, :]), axis=0)

        robust_z = cp.abs(flat - med[None, :]) / (1.4826 * mad[None, :] + eps)

        sig_flat = robust_z > mad_threshold
        sig_flat[:, mad < eps] = False

        sig_mask = sig_flat.reshape(nx, ny, z_size)

    if not bool(cp.any(sig_mask).item()):
        return cp.zeros((x_size, y_size, z_size), dtype=cp.bool_), 0

    # ------------------------------------------------------------
    # 2. Connected component labeling on GPU.
    #
    # Important:
    # structure only connects in x-y plane, not across z.
    # This preserves your old per-z connected-component behavior.
    # ------------------------------------------------------------
    structure = cp.zeros((3, 3, 3), dtype=cp.bool_)
    structure[:, :, 1] = cp.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=cp.bool_,
    )

    labels, nlab = cupy_ndimage.label(sig_mask, structure=structure)
    nlab = int(nlab)

    if nlab == 0:
        return cp.zeros((x_size, y_size, z_size), dtype=cp.bool_), 0

    # ------------------------------------------------------------
    # 3. Filter small components.
    # ------------------------------------------------------------
    label_flat = labels.ravel()

    sizes = cp.bincount(label_flat, minlength=nlab + 1)
    keep_label = sizes >= min_component_size
    keep_label[0] = False

    kept_labels = cp.nonzero(keep_label)[0]
    num_regions_total = int(kept_labels.size)

    if num_regions_total == 0:
        return cp.zeros((x_size, y_size, z_size), dtype=cp.bool_), 0

    # ------------------------------------------------------------
    # 4. Compute component bbox on GPU.
    # ------------------------------------------------------------
    rr, cc, zz = cp.nonzero(labels > 0)
    labs = labels[rr, cc, zz].astype(cp.int32)

    valid = keep_label[labs]
    rr = rr[valid].astype(cp.int32)
    cc = cc[valid].astype(cp.int32)
    zz = zz[valid].astype(cp.int32)
    labs = labs[valid]

    inf = cp.asarray(10**9, dtype=cp.int32)

    min_r = cp.full(nlab + 1, inf, dtype=cp.int32)
    max_r = cp.full(nlab + 1, -1, dtype=cp.int32)
    min_c = cp.full(nlab + 1, inf, dtype=cp.int32)
    max_c = cp.full(nlab + 1, -1, dtype=cp.int32)
    min_z = cp.full(nlab + 1, inf, dtype=cp.int32)
    max_z = cp.full(nlab + 1, -1, dtype=cp.int32)

    cp.minimum.at(min_r, labs, rr)
    cp.maximum.at(max_r, labs, rr)
    cp.minimum.at(min_c, labs, cc)
    cp.maximum.at(max_c, labs, cc)
    cp.minimum.at(min_z, labs, zz)
    cp.maximum.at(max_z, labs, zz)

    labs_keep = kept_labels.astype(cp.int32)

    min_r = min_r[labs_keep]
    max_r = max_r[labs_keep]
    min_c = min_c[labs_keep]
    max_c = max_c[labs_keep]
    min_z = min_z[labs_keep]
    max_z = max_z[labs_keep]

    # Because we used no z-connectivity, each component should live in one z slice.
    # But keep min_z/max_z for safety.
    dx = int(cp.ceil(cp.asarray(1.5 * r)).item())

    min_x = cp.maximum(0, xG[min_r] - dx)
    max_x = cp.minimum(x_size - 1, xG[max_r] + dx)
    min_y = cp.maximum(0, yG[min_c] - dx)
    max_y = cp.minimum(y_size - 1, yG[max_c] + dx)

    min_z = cp.maximum(0, min_z)
    max_z = cp.minimum(z_size - 1, max_z)

    # ------------------------------------------------------------
    # 5. Dense mask construction using 3D difference array.
    #
    # This avoids Python loop like:
    #     bad_mask[min_x:max_x+1, min_y:max_y+1, z] = True
    #
    # Since components do not connect across z, usually min_z == max_z.
    # But this implementation supports z intervals too.
    # ------------------------------------------------------------
    diff = cp.zeros((x_size + 1, y_size + 1, z_size + 1), dtype=cp.int32)

    x0 = min_x.astype(cp.int32)
    x1 = (max_x + 1).astype(cp.int32)
    y0 = min_y.astype(cp.int32)
    y1 = (max_y + 1).astype(cp.int32)
    z0 = min_z.astype(cp.int32)
    z1 = (max_z + 1).astype(cp.int32)

    cp.add.at(diff, (x0, y0, z0), 1)
    cp.add.at(diff, (x1, y0, z0), -1)
    cp.add.at(diff, (x0, y1, z0), -1)
    cp.add.at(diff, (x0, y0, z1), -1)

    cp.add.at(diff, (x1, y1, z0), 1)
    cp.add.at(diff, (x1, y0, z1), 1)
    cp.add.at(diff, (x0, y1, z1), 1)
    cp.add.at(diff, (x1, y1, z1), -1)

    bad_mask = cp.cumsum(
        cp.cumsum(
            cp.cumsum(diff, axis=0),
            axis=1,
        ),
        axis=2,
    )

    bad_mask = bad_mask[:x_size, :y_size, :z_size] > 0

    return bad_mask, num_regions_total


def get_quantile_threshold(arr, q=0.95):
    """
    Robust quantile threshold on CuPy array.
    """
    flat = cp.asarray(arr).ravel()
    if flat.size == 0:
        return cp.asarray(0, dtype=cp.float32)
    k = int(cp.clip(cp.floor((flat.size - 1) * q), 0, flat.size - 1))
    part = cp.partition(flat, k)
    return part[k]


def build_bad_cp_mask_from_cp_error(
    err_cp,
    mad_threshold=3.0,
    min_component_size=2,
    exclude_border_if_large=True,
    large_cp_threshold=144,
    eps=1e-12,
):
    """
    Detect abnormal control points from err_cp.

    Parameters
    ----------
    err_cp : cp.ndarray, shape (nx_cp, ny_cp, z)
        Local residual error sampled on control points.

    Returns
    -------
    bad_cp_mask : cp.ndarray, bool, shape (nx_cp, ny_cp, z)
        True means this control point is considered wrong.

    num_regions_total : int
        Number of connected abnormal components after size filtering.

    Notes
    -----
    - Connectivity is only in xy plane, not across z.
    - This keeps different moving slices independent.
    """
    if not isinstance(err_cp, cp.ndarray):
        err_cp = cp.asarray(err_cp)

    err_cp = err_cp.astype(cp.float32, copy=False)
    nx, ny, z_size = err_cp.shape

    if err_cp.size == 0:
        return cp.zeros_like(err_cp, dtype=cp.bool_), 0

    sig_mask = cp.zeros((nx, ny, z_size), dtype=cp.bool_)

    use_inner = exclude_border_if_large and nx * ny > large_cp_threshold and nx > 2 and ny > 2

    if use_inner:
        err_use = err_cp[1:-1, 1:-1, :]
        flat = err_use.reshape(-1, z_size)

        med = cp.median(flat, axis=0)
        mad = cp.median(cp.abs(flat - med[None, :]), axis=0)

        robust_z = cp.abs(flat - med[None, :]) / (1.4826 * mad[None, :] + eps)
        sig_flat = robust_z > mad_threshold

        # Important: preserve old MAD=0 behavior.
        # If MAD is zero, non-median values should still be considered abnormal.
        zero_mad = mad < eps
        if bool(cp.any(zero_mad).item()):
            sig_flat[:, zero_mad] = flat[:, zero_mad] != med[None, zero_mad]

        sig_mask[1:-1, 1:-1, :] = sig_flat.reshape(nx - 2, ny - 2, z_size)

    else:
        flat = err_cp.reshape(-1, z_size)

        med = cp.median(flat, axis=0)
        mad = cp.median(cp.abs(flat - med[None, :]), axis=0)

        robust_z = cp.abs(flat - med[None, :]) / (1.4826 * mad[None, :] + eps)
        sig_flat = robust_z > mad_threshold

        zero_mad = mad < eps
        if bool(cp.any(zero_mad).item()):
            sig_flat[:, zero_mad] = flat[:, zero_mad] != med[None, zero_mad]

        sig_mask = sig_flat.reshape(nx, ny, z_size)

    if not bool(cp.any(sig_mask).item()):
        return cp.zeros_like(sig_mask, dtype=cp.bool_), 0

    # 4-neighbor connectivity in xy only, no connection across z.
    structure = cp.zeros((3, 3, 3), dtype=cp.bool_)
    structure[:, :, 1] = cp.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=cp.bool_,
    )

    labels, nlab = cupy_ndimage.label(sig_mask, structure=structure)
    nlab = int(nlab)

    if nlab == 0:
        return cp.zeros_like(sig_mask, dtype=cp.bool_), 0

    sizes = cp.bincount(labels.ravel(), minlength=nlab + 1)
    keep_label = sizes >= int(min_component_size)
    keep_label[0] = False

    bad_cp_mask = keep_label[labels]
    num_regions_total = int(cp.sum(keep_label).item())

    return bad_cp_mask.astype(cp.bool_), num_regions_total


# 2. Cross-resolution registration helpers
def compose_phase_from_motion(phase_current, motion_current, zRatio, zRatio_hr):
    """
    Compose phase_current and motion_current into the final mapping phase_new.
    motion_current is defined in moving-grid coordinates.
    phase_new is defined in reference-grid coordinates.
    """
    phase_update = motion_current.copy()
    ####Update 2026/4/22 needn't to adjust the
    # phase_update[..., 2] = phase_update[..., 2] / zRatio_hr
    phase_new = phase_current + phase_update
    return phase_new


def resample_exclude_mask(mask, output_shape, threshold=0.5):
    """
    Resize exclusion mask.

    Parameters
    ----------
    mask : array
        Original exclusion mask, where:
        1 / True = exclude
        0 / False = keep
    output_shape : tuple
        Target shape.
    threshold : float
        Threshold after interpolation.

    Returns
    -------
    mask_out : cp.ndarray, bool
        True = exclude, False = keep
    """
    mask_rs = imresize(cp.asarray(mask, dtype=cp.float32), output_shape=output_shape)
    return mask_rs > threshold


def initialize_motion_for_layer(
    option, layer, layer_num, SZ, x, y, z, pyramid="anisotropic", prev_motion=None
):
    """
    Initialize motion field for current layer.
    """
    if layer == layer_num:
        if "motion" in option and option["motion"] is not None:
            motion_init = cp.asarray(option["motion"], dtype=cp.float32)
            motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
            motion_current[..., 0] = imresize(motion_init[..., 0], output_shape=(x, y, z)) / (
                SZ[0] / x
            )
            motion_current[..., 1] = imresize(motion_init[..., 1], output_shape=(x, y, z)) / (
                SZ[1] / y
            )
            motion_current[..., 2] = imresize(motion_init[..., 2], output_shape=(x, y, z)) / (
                SZ[2] / z
            )
        else:
            motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
    else:
        motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
        motion_current[..., 0] = (
            imresize(prev_motion[..., 0], output_shape=(x, y, z), method="bilinear") * 2
        )
        motion_current[..., 1] = (
            imresize(prev_motion[..., 1], output_shape=(x, y, z), method="bilinear") * 2
        )
        motion_current[..., 2] = imresize(
            prev_motion[..., 2], output_shape=(x, y, z), method="bilinear"
        )
        motion_current = cp.asarray(motion_current, dtype=cp.float32)

    return motion_current


def initialize_phase_for_layer(option, SZ, layer, x, y, z, zRatio, zRatio_hr):
    """
    Initialize phase field for current layer.
    """
    if "phase" in option and option["phase"] is not None:
        phase_init = cp.asarray(option["phase"], dtype=cp.float32)
        phase_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
        phase_current[..., 0] = imresize(phase_init[..., 0], output_shape=(x, y, z)) / (SZ[0] / x)
        phase_current[..., 1] = imresize(phase_init[..., 1], output_shape=(x, y, z)) / (SZ[1] / y)
        phase_current[..., 2] = imresize(phase_init[..., 2], output_shape=(x, y, z)) / (SZ[2] / z)
    else:
        X, Y, Z = cp.indices((x, y, z))
        phase_current = cp.stack([X, Y, Z * zRatio / zRatio_hr], axis=-1).astype(cp.float32)

    return phase_current


def make_control_point_grid(x, y, z, r):
    """
    Build control-point grid for LK update.
    """
    xG = cp.arange(r, x - 1, step=2 * r + 1)
    yG = cp.arange(r, y - 1, step=2 * r + 1)
    zG = cp.arange(0, z)
    xG_grid, yG_grid, zG_grid = cp.meshgrid(xG, yG, zG, indexing="ij")
    return xG, yG, zG, xG_grid, yG_grid, zG_grid


def compute_valid_mask_on_moving_grid(phase_new, H_mask_ref_layer, mask_mov_layer):
    """
    Parameters
    ----------
    phase_new : mapping from moving grid to reference continuous coordinates
    H_mask_ref_layer : continuous interpolator of reference exclusion mask
    mask_mov_layer : bool mask on moving grid, True means exclude

    Returns
    -------
    valid_mask : bool
        True means this moving-grid voxel is valid for data term.
    """
    # reference exclusion mask sampled onto moving grid
    if H_mask_ref_layer is None:
        ref_excluded = cp.zeros(mask_mov_layer.shape, dtype=cp.bool_)
    else:
        ref_excluded = apply_H_to_matrix_gpu(phase_new, H_mask_ref_layer) > 0.5

    mov_excluded = mask_mov_layer > 0
    valid_mask = (~mov_excluded) & (~ref_excluded)
    return valid_mask


# 3. Core optimizer for one layer
def optimize_layer_cross_resolution(
    data_mov_layer,
    data_ref_layer,
    H_ref_layer,
    H_mask_ref_layer,
    mask_mov_layer,
    phase_current,
    motion_init,
    xG_grid,
    yG_grid,
    zG_grid,
    r,
    iterNum,
    movRange,
    smoothPenalty,
    smoothPenaltySum,
    zRatio,
    zRatio_hr,
    tol=1e-3,
    verbose=False,
    layer=None,
):
    """
    Optimize one pyramid layer in the cross-resolution setting.

    Returns
    -------
    result : dict
        {
            "motion": ...,
            "phase_new": ...,
            "data_ref_sampled": ...,
            "residual": ...,
            "error_init": ...,
            "error_last": ...,
            "current_error": ...,
            "diff_error": ...,
            "penalty_error": ...,
            "valid_mask": ...
        }
    """
    x, y, z = data_mov_layer.shape
    motion_current = motion_init.copy()
    oldError = cp.inf * cp.ones(3, dtype=cp.float32)

    error_init = None
    error_last = None
    phase_new = None
    data_ref_sampled = None
    valid_mask = None
    diffError = None
    penaltyError = None
    currentError = None

    AverageFilter = cp.ones((2 * r + 1, 2 * r + 1, 1), dtype=cp.float32)
    Xcp = xG_grid.astype(cp.float32)
    Ycp = yG_grid.astype(cp.float32)
    Zcp = zG_grid.astype(cp.float32)

    phase_identity_cp = cp.stack([Xcp, Ycp, Zcp], axis=-1)
    for it in range(iterNum):
        old_motion = motion_current.copy()

        phase_new = compose_phase_from_motion(phase_current, motion_current, zRatio, zRatio_hr)
        data_ref_sampled = apply_H_to_matrix_gpu(phase_new, H_ref_layer)
        valid_mask = compute_valid_mask_on_moving_grid(phase_new, H_mask_ref_layer, mask_mov_layer)

        residual = data_mov_layer - data_ref_sampled
        residual = calculate.imfilter(
            residual, cp.ones((3, 3, 1), dtype=cp.float32) / 9, "replicate", "same", "corr"
        )
        residual[~valid_mask] = 0

        if it == 0:
            error_init = residual**2

        error_last = residual**2

        # change the motion current to phase new 20260607 15:54
        neiDiff = getNeiDiff(motion_current[xG_grid, yG_grid, zG_grid, :], 1)
        neiDiff[:,:,2] *= zRatio_hr
        # phase_cp = phase_new[xG_grid,yG_grid,zG_grid,:]
        # phase_residual_cp = phase_cp - phase_identity_cp
        # neiDiff = getNeiDiff(phase_residual_cp, 1)
        neiSum = smoothPenaltySum * neiDiff

        diffError, penaltyError = calError(residual, neiDiff, smoothPenaltySum)
        currentError = diffError + penaltyError

        if verbose:
            print(
                f"[layer={layer}] iter={it:02d} "
                f"error={float(currentError):.4f} "
                f"diff={float(diffError):.4f} "
                f"penalty={float(penaltyError):.4f}"
            )

        # stopping
        if it == iterNum - 1:
            break
        if cp.sum(oldError <= currentError) > 1:
            if verbose:
                print(f"[layer={layer}] stop: error increased repeatedly")
            break
        if cp.abs(oldError[-1] - currentError) < tol:
            if verbose:
                print(f"[layer={layer}] stop: error change below tol")
            break

        oldError[:-1] = oldError[1:]
        oldError[-1] = currentError

        # spatial gradient on reference, evaluated at phase_new
        Ix, Iy, Iz = getSpatialGradientInOrgGrid(data_ref_layer, phase_new)
        Ix[~valid_mask] = 0
        Iy[~valid_mask] = 0
        Iz[~valid_mask] = 0
        Iz /= zRatio_hr

        Ixx = calculate.imfilter(Ix**2, AverageFilter, "replicate", "same", "corr")
        Ixy = calculate.imfilter(Ix * Iy, AverageFilter, "replicate", "same", "corr")
        Ixz = calculate.imfilter(Ix * Iz, AverageFilter, "replicate", "same", "corr")
        Iyy = calculate.imfilter(Iy**2, AverageFilter, "replicate", "same", "corr")
        Iyz = calculate.imfilter(Iy * Iz, AverageFilter, "replicate", "same", "corr")
        Izz = calculate.imfilter(Iz**2, AverageFilter, "replicate", "same", "corr")
        Ixt = calculate.imfilter(Ix * residual, AverageFilter, "replicate", "same", "corr")
        Iyt = calculate.imfilter(Iy * residual, AverageFilter, "replicate", "same", "corr")
        Izt = calculate.imfilter(Iz * residual, AverageFilter, "replicate", "same", "corr")

        Ixx = Ixx[xG_grid, yG_grid, zG_grid]
        Ixy = Ixy[xG_grid, yG_grid, zG_grid]
        Ixz = Ixz[xG_grid, yG_grid, zG_grid]
        Iyy = Iyy[xG_grid, yG_grid, zG_grid]
        Iyz = Iyz[xG_grid, yG_grid, zG_grid]
        Izz = Izz[xG_grid, yG_grid, zG_grid]
        Ixt = Ixt[xG_grid, yG_grid, zG_grid]
        Iyt = Iyt[xG_grid, yG_grid, zG_grid]
        Izt = Izt[xG_grid, yG_grid, zG_grid]

        motion_update = getFlow3_withPenalty6(
            Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum, verbose
        )

        # step clipping
        motion_norm = cp.sqrt(cp.sum(motion_update**2, axis=3))
        motion_norm = cp.maximum(motion_norm / movRange, 1.0)
        motion_update = motion_update / motion_norm[..., cp.newaxis]

        ## update 2026/4/22 we need to use the motion in coordinates space instead of phisical space
        motion_update[..., 2] = motion_update[..., 2] / zRatio_hr

        motion_current_CP = motion_current[xG_grid, yG_grid, zG_grid, :] + motion_update
        grid_dense = cp.meshgrid(
            *[cp.arange(n, dtype=cp.float32) for n in data_mov_layer.shape],
            indexing="ij",
            sparse=False,
        )
        coords_new = compute_new_grid(grid_dense, r, motion_current_CP.shape)

        for d in range(3):
            temp_phi = cp.asarray(motion_current_CP[..., d])
            motion_current[..., d] = interp.interp3Grid(temp_phi, coords_new).reshape(x, y, z)

        max_diff_motion = cp.max(cp.abs(motion_current - old_motion))
        if max_diff_motion < tol:
            if verbose:
                print(f"[layer={layer}] stop: motion update below tol")
            break
        # visualization.visualize_2d_image(data_ref_sampled.get(),title = "[iter {iter}]: mapping image")

    # final recompute
    phase_new = compose_phase_from_motion(phase_current, motion_current, zRatio, zRatio_hr)
    data_ref_sampled = apply_H_to_matrix_gpu(phase_new, H_ref_layer)
    valid_mask = compute_valid_mask_on_moving_grid(phase_new, H_mask_ref_layer, mask_mov_layer)

    ##################################################################################################
    # visualization.visualize_2d_image(valid_mask.get(),autocontrast=False,title = f"[layer:{layer}] valid mask")
    ##################################################################################################

    residual = data_mov_layer - data_ref_sampled
    residual = calculate.imfilter(
        residual, cp.ones((3, 3, 1), dtype=cp.float32) / 9, "replicate", "same", "corr"
    )
    residual[~valid_mask] = 0
    error_last = residual**2

    neiDiff = getNeiDiff(motion_current[xG_grid, yG_grid, zG_grid, :], 1)
    diffError, penaltyError = calError(residual, neiDiff, smoothPenaltySum)
    currentError = diffError + penaltyError

    return {
        "motion": motion_current,
        "phase_new": phase_new,
        "data_ref_sampled": data_ref_sampled,
        "residual": residual,
        "error_init": error_init if error_init is not None else error_last.copy(),
        "error_last": error_last,
        "current_error": currentError,
        "diff_error": diffError,
        "penalty_error": penaltyError,
        "valid_mask": valid_mask,
    }


# =========================================================
# 4. Wrong-region correction for one layer
# =========================================================


def correct_wrong_regions_one_layer(
    data_mov_layer,
    data_ref_layer,
    H_ref_layer,
    H_mask_ref_layer,
    mask_mov_layer,
    mask_ref_layer,
    phase_current,
    motion_init_layer,
    xG,
    yG,
    xG_grid,
    yG_grid,
    zG_grid,
    r,
    iterNum,
    movRange,
    smoothPenalty,
    smoothPenaltySum,
    zRatio,
    zRatio_hr,
    error_metric="mse",
    mad_threshold=3.0,
    min_component_size=2,
    bad_region_exclude_mode="direct",
    verbose=False,
    layer=None,
    # Params for constructing trap mask in reference space
    expand_radius_xy=2.0,
    sigma_grad=1.0,
    intensity_k=2.0,
    quantile_threshold_q=0.8,
    motion_inpaint_enable=True,
    motion_inpaint_iters=30,
    motion_inpaint_neighbor_mode="4",
):
    """
    Run normal optimization, detect bad regions, mask them out, rerun, and accept
    correction only if final objective improves.

    bad_region_exclude_mode:
        - "direct": exclude all detected bad regions
        - "highresidual": exclude only the top residual fraction inside bad regions
    """
    # -----------------------------------------------------
    # Pass 1: normal optimization
    # -----------------------------------------------------
    res0 = optimize_layer_cross_resolution(
        data_mov_layer=data_mov_layer,
        data_ref_layer=data_ref_layer,
        H_ref_layer=H_ref_layer,
        H_mask_ref_layer=H_mask_ref_layer,
        mask_mov_layer=mask_mov_layer,
        phase_current=phase_current,
        motion_init=motion_init_layer,
        xG_grid=xG_grid,
        yG_grid=yG_grid,
        zG_grid=zG_grid,
        r=r,
        iterNum=iterNum,
        movRange=movRange,
        smoothPenalty=smoothPenalty,
        smoothPenaltySum=smoothPenaltySum,
        zRatio=zRatio,
        zRatio_hr=zRatio_hr,
        verbose=verbose,
        layer=layer,
    )
    motion0 = res0["motion"]
    current_error0 = res0["current_error"]
    residual0 = res0["residual"]
    valid_mask0 = res0["valid_mask"]

    # -----------------------------------------------------
    # Detect bad regions on moving grid from residual
    # -----------------------------------------------------
    err_cp, _ = get_local_error_on_control_points(
        residual0, xG_grid, yG_grid, zG_grid, kernelsize=2 * r + 1, metric=error_metric
    )

    bad_mask, num_regions = build_bad_region_mask_from_cp_error(
        err_cp,
        r,
        xG,
        yG,
        x_size=data_mov_layer.shape[0],
        y_size=data_mov_layer.shape[1],
        mad_threshold=mad_threshold,
        min_component_size=min_component_size,
    )
    # visualization.visualize_2d_image(bad_mask.get(),autocontrast=False)
    if verbose:
        print(f"[layer={layer}] detected wrong regions: {num_regions}")

    if num_regions == 0:
        return res0

    # Optionally refine bad-mask with dense residual threshold inside detected regions
    if bad_region_exclude_mode == "highresidual":
        error_init0 = res0["error_init"]
        error_last0 = res0["error_last"]
        # visualization.visualize_2d_image(error_init0.get(),autocontrast=True,title=f"[layer={layer}] dense error before thresholding")
        # visualization.visualize_2d_image(error_last0.get(),autocontrast=True,title=f"[layer={layer}] dense error before thresholding")

        dense_err = error_init0 - error_last0
        dense_err = calculate.imfilter(
            dense_err, cp.ones((3, 3, 1), dtype=cp.float32) / 9, "replicate", "same", "corr"
        )
        dense_err[~bad_mask] = 0
        # visualization.visualize_2d_image(dense_err.get(),autocontrast=True,title=f"[layer={layer}] dense error before thresholding")
        th = (
            get_quantile_threshold(dense_err[bad_mask], q=quantile_threshold_q)
            if cp.any(bad_mask)
            else 0
        )
        bad_mask = bad_mask & (dense_err >= th)

    # visualization.visualize_2d_image(bad_mask.get(),autocontrast=False)
    # Only exclude regions that were originally valid in moving mask
    bad_mask = bad_mask & valid_mask0
    # visualization.visualize_2d_image(bad_mask.get(),autocontrast=False,title=f"[layer={layer}] bad region mask on moving grid")
    trap_mask_ref = build_reference_trap_mask_from_bad_moving_fast_roi(
        bad_mask=bad_mask,
        phase_new=res0["phase_new"],
        data_ref_layer=data_ref_layer,
        z_ratio_ref=zRatio_hr,
        expand_radius_xy=expand_radius_xy / (2.0**layer),
        sigma_grad=sigma_grad,
        intensity_k=intensity_k,
    )
    # visualization.visualize_3d_image(trap_mask_ref.get(),title=f"[layer={layer}] trap mask in reference space")
    corrected_mask_ref = mask_ref_layer | trap_mask_ref

    H_mask_ref_layer_corrected = generate_continuous_H_gpu(
        corrected_mask_ref.astype(cp.float32), zRatio=1
    )

    n_valid_old = int(cp.sum(valid_mask0).item())
    valid_mask1 = compute_valid_mask_on_moving_grid(
        res0["phase_new"], H_mask_ref_layer_corrected, mask_mov_layer
    )
    n_valid_new = int(cp.sum(valid_mask1).item())
    valid_ratio = n_valid_new / max(n_valid_old, 1)
    if verbose:
        print(f"[layer={layer}] mask kept ratio after correction: {valid_ratio:.3f}")

    if valid_ratio < 0.4:
        if verbose:
            print(f"[layer={layer}] skip correction: corrected mask too small")
        return res0

    # -----------------------------------------------------
    # Pass 2: robust rerun using corrected mask
    # -----------------------------------------------------
    res1 = optimize_layer_cross_resolution(
        data_mov_layer=data_mov_layer,
        data_ref_layer=data_ref_layer,
        H_ref_layer=H_ref_layer,
        H_mask_ref_layer=H_mask_ref_layer_corrected,
        mask_mov_layer=mask_mov_layer,
        phase_current=phase_current,
        motion_init=motion_init_layer,
        xG_grid=xG_grid,
        yG_grid=yG_grid,
        zG_grid=zG_grid,
        r=r,
        iterNum=iterNum,
        movRange=movRange,
        smoothPenalty=smoothPenalty,
        smoothPenaltySum=smoothPenaltySum,
        zRatio=zRatio,
        zRatio_hr=zRatio_hr,
        verbose=verbose,
        layer=layer,
    )

    # -----------------------------------------------------
    # Pass 3: refinement with original moving mask
    # -----------------------------------------------------
    res2 = optimize_layer_cross_resolution(
        data_mov_layer=data_mov_layer,
        data_ref_layer=data_ref_layer,
        H_ref_layer=H_ref_layer,
        H_mask_ref_layer=H_mask_ref_layer,
        mask_mov_layer=mask_mov_layer,
        phase_current=phase_current,
        motion_init=res1["motion"],
        xG_grid=xG_grid,
        yG_grid=yG_grid,
        zG_grid=zG_grid,
        r=r,
        iterNum=iterNum,
        movRange=movRange,
        smoothPenalty=smoothPenalty,
        smoothPenaltySum=smoothPenaltySum,
        zRatio=zRatio,
        zRatio_hr=zRatio_hr,
        verbose=verbose,
        layer=layer,
    )
    # visualization.visualize_2d_image(res0["data_ref_sampled"].get(),title="first processed")
    # visualization.visualize_2d_image(res1["data_ref_sampled"].get(),title="after mask")
    # visualization.visualize_2d_image(res2["data_ref_sampled"].get(),title="removed mask")
    current_error2 = res2["current_error"]

    if verbose:
        print(
            f"[layer={layer}] compare original vs corrected: "
            f"{float(current_error0):.4f} -> {float(current_error2):.4f}"
        )

    if current_error2 < current_error0:
        if verbose:
            print(f"[layer={layer}] accept corrected result")
        return res2
    else:
        if verbose:
            print(f"[layer={layer}] keep original result")
        return res0


# =========================================================
# 5. Main public function
# =========================================================


def getMotion_v2(data_mov, data_ref, option, verbose=False):
    """
    Cross-resolution registration with wrong-region correction.

    Parameters
    ----------
    data_mov : np.ndarray or cp.ndarray
        Moving image, shape (H, W, D_mov)
    data_ref : np.ndarray or cp.ndarray
        Reference image, shape (H_hr, W_hr, D_ref)
    option : dict
        Required fields:
            mask_ref, mask_mov, layer, iter, r, zRatio, zRatio_HR,
            smoothPenalty, movRange
        Optional:
            motion, phase, tol,
            wrong_region_enable,
            wrong_region_metric,
            wrong_region_mad_threshold,
            wrong_region_min_component_size,
            wrong_region_exclude_mode
    verbose : bool

    Returns
    -------
    phase_new : np.ndarray
    motion_current : np.ndarray
    data_ref_sampled : np.ndarray
    """
    # -----------------------------------------------------
    # Options
    # -----------------------------------------------------
    option["mask_ref"] = cp.asarray(option["mask_ref"], dtype=cp.float32)
    option["mask_mov"] = cp.asarray(option["mask_mov"], dtype=cp.float32)

    layer_num = int(option["layer"])
    iterNum = int(option["iter"])
    r = int(option["r"])
    zRatio_raw = float(option.get("zRatio", 1.0))
    zRatio_HR = float(option["zRatio_HR"])
    smoothPenalty = float(option["smoothPenalty"])
    movRange = float(option.get("movRange", 5.0))
    tol = float(option.get("tol", 1e-3))

    wrong_region_enable = bool(option.get("wrong_region_enable", True))
    wrong_region_metric = option.get("wrong_region_metric", "mse")
    wrong_region_mad_threshold = float(option.get("wrong_region_mad_threshold", 3.0))
    wrong_region_min_component_size = int(option.get("wrong_region_min_component_size", 2))
    wrong_region_exclude_mode = option.get("wrong_region_exclude_mode", "direct")
    expand_radius_xy = float(option.get("expand_radius_xy", 2.0))
    sigma_grad = float(option.get("sigma_grad", 1.0))
    intensity_k = float(option.get("intensity_k", 2.0))
    quantile_threshold_q = float(option.get("quantile_threshold_q", 0.8))

    data_mov = cp.asarray(data_mov, dtype=cp.float32)
    data_ref = cp.asarray(data_ref, dtype=cp.float32)

    SZ = data_mov.shape
    SZ_HR = data_ref.shape

    motion_current = None
    phase_new = None
    data_ref_sampled = None

    # -----------------------------------------------------
    # Pyramid: coarse -> fine
    # -----------------------------------------------------
    for layer in range(layer_num, -1, -1):
        if verbose:
            print(f"\n========== start layer {layer}/{layer_num} ==========")
        x = int(SZ[0] / (2**layer))
        y = int(SZ[1] / (2**layer))
        z = SZ[2]

        # reference resolution
        x_hr = int(SZ_HR[0] / (2**layer))
        y_hr = int(SZ_HR[1] / (2**layer))
        z_hr = SZ_HR[2]

        data_mov_layer = imresize(data_mov, output_shape=(x, y, z))
        data_ref_layer = imresize(data_ref, output_shape=(x_hr, y_hr, z_hr))

        mask_mov_layer = resample_exclude_mask(option["mask_mov"], output_shape=(x, y, z))
        mask_ref_layer = resample_exclude_mask(option["mask_ref"], output_shape=(x_hr, y_hr, z_hr))

        zRatio = zRatio_raw / (2**layer)
        zRatio_hr = zRatio_HR / (2**layer)
        H_ref_layer = generate_continuous_H_gpu(data_ref_layer, zRatio=1)
        H_mask_ref_layer = generate_continuous_H_gpu(mask_ref_layer.astype(cp.float32), zRatio=1)

        # init motion / phase
        motion_init_layer = initialize_motion_for_layer(
            option=option,
            layer=layer,
            layer_num=layer_num,
            SZ=SZ,
            x=x,
            y=y,
            z=z,
            prev_motion=motion_current,
        )
        phase_current = initialize_phase_for_layer(
            option=option, SZ=SZ, layer=layer, x=x, y=y, z=z, zRatio=zRatio, zRatio_hr=zRatio_hr
        )

        # keep a copy for possible correction restart
        motion_init_this_layer = motion_init_layer.copy()

        patchConnectNum = (2 * r + 1) ** 2
        smoothPenaltySum = smoothPenalty * patchConnectNum

        xG, yG, zG, xG_grid, yG_grid, zG_grid = make_control_point_grid(x, y, z, r)

        if wrong_region_enable:
            result = correct_wrong_regions_one_layer(
                data_mov_layer=data_mov_layer,
                data_ref_layer=data_ref_layer,
                H_ref_layer=H_ref_layer,
                H_mask_ref_layer=H_mask_ref_layer,
                mask_mov_layer=mask_mov_layer,
                mask_ref_layer=mask_ref_layer,
                phase_current=phase_current,
                motion_init_layer=motion_init_this_layer,
                xG=xG,
                yG=yG,
                xG_grid=xG_grid,
                yG_grid=yG_grid,
                zG_grid=zG_grid,
                r=r,
                iterNum=iterNum,
                movRange=movRange,
                smoothPenalty=smoothPenalty,
                smoothPenaltySum=smoothPenaltySum,
                zRatio=zRatio,
                zRatio_hr=zRatio_hr,
                error_metric=wrong_region_metric,
                mad_threshold=wrong_region_mad_threshold,
                min_component_size=wrong_region_min_component_size,
                bad_region_exclude_mode=wrong_region_exclude_mode,
                verbose=verbose,
                layer=layer,
                expand_radius_xy=expand_radius_xy,
                sigma_grad=sigma_grad,
                intensity_k=intensity_k,
                quantile_threshold_q=quantile_threshold_q,
            )
        else:
            result = optimize_layer_cross_resolution(
                data_mov_layer=data_mov_layer,
                data_ref_layer=data_ref_layer,
                H_ref_layer=H_ref_layer,
                H_mask_ref_layer=H_mask_ref_layer,
                mask_mov_layer=mask_mov_layer,
                phase_current=phase_current,
                motion_init=motion_init_this_layer,
                xG_grid=xG_grid,
                yG_grid=yG_grid,
                zG_grid=zG_grid,
                r=r,
                iterNum=iterNum,
                movRange=movRange,
                smoothPenalty=smoothPenalty,
                smoothPenaltySum=smoothPenaltySum,
                zRatio=zRatio,
                zRatio_hr=zRatio_hr,
                tol=tol,
                verbose=verbose,
                layer=layer,
            )

        motion_current = result["motion"]
        phase_new = result["phase_new"]
        data_ref_sampled = result["data_ref_sampled"]

    # -----------------------------------------------------
    # Return as numpy
    # -----------------------------------------------------
    if hasattr(motion_current, "get"):
        motion_current_np = cp.asnumpy(motion_current)
    else:
        motion_current_np = np.asarray(motion_current)

    if hasattr(phase_new, "get"):
        phase_new_np = cp.asnumpy(phase_new)
    else:
        phase_new_np = np.asarray(phase_new)

    if hasattr(data_ref_sampled, "get"):
        data_ref_sampled_np = cp.asnumpy(data_ref_sampled)
    else:
        data_ref_sampled_np = np.asarray(data_ref_sampled)

    return phase_new_np, motion_current_np, data_ref_sampled_np
