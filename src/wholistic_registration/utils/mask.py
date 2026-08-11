import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_opening

from .preprocess import robust_mean_std


def bwareafilt3_wei(vol, size_range):
    """
    3D binary area filtering function that keeps only connected components
    within a specified size range.

    Parameters:
    -----------
    vol : numpy.ndarray
        Input 3D volume (binary or numeric). Non-zero values are treated as foreground.
    size_range : list or tuple
        Two-element container [min_size, max_size] specifying the acceptable
        size range for connected components (in number of voxels).

    Returns:
    --------
    numpy.ndarray
        Filtered 3D boolean volume containing only components within size range.
    """
    # Convert input volume to boolean array
    # Any non-zero values become True, zeros become False
    vol = vol.astype(bool)

    # Label connected components in 3D using 26-connectivity
    # Each separate connected component gets a unique integer label
    # Background (False voxels) remains labeled as 0
    labeled_vol = label(vol, connectivity=3)

    # Extract properties of each labeled region
    # This gives us access to area, coordinates, and other properties
    # for each connected component
    regions = regionprops(labeled_vol)

    # Initialize output volume as all False (background)
    # This will store only the components that pass the size filter
    filtered_vol = np.zeros_like(vol, dtype=bool)

    # Iterate through each connected component and apply size filter
    for region in regions:
        # Get the area (number of voxels) of this connected component
        area = region.area

        # Check if component size falls within acceptable range (inclusive)
        if size_range[0] <= area <= size_range[1]:
            # If size is acceptable, set all voxels of this component to True
            # region.coords contains [row, col, depth] coordinates of all voxels
            # .T transposes to get separate arrays for each dimension
            filtered_vol[tuple(region.coords.T)] = True

    return filtered_vol


def getMask(dat_mov, thres_factor):
    """
    Generate a binary mask from 3D data using statistical thresholding
    and morphological operations.

    This function creates a mask by identifying voxels that deviate significantly
    from the mean (outliers) and then applies morphological operations to clean
    up the mask by removing noise and filling small gaps.

    Parameters:
    -----------
    dat_mov : numpy.ndarray
        Input 3D data volume (typically image data).
    thres_factor : float
        Threshold factor for statistical outlier detection. Higher values create
        more restrictive masks (fewer voxels selected).

    Returns:
    --------
    numpy.ndarray
        Binary mask (boolean array) with same shape as input data.
    """
    # Convert input data to float32 for consistent numerical operations
    dat_mov = dat_mov.astype(np.float32)

    # Robust (top-percentile-clipped) statistics instead of plain mean/std:
    # the outliers this mask exists to catch would otherwise be part of the
    # estimate itself — a single large bright artifact inflates sigma by
    # orders of magnitude (measured: 10 -> 1013), leaving any coexisting dim
    # artifact at |z| ~ 1 and invisible to every reasonable thres_factor.
    mu, sigma = robust_mean_std(dat_mov)

    # Normalize data by converting to z-scores and taking absolute values
    # This measures how many standard deviations each voxel is from the mean
    normalized = np.abs((dat_mov - mu) / sigma)

    # Create initial binary mask by thresholding normalized values
    # Voxels with z-scores above threshold are considered significant
    mask = normalized > thres_factor

    # Define structuring element for morphological operations
    # 4x4x1 kernel - processes in 2D slices along the z-axis
    selem = np.ones((3, 3, 1), dtype=bool)

    # Apply binary opening: erosion followed by dilation
    # This removes small noise artifacts and thin connections
    mask = binary_opening(mask, selem)

    # Apply binary dilation to restore size and fill small gaps
    # This expands the remaining structures after opening
    mask = binary_dilation(mask, selem)

    return mask
