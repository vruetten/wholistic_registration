"""

version : 0.1
file name: IO.py

Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) and Virginia Ruetten (Janelia) for python
Last Update Date : 2025/8/05

Overview:
    This module provides functions to read metadata and frames from ND2 files, specifically designed for handling 3D imaging data. It includes functionality to read metadata, extract specific frames, and handle both single and multiple frame requests efficiently.


functions:
    - readMeta(filePath, Ifprint=True): Reads metadata from an ND2 file and optionally prints Z ratio and data size.
    - readFrame(filePath, frame, channel=0, to_memory=True): Reads specified frames from an ND2 file, supporting both single and multiple frame requests, and handles 5D data structures.
    - downsample_tiff_series(tiff_folder_or_list, xy_down=4, batch_processing=False, batch_size=50, verbose=True): Efficiently downsample a series of TIFF files using dask for lazy loading and parallel processing.

"""

import json
import os
import shutil
from glob import glob

import dask
import dask.array as da
import dask.delayed
import nd2
import numpy as np
import tifffile
import toml
import zarr


def readMeta(filePath, Ifprint=True):
    """
    Reads metadata from an ND2 file and optionally prints Z ratio and data size.

    Parameters:
        filePath (str): Path to the ND2 file.
        Ifprint (bool): Whether to print Z ratio and data size. Default is True.

    Returns:
        metadata (nd2.Metadata): Metadata object containing information about the ND2 file.

    """
    with nd2.ND2File(filePath) as f:
        metadata = f.metadata
        channels = metadata.channels[0]
        if Ifprint:
            # get Zratio
            axesCalibration = channels.volume.axesCalibration
            zRatio = axesCalibration[2] / axesCalibration[0]
            print("Z ratio is", zRatio)
            # get size
            print("Data size is", channels.volume.voxelCount)
            # get total frames
            print("Total frames is", f.sizes["T"])

    return metadata


def readMeta_new(filePath, Ifprint=True):
    """
    Reads metadata from an ND2 file and optionally prints Z ratio and data size.

    Parameters:
        filePath (str): Path to the ND2 file.
        Ifprint (bool): Whether to print Z ratio and data size. Default is True.

    Returns:
        metadata (nd2.Metadata): Metadata object containing information about the ND2 file.

    """
    with nd2.ND2File(filePath) as ndf:
        data = ndf.to_dask()
        data_shape = data.shape
        data_dtype = data.dtype  # Get the actual data type

        if hasattr(ndf.metadata, "channels"):
            resolutionxyz = ndf.metadata.channels[0].volume.axesCalibration
            spacing_x = resolutionxyz[0]
            spacing_y = resolutionxyz[1]

            nchannels = len(ndf.metadata.channels)
            voxelCount = ndf.metadata.channels[0].volume.voxelCount
            nframes = ndf.shape[0]

            nxpix = voxelCount[0]
            nypix = voxelCount[1]
            if len(voxelCount) > 2:
                nzpix = voxelCount[2]
                spacing_z = resolutionxyz[2]
                zRatio = spacing_z / spacing_x

            else:
                spacing_z = 1
                zRatio = 1
                nzpix = 1

        metadata = ndf.metadata
        channels = metadata.channels[0]

        multichannel = True if nchannels > 1 else False
        multiz = True if nzpix > 1 else False
        multiframe = True if nframes > 1 else False
        if not multiz:
            # (T, C, Y, X)->(T, 1, C, Y, X)
            data_shape = (data_shape[0], 1, data_shape[1], data_shape[2], data_shape[3])
        try:
            avgdiff = ndf.experiment[0].parameters.periodDiff.avg / 1000
            framerate = 1 / avgdiff
        except:
            t0 = ndf.frame_metadata(0).channels[0].time.relativeTimeMs
            t1 = ndf.frame_metadata(1).channels[0].time.relativeTimeMs
            dt_ms = t1 - t0
            if dt_ms <= 0:
                raise ValueError("Invalid timestamps: Δt <= 0")
            framerate = 1000.0 / dt_ms

        if Ifprint:
            # get Zratio
            zRatio = spacing_z / spacing_x
            print("Z ratio is", zRatio)
            # get size
            print("Data size is", [nxpix, nypix, nzpix])
            # get total frames
            print("Total frames is", ndf.sizes["T"])

    metadata_dict = {  # needed for ImageJ
        "Pixels": {
            "PhysicalSizeX": spacing_x,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": spacing_y,
            "PhysicalSizeYUnit": "um",
            "PhysicalSizeZ": spacing_z,
            "PhysicalSizeZUnit": "um",
        },
        "loop": True,
        "fps": framerate,
        "zRatio": zRatio,
        "nframes": nframes,
        "nchannels": nchannels,
        "resolutionxyz": resolutionxyz,
        "data_shape": data_shape,
        "dtype": data_dtype,  # Data type (e.g., uint16, float32)
        "bytes_per_pixel": data_dtype.itemsize,  # Bytes per pixel
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "spacing_z": spacing_z,
        "axes": "TZCYX",
        "SizeC": nchannels,
        "SizeT": nframes,
        "SizeZ": nzpix,
        "SizeX": nxpix,
        "SizeY": nypix,
        "multichannel": multichannel,
        "multiz": multiz,
        "multiframe": multiframe,
        "nzpix": nzpix,
        "nxpix": nxpix,
        "nypix": nypix,
    }

    return metadata_dict


def get_framerate(filePath):
    with nd2.ND2File(filePath) as f:
        t0 = f.frame_metadata(0).channels[0].time.relativeTimeMs
        t1 = f.frame_metadata(1).channels[0].time.relativeTimeMs
        dt_ms = t1 - t0
        if dt_ms <= 0:
            raise ValueError("Invalid timestamps: Δt <= 0")
        framerate = 1000.0 / dt_ms
        return framerate, dt_ms


def getTotalFrames(filePath):
    with nd2.ND2File(filePath) as f:
        frame = f.sizes["T"]
    return frame


def readND2Frame(
    filePath, frames, slices=None, channel=0, xy_down=1, to_memory=True, verbose=False
):
    """
    ND2 reader with high-quality XY downsampling:
      - Z slice selection (via `slices`)
      - XY resampling using skimage.resize with anti-aliasing
    """

    if channel is None:
        channel = slice(None)

    # Normalize indexing to preserve dimensions
    def normalize_index(idx, name):
        """Convert various index types to dimension-preserving format"""
        if idx is None:
            return slice(None)
        elif isinstance(idx, (int, np.integer)):
            # Convert int to dimension-preserving slice; stop=None when idx==-1
            # (slice(-1, 0) would be an empty selection)
            return slice(idx, idx + 1 if idx != -1 else None)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            # For lists, ensure at least 2 elements to preserve dimensions
            idx_list = list(idx)
            if len(idx_list) == 1:
                # Duplicate the single element to preserve dimension
                i0 = idx_list[0]
                return slice(i0, i0 + 1 if i0 != -1 else None)
            return idx_list
        elif isinstance(idx, slice):
            return idx
        else:
            raise ValueError(f"Invalid {name} index type: {type(idx)}")

    frames = normalize_index(frames, "frames")
    slices = normalize_index(slices, "slices")
    channel = normalize_index(channel, "channel")

    with nd2.ND2File(filePath) as f:
        sizes = f.sizes
        dims = list(sizes.keys())
        is_5d = len(dims) == 5
        metadata = readMeta_new(filePath, Ifprint=verbose)
        nframes = metadata["nframes"]
        nchannels = metadata["nchannels"]
        nzpix = metadata["nzpix"]

        # ------------------------------------
        # Read ND2 data (lazy dask tensor) - ALL FRAMES AT ONCE
        # ------------------------------------
        dask_data = f.to_dask()
        multiframe = True if nframes > 1 else False
        multiz = True if nzpix > 1 else False
        multichannel = True if nchannels > 1 else False
        if verbose:
            print(f"multiframe is {multiframe}")
            print(f"multiz is {multiz}")
            print(f"multichannel is {multichannel}")
            print(f"nframes is {nframes}")
            print(f"nzpix is {nzpix}")
            print(f"nchannels is {nchannels}")

        # Add dimensions if they don't exist to ensure 5D structure
        if not multiframe:
            dask_data = dask_data[None]  # Add T dimension

        if not multiz:
            dask_data = dask_data[:, None]  # Add Z dimension

        if not multichannel:
            dask_data = dask_data[:, :, None]  # Add C dimension
        if verbose:
            print(
                f"After adding dimensions: dask_data.shape = {dask_data.shape}, len(data.shape) = {len(dask_data.shape)}"
            )

        # Apply indexing while preserving dimensions
        dask_data = dask_data[frames, ...]
        dask_data = dask_data[:, slices, ...]
        dask_data = dask_data[
            :,
            :,
            channel,
        ]

        # check if 5D
        if len(dask_data.shape) != 5:
            raise ValueError(
                f"After slicing: dask_data.shape = {dask_data.shape}, len(data.shape) = {len(dask_data.shape)}"
            )

        if verbose:
            print(f"After slicing: dask_data.shape = {dask_data.shape}")

            print(f"dask_data.shape is {dask_data.shape}")

        T, Z, C, Y, X = dask_data.shape

        # ------------------------------------
        # XY downsample using binning/averaging (dask-native)
        # ------------------------------------
        if xy_down > 1:
            if verbose:
                print(f"Downsampling data by {xy_down}x")
            dask_data = downsample(dask_data, xy_down)

        # Only compute at the very end if requested
        data = dask_data.compute().astype(np.float32) if to_memory else dask_data

        return data


def downsample(data_tzcyx, xy_down=4):
    T, Z, C, Y, X = data_tzcyx.shape
    newY = (Y // xy_down) * xy_down
    newX = (X // xy_down) * xy_down
    data_tzcyx = data_tzcyx[:, :, :, :newY, :newX]
    data_tzcyx = data_tzcyx.reshape(T, Z, C, newY // xy_down, xy_down, newX // xy_down, xy_down)
    data_tzcyx = data_tzcyx.mean(axis=(4, 6))
    return data_tzcyx


def downsample_tiff_series(
    tiff_folder_or_list, xy_down=4, batch_processing=False, batch_size=50, verbose=True
):
    """
    Efficiently downsample a series of TIFF files using dask for lazy loading and parallel processing.

    Parameters:
        tiff_folder_or_list (str or list): Either a folder path containing TIFF files,
                                          or a list of TIFF file paths
        xy_down (int): Downsampling factor for X and Y dimensions. Default is 4.
        batch_processing (bool): If True, process TIFFs in batches for very large datasets.
                               If False, process all at once. Default is False.
        batch_size (int): Number of TIFFs to process in each batch when batch_processing=True.
                         Default is 50.
        verbose (bool): Whether to print progress information. Default is True.

    Returns:
        dask.array: Downsampled data array with shape (T, Z, C, Y_ds, X_ds) where T is time,
                   Z is depth, C is channel, and Y_ds/X_ds are downsampled spatial dimensions.
                   Call .compute() on the result to get the actual numpy array.

    Example:
        # Process all TIFFs in a folder
        result = downsample_tiff_series("/path/to/tiff/folder", xy_down=4)
        downsampled_data = result.compute()

        # Process a specific list of TIFF files
        tiff_files = ["/path/to/file1.tif", "/path/to/file2.tif"]
        result = downsample_tiff_series(tiff_files, xy_down=2)

        # For very large datasets, use batch processing
        result = downsample_tiff_series("/path/to/large/dataset",
                                       xy_down=4, batch_processing=True, batch_size=100)
    """

    # Handle input: folder path or list of files
    if isinstance(tiff_folder_or_list, str):
        if os.path.isdir(tiff_folder_or_list):
            tiff_files = glob(os.path.join(tiff_folder_or_list, "*.tif"))
            tiff_files.extend(glob(os.path.join(tiff_folder_or_list, "*.tiff")))
            tiff_files.sort()  # Ensure consistent ordering
        else:
            raise ValueError(f"Directory not found: {tiff_folder_or_list}")
    elif isinstance(tiff_folder_or_list, list):
        tiff_files = tiff_folder_or_list
    else:
        raise ValueError("Input must be either a directory path or a list of TIFF file paths")

    if len(tiff_files) == 0:
        raise ValueError("No TIFF files found")

    if verbose:
        print(f"Found {len(tiff_files)} TIFF files")

    # Read one TIFF to get the shape and dtype
    sample_data = tifffile.imread(tiff_files[0])
    if verbose:
        print(f"Sample TIFF shape: {sample_data.shape}, dtype: {sample_data.dtype}")

    @dask.delayed
    def load_tiff(tiff_path):
        """Load a single TIFF file"""
        return tifffile.imread(tiff_path)

    if not batch_processing:
        # Process all TIFFs at once (recommended for moderate datasets)
        if verbose:
            print("Processing all TIFFs at once...")

        # Create delayed objects for each TIFF
        delayed_arrays = [load_tiff(tiff) for tiff in tiff_files]

        # Convert to dask arrays and stack them with optimal chunking
        if len(sample_data.shape) == 3:  # 3D TIFFs (Z, Y, X)
            Z, Y, X = sample_data.shape
            # Chunk size: process a few timepoints at once, keep spatial dims manageable
            chunk_t = min(4, len(tiff_files))
            dask_arrays = [
                da.from_delayed(delayed_arr, shape=(Z, Y, X), dtype=sample_data.dtype)
                for delayed_arr in delayed_arrays
            ]
            # Stack along time axis to create (T, Z, Y, X)
            data_tzyx = da.stack(dask_arrays, axis=0)
            # Rechunk for better performance: chunk along time, keep spatial dims together
            data_tzyx = data_tzyx.rechunk((chunk_t, Z, Y // 2, X // 2))
            # Add channel dimension to make it (T, Z, C, Y, X) for downsample
            data_tzcyx = data_tzyx[:, :, None, :, :]
        else:  # 2D TIFFs (Y, X)
            Y, X = sample_data.shape
            # Process more 2D timepoints at once
            chunk_t = min(8, len(tiff_files))
            dask_arrays = [
                da.from_delayed(delayed_arr, shape=(Y, X), dtype=sample_data.dtype)
                for delayed_arr in delayed_arrays
            ]
            # Stack along time axis to create (T, Y, X)
            data_tyx = da.stack(dask_arrays, axis=0)
            # Rechunk for better performance
            data_tyx = data_tyx.rechunk((chunk_t, Y // 2, X // 2))
            # Add Z and C dimensions to make it (T, Z, C, Y, X) for downsample
            data_tzcyx = data_tyx[:, None, None, :, :]

        if verbose:
            print(f"Dask array shape: {data_tzcyx.shape}")
            print(f"Dask array chunks: {data_tzcyx.chunks}")

        # Downsample using the existing downsample function
        data_ds = downsample(data_tzcyx, xy_down=xy_down)

        if verbose:
            print(f"Downsampled shape: {data_ds.shape}")

        return data_ds

    else:
        # Batch processing for very large datasets
        if verbose:
            print(f"Processing TIFFs in batches of {batch_size}...")

        results = []
        num_batches = (len(tiff_files) + batch_size - 1) // batch_size

        for i in range(0, len(tiff_files), batch_size):
            batch_tiffs = tiff_files[i : i + batch_size]
            batch_num = i // batch_size + 1

            if verbose:
                print(f"Processing batch {batch_num}/{num_batches} ({len(batch_tiffs)} files)")

            # Create delayed objects for this batch
            batch_delayed = [load_tiff(tiff) for tiff in batch_tiffs]

            # Convert to dask arrays and stack
            if len(sample_data.shape) == 3:  # 3D TIFFs
                Z, Y, X = sample_data.shape
                batch_arrays = [
                    da.from_delayed(delayed_arr, shape=(Z, Y, X), dtype=sample_data.dtype)
                    for delayed_arr in batch_delayed
                ]
                batch_data = da.stack(batch_arrays, axis=0)[:, :, None, :, :]
            else:  # 2D TIFFs
                Y, X = sample_data.shape
                batch_arrays = [
                    da.from_delayed(delayed_arr, shape=(Y, X), dtype=sample_data.dtype)
                    for delayed_arr in batch_delayed
                ]
                batch_data = da.stack(batch_arrays, axis=0)[:, None, None, :, :]

            # Downsample this batch
            batch_ds = downsample(batch_data, xy_down=xy_down)

            # Compute and store result
            batch_result = batch_ds.compute()
            results.append(batch_result)

            if verbose:
                print(f"Batch {batch_num} completed, shape: {batch_result.shape}")

        # Concatenate all batch results and convert back to dask array
        final_result = np.concatenate(results, axis=0)
        if verbose:
            print(f"Final concatenated result shape: {final_result.shape}")

        # Convert back to dask array for consistency
        return da.from_array(
            final_result,
            chunks=(
                min(8, final_result.shape[0]),
                final_result.shape[1],
                final_result.shape[2],
                final_result.shape[3] // 2,
                final_result.shape[4] // 2,
            ),
        )


def reset_dir(path, force=False):
    # FIX: added `force` flag (default False) so callers can reset without blocking
    # on interactive input. The old version used input() which breaks automated pipelines.
    if os.path.exists(path):
        if force:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            ans = input(f"Directory '{path}' exists. Delete it? [y/N]: ").strip().lower()
            if ans in ["y", "yes"]:
                shutil.rmtree(path)
                os.makedirs(path)
                print("Directory reset.")
            else:
                print("Skipped.")
    else:
        os.makedirs(path)


def downsample_tifs_dask(
    input_folder, output_folder, downsample_xy=4, downsample_t=1, n_workers=4, verbose=True
):
    """
    Read TIF files, downsample, and save to new folder using Dask for parallelization.

    Parameters
    ----------
    input_folder : str
        Path to folder containing input TIF files
    output_folder : str
        Path to folder where downsampled TIFs will be saved
    downsample_xy : int
        Downsampling factor for X and Y dimensions
    downsample_t : int
        Downsampling factor for time dimension (take every Nth frame)
    n_workers : int
        Number of parallel workers for Dask
    """
    import dask
    import numpy as np
    import tifffile
    from dask import delayed

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of TIF files
    tif_files = sorted([f for f in os.listdir(input_folder) if f.endswith((".tif", ".tiff"))])
    tif_files = tif_files[::downsample_t]

    if verbose:
        print(f"Found {len(tif_files)} TIF files to process")

    @delayed
    def process_single_tif(filename):
        """Process a single TIF file: read, downsample, save."""
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read the TIF file
        with tifffile.TiffFile(input_path) as tif:
            image = tif.asarray()
            # Try to get existing metadata
            try:
                original_metadata = tif.imagej_metadata or {}
            except:
                original_metadata = {}

        # Determine dimensions and downsample accordingly
        # Expected format: TZCYX (5D) or ZCYX (4D) or ZYX (3D) or YX (2D)
        ndim = image.ndim

        if ndim == 5:  # TZCYX
            image = image[:, :, :, ::downsample_xy, ::downsample_xy]
        elif ndim == 4:  # ZCYX or TCYX
            image = image[:, :, ::downsample_xy, ::downsample_xy]
        elif ndim == 3:  # ZYX or TYX
            image = image[:, ::downsample_xy, ::downsample_xy]
        elif ndim == 2:  # YX
            image = image[::downsample_xy, ::downsample_xy]

        # Ensure 5D for saving (TZCYX)
        while image.ndim < 5:
            image = image[np.newaxis, ...]

        # Update metadata with new shape and adjusted spacing
        metadata = {
            "spacing_x": original_metadata.get("spacing_x", 1.0) * downsample_xy,
            "spacing_y": original_metadata.get("spacing_y", 1.0) * downsample_xy,
            "data_shape": image.shape,
            "downsample_xy": downsample_xy,
            "downsample_t": downsample_t,
        }

        # Save using tifffile (same format as saveTiff_new)
        spacing_x = metadata["spacing_x"]
        spacing_y = metadata["spacing_y"]

        with tifffile.TiffWriter(output_path, imagej=True) as tif_writer:
            tif_writer.write(
                image, metadata=metadata, resolution=(1.0 / spacing_x, 1.0 / spacing_y)
            )

        return filename

    # Create delayed tasks for all files
    tasks = [process_single_tif(f) for f in tif_files]

    # Execute in parallel with progress bar
    if verbose:
        print(f"Processing {len(tasks)} files with {n_workers} workers...")

    from dask.diagnostics import ProgressBar

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=n_workers):
            results = dask.compute(*tasks)
    if verbose:
        print(f"   Processed {len(results)} files")
        print(f"   Output folder: {output_folder}")
    return list(results)


def downsample_nd2_to_tiff_folder(
    nd2_path,
    output_folder,
    ds_xy=1,
    ds_t=1,
    frame_list=None,
    slices=None,
    channel=0,
    n_workers=4,
    dtype=np.float32,
    verbose=True,
):
    """
    Downsample ND2 file using Dask (stride-based) and save as TIFF series in parallel.

    Output:
        One TIFF per frame, shape (Z, C, Y, X)
        Compatible with both 2D and 3D ND2 files.

    Parameters
    ----------
    nd2_path : str
        Path to ND2 file
    output_folder : str
        Path to save downsampled TIFF series
    ds_xy : int
        XY downsampling factor
    ds_t : int
        Temporal downsampling factor (take every ds_t frame)
    frame_list : list[int] | None
        Only process frames in this list
    slices : list | slice | None
        Z slice selection (ignored safely for 2D ND2)
    channel : int | list | None
        Channel selection
    n_workers : int
        Number of threads
    dtype : np.dtype
        Output dtype
    verbose : bool
    """
    from dask import compute, delayed
    from dask.diagnostics import ProgressBar

    os.makedirs(output_folder, exist_ok=True)

    def normalize_index(idx):
        if idx is None:
            return slice(None)
        if isinstance(idx, (int, np.integer)):
            return slice(idx, idx + 1 if idx != -1 else None)
        return idx

    def ensure_5d_tzcyx(dask_data, dims, verbose=False):
        """Normalize ND2 dask array to shape (T, Z, C, Y, X), works for 2D/3D"""
        axis_map = {d: i for i, d in enumerate(dims)}

        if "T" not in axis_map:
            dask_data = dask_data[None, ...]
            dims = ["T"] + dims
            axis_map = {d: i for i, d in enumerate(dims)}

        if "Z" not in axis_map:
            z_axis = axis_map["T"] + 1
            dask_data = dask_data[(slice(None),) * z_axis + (None,)]
            dims = dims[:z_axis] + ["Z"] + dims[z_axis:]
            axis_map = {d: i for i, d in enumerate(dims)}

        if "C" not in axis_map:
            c_axis = axis_map["Z"] + 1
            dask_data = dask_data[(slice(None),) * c_axis + (None,)]
            dims = dims[:c_axis] + ["C"] + dims[c_axis:]
            axis_map = {d: i for i, d in enumerate(dims)}

        # reorder to TZCYX
        target_order = ["T", "Z", "C", "Y", "X"]
        perm = [axis_map[d] for d in target_order]
        dask_data = dask_data.transpose(*perm)
        if verbose:
            print(f"[INFO] Normalized ND2 shape -> TZCYX: {dask_data.shape}")
        return dask_data

    with nd2.ND2File(nd2_path) as f:
        dask_data = f.to_dask()
        dims = list(f.sizes.keys())
        dask_data = ensure_5d_tzcyx(dask_data, dims, verbose)
        if frame_list is not None:
            frame_list = frame_list[::ds_t]
        else:
            frame_list = list(range(dask_data.shape[0]))
            frame_list = frame_list[::ds_t]
        dask_data = dask_data[frame_list]
        dask_data = dask_data[
            slice(None, None, 1),  # T
            normalize_index(slices),  # Z
            normalize_index(channel),  # C
            slice(None, None, ds_xy),  # Y
            slice(None, None, ds_xy),  # X
        ]
        if verbose:
            print(f"[INFO] After slicing/downsample: {dask_data.shape}")

    tasks = []
    T = dask_data.shape[0]

    @delayed
    def save_frame_outer(frame_data, out_folder, fidx, ch, dtype):
        vol = frame_data.astype(dtype)
        out_name = f"vol_ch{ch}_downsample_{fidx:06d}.tif"
        out_path = os.path.join(out_folder, out_name)
        tifffile.imwrite(out_path, vol, imagej=True)
        return out_path

    # FIX: removed dead commented-out save_frame closure that was superseded by save_frame_outer
    for i in range(T):
        frame_idx = frame_list[i]
        tasks.append(
            save_frame_outer(dask_data[i], output_folder, fidx=frame_idx, ch=channel, dtype=dtype)
        )
    if verbose:
        print(f"[INFO] Processing {T} frames with {n_workers} threads...")
    with ProgressBar():
        compute(*tasks, scheduler="threads", num_workers=n_workers)
    if verbose:
        print(f"[DONE] ND2 downsampled TIFFs saved to {output_folder}")
    return [
        os.path.join(output_folder, f"vol_ch{channel}_downsample_{frame_list[i]:06d}.tif")
        for i in range(T)
    ]


def saveTiff(image_list, config_path, save_path):
    """
    Saves a list of image frames as a multi-page TIFF file and embeds configuration
    data from a TOML file into the TIFF metadata.

    Parameters:
        image_list (list[np.ndarray]): A list where each element is a 2D or 3D NumPy array
                                       representing one image frame (H, W) or (H, W, C).
        config_path (str): Path to the TOML configuration file.
        save_path (str): Path to save the resulting TIFF file.

    Returns:
        None

    Notes:
        - The content of the TOML file will be serialized into a JSON string and stored
          in the TIFF ImageDescription tag for later retrieval.
        - All images will be converted to uint8 before saving if they are not already.
        - The function uses the tifffile library for writing multi-page TIFF files.
    """
    config_data = toml.load(config_path)

    import json

    config_str = json.dumps(config_data, ensure_ascii=False)

    # check the list
    for i, img in enumerate(image_list):
        if not isinstance(img, np.ndarray):
            raise ValueError(f"element{i} is not a image")

    tifffile.imwrite(save_path, image_list, description=config_str, bigtiff=True)


def saveTiff_new(image, save_path, config_path=None, metadata=None, verbose=True):
    """
    Save 5D (TZCYX) image as ImageJ-compatible TIFF.

    Note: unlike saveTiff(), this stores config_path as a string in the TIFF
    description field (not the serialized JSON contents). Use readTiff() to
    retrieve the path, then load the config separately with toml.load().
    """
    if image.ndim == 2:
        image = image[None, None, None, :, :]
    if image.ndim != 5:
        raise ValueError("All saved should be 5D (TZCYX)")

    if metadata is not None:
        spacing_x = metadata["spacing_x"]
        spacing_y = metadata["spacing_y"]
        # copy so the caller's dict is not mutated
        metadata = {**metadata, "data_shape": image.shape}
    else:
        metadata = {}
        spacing_x = 1
        spacing_y = 1
        metadata["spacing_x"] = spacing_x
        metadata["spacing_y"] = spacing_y
        metadata["data_shape"] = image.shape

    if verbose:
        print(f"Saving TIFF file to {save_path}")
        print(f"  - shape: {image.shape}")
        print(f"  - spacing: {spacing_x}, {spacing_y}")
        if config_path is not None:
            print(f"  - config: {config_path}")

    with tifffile.TiffWriter(save_path, imagej=True) as tif:
        tif.write(
            image,
            metadata=metadata,
            resolution=(1.0 / spacing_x, 1.0 / spacing_y),
            description=config_path,
        )


def saveZarr(mem_data, ca_data, reference, config_path, save_path, chunks=(1, 512, 512)):
    """
    Save membrane channel, calcium channel, and reference image into one Zarr store.

    Parameters:
        mem_data (np.ndarray): Membrane channel data, shape (T, H, W) or (T, H, W, C).
        ca_data  (np.ndarray): Calcium channel data, shape (T, H, W) or (T, H, W, C).
        reference (np.ndarray): Reference image (2D).
        config_path (str): Path to the TOML configuration file.
        save_path (str): Path to save the resulting Zarr dataset (directory).
        chunks (tuple): Chunk size for Zarr storage, default (1, 512, 512).

    Returns:
        None

    """
    config_data = toml.load(config_path)
    config_str = json.dumps(config_data, ensure_ascii=False)

    # ensure numpy array
    mem_data = np.asarray(mem_data, dtype=np.float32)
    ca_data = np.asarray(ca_data, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    # open zarr root
    root = zarr.open(save_path, mode="w")

    # create datasets
    root.create_dataset("membrane", data=mem_data, chunks=chunks, overwrite=True)
    root.create_dataset("calcium", data=ca_data, chunks=chunks, overwrite=True)
    root.create_dataset(
        "reference", data=reference, overwrite=True
    )  # usually 2D, so no chunks needed

    # save config
    root.attrs["config"] = config_str

    print(f"Saved Zarr dataset at {save_path}")
    print(f"  - membrane: {mem_data.shape}")
    print(f"  - calcium : {ca_data.shape}")
    print(f"  - reference: {reference.shape}")


def saveZarr_fast(
    mem_data,
    ca_data,
    reference,
    config_path,
    save_path,
    chunks=(16, 512, 512),
    compressor=None,
    single_file=False,
):
    """
    Fast Zarr saving for membrane, calcium and reference data.

    Parameters:
        mem_data (np.ndarray): Membrane channel, shape (T,H,W) or (T,H,W,C)
        ca_data (np.ndarray): Calcium channel, shape (T,H,W) or (T,H,W,C)
        reference (np.ndarray): Reference image, shape (H,W)
        config_path (str): TOML configuration file path
        save_path (str): Output Zarr directory or file (if single_file=True)
        chunks (tuple): Chunk size for Zarr
        compressor: Zarr compressor (default: fast Blosc zstd)
        single_file (bool): Whether to save as single file (ZipStore)

    Returns:
        None
    """
    import json

    import toml
    import zarr
    from numcodecs import Blosc

    # default compressor
    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)

    # load config
    config_data = toml.load(config_path)
    config_str = json.dumps(config_data, ensure_ascii=False)

    # switch to numpy
    mem_data = np.asarray(mem_data, dtype=np.float32)
    ca_data = np.asarray(ca_data, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    # open Zarr store
    if single_file:
        store = zarr.ZipStore(save_path + ".zip", mode="w")
    else:
        store = save_path
    try:
        root = zarr.open(store, mode="w")

        # create datasets
        root.create_dataset(
            "membrane", data=mem_data, chunks=chunks, compressor=compressor, overwrite=True
        )
        root.create_dataset(
            "calcium", data=ca_data, chunks=chunks, compressor=compressor, overwrite=True
        )
        root.create_dataset("reference", data=reference, compressor=compressor, overwrite=True)

        # save config
        root.attrs["config"] = config_str
    finally:
        # ZipStore only writes the zip central directory on close(); without this
        # the archive is unreadable until interpreter shutdown (and corrupt on crash)
        if single_file:
            store.close()

    print(f"Saved fast Zarr at {save_path}")
    print(f"  - membrane: {mem_data.shape}")
    print(f"  - calcium : {ca_data.shape}")
    print(f"  - reference: {reference.shape}")


# FIX: renamed from readTifff (triple-f typo) to readTiff
def readTiff(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        images = tif.asarray()
        desc = tif.pages[0].tags.get("ImageDescription")
        if desc is not None:
            desc = desc.value
    return images, desc


def write_volume_as_ome_tiff(
    volume, out_dir, ch_idx, frame_idx, configPath, spacing_x=1.0, spacing_y=1.0
):
    """
    volume: np.ndarray, shape (Z,Y,X) or (Y,X)
    out_dir: target directory
    ch_idx: integer channel id used in filename
    frame_idx: integer or string like '100~200'
    """
    if isinstance(frame_idx, int):
        frame_tag = f"{frame_idx:06d}"
    elif isinstance(frame_idx, str):
        cleaned = frame_idx.replace("-", "~").replace("_", "~")
        parts = cleaned.split("~")
        if len(parts) == 1:
            try:
                frame_tag = f"{int(parts[0]):06d}"
            except:
                raise ValueError(f"Invalid frame_idx string: {frame_idx}")

        elif len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                frame_tag = f"{a:06d}_{b:06d}"
            except:
                raise ValueError(f"Invalid frame_idx string: {frame_idx}")
        elif len(parts) == 5:
            try:
                a, b, c, d = int(parts[0]), int(parts[1]), int(parts[3]), int(parts[4])
                frame_tag = f"{a:04d}~{b:04d}_vs_{c:04d}~{d:04d}"
            except:
                raise ValueError(f"Invalid frame_idx string: {frame_idx}")
        else:
            raise ValueError(f"Invalid frame_idx format: {frame_idx}")

    else:
        raise TypeError("frame_idx must be int or string")

    if volume.ndim == 2:
        zvol = volume[np.newaxis, :, :]
    elif volume.ndim == 3:
        zvol = volume
    else:
        raise ValueError("volume must be 2D or 3D (Z,Y,X)")
    img5d = zvol[np.newaxis, :, np.newaxis, :, :]  # (1,Z,1,Y,X)
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"vol_{ch_idx}_{frame_tag}.tif")
    metadata = {"spacing_x": spacing_x, "spacing_y": spacing_y, "data_shape": img5d.shape}

    saveTiff_new(img5d, fname, config_path=configPath, metadata=metadata, verbose=False)

    return fname


def write_multichannel_volume_as_ome_tiff(
    volume, out_dir, frame_idx, configPath=None, label=None, spacing_x=1.0, spacing_y=1.0
):
    """
    volume: list of c arrays, each (Z,Y,X)
        ch0, ch1, ch2

    output OME TIFF shape: (1, Z, c, Y, X)
    """

    processed = []
    for v in volume:
        if v.ndim == 2:
            v = v[np.newaxis, :, :]
        if v.dtype == bool:
            v = v.astype(np.uint8)
        elif v.dtype not in [np.uint8, np.float32]:
            v = v.astype(np.float32)
        processed.append(v)

    img5d = np.stack(processed, axis=0)  # (3, Z, Y, X)
    img5d = img5d[np.newaxis, :, :, :, :]  # (1,3,Z,Y,X)
    img5d = np.transpose(img5d, (0, 2, 1, 3, 4))  # → (1,Z,3,Y,X)

    fname = os.path.join(out_dir, f"vol_{label}_{frame_idx:06d}.tif")

    metadata = {"spacing_x": spacing_x, "spacing_y": spacing_y, "data_shape": img5d.shape}

    saveTiff_new(img5d, fname, config_path=configPath, metadata=metadata, verbose=False)


def read_reg_tiff(folder, frame_idx, ch_idx):
    fname = os.path.join(folder, f"vol_ch{ch_idx}_{frame_idx:06d}.tif")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Cannot find {fname}")
    vol = tifffile.imread(fname)  # (Z,Y,X)
    return vol
