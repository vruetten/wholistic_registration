import os

import dask
import tifffile as tf


def save_zarr_as_tiffs_simple(
    zarr_array, output_dir, n_frames=None, xy_downsample=2, channel=0, metadata=None
):
    """
    Simple function: zarr frame -> downsample -> save as TIFF
    No batching, no complex graphs - just straightforward processing
    """
    if n_frames is None:
        n_frames = zarr_array.shape[0]

    # Get spacing info
    if metadata is not None:
        spacing_x = metadata.get("spacing_x", 1) * xy_downsample
        spacing_y = metadata.get("spacing_y", 1) * xy_downsample
        # embed the downsample-corrected spacing (copy: don't mutate the caller's dict)
        metadata = {**metadata, "spacing_x": spacing_x, "spacing_y": spacing_y}
    else:
        spacing_x = xy_downsample
        spacing_y = xy_downsample
        metadata = {}
        metadata["spacing_x"] = spacing_x
        metadata["spacing_y"] = spacing_y
        metadata["data_shape"] = zarr_array.shape

    def downsample_xy(data, ds_factor):
        if ds_factor == 1:
            return data
        Y, X = data.shape[-2:]
        new_Y = (Y // ds_factor) * ds_factor
        new_X = (X // ds_factor) * ds_factor
        data_cropped = data[..., :new_Y, :new_X]
        if len(data.shape) == 3:  # (Z, Y, X)
            Z = data.shape[0]
            data_ds = data_cropped.reshape(
                Z, new_Y // ds_factor, ds_factor, new_X // ds_factor, ds_factor
            )
            return data_ds.mean(axis=(2, 4))
        else:  # (Y, X)
            data_ds = data_cropped.reshape(
                new_Y // ds_factor, ds_factor, new_X // ds_factor, ds_factor
            )
            return data_ds.mean(axis=(1, 3))

    @dask.delayed
    def save_single_frame(frame_idx):
        """Load one frame, downsample it, save as TIFF"""
        # Load single frame - zarr indexing returns numpy array directly
        frame_data = zarr_array[frame_idx]

        # Downsample
        if xy_downsample > 1:
            frame_data = downsample_xy(frame_data, xy_downsample)

        # Save
        filename = f"vol_ch{channel}_{frame_idx:06d}.tif"
        filepath = os.path.join(output_dir, filename)
        tf.imwrite(
            filepath,
            frame_data,
            compression="lzw",
            resolution=(1.0 / spacing_x, 1.0 / spacing_y),
            metadata=metadata,
        )
        return filepath

    # Create one delayed task per frame
    delayed_tasks = [save_single_frame(i) for i in range(n_frames)]
    return delayed_tasks
