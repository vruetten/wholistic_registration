# %%
# %load_ext autoreload
# %autoreload 2
from wholistic_registration.core import main_function

configFile = "./code/wholistic_registration/configs/config_0120.toml"

# Define data path and the normal config
main_function.DefineParams(
    configFile=configFile,
    inputFile="/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/nd2/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2",
    outputFile="/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/reg/",
    downsampleXY=1,
    dual_channel=True,
    # downsampleZ=[5, 6, 7], #choose which planes to use
    window_size=10,  ##  minutes
    mid_window_size=5,  ## minutes
    verbose=True,
    function="raw",
    batch_size=60,  # in frames
    mid_stride=1,  # in frames
    reference_chunk=50,
    preprocess=False,  # WHETHER TO NORMALIZE THE IMAGE TO [0,255]
    thresFactor=5,  # IMMUNE CELL THRESHOLD - REMOVE REGIONS THAT ARE NSTD OF THE WHOLE IMAGE
    maskRange=[
        5,
        4000,
    ],  # IF ABSOLUTE VALUE OF THE PIXELS IS GREATER THAN THIS, WE WILL MASK THE PIXEL # NOT REALLY USED - THIS SHOULD BE SAVED
    layer=3,  # PYRAMID LEVELS
    r=5,  # THIS SHOULD BE IN MICRONS -
    iter=10,  # MAX ITERATIONS PER PYRAMID LEVEL -  OTHER STOPPING CRITERIA? EXPOSED TO USER
    smoothPenalty=0.08,  # THE COEFFICIENT OF THE SMOOTHNESS PENALTY TERM. A LARGER SMOOTHNESS PENALTY MEANS MORE SMOOTH MOTION WE WILL GET AND CORRESPONDINGLY THE ERROR OF INTENSITY WILL INCREASE
    tolerance=1e-3,  # THE TOLERANCE TO STOP THE ITERATION, DEFAULT IS 1E-3
)
# %%


###########################################################################################
### main process
# Do registration
# main_function.Registration_v3(
#     configFile,
#     parallel=False
# )

# %%
# reliable analysis
# main_function.ReliableAnalysis(
#     configFile,
# )


###########################################################################################
### visualization
# consistent of the mask
# # ## create downsample data
# main_function.create_downsample_dataset_v3(
#     configFile,
# downsampleFilePath='/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/reg_ds/',
#     ds_XY=4,
#     ds_T=1,
#     block_size=50
# )


# downsample calcliu
# downsampel membrance
# downsample reference
# downsample mask

#### extrac funct

# %%
import os


def downsample_tifs_dask(input_folder, output_folder, downsample_xy=4, downsample_t=1, n_workers=4):
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
            # Downsample T and XY
            image = image[::downsample_t, :, :, ::downsample_xy, ::downsample_xy]
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
    print(f"Processing {len(tasks)} files with {n_workers} workers...")

    from dask.diagnostics import ProgressBar

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=n_workers):
            results = dask.compute(*tasks)

    print(f"✅ Processed {len(results)} files")
    print(f"   Output folder: {output_folder}")
    return list(results)


reg_mem_path = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/reg/membrane"
output_folder = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/reg_ds/membrane"
# Example usage:
downsample_tifs_dask(
    input_folder=reg_mem_path,
    output_folder=output_folder,
    downsample_xy=4,
    downsample_t=1,
    n_workers=8,
)
