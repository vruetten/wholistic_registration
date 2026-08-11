import multiprocessing as mp
import os
import shutil

import h5py
import numpy as np
import toml

from ..utils import IO, reference, registration


####################################################################################################################
##  Define the parameters
##  haven't update the params of the reliable analysis.
def DefineParams(
    configFile="./configs/config.toml",
    inputFile=None,
    outputFile=None,
    downsampleXY=1,
    downsampleZ=-1,
    frame_downsample=1,  # We will process the dataset every {frame_downsample} frames
    frames=None,  # The frame we will process, the default is all the frames, you could also provide a input like: (t0,t1), and we will process frame[t0] ~ frame[t1] (beginning at 0), it will also be influenced by frame_downsample
    dual_channel=False,
    function="raw",
    k=0.0,
    batch_size=100,
    time_measurement="minute",  # frame
    mid_window_size=100,  # this can be in minutes or frames (depends on time_measurement)
    window_size=1,  # this can be in minutes or frames (depends on time_measurement)
    mid_stride=10,
    reference_chunk=1,  # this can be in minutes or frames (depends on time_measurement)
    preprocess=False,
    thresFactor=5,
    maskRange=[5, 4000],
    layer=3,
    r=5,
    iter=10,
    smoothPenalty=0.08,
    tolerance=1e-3,
    patch_sigma=3.0,
    offset_radius=5,
    structure_tau=0.5,
    structure_beta=0.1,
    eps=1e-6,
    save_ref=True,
    save_motion=False,
    verbose=True,
):
    """
    General parameters:
    -configFile
        Path to the configuration file. Default: './configs/config.toml'
    -inputFile
        Path to the input ND2 file containing the video data.
    -outputFile
        Path to the output directory where registered data will be saved.
    -preprocess
        Whether to normalize images to the range [0, 255]. Default: False
    -save_ref
        Whether to save reference images. Default: True
    -save_motion
        Whether to save motion fields. Default: False
    -verbose
        Whether to print detailed information during processing. Default: True

    [downsample]
    Params:
    -downsampleXY
        The coefficient of the downsampling on X or Y dimension.
        If downsampleXY = 4 & origin shape of 1 slice is (X,Y), then after downsampling the shape should be (X/4,Y/4)
    -downsampleZ
        A list of the z slices we will use.
        If downsampleZ=[4,5,6,7], we will use the 5th, 6th, 7th, 8th slices of the whole data.
        If downsampleZ=-1, we will use all of the z slices

    [channels]
    Params:
    -dual_channel
        Whether to use two channels to do registration
        If true, we will use the "membrane_channel+k*function(Ca_channel)" to do registration
    -function
        The method we process the Ca_channel. It can be: "sqrt", "log2", "log10" or "raw"
        "sqrt": square root
        "log2": log2(1+x)
        "log10": log10(1+x)
        "raw": x
    -k
        The coefficient multiplied after the transformation of Ca channel data:
        the larger this value, the greater the proportion of the calcium channel, making it more susceptible to changes in the Ca channel.
    Example:
    1.
        dual_channel=true
        function="raw"
        k=0.3
    2.
        dual_channel=true
        function="log10"
        k=300

    [reference]
    Params:
    -time_measurement
        'minute' or 'frame'
            if time_measurement='minute', then the params "window_size", "mid_window_size", and "reference_chunk" are in minutes
            if time_measurement='frame', then the params "window_size", "mid_window_size", and "reference_chunk" are in frames
    -pick_reference_auto
        Whether pick the reference image from moving image
        If true, it will pick the reference image from the moving video each several frames
        If false, you need to give a reference image
    -window_size
        The size of time we will use to process each time
    -mid_window_size
        The size of time we will use to pick the initial reference from the middle block
    -reference_chunk
        The size of frames we will use to compute reference
    -mid_stride
        We needn't use all of the frames of the middle window to compute reference, so we pick frames every mid_stride frames

    [frames]
    Params:
      -frame_downsample
          We will process the dataset every {frame_downsample} frames
          e.g.: frame_downsample=3, it means we will process frame[0], frame[3], frame[6], and so on.
      -frames
          The frame we will process, the default is all the frames, you could also provide a input like: (t0,t1),
          and we will process frame[t0] ~ frame[t1] (beginning at 0), it will also be influenced by frame_downsample
    Example:
      frame_downsample=2, frames=(0,999)
      Then we will process the frame:0, 2, 4, 6, ...,998 (total 500 frames)

    [mask]
    Params:
    -thresFactor
        The threshold value
        pixels greater than thresFactor times the standard deviation are regarded as outliers, and we will initially mask these points.
    -maskRange
        The pixel range of the mask region
        Only the pixels in the maskRange will be masked.
        Remark: This is the absolute value of the pixels, so it depends on the bit depth of the image's pixels.
        For example, if your image range is [0, 255], then your reasonable maskRange should be at least a subset of [0, 255].
        If you don't want to filter out any mask points, you can set this range to be very large.

    [pyramid]
    Params:
    -layer
        The layer of pyramid
        A larger layer means that the algorithm is better at capturing large-scale displacements
        and correspondingly, its ability to capture some small deformations will be slightly diminished.
    -r
        The radius of the patch
        The size of each patch is (2r+1)*(2r+1), and each patch has one control point.
        A smaller r means more control points and more easier to capture noise.
    -iter
        The num of maximum iterations of each layer.
    -smoothPenalty
        The coefficient of the smoothness penalty term.
        A larger smoothPenalty means more smooth motion we will get and correspondingly the error of intensity will increase
    -tolerance
        The tolerance to stop the iteration, default is 1e-3

    [processing]
    Params:
    -batch_size
        The number of frames we will process each time. If the memory of your GPU is not enough, you can try to reduce this value.
        The default value is 100

    [ReliableAnalysis]
    Params:
    -patch_sigma : float
        Gaussian smoothing applied to the image before computing patch-based SSDs.
        Larger values suppress noise but also blur fine structures.
    -offset_radius : int (pixels)
        Distance of neighborhood offsets for MIND descriptor computation.
        Determines the size of the local neighborhood used to capture structure patterns.
    -norm_percentile : float (0-100)
        Percentile used to normalize the difference map.
        Only pixels within structured regions are considered.
        Controls the sensitivity of the resulting misalignment map (lower → more sensitive).
    -structure_tau : float (0-1)
        Structure activation threshold for soft gating.
        Controls the location of the sigmoid transition applied to the
        MIND-derived structure map. Pixels with structure response
        S ≳ structure_tau are treated as foreground (structurally meaningful),
        while S ≲ structure_tau are progressively suppressed as background.
        Larger values make the mask more conservative, retaining only
        high-structure regions.
    -structure_beta : float
        Softness (slope) parameter of the structure sigmoid.
        Determines how sharp the transition is around structure_tau.
        Smaller values yield a steeper, near-binary mask (hard gating),
        while larger values produce a smoother, more tolerant weighting
        that gradually attenuates low-structure regions.
    """
    ## read the metadata
    print("Reading meta data")
    if inputFile is None:
        raise ValueError("[ERROR]inputFile must not be None. A valid file path is required.")

    elif inputFile is not None:
        meta = IO.readMeta_new(inputFile, Ifprint=verbose)
        nchannels = meta["nchannels"]
        nframes = meta["nframes"]
        data_shape = meta["data_shape"]
        resolutionxyz = meta["resolutionxyz"]
        spacing_x = meta["spacing_x"]
        spacing_y = meta["spacing_y"]
        spacing_z = meta["spacing_z"]
        framerate = meta["fps"]
        zRatio = meta["zRatio"]
        data_dtype = meta["dtype"]
        bytes_per_pixel = meta["bytes_per_pixel"]

    ## load the default config file
    import os

    print("Loading the default config file")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(current_dir), "configs", "config_default.toml")
    config = toml.load(config_path)

    # change the meta data
    config["MetaData"]["nchannels"] = nchannels
    config["MetaData"]["zRatio"] = zRatio
    config["MetaData"]["SIZE"] = data_shape
    config["MetaData"]["frames"] = nframes
    config["MetaData"]["dtype"] = str(data_dtype)  # Store as string for TOML compatibility
    config["MetaData"]["bytes_per_pixel"] = bytes_per_pixel
    # Calculate single frame dimension
    if len(data_shape) == 5:
        # 5D data: (T, Z, C, Y, X), check Z dimension (index 1)
        z_dim = data_shape[1]
        if z_dim > 1:
            # 3D single frame: (Z, Y, X)
            single_frame_dim = 3
        else:
            # 2D single frame: (Y, X), Z=1 is just padding
            single_frame_dim = 2
    else:
        raise ValueError("data_shape should be 5D (T, Z, C, Y, X)")
    config["MetaData"]["Dim"] = single_frame_dim
    config["MetaData"]["voxelsize"] = resolutionxyz
    config["MetaData"]["fps"] = framerate
    config["MetaData"]["spacing_x"] = spacing_x
    config["MetaData"]["spacing_y"] = spacing_y
    config["MetaData"]["spacing_z"] = spacing_z

    # change the downsample config
    config["downsample"]["downsampleXY"] = downsampleXY
    config["downsample"]["downsampleZ"] = downsampleZ

    # change the frames config
    if frames is None:
        config["frames"]["frames"] = list(range(0, nframes, frame_downsample))
    else:
        config["frames"]["frames"] = list(range(frames[0], frames[1], frame_downsample))

    config["frames"]["frame_downsample"] = frame_downsample

    # change the filepath
    config["file_path"]["input_path"] = inputFile
    config["file_path"]["registrated_path"] = outputFile
    config["file_path"]["mask_path"] = outputFile + "/mask"

    # change the dual_channels config
    config["channels"]["dual_channel"] = dual_channel
    config["channels"]["function"] = function
    config["channels"]["k"] = k

    # change the reference config
    config["reference"]["time_measurement"] = time_measurement  # default is 'minute'
    config["reference"]["pick_reference_auto"] = False
    config["reference"]["mid_window_size"] = mid_window_size
    config["reference"]["window_size"] = window_size
    config["reference"]["mid_stride"] = mid_stride
    config["reference"]["reference_chunk"] = reference_chunk
    if reference_chunk > window_size:
        raise ValueError("[ERROR]reference_chunk should be less than or equal to window_size.")
    if time_measurement == "minute":
        mid_window_size_minutes = mid_window_size
        window_size_minutes = window_size
        reference_chunk_minutes = reference_chunk

        mid_window_size_frames = int(mid_window_size * 60 * framerate)
        window_size_frames = int(window_size * 60 * framerate)
        reference_chunk_frames = int(reference_chunk * 60 * framerate)

    elif time_measurement == "frame":
        mid_window_size_minutes = mid_window_size / (60 * framerate)
        window_size_minutes = window_size / (60 * framerate)
        reference_chunk_minutes = reference_chunk / (60 * framerate)

        mid_window_size_frames = mid_window_size
        window_size_frames = window_size
        reference_chunk_frames = reference_chunk
    else:
        raise ValueError("[ERROR]time_measurement should be 'minute' or 'frame'.")
    # change the preprocess config
    config["preprocess"]["normalize_to_0_255"] = preprocess

    # change the processing config
    config["processing"]["batch_size"] = batch_size

    # change the mask config
    config["mask"]["thresFactor"] = thresFactor
    config["mask"]["maskRange"] = maskRange

    # change the pyramid config
    config["pyramid"]["layer"] = layer
    config["pyramid"]["r"] = r
    config["pyramid"]["iter"] = iter
    config["pyramid"]["smoothPenalty"] = smoothPenalty
    config["pyramid"]["tolerance"] = tolerance

    # change the ReliableAnalysis config
    config["ReliableAnalysis"]["patch_sigma"] = patch_sigma
    config["ReliableAnalysis"]["offset_radius"] = offset_radius
    config["ReliableAnalysis"]["structure_tau"] = structure_tau
    config["ReliableAnalysis"]["structure_beta"] = structure_beta
    config["ReliableAnalysis"]["eps"] = eps

    # change the save config
    config["save_config"]["save_ref"] = save_ref
    config["save_config"]["save_motion"] = save_motion

    print("[INFO]Config created =====> Saving the config")
    # Create directory if it doesn't exist
    config_dir = os.path.dirname(os.path.abspath(configFile))
    if config_dir and not os.path.exists(config_dir):
        IO.reset_dir(config_dir)
        if verbose:
            print(f"[INFO]Created directory: {config_dir}")

    with open(configFile, "w") as f:
        toml.dump(config, f)
    print("[INFO]Config saved =====> Done")
    if verbose == True:
        print("\nConfiguration summary:")
        print("--------------------------------------------------")

        print("[MetaData]")
        print(f"  zRatio: {config['MetaData']['zRatio']}")
        print(f"  SIZE:   {config['MetaData']['SIZE']}")
        print(f"  frames: {config['MetaData']['frames']}")
        print(f"  Dim:    {config['MetaData']['Dim']}")
        print(f"  voxelsize: {config['MetaData']['voxelsize']}")
        print(f"  frame rate: {config['MetaData']['fps']} fps")

        print("\n[Downsample]")
        print(f"  XY: {config['downsample']['downsampleXY']}")
        if config["downsample"]["downsampleZ"] == -1:
            print("  Z : All the z slices")
        else:
            print(f"  Z : {config['downsample']['downsampleZ']}")

        print("\n[File Path]")
        print(f"  input_path :  {config['file_path']['input_path']}")
        print(f"  registrated_path: {config['file_path']['registrated_path']}")

        print("\n[Channels]")
        print(f"  dual_channel : {config['channels']['dual_channel']}")
        print(f"  function     : {config['channels']['function']}")
        print(f"  k            : {config['channels']['k']}")

        print("\n[Processing]")
        print(f"  batch_size        : {config['processing']['batch_size']} frames per batch")

        print("\n[Frames]")
        print(
            f" Processing frames: from frame {config['frames']['frames'][0]} to frame {config['frames']['frames'][-1]} with stride {frame_downsample}, totoal frames is {len(config['frames']['frames'])}"
        )

        print("\n[Reference]")
        print(
            f"  mid_window_size      : {mid_window_size_minutes} minutes {mid_window_size_frames} frames (process {int(mid_window_size_frames / frame_downsample)} frames)"
        )
        print(
            f"  window_size          : {window_size_minutes} minutes {window_size_frames} frames (process {int(window_size_frames / frame_downsample)} frames)"
        )
        print(
            f"  reference_chunk      : {reference_chunk_minutes} minutes {reference_chunk_frames} frames (use {int(reference_chunk_frames / frame_downsample)} frames)"
        )

        print("\n[Preprocess]")
        print(f"  normalize_to_0_255 : {config['preprocess']['normalize_to_0_255']}")

        print("\n[Mask]")
        print(f"  thresFactor : {config['mask']['thresFactor']}")
        print(f"  maskRange   : {config['mask']['maskRange']}")

        print("\n[Pyramid]")
        print(f"  layer         : {config['pyramid']['layer']}")
        print(f"  r             : {config['pyramid']['r']}")
        print(f"  iter          : {config['pyramid']['iter']}")
        print(f"  smoothPenalty : {config['pyramid']['smoothPenalty']}")
        print(f"  tolerance     : {config['pyramid']['tolerance']}")

        print("\n[ReliableAnalysis]")
        print(f"  patch_sigma(0~1)       : {config['ReliableAnalysis']['patch_sigma']}")
        print(f"  offset_radius          : {config['ReliableAnalysis']['offset_radius']}")
        print(f"  structure_tau(0~1)     : {config['ReliableAnalysis']['structure_tau']}")
        print(f"  structure_beta         : {config['ReliableAnalysis']['structure_beta']}")

        # ========== Resource Estimation Section ==========
        print("\n[Resource Estimation]")

        # Get data dimensions
        T, Z, C, Y, X = config["MetaData"]["SIZE"]
        fps = config["MetaData"]["fps"]
        bytes_per_pixel = config["MetaData"]["bytes_per_pixel"]
        data_dtype = config["MetaData"]["dtype"]

        # Calculate downsampled dimensions
        ds_xy = config["downsample"]["downsampleXY"]
        ds_z = config["downsample"]["downsampleZ"]
        if ds_z == -1:
            Z_down = Z
        else:
            Z_down = len(ds_z) if isinstance(ds_z, list) else Z

        Y_down = Y // ds_xy
        X_down = X // ds_xy

        # Single frame size (one channel, downsampled)
        single_frame_bytes = Z_down * Y_down * X_down * bytes_per_pixel
        single_frame_mb = single_frame_bytes / (1024**2)
        single_frame_gb = single_frame_bytes / (1024**3)

        # Mid window calculation (for initial reference)
        mid_window = config["reference"]["mid_window_size"]
        mid_stride = config["reference"]["mid_stride"]
        if config["reference"]["time_measurement"] == "minute":
            mid_window_frames = int(mid_window * 60 * fps)
        elif config["reference"]["time_measurement"] == "frame":
            mid_window_frames = int(mid_window)
        else:
            raise ValueError("[ERROR]Invalid time_measurement in reference config.")
        mid_window_frames_used = mid_window_frames // mid_stride  # Frames actually loaded

        # GPU memory for mid_window reference calculation (both channels loaded)
        n_channels_loaded = 2 if config["channels"]["dual_channel"] else 1
        mid_window_gpu_bytes = mid_window_frames_used * single_frame_bytes * n_channels_loaded
        mid_window_gpu_gb = mid_window_gpu_bytes / (1024**3)

        # Batch processing memory (per batch)
        batch_size = config["processing"]["batch_size"]
        batch_gpu_bytes = batch_size * single_frame_bytes * n_channels_loaded
        batch_gpu_gb = batch_gpu_bytes / (1024**3)

        # Registration working memory (rough estimate: 3-5x single frame for intermediate arrays)
        registration_overhead_factor = 5
        registration_working_gb = single_frame_gb * registration_overhead_factor * n_channels_loaded

        # Total GPU memory estimate
        total_gpu_needed_mid = mid_window_gpu_gb + registration_working_gb
        total_gpu_needed_batch = batch_gpu_gb + registration_working_gb

        # CPU/System memory (need to hold full dataset + working memory)
        full_dataset_bytes = T * Z * Y * X * bytes_per_pixel * C
        full_dataset_gb = full_dataset_bytes / (1024**3)
        cpu_working_memory_gb = (
            max(total_gpu_needed_mid, total_gpu_needed_batch) * 2
        )  # Buffer for CPU operations
        total_cpu_recommended = full_dataset_gb + cpu_working_memory_gb

        print(f"  Data dimensions (T,Z,C,Y,X): {(T, Z, C, Y, X)}")
        print(f"  Data type                 : {data_dtype} ({bytes_per_pixel} bytes/pixel)")
        print(
            f"  Downsampled frame size    : {Z_down} x {Y_down} x {X_down} = {single_frame_mb:.1f} MB/frame/channel"
        )
        print("  ")
        print("  --- Initial Reference (mid_window) ---")
        print(
            f"  Mid window: {mid_window_size_minutes} min = {mid_window_frames} frames (using every {mid_stride}th = {mid_window_frames_used} frames)"
        )
        print(
            f"  GPU memory for mid_window : {mid_window_gpu_gb:.1f} GB ({n_channels_loaded} channel(s))"
        )
        print(f"  + Registration overhead   : ~{registration_working_gb:.1f} GB")
        print(f"  = Total GPU needed (ref)  : ~{total_gpu_needed_mid:.1f} GB")
        print("  ")
        print("  --- Batch Processing ---")
        print(f"  Batch size: {batch_size} frames")
        print(
            f"  GPU memory per batch      : {batch_gpu_gb:.1f} GB ({n_channels_loaded} channel(s))"
        )
        print(f"  + Registration overhead   : ~{registration_working_gb:.1f} GB")
        print(f"  = Total GPU needed (batch): ~{total_gpu_needed_batch:.1f} GB")
        print("  ")
        print("  --- System (CPU) Memory ---")
        print(f"  Full dataset size         : {full_dataset_gb:.1f} GB")
        print(f"  Recommended system RAM    : ~{total_cpu_recommended:.0f} GB")
        print("  ")

        # Warnings
        if total_gpu_needed_mid > 40:
            print(f"  ⚠️  WARNING: mid_window requires {total_gpu_needed_mid:.1f} GB GPU memory!")
            print(
                "      Consider: reducing mid_window_size, increasing mid_stride, or increasing downsampleXY"
            )
        if total_gpu_needed_batch > 40:
            print(
                f"  ⚠️  WARNING: batch processing requires {total_gpu_needed_batch:.1f} GB GPU memory!"
            )
            print("      Consider: reducing batch_size or increasing downsampleXY")

        print("--------------------------------------------------")
        print("[INFO]Configuration loaded successfully.\n")


####################################################################################################################
##  main process
def Registration_v3(configPath="./configs/config.toml", parallel=True):
    """
    Full 2D/3D registration pipeline for multi-channel volumetric time-lapse data.

    Parameters
    ----------
    configPath : str, default='./configs/config.toml'
        Path to the configuration TOML file generated by `DefineParams`.
    parallel : bool, default=True
        Whether to run forward/backward registration in parallel using multiprocessing.
        If True, uses GPU 0 and GPU 1 (if available) for backward and forward processing.

    Returns
    -------
    None
        The function writes outputs to disk:
        - Registered volumes per channel as OME-TIFF
          Naming: vol_chN_XXXXXX.tif (6-digit frame index)
        - Reference images (if save_ref=True)
        - Motion fields (if save_motion=True) as HDF5

    Notes
    -----
    Pipeline Steps:
    1. Load configuration and metadata from configPath.
    2. Determine dataset dimensions and downsampling factors.
    3. Compute initial reference image from middle block.
    4. Register middle block frames in batches.
    5. Maintain reference windows for forward/backward processing.
    6. Process frames forward and backward from middle chunk (serial or parallel).
    7. Save registered volumes, reference images, and motion fields.

    Example
    -------
    Registration_v3(configPath='./configs/config.toml', parallel=True)
    """

    # For CUDA compatibility, use 'spawn' or 'forkserver'
    mp.set_start_method("spawn", force=True)
    # user-provided save function (assumed imported already)
    # load config
    config = toml.load(configPath)

    # basic params
    output_root = config["file_path"]["registrated_path"]
    movingFilePath = config["file_path"]["input_path"]

    # get basic meta info
    Dim = config["MetaData"]["Dim"]
    total_frames = int(config["MetaData"]["frames"])
    dual_channel = config["channels"]["dual_channel"]

    # frames to process
    process_frames = config["frames"]["frames"]
    frame_downsample = config["frames"]["frame_downsample"]
    total_process_frames = len(process_frames)

    # downsample params and the frames we need to process
    downsampleXY = config["downsample"]["downsampleXY"]
    downsampleZ = config["downsample"]["downsampleZ"]
    window_size = config["reference"]["window_size"]

    if config["reference"]["time_measurement"] == "minute":
        window_size_frames = int(
            window_size * 60 * config["MetaData"]["fps"] / frame_downsample
        )  # convert minutes to frames
    elif config["reference"]["time_measurement"] == "frame":
        window_size_frames = int(window_size / frame_downsample)  # already in frames
    else:
        raise ValueError("[ERROR]Invalid time_measurement in reference config.")

    # Process downsampleZ: if -1, use all Z slices
    if Dim == 3:
        if downsampleZ == -1:
            SIZE = config["MetaData"]["SIZE"]
            Z_full = int(SIZE[1])
            downsampleZ = list(range(Z_full))
    elif Dim == 2:
        downsampleZ = [0]
    else:
        raise ValueError("[ERROR]Invalid Dim in MetaData config.")
    # prepare output directories (one folder per channel)
    print("[INFO]Preparing output directories...")
    out_mem = os.path.join(output_root, "membrane")
    out_ref = os.path.join(output_root, "reference")
    out_mot = os.path.join(output_root, "motion")
    IO.reset_dir(out_mem)
    if dual_channel:
        out_ca = os.path.join(output_root, "calcium")
        out_ca_ref = os.path.join(output_root, "calcium_reference")
        IO.reset_dir(out_ca)
        IO.reset_dir(out_ca_ref)
    else:
        out_ca = None
    save_ref = bool(config["save_config"]["save_ref"])
    save_motion = bool(config["save_config"]["save_motion"])

    if save_ref:
        IO.reset_dir(out_ref)
    if save_motion:
        IO.reset_dir(out_mot)

    # Channel index mapping for file naming
    # Using channel indices as they appear in the ND2 file
    # membrane channel = 1 in ND2 file, calcium channel = 0 in ND2 file
    channel_index_map = {"membrane": "ch1", "calcium": "ch0"}

    # ---------------------------
    # Step A: Process the middle chunk to build initial reference
    # ---------------------------
    print("[INFO]Processing middle chunk to build initial reference...")
    batch_size = config["processing"][
        "batch_size"
    ]  # Frames per reference update and process frames to save memory
    mid_window_size = int(
        config["reference"]["mid_window_size"]
    )  # Window size for initial reference in minutes
    if config["reference"]["time_measurement"] == "minute":
        mid_window_size_frames = int(
            mid_window_size * 60 * config["MetaData"]["fps"] / frame_downsample
        )  # Convert to frames
    elif config["reference"]["time_measurement"] == "frame":
        mid_window_size_frames = int(mid_window_size / frame_downsample)  # Already in frames\
        mid_window_size = mid_window_size / (
            60 * config["MetaData"]["fps"]
        )  # Convert to minutes for printing
    else:
        raise ValueError("[ERROR]Invalid time_measurement in reference config.")
    # Calculate the middle window position
    half_chunk = mid_window_size_frames // 2
    total_mid = total_process_frames // 2  # Middle frame of entire dataset
    # Clamp the window to the dataset: a mid_window_size larger than the
    # recording would otherwise make mid_start negative (Python indexing wraps,
    # silently pulling end-of-recording frames into the "middle" reference)
    # and mid_end overrun into an IndexError.
    mid_start = max(0, total_mid - half_chunk)  # Start frame of middle window
    mid_end = min(mid_start + mid_window_size_frames, total_process_frames)  # End frame

    # Read middle block from ND2 file
    # Channel 1 = membrane, channel 0 = calcium
    mid_stride = int(config["reference"]["mid_stride"])
    frames_mid = list(
        range(mid_start, min(mid_end + 1, len(process_frames)))
    )  # List of frames in middle window

    # load all of the frames to the memory is so memory-comsuming, so we downsample the frames
    mem_mid_downsample = IO.readND2Frame(
        movingFilePath,
        [process_frames[i] for i in list(range(mid_start, mid_end, mid_stride))],
        downsampleZ,
        channel=1,
        xy_down=downsampleXY,
        verbose=False,
    )
    # Remove singleton dimensions
    mem_mid_downsample = np.squeeze(mem_mid_downsample)

    if dual_channel:
        ca_mid_downsample = IO.readND2Frame(
            movingFilePath,
            [process_frames[i] for i in list(range(mid_start, mid_end, mid_stride))],
            downsampleZ,
            channel=0,
            xy_down=downsampleXY,
            verbose=False,
        )
        ca_mid_downsample = np.squeeze(ca_mid_downsample)
    else:
        ca_mid_downsample = None
    print(
        f"[INFO]Loaded middle block frames {process_frames[frames_mid[0]]} to {process_frames[frames_mid[-1]]} with frames downsample {frame_downsample} ({mid_window_size} minutes)"
    )

    # Compute initial reference image from middle block
    ref_img, indsort = reference.compute_reference_from_block(
        mem_mid_downsample, config, ca_mid_downsample
    )
    IO.write_volume_as_ome_tiff(
        ref_img,
        out_ref,
        "ref",
        f"{process_frames[frames_mid[0]]}_{process_frames[frames_mid[-1]]}",
        configPath,
    )
    if dual_channel:
        IO.write_volume_as_ome_tiff(
            np.mean(ca_mid_downsample[indsort, :], axis=0),
            out_ref,
            "Ca_ref",
            f"{process_frames[frames_mid[0]]}_{process_frames[frames_mid[-1]]}",
            configPath,
        )
    print(
        f"[INFO]Computed initial reference image from middle block frames {process_frames[frames_mid[0]]} to {process_frames[frames_mid[-1]]} ({mid_window_size} minutes)"
    )

    # Register the middle block using either 2D or 3D registration based on dataset dimension
    print(f"[INFO]Registering middle block...(every {batch_size} frames as a batch)")

    # storage the head and tail of the middle block, we will use them when we process forward and backward
    from collections import deque

    reference_chunk = config["reference"]["reference_chunk"]
    if config["reference"]["time_measurement"] == "minute":
        reference_chunk = int(
            reference_chunk * 60 * config["MetaData"]["fps"] / frame_downsample
        )  # convert minutes to frames
    elif config["reference"]["time_measurement"] == "frame":
        reference_chunk = int(reference_chunk / frame_downsample)  # already in frames
    else:
        raise ValueError("[ERROR]Invalid time_measurement in reference config.")

    head_mem = deque(maxlen=reference_chunk)
    head_ca = deque(maxlen=reference_chunk)
    tail_mem = deque(maxlen=reference_chunk)
    tail_ca = deque(maxlen=reference_chunk)
    num_frames = len(frames_mid)

    for i in range(0, num_frames, batch_size):
        batch_frames = [
            process_frames[j]
            for j in list(range(mid_start + i, mid_start + min(num_frames, i + batch_size)))
        ]
        mem_batch = IO.readND2Frame(
            movingFilePath,
            batch_frames,
            downsampleZ,
            channel=1,
            xy_down=downsampleXY,
            verbose=False,
        )
        mem_batch = np.squeeze(mem_batch)

        if dual_channel:
            ca_batch = IO.readND2Frame(
                movingFilePath,
                batch_frames,
                downsampleZ,
                channel=0,
                xy_down=downsampleXY,
                verbose=False,
            )
            ca_batch = np.squeeze(ca_batch)
        else:
            ca_batch = None
        if Dim == 3 and len(downsampleZ) > 1:
            mem_reg, ca_reg, _, _, motion = registration.wbi_registration_3d(
                mem_batch, configPath, ref_img, frame=batch_frames, moving_Ca_image=ca_batch
            )
        else:
            mem_reg, ca_reg, _, _, motion = registration.wbi_registration_2d(
                mem_batch, configPath, ref_img, frame=batch_frames, moving_Ca_image=ca_batch
            )
        if i == 0:
            motion_start = motion[0].copy()
        if i + batch_size >= num_frames:
            motion_end = motion[-1].copy()
        # save immediately
        for k, fid in enumerate(batch_frames):
            IO.write_volume_as_ome_tiff(
                mem_reg[k], out_mem, channel_index_map["membrane"], fid, configPath
            )
            if dual_channel:
                IO.write_volume_as_ome_tiff(
                    ca_reg[k], out_ca, channel_index_map["calcium"], fid, configPath
                )

            if save_motion:
                with h5py.File(os.path.join(out_mot, f"motion_{fid:06d}.h5"), "w") as hf:
                    hf.create_dataset("motion", data=motion[k], compression="gzip")

        # ---------- maintain reference windows ----------
        if i == 0:
            for k in range(len(mem_reg)):
                if len(head_mem) < reference_chunk:
                    head_mem.append(mem_reg[k])
                    if dual_channel:
                        head_ca.append(ca_reg[k])

        for k in range(len(mem_reg)):
            tail_mem.append(mem_reg[k])
            if dual_channel:
                tail_ca.append(ca_reg[k])

        del mem_batch, ca_batch, mem_reg, ca_reg, motion

    # ---------------------------
    # Step B & C: Process forward and backward from the middle chunk
    # ---------------------------

    ## Detect GPU availability using CuPy
    try:
        from utils import cp

        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()  # Get number of available GPUs
            print(f"[INFO]Detected {num_gpus} GPU(s).")
        except Exception:
            num_gpus = 0
            print("[ERROR]Failed to query GPU devices via CuPy.")
            print("[INFO]Falling back to serial mode.")
    except ImportError:
        num_gpus = 0
        print("[ERROR]CuPy is not installed or failed to import.")
        print("[INFO]Falling back to serial mode.")

    print(
        f"[INFO]Processing forward and backward every {window_size} minutes ({window_size_frames} frames)"
    )

    # =========================
    # Process in serial or parallel mode
    # =========================
    if not parallel or num_gpus < 2:
        # Serial processing: process backward direction first, then forward direction
        print("[INFO]Running in serial mode...")

        # Process backward from middle chunk to beginning (frames mid_start to 0)
        process_directional_chunks(
            direction="backward",  # Process frames in reverse order
            frame_list=process_frames,
            start_frame=mid_start,  # Starting from the beginning of the middle chunk
            end_frame=0,  # Ending at the first frame of the dataset
            init_mem_windows=head_mem,  # Initial membrane reference windows
            init_ca_windows=head_ca,  # Initial calcium reference windows
            device_id=None,  # Use CPU (no GPU)
            configPath=configPath,  # Path to configuration file
            config=config,  # Configuration dictionary
            movingFilePath=movingFilePath,  # Path to input ND2 file
            out_mem=out_mem,  # Output directory for membrane channel
            out_ca=out_ca,  # Output directory for calcium channel
            out_ref=out_ref,  # Output directory for reference images
            out_mot=out_mot,  # Output directory for motion fields
            channel_index_map=channel_index_map,  # Mapping of channel names to indices
            save_motion=save_motion,  # Whether to save motion fields
            save_ref=save_ref,  # Whether to save reference images
            Dim=Dim,  # Dimensionality (2D or 3D)
            window_size_frames=window_size_frames,  # Window size in frames
            downsampleZ=downsampleZ,  # Z downsampling parameters
            downsampleXY=downsampleXY,  # XY downsampling factor
            batch_size=batch_size,  # Batch size for processing
            reference_chunk=reference_chunk,  # Reference chunk size
            motion_init=motion_start,
        )

        # Process forward from middle chunk to end (frames mid_end+1 to total_frames)
        process_directional_chunks(
            direction="forward",  # Process frames in normal order
            frame_list=process_frames,
            # mid_end + 1, not mid_end: `frames_mid` above is INCLUSIVE of mid_end
            # (range(mid_start, mid_end + 1)) so the middle block already registered
            # that frame against the high-quality middle reference, while
            # build_chunks forward covers range(start_frame, end_frame). Starting at
            # mid_end would re-register it against the rolling window and overwrite
            # the better result.
            start_frame=mid_end + 1,  # Starting just after the end of the middle chunk
            end_frame=total_process_frames,  # Ending at the last frame of the dataset
            init_mem_windows=tail_mem,  # Initial membrane reference windows
            init_ca_windows=tail_ca,  # Initial calcium reference windows
            device_id=None,  # Use CPU (no GPU)
            configPath=configPath,
            config=config,
            movingFilePath=movingFilePath,
            out_mem=out_mem,
            out_ca=out_ca,
            out_ref=out_ref,
            out_mot=out_mot,
            channel_index_map=channel_index_map,
            save_motion=save_motion,
            save_ref=save_ref,
            Dim=Dim,
            window_size_frames=window_size_frames,
            downsampleZ=downsampleZ,
            downsampleXY=downsampleXY,
            batch_size=batch_size,  # Batch size for processing
            reference_chunk=reference_chunk,  # Reference chunk size
            motion_init=motion_end,
        )

    else:
        # Parallel processing: process both directions simultaneously
        print("[INFO]Running in parallel mode...")

        # Create process for backward direction processing
        p_bwd = mp.Process(
            target=process_directional_chunks,
            kwargs=dict(
                direction="backward",
                frame_list=process_frames,
                start_frame=mid_start,
                end_frame=0,
                init_mem_windows=head_mem,
                init_ca_windows=head_ca,
                device_id=0,  # Use GPU 0 for backward processing
                configPath=configPath,
                config=config,
                movingFilePath=movingFilePath,
                out_mem=out_mem,
                out_ca=out_ca,
                out_ref=out_ref,
                out_mot=out_mot,
                channel_index_map=channel_index_map,
                save_motion=save_motion,
                save_ref=save_ref,
                Dim=Dim,
                window_size_frames=window_size_frames,
                downsampleZ=downsampleZ,
                downsampleXY=downsampleXY,
                batch_size=batch_size,  # Batch size for processing
                reference_chunk=reference_chunk,  # Reference chunk size
                motion_init=motion_start,
            ),
        )

        # Create process for forward direction processing
        p_fwd = mp.Process(
            target=process_directional_chunks,
            kwargs=dict(
                direction="forward",
                frame_list=process_frames,
                start_frame=mid_end + 1,  # see the serial branch: frames_mid includes mid_end
                end_frame=total_process_frames,
                init_mem_windows=tail_mem,
                init_ca_windows=tail_ca,
                device_id=1,  # Use GPU 1 for forward processing
                configPath=configPath,
                config=config,
                movingFilePath=movingFilePath,
                out_mem=out_mem,
                out_ca=out_ca,
                out_ref=out_ref,
                out_mot=out_mot,
                channel_index_map=channel_index_map,
                save_motion=save_motion,
                save_ref=save_ref,
                Dim=Dim,
                window_size_frames=window_size_frames,
                downsampleZ=downsampleZ,
                downsampleXY=downsampleXY,
                batch_size=batch_size,  # Batch size for processing
                reference_chunk=reference_chunk,  # Reference chunk size
                motion_init=motion_end,
            ),
        )

        p_bwd.start()
        p_fwd.start()
        p_bwd.join()
        p_fwd.join()


def ReliableAnalysis(
    configPath: str = None,
):
    """
    Main entry function for computing spatial, temporal, and accumulative reliability masks.

    Loads configuration, defines correlation function, and calls ComputMask
    to compute reliability masks.

    Directory structure under `registrated_path` must be:
        root_dir/
            ├── membrane/      # Registered membrane images (channel 1) (e.g. vol_ch1_000000.tif)
            ├── calcium/       # Registered calcium images (channel 0) (e.g. vol_ch0_000000.tif)
            └── reference/     # Reference images used for registration (e.g. vol_ch0_000000.tif)

    And after running this function, we will add a folder named 'mask' under root_dir

    Parameters
    ----------
    configPath : str
        Path to TOML configuration file.

    Returns
    -------
    None

    Notes
    -----
    - For dual-channel data, calcium channel is transformed using config['channels']['function']
      and weighted by config['channels']['k'] before correlation.
    - ComputMask handles the main mask computation, including SSIM comparison with references.
    """
    # Validate input parameters
    if configPath is None:
        raise ValueError("configPath must be provided")

    # Load configuration parameters
    config = toml.load(configPath)

    # Define directory paths
    root_dir = config["file_path"]["registrated_path"]
    out_dir = config["file_path"]["mask_path"]

    # Create subdirectory paths using os.path.join for platform compatibility
    mem_dir = os.path.join(root_dir, "membrane")  # Channel 1 in registration process
    ca_dir = os.path.join(root_dir, "calcium")  # Channel 0 in registration process
    ref_dir = os.path.join(root_dir, "reference")

    # Count total number of frames from membrane channel directory
    frames = sorted(os.listdir(mem_dir))
    T = config["MetaData"]["frames"]
    dual_channel = config["channels"]["dual_channel"]

    # Define correlation function based on channel configuration
    def compute_cor_fn(mem, ca):
        """
        Computes correlation map from membrane and calcium channels.

        Parameters:
        -----------
        mem : np.ndarray
            Membrane channel frame (shape: Z, Y, X)
        ca : np.ndarray
            Calcium channel frame (shape: Z, Y, X)

        Returns:
        --------
        cor : np.ndarray
            Correlation map (shape: Z, Y, X)
        """
        if config["channels"]["dual_channel"]:
            # For dual-channel data, transform calcium and combine with membrane
            k = config["channels"]["k"]  # Weight factor for calcium channel
            function = config["channels"]["function"]  # Transformation function
            ca_transformed = registration.transform(ca, k, function)
            cor = mem + ca_transformed
        else:
            # For single-channel data, use only membrane channel
            cor = mem
        return cor

    # Import ComputMask function from utils.reliableAnalysis module
    from ..utils.reliableAnalysis import ComputeMask_v2

    # Call ComputMask to perform actual mask computation
    ComputeMask_v2(
        mem_dir=mem_dir,
        ca_dir=ca_dir,
        ref_dir=ref_dir,
        out_dir=out_dir,
        dual_channel=dual_channel,
        frames=config["frames"]["frames"],
        config=config["ReliableAnalysis"],
        compute_cor_fn=compute_cor_fn,
        configPath=configPath,
    )


####################################################################################################################
##  create downsample dataset
def create_downsample_dataset_v3(
    configPath="./configs/config.toml",
    downsampleFilePath="./registrated_downsample",
    ds_XY=1,
    ds_T=1,
    block_size=50,
    verbose=True,
):
    """
    Create a downsampled dataset (v3) from registered and raw ND2 data.

    This version:
    - Uses lazy dask arrays for TIFF series to reduce peak memory usage
    - Computes per block to avoid loading all frames at once
    - Outputs membrane, calcium, and mask volumes
    - Maintains XY and time downsampling

    Parameters
    ----------
    configPath : str
        Path to TOML configuration file (must contain input_path, registrated_path, mask_path, and metadata).
    downsampleFilePath : str
        Output directory for downsampled dataset.
    ds_XY : int
        XY downsampling factor applied on top of base downsample in config.
    ds_T : int
        Temporal downsampling factor (frames skipped along time axis).
    block_size : int
        Number of frames to process per block to reduce memory consumption.
    verbose : bool
        Whether to print progress information.

    Outputs
    -------
    downsampleFilePath/
        membrane/    : raw_membrane + registered_membrane + reference
        calcium/     : raw_calcium + registered_calcium
        mask/        : reference + registered_membrane + mask

    Usage
    -----
    create_downsample_dataset_v3(
        configPath='./configs/config.toml',
        downsampleFilePath='./registrated_downsample',
        ds_XY=4, ds_T=2, block_size=50, verbose=True
    )
    """
    # ---------------- load config ----------------
    config = toml.load(configPath)
    raw_path = config["file_path"]["input_path"]
    reg_path = config["file_path"]["registrated_path"]
    mask_path = config["file_path"]["mask_path"]
    total_frames = config["MetaData"]["frames"]
    dual_channel = config["channels"]["dual_channel"]
    base_dsXY = config["downsample"]["downsampleXY"]
    base_dsZ = config["downsample"]["downsampleZ"]
    if config["MetaData"]["Dim"] == 3:
        base_dsZ = list(range(config["MetaData"]["SIZE"][1]))
    elif config["MetaData"]["Dim"] == 2:
        base_dsZ = [0]
    else:
        raise ValueError(
            f"[ERROR] Invalid Dim value in config. Must be 2 or 3. Dim={config['MetaData']['Dim']}"
        )

    raw_dsXY = base_dsXY * ds_XY

    # ---------------- output dirs ----------------
    if os.path.exists(downsampleFilePath):
        if verbose:
            print(f"[INFO] Removing existing directory: {downsampleFilePath}")
        shutil.rmtree(downsampleFilePath)

    mem_out_dir = os.path.join(downsampleFilePath, "membrane")
    cal_out_dir = os.path.join(downsampleFilePath, "calcium")
    mask_out_dir = os.path.join(downsampleFilePath, "mask")
    IO.reset_dir(mem_out_dir)
    IO.reset_dir(cal_out_dir)
    IO.reset_dir(mask_out_dir)
    # ---------------- time indices ----------------
    time_index = config["frames"]["frames"][::ds_T]
    T_ds = len(time_index)
    if verbose:
        print(f"[INFO] Total frames after T-downsample: {T_ds}")

    # ---------------- build reference index ----------------
    from ..utils.reliableAnalysis import build_reference_index

    reg_mem_path = os.path.join(reg_path, "membrane")
    reg_cal_path = os.path.join(reg_path, "calcium")
    reg_ref_path = os.path.join(reg_path, "reference")
    ref_map, _ = build_reference_index(reg_ref_path)

    # ---------------- build lazy dask arrays ----------------
    if verbose:
        print("[INFO] Building lazy downsampled dask arrays...")
    reg_mem_ds = IO.downsample_tiff_series(
        [os.path.join(reg_mem_path, f"vol_ch1_{i:06d}.tif") for i in time_index],
        xy_down=ds_XY,
        batch_processing=False,
        verbose=verbose,
    )
    reg_cal_ds = IO.downsample_tiff_series(
        [os.path.join(reg_cal_path, f"vol_ch0_{i:06d}.tif") for i in time_index],
        xy_down=ds_XY,
        batch_processing=False,
        verbose=verbose,
    )
    reg_ref_ds = IO.downsample_tiff_series(
        [ref_map[i] for i in time_index], xy_down=ds_XY, batch_processing=False, verbose=verbose
    )
    mask_ds = IO.downsample_tiff_series(
        [os.path.join(mask_path, f"vol_mask_{i:06d}.tif") for i in time_index],
        xy_down=ds_XY,
        batch_processing=False,
        verbose=verbose,
    )

    # ---------------- block-wise compute & write ----------------
    num_blocks = (T_ds + block_size - 1) // block_size
    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, T_ds)
        if verbose:
            print(f"[INFO] Processing block {b + 1}/{num_blocks} frames [{start}:{end}]")

        # ---- compute this block only ----
        reg_mem_block = reg_mem_ds[start:end].compute()
        reg_cal_block = reg_cal_ds[start:end].compute()
        reg_ref_block = reg_ref_ds[start:end].compute()
        mask_block = mask_ds[start:end].compute()

        # Keep single channel only if necessary
        if reg_mem_block.ndim == 4:
            reg_mem_block = reg_mem_block[..., 0]
        if reg_cal_block.ndim == 4:
            reg_cal_block = reg_cal_block[..., 0]
        if reg_ref_block.ndim == 4:
            reg_ref_block = reg_ref_block[..., 0]
        if mask_block.ndim == 4:
            mask_block = mask_block[..., 0]

        # ---- raw ND2 ----
        raw_frames = time_index[start:end]
        try:
            raw_mem = IO.readND2Frame(
                raw_path,
                frames=raw_frames,
                slices=base_dsZ,
                channel=1,
                xy_down=raw_dsXY,
                verbose=False,
            )
            raw_mem = np.squeeze(raw_mem, axis=0) if raw_mem.shape[0] == 1 else raw_mem

            raw_cal = IO.readND2Frame(
                raw_path,
                frames=raw_frames,
                slices=base_dsZ,
                channel=0,
                xy_down=raw_dsXY,
                verbose=False,
            )
            raw_cal = np.squeeze(raw_cal, axis=0) if raw_cal.shape[0] == 1 else raw_cal
        except Exception as e:
            print(f"[ERROR] Failed to read raw ND2 frames {raw_frames}: {e}")
            continue

        # ---- write files ----
        for i, t in enumerate(raw_frames):
            try:
                IO.write_multichannel_volume_as_ome_tiff(
                    volume=[
                        raw_mem[i].squeeze(),
                        reg_mem_block[i].squeeze(),
                        reg_ref_block[i].squeeze(),
                    ],
                    out_dir=mem_out_dir,
                    frame_idx=t,
                    configPath=configPath,
                    label="membrane_downsample",
                )
                IO.write_multichannel_volume_as_ome_tiff(
                    volume=[raw_cal[i].squeeze(), reg_cal_block[i].squeeze()],
                    out_dir=cal_out_dir,
                    frame_idx=t,
                    configPath=configPath,
                    label="calcium_downsample",
                )
                IO.write_multichannel_volume_as_ome_tiff(
                    volume=[
                        reg_ref_block[i].squeeze(),
                        reg_mem_block[i].squeeze(),
                        mask_block[i].squeeze(),
                    ],
                    out_dir=mask_out_dir,
                    frame_idx=t,
                    configPath=configPath,
                    label="mask_downsample",
                )
            except Exception as e:
                print(f"[ERROR] Failed to write frame {t}: {e}")

        if verbose:
            print(
                f"[INFO] Block {b + 1}/{num_blocks} finished. Shapes: "
                f"mem={reg_mem_block.shape}, cal={reg_cal_block.shape}, ref={reg_ref_block.shape}, mask={mask_block.shape}"
            )

    if verbose:
        print("[ALL DONE] Downsampled dataset (v3) created successfully.")


def create_downsample_dataset_v4(
    configPath="./configs/config.toml",
    downsampleFilePath="./registrated_downsample",
    ds_XY=1,
    ds_T=1,
    n_workers=4,
    verbose=True,
):
    """
    Create a downsampled dataset (v3) from registered and raw ND2 data.

    This version:
    - Uses lazy dask arrays for TIFF series to reduce peak memory usage
    - Computes per block to avoid loading all frames at once
    - Outputs membrane, calcium, and mask volumes
    - Maintains XY and time downsampling

    Parameters
    ----------
    configPath : str
        Path to TOML configuration file (must contain input_path, registrated_path, mask_path, and metadata).
    downsampleFilePath : str
        Output directory for downsampled dataset.
    ds_XY : int
        XY downsampling factor applied on top of base downsample in config.
    ds_T : int
        Temporal downsampling factor (frames skipped along time axis).
    block_size : int
        Number of frames to process per block to reduce memory consumption.
    verbose : bool
        Whether to print progress information.

    Outputs
    -------
    downsampleFilePath/
        raw_membrane/: raw_membrane
        membrane/    : registered_membrane
        raw_calcium/ : raw_calcium
        calcium/     : registered_calcium
        mask/        : reference
        reference/   : reference

    Usage
    -----
    create_downsample_dataset_v4(
        configPath='./configs/config.toml',
        downsampleFilePath='./registrated_downsample',
        ds_XY=4, ds_T=2, num_workers = 8, verbose=True
    )
    And you could see the result with FIJI by the code in the folder /macros ("ShowCaDownsample.ijm" and "ShowMemDownsample.ijm")

    """
    # ---------------- load config ----------------
    config = toml.load(configPath)
    raw_path = config["file_path"]["input_path"]
    reg_path = config["file_path"]["registrated_path"]
    mask_path = config["file_path"]["mask_path"]
    dual_channel = config["channels"]["dual_channel"]
    base_dsXY = config["downsample"]["downsampleXY"]
    base_dsZ = config["downsample"]["downsampleZ"]
    frames = config["frames"]["frames"]
    if config["MetaData"]["Dim"] == 3:
        base_dsZ = list(range(config["MetaData"]["SIZE"][1]))
    elif config["MetaData"]["Dim"] == 2:
        base_dsZ = [0]
    else:
        raise ValueError(
            f"[ERROR] Invalid Dim value in config. Must be 2 or 3. Dim={config['MetaData']['Dim']}"
        )
    dual_channel = config["channels"]["dual_channel"]

    raw_dsXY = base_dsXY * ds_XY

    from ..utils.reliableAnalysis import build_reference_index

    reg_mem_path = os.path.join(reg_path, "membrane")
    reg_cal_path = os.path.join(reg_path, "calcium")
    reg_ref_path = os.path.join(reg_path, "reference")
    ref_map, _ = build_reference_index(reg_ref_path)

    if os.path.exists(downsampleFilePath):
        if verbose:
            print(f"[INFO] Removing existing directory: {downsampleFilePath}")
        shutil.rmtree(downsampleFilePath)

    mem_out_dir = os.path.join(downsampleFilePath, "membrane")
    mask_out_dir = os.path.join(downsampleFilePath, "mask")
    raw_mem_out_dir = os.path.join(downsampleFilePath, "raw_membrane")
    ref_out_dir = os.path.join(downsampleFilePath, "reference")
    IO.reset_dir(mem_out_dir)
    IO.reset_dir(ref_out_dir)
    IO.reset_dir(mask_out_dir)
    IO.reset_dir(raw_mem_out_dir)
    # ---------------- build lazy dask arrays ----------------

    IO.downsample_tifs_dask(reg_ref_path, ref_out_dir, ds_XY, 1, n_workers, verbose)
    IO.downsample_tifs_dask(reg_mem_path, mem_out_dir, ds_XY, ds_T, n_workers, verbose)
    IO.downsample_tifs_dask(mask_path, mask_out_dir, ds_XY, 1, n_workers, verbose)
    IO.downsample_nd2_to_tiff_folder(
        raw_path,
        raw_mem_out_dir,
        raw_dsXY,
        ds_T,
        frames,
        base_dsZ,
        1,
        n_workers=n_workers,
        verbose=verbose,
    )

    if dual_channel:
        cal_out_dir = os.path.join(downsampleFilePath, "calcium")
        raw_cal_out_dir = os.path.join(downsampleFilePath, "raw_calcium")
        IO.reset_dir(cal_out_dir)
        IO.reset_dir(raw_cal_out_dir)
        IO.downsample_tifs_dask(reg_cal_path, cal_out_dir, ds_XY, ds_T, n_workers, verbose=verbose)
        IO.downsample_nd2_to_tiff_folder(
            raw_path,
            raw_cal_out_dir,
            raw_dsXY,
            ds_T,
            frames,
            base_dsZ,
            0,
            n_workers=n_workers,
            verbose=verbose,
        )


####################################################################################################################
####################################################################################################################
## helper functions


def write_volume_as_ome_tiff(
    volume, out_dir, ch_idx, frame_idx, configPath, spacing_x=1.0, spacing_y=1.0
):
    """
    volume: np.ndarray, shape (Z,Y,X) for 3D or (Y,X) for 2D
    out_dir: target directory
    ch_idx: integer channel id used in filename
    frame_idx: integer frame index
    """
    if volume.ndim == 2:
        # make (Z=1, Y, X)
        zvol = volume[np.newaxis, :, :]
    elif volume.ndim == 3:
        zvol = volume
    else:
        raise ValueError("volume must be 2D or 3D (Z,Y,X)")

    # convert to TZCYX: T=1, Z, C=1, Y, X
    t = 1
    Zv, Yv, Xv = zvol.shape
    img5d = zvol[np.newaxis, :, np.newaxis, :, :]  # shape (1,Z,1,Y,X)

    fname = os.path.join(out_dir, f"vol_ch{ch_idx}_{frame_idx:06d}.tif")
    # optional metadata: include spacing if available
    metadata = {"spacing_x": spacing_x, "spacing_y": spacing_y, "data_shape": img5d.shape}
    # call the provided save function
    IO.saveTiff_new(img5d, fname, config_path=configPath, metadata=metadata, verbose=False)


def build_chunks(start_frame, end_frame, window_size, direction):
    """
    Returns a list of frame-index lists.
    Applies the rule:
      - residual chunk < 0.5 * window_size is merged to neighbor
    """

    assert direction in ("forward", "backward")

    if direction == "forward":
        frames = list(range(start_frame, end_frame))
    elif direction == "backward":
        frames = list(range(start_frame - 1, end_frame - 1, -1))
    else:
        raise ValueError("direction must be 'forward' or 'backward'")
    chunks = []
    i = 0
    N = len(frames)

    while i < N:
        chunk = frames[i : i + window_size]
        chunks.append(chunk)
        i += window_size

    # ---- handle residual chunk ----
    if len(chunks) >= 2:
        last_chunk = chunks[-1]
        if len(last_chunk) < window_size // 2:
            # merge into previous
            chunks[-2].extend(last_chunk)
            chunks.pop(-1)

    return chunks


def process_directional_chunks(
    direction,
    frame_list,
    start_frame,
    end_frame,
    init_mem_windows,
    init_ca_windows,
    device_id,
    configPath,
    config,
    movingFilePath,
    out_mem,
    out_ca,
    out_ref,
    out_mot,
    channel_index_map,
    save_motion,
    save_ref,
    Dim,
    window_size_frames,
    reference_chunk,
    downsampleZ,
    downsampleXY,
    batch_size=50,
    motion_init=None,
):
    """
    Process frames in chunks in a specified direction (forward or backward) with batch processing and
    sliding reference windows for registration.

    This function:
    - Divides frames into chunks to reduce peak memory usage.
    - Registers membrane and calcium channels to a computed reference image.
    - Maintains a rolling reference window (deque) for computing the reference image.
    - Supports both forward and backward processing directions.
    - Saves registered images and optional motion fields.

    Parameters:
    -----------
    direction : str
        'forward' or 'backward'
    start_frame, end_frame : int
        Frame range to process
    init_mem_windows, init_ca_windows : list or np.ndarray
        Initial reference windows
    device_id : int or None
        GPU device ID
    configPath : str
        Path to config
    config : dict
        Configuration dictionary
    movingFilePath : str
        Path to ND2 file
    out_mem, out_ca, out_ref, out_mot : str
        Output directories
    channel_index_map : dict
        {'membrane': idx, 'calcium': idx}
    save_motion, save_ref : bool
        Whether to save motion or reference images
    Dim : int
        2 or 3
    window_size_frames : int
        Chunk size
    reference_chunk : int
        Size of rolling reference window
    downsampleZ, downsampleXY : int/float
        Downsampling factors
    batch_size : int
        Number of frames per batch
    """

    import os
    from collections import deque

    import h5py
    import numpy as np

    # Route through the package shim (numpy fallback when CuPy/CUDA is absent)
    # instead of importing cupy directly: this function runs in the SERIAL path
    # too, where a hard cupy import aborted CPU-only runs right after the middle
    # block had been written.
    from ..utils import CUPY_AVAILABLE, cp

    if device_id is not None and CUPY_AVAILABLE:
        cp.cuda.Device(device_id).use()

    # -------------------------------
    # Initialize reference windows as deque
    # -------------------------------
    ref_windows_mem = deque(init_mem_windows, maxlen=reference_chunk)
    ref_windows_ca = deque(init_ca_windows, maxlen=reference_chunk)

    # Build frame chunks
    chunks = build_chunks(
        start_frame=start_frame,
        end_frame=end_frame,
        window_size=window_size_frames,
        direction=direction,
    )

    for frames in chunks:
        print(
            f"[{direction}] Processing chunk {frame_list[frames[0]]}~{frame_list[frames[-1]]} (n={len(frames)})"
        )
        # -------------------------------
        # Compute reference once per chunk
        # -------------------------------
        ref_img, indsort = reference.compute_reference_from_block(
            list(ref_windows_mem), config, list(ref_windows_ca)
        )

        # renew the windows:
        ref_windows_mem.clear()
        ref_windows_ca.clear()

        if save_ref:
            if direction == "forward":
                IO.write_volume_as_ome_tiff(
                    ref_img.copy(),
                    out_ref,
                    "ref",
                    f"{frame_list[frames[0]]}~{frame_list[frames[-1]]}",
                    configPath,
                )
            else:
                IO.write_volume_as_ome_tiff(
                    ref_img.copy(),
                    out_ref,
                    "ref",
                    f"{frame_list[frames[-1]]}~{frame_list[frames[0]]}",
                    configPath,
                )

        # -------------------------------
        # Process in batches
        # -------------------------------
        num_frames = len(frames)
        dual_channel = config["channels"]["dual_channel"]
        for i in range(0, num_frames, batch_size):
            batch_frames = [frame_list[i] for i in frames[i : i + batch_size]]
            print(f"    batch {batch_frames[0]}~{batch_frames[-1]} (n={len(batch_frames)})")

            # Read frames
            mem_batch = np.squeeze(
                IO.readND2Frame(
                    movingFilePath,
                    batch_frames,
                    downsampleZ,
                    channel=1,
                    xy_down=downsampleXY,
                    verbose=False,
                )
            )
            if dual_channel:
                ca_batch = np.squeeze(
                    IO.readND2Frame(
                        movingFilePath,
                        batch_frames,
                        downsampleZ,
                        channel=0,
                        xy_down=downsampleXY,
                        verbose=False,
                    )
                )
            else:
                ca_batch = None
            # Registration
            if Dim == 3 and len(downsampleZ) > 1:
                mem_reg, ca_reg, _, _, motion = registration.wbi_registration_3d(
                    mem_batch,
                    configPath,
                    ref_img,
                    frame=batch_frames,
                    direction=direction,
                    motion_init=motion_init,
                    moving_Ca_image=ca_batch,
                )
            else:
                mem_reg, ca_reg, _, _, motion = registration.wbi_registration_2d(
                    mem_batch,
                    configPath,
                    ref_img,
                    frame=batch_frames,
                    direction=direction,
                    motion_init=motion_init,
                    moving_Ca_image=ca_batch,
                )
            motion_init = motion[-1]
            # Save batch results immediately
            for k, fid in enumerate(batch_frames):
                IO.write_volume_as_ome_tiff(
                    mem_reg[k], out_mem, channel_index_map["membrane"], fid, configPath
                )
                if dual_channel:
                    IO.write_volume_as_ome_tiff(
                        ca_reg[k], out_ca, channel_index_map["calcium"], fid, configPath
                    )
                if save_motion:
                    with h5py.File(os.path.join(out_mot, f"motion_{fid:06d}.h5"), "w") as hf:
                        hf.create_dataset("motion", data=motion[k], compression="gzip")

                # -------------------------------
                # Update rolling reference window
                # -------------------------------
                # if direction == 'forward':
                ref_windows_mem.append(mem_reg[k])
                if dual_channel:
                    ref_windows_ca.append(ca_reg[k])
                # elif direction == 'backward':  # backward
                #     if len(ref_windows_mem) < reference_chunk:
                #         ref_windows_mem.append(mem_reg[k])
                #         ref_windows_ca.append(ca_reg[k])
                # else:
                #     raise ValueError("direction must be 'forward' or 'backward'")
            # Free batch memory
            del mem_batch, ca_batch, mem_reg, ca_reg, motion
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

    print(f"[{direction}] Processing complete.")
