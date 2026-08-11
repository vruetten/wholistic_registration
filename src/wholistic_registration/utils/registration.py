"""

version : 0.3
file name : registration.py

Last Update Date : 2025/12/2

Overview:
    The whole pipeline of registration

Functions:
    - wbi_registration
"""

import numpy as np
import toml

from . import calFlow3d_Wei_v1, mask, option, reference
from . import preprocess as prep


def transform(image, k=1, method="raw"):
    if method == "raw":
        return k * image
    elif method == "sqrt":
        return k * np.sqrt(image)
    elif method == "log2":
        return k * np.log2(1 + image)
    elif method == "log10":
        return k * np.log10(1 + image)
    else:
        raise ValueError(f"Unknown method to process the image:{method}")


def wbi_registration_2d(
    moving_membrane_image,
    config_file,
    reference_image=None,
    motion_init=None,
    verbose=True,
    frame=None,
    direction="forward",
    moving_Ca_image=None,
):
    """Load the config file"""
    config = toml.load(config_file)
    """Get frames"""
    if len(moving_membrane_image.shape) == 2:
        moving_membrane_image = np.expand_dims(moving_membrane_image, axis=0)
        if moving_Ca_image is not None:
            moving_Ca_image = np.expand_dims(moving_Ca_image, axis=0)
    nimg, Ly, Lx = moving_membrane_image.shape
    frames = range(nimg)

    """Get reference image"""
    # if we pick reference image from moving_image
    refer = config["reference"]
    if refer["pick_reference_auto"]:
        membrane_ref_1plane, indsort = reference.pick_initial_reference(
            moving_membrane_image, max_corr_frames=refer["chunk_size"]
        )
    else:
        membrane_ref_1plane = reference_image

    # If to use two channel data
    channels = config["channels"]
    if channels["dual_channel"] and moving_Ca_image is None:
        raise ValueError("[ERROR] Calcium image stack shouldn't be empty")

    if (
        channels["dual_channel"] and refer["pick_reference_auto"] and channels["k"] != 0
    ):  # get the corresponding planes in Ca channel
        Ca_data_reshape = np.reshape(moving_Ca_image, (len(frames), -1))
        Ca_average = np.mean(Ca_data_reshape[indsort, :], axis=0)
        Ca_ref_1plane = np.reshape(Ca_average, moving_membrane_image.shape[1:])
        Ca_ref_1plane_transform = transform(Ca_ref_1plane, channels["k"], channels["function"])

        # record the mean and std of Ca channel(we will use these two values to normalize the moving Ca image)
        ref_mean = np.mean(Ca_ref_1plane_transform)
        ref_std = np.std(Ca_ref_1plane_transform)

        # get the reference data(1 plane)
        dat_ref_1plane = membrane_ref_1plane + Ca_ref_1plane_transform

    else:
        dat_ref_1plane = membrane_ref_1plane

    # visualization.visualize_2d_image(dat_ref_1plane,title="Reference Image")
    # stack the refence to get a fake 3D image
    dat_ref = np.stack([dat_ref_1plane] * 2, axis=2)

    """Get mask"""
    maskConfig = config["mask"]
    maskRange = maskConfig["maskRange"]
    thresFactor = maskConfig["thresFactor"]

    option["mask_ref"] = mask.getMask(dat_ref, thresFactor)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], maskRange)

    """Do registration"""
    # initial list to store the result
    mem_channel = []
    Ca_channel = []
    errors = []
    motions = []

    # inital the motion
    if motion_init is None:
        option["motion"] = np.zeros([dat_ref.shape[0], dat_ref.shape[1], 2, 3])
    else:
        option["motion"] = np.stack([motion_init] * 2, axis=2)
    # initial the pyramid parameters
    pyramid = config["pyramid"]
    option["r"] = pyramid["r"]
    option["layer"] = pyramid["layer"]
    option["iter"] = pyramid["iter"]
    option["movRange"] = 5.0
    smoothPenalty_raw = pyramid["smoothPenalty"]
    option["tol"] = pyramid["tolerance"]
    # get smoothPenalty
    Pnltfactor = prep.getSmPnltNormFctr(dat_ref, option)
    smoothPenalty = Pnltfactor * smoothPenalty_raw
    option["smoothPenalty"] = smoothPenalty

    # do registration
    for i in frames:
        # get dat_mov
        mem_1plane = moving_membrane_image[i, :, :]
        dat_mem = np.stack([mem_1plane] * 2, axis=2)

        if channels["dual_channel"]:
            Ca_1plane = moving_Ca_image[i, :, :]
            dat_ca = np.stack([Ca_1plane] * 2, axis=2)

        if channels["dual_channel"]:
            dat_ca_tran = transform(dat_ca, channels["k"], channels["function"])
            # normalize to the mean and std of the reference
            if refer["pick_reference_auto"]:
                dat_ca_tran = prep.normalize_std(ref_mean, ref_std, dat_ca_tran)
            dat_mov = dat_mem + dat_ca_tran
        else:
            dat_mov = dat_mem

        # get mask_mov
        option["mask_mov"] = mask.getMask(dat_mov, thresFactor)
        option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

        # get motion
        motion_current, _, new_coords, error_logs = calFlow3d_Wei_v1.getMotion(
            dat_mov, dat_ref, option
        )
        if channels["dual_channel"]:
            corrected_ca = calFlow3d_Wei_v1.correctMotion(dat_ca, motion_current)
        corrected_mem = calFlow3d_Wei_v1.correctMotion(dat_mem, motion_current)
        corrected_mov = calFlow3d_Wei_v1.correctMotion(dat_mov, motion_current)
        initial_error = np.mean((dat_mov - dat_ref) ** 2)
        eventual_error = np.mean((corrected_mov - dat_ref) ** 2)

        # print error
        if verbose == True:
            if frame is None:
                print(
                    f"        Frame: {i + 1}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}"
                )
            else:
                print(
                    f"        Frame: {frame[i]}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}"
                )

        error = {"initial_error": initial_error, "eventual_error": eventual_error}

        # store the result
        errors.append(error)
        mem_channel.append(corrected_mem[:, :, 0])
        if channels["dual_channel"]:
            Ca_channel.append(corrected_ca[:, :, 0])
        motions.append(motion_current[:, :, 0, :])

    # Convert to NumPy arrays directly (avoid GPU memory allocation)
    mem_out = np.asarray(mem_channel)
    ca_out = np.asarray(Ca_channel) if Ca_channel else np.array([])
    motion_out = np.asarray(motions)
    return mem_out, ca_out, dat_ref, errors, motion_out


def wbi_registration_3d(
    moving_membrane_image,
    config_file,
    reference_image=None,
    motion_init=None,
    verbose=True,
    frame=None,
    direction="forward",
    moving_Ca_image=None,
):
    """Load the config file"""
    config = toml.load(config_file)
    if len(moving_membrane_image.shape) == 3:
        moving_membrane_image = np.expand_dims(moving_membrane_image, axis=0)
        if moving_Ca_image is not None:
            moving_Ca_image = np.expand_dims(moving_Ca_image, axis=0)
    nimg, Lz, Ly, Lx = moving_membrane_image.shape
    frames = range(nimg)

    """Get frames"""
    if len(moving_membrane_image.shape) == 4:
        nimg, Lz, Ly, Lx = moving_membrane_image.shape
    elif len(moving_membrane_image.shape) == 3:
        Lz, Ly, Lx = moving_membrane_image.shape
        nimg = 1
    else:
        raise ValueError("moving_membrane_image must be 3D or 4D array")
    frames = range(nimg)

    """Get reference image"""
    # if we pick reference image from moving_image
    refer = config["reference"]
    if refer["pick_reference_auto"]:
        membrane_ref, indsort = reference.pick_initial_reference(
            moving_membrane_image, max_corr_frames=refer["chunk_size"]
        )
    else:
        membrane_ref = reference_image

    # If to use two channel data
    channels = config["channels"]
    if channels["dual_channel"] and moving_Ca_image is None:
        raise ValueError("[ERROR] Calcium image stack shouldn't be empty")

    if channels["dual_channel"] and refer["pick_reference_auto"] and channels["k"] != 0:
        # get the corresponding planes in Ca channel
        Ca_data_reshape = np.reshape(moving_Ca_image, (len(frames), -1))
        Ca_average = np.mean(Ca_data_reshape[indsort, :], axis=0)
        Ca_ref = np.reshape(Ca_average, moving_membrane_image.shape[1:])
        Ca_ref_transform = transform(Ca_ref, channels["k"], channels["function"])

        # record the mean and std of Ca channel(we will use these two values to normalize the moving Ca image)
        ref_mean = np.mean(Ca_ref_transform)
        ref_std = np.std(Ca_ref_transform)

        # get the reference data(1 plane)
        dat_ref = (membrane_ref + Ca_ref_transform).transpose(2, 1, 0)

    else:
        dat_ref = membrane_ref.transpose(2, 1, 0)

    # visualization.visualize_2d_image(dat_ref_1plane,title="Reference Image")
    maskConfig = config["mask"]
    maskRange = maskConfig["maskRange"]
    thresFactor = maskConfig["thresFactor"]

    option["mask_ref"] = mask.getMask(dat_ref, thresFactor)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], maskRange)

    """Do registration"""
    # initial list to store the result
    mem_channel = []
    Ca_channel = []
    errors = []
    motions = []

    # inital the motion
    if motion_init is None:
        option["motion"] = np.zeros([Lx, Ly, Lz, 3])
    else:
        # motion_init arrives in this function's OUTPUT layout (z, y, x, 3) —
        # every in-pipeline caller feeds back a field from `motions.append(
        # motion_current.transpose(2, 1, 0, 3))` below — but getMotion consumes
        # option["motion"] in its internal (x, y, z, 3) layout. Transpose back;
        # the component axis stays as-is (component c displaces axis c).
        # Without this, getMotion silently imresize-stretched the transposed
        # field into shape AND divided components by wrong SZ/dim factors.
        option["motion"] = np.asarray(motion_init).transpose(2, 1, 0, 3)
    # initial the pyramid parameters
    pyramid = config["pyramid"]
    option["r"] = pyramid["r"]
    option["layer"] = pyramid["layer"]
    option["iter"] = pyramid["iter"]
    option["movRange"] = 5.0
    option["tol"] = pyramid["tolerance"]
    # getMotion scales its z gradients / z smoothness by option["zRatio"]; only the
    # config carries the ratio measured from the acquisition (DefineParams writes it
    # from the ND2 metadata), so without this the solver silently used the module
    # default in utils/__init__.py for every dataset.
    option["zRatio"] = config["MetaData"].get("zRatio", option["zRatio"])
    smoothPenalty_raw = pyramid["smoothPenalty"]

    # get smoothPenalty
    Pnltfactor = prep.getSmPnltNormFctr(dat_ref, option)
    smoothPenalty = Pnltfactor * smoothPenalty_raw
    option["smoothPenalty"] = smoothPenalty

    for i in frames:
        # get dat_mov
        dat_mem = moving_membrane_image[i, ...]

        if channels["dual_channel"]:
            dat_ca = moving_Ca_image[i, ...]

        if channels["dual_channel"]:
            dat_ca_tran = transform(dat_ca, channels["k"], channels["function"])
            # normalize to the mean and std of the reference
            if refer["pick_reference_auto"]:
                dat_ca_tran = prep.normalize_std(ref_mean, ref_std, dat_ca_tran)
            dat_mov = (dat_mem + dat_ca_tran).transpose(2, 1, 0)
        else:
            dat_mov = dat_mem.transpose(2, 1, 0)

        # get mask_mov
        option["mask_mov"] = mask.getMask(dat_mov, thresFactor)
        option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

        # get motion
        motion_current, _, new_coords, error_logs = calFlow3d_Wei_v1.getMotion(
            dat_mov, dat_ref, option
        )
        if channels["dual_channel"]:
            corrected_ca = calFlow3d_Wei_v1.correctMotion(dat_ca.transpose(2, 1, 0), motion_current)
        corrected_mem = calFlow3d_Wei_v1.correctMotion(dat_mem.transpose(2, 1, 0), motion_current)
        corrected_mov = calFlow3d_Wei_v1.correctMotion(dat_mov, motion_current)
        initial_error = np.mean((dat_mov - dat_ref) ** 2)
        eventual_error = np.mean((corrected_mov - dat_ref) ** 2)
        # print error
        if verbose == True:
            if frame is None:
                print(
                    f"        Frame: {i + 1}\tInitial Error is:{initial_error:.4f}\tEventual Error: {eventual_error:.4f}"
                )
            else:
                print(
                    f"        Frame: {frame[i]}\tInitial Error is:{initial_error:.4f}\tEventual Error: {eventual_error:.4f}"
                )
        error = {"initial_error": initial_error, "eventual_error": eventual_error}

        # store the result
        errors.append(error)
        mem_channel.append(corrected_mem.transpose(2, 1, 0))
        if channels["dual_channel"]:
            Ca_channel.append(corrected_ca.transpose(2, 1, 0))
        motions.append(motion_current.transpose(2, 1, 0, 3))

    # Convert to NumPy arrays directly (avoid GPU memory allocation)
    mem_out = np.asarray(mem_channel)
    ca_out = np.asarray(Ca_channel) if Ca_channel else np.array([])
    motion_out = np.asarray(motions)
    return mem_out, ca_out, dat_ref, errors, motion_out


def register_one_frame(configFilePath, mem_img, ca_img, ref_pool, idx, verbose=True):
    """Register one mem+Ca frame to the reference generated from the pool"""
    config = toml.load(configFilePath)
    ref_img = reference.compute_reference_from_block(
        np.array([m for m in ref_pool["mem"]]), np.array([c for c in ref_pool["ca"]]), config
    )
    if config["MetaData"]["Dim"] == 3:
        mem_reg, ca_reg, _, _, motion_reg = wbi_registration_3d(
            np.expand_dims(mem_img, axis=0),
            np.expand_dims(ca_img, axis=0),
            configFilePath,
            ref_img,
            verbose=verbose,
            frame=idx,
        )
    else:
        mem_reg, ca_reg, _, _, motion_reg = wbi_registration_2d(
            np.expand_dims(mem_img, axis=0),
            np.expand_dims(ca_img, axis=0),
            configFilePath,
            ref_img,
            verbose=verbose,
            frame=idx,
        )
    return mem_reg[0], ca_reg[0], ref_img, motion_reg[0]
