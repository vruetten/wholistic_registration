"""

version : 0.1
file name : imresize.py

Algorithm Author : Fatheral , Yunfeng Chi
Code Author : Yunfeng Chi (Tsinghua University) ,but the origin file is on the website https://github.com/fatheral/matlab_imresize (By Fatheral). Yunfeng only made some tiny changes to make the code adjust to this package
Last Update Date : 2025/4/16

Overview:
    We want to achieve the same performace on the downsampling of the 3D image as the matlab function imresize3.
    But the tools at present still remains a huge diff,such scipy.ndimage.zoom ,scipy.misc.imresize and PIL.image
    So Fatheral did the transform and achieve a almost identical performance and upload his code on the website https://github.com/fatheral/matlab_imresize
    We changed it into a version which runs on the GPU with package cupy to accerlate the pipeline and the dimensions of input is changed to 3.
    Anyway thanks to the contribution of Fatheral

functions:


"""

from math import ceil

from . import cp


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(3):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(3):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = cp.array(x).astype(cp.float64)
    lessthanzero = cp.logical_and((x >= -1), x < 0)
    greaterthanzero = cp.logical_and((x <= 1), x >= 0)
    f = cp.multiply((x + 1), lessthanzero) + cp.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = cp.array(x).astype(cp.float64)
    absx = cp.absolute(x)
    absx2 = cp.multiply(absx, absx)
    absx3 = cp.multiply(absx2, absx)
    f = cp.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + cp.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (absx > 1) & (absx <= 2)
    )
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = cp.arange(1, out_length + 1).astype(cp.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = cp.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = cp.expand_dims(left, axis=1) + cp.arange(P) - 1
    indices = ind.astype(cp.int32)
    weights = h(cp.expand_dims(u, axis=1) - indices - 1)
    weights = cp.divide(weights, cp.expand_dims(cp.sum(weights, axis=1), axis=1))
    aux = cp.concatenate((cp.arange(in_length), cp.arange(in_length - 1, -1, step=-1))).astype(
        cp.int32
    )
    indices = aux[cp.mod(indices, aux.size)]
    ind2store = cp.nonzero(cp.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim, work_dtype=cp.float64):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = cp.zeros(out_shape, dtype=work_dtype)
    weights = weights.astype(work_dtype, copy=False)

    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(work_dtype)
                outimg[i_w, i_img] = cp.sum(cp.multiply(cp.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(work_dtype)
                outimg[i_img, i_w] = cp.sum(cp.multiply(cp.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 2:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, :, ind].astype(work_dtype)
                outimg[i_img, i_w] = cp.sum(cp.multiply(cp.squeeze(im_slice, axis=2), w.T), axis=1)

    if inimg.dtype == cp.uint8:
        outimg = cp.clip(outimg, 0, 255)
        return cp.around(outimg).astype(cp.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim, work_dtype=cp.float64):
    wshape = weights.shape
    weights = weights.astype(work_dtype, copy=False)
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = cp.sum(weights * (inimg[indices].squeeze(axis=1)).astype(work_dtype), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = cp.sum(weights * (inimg[:, indices].squeeze(axis=2)).astype(work_dtype), axis=2)
    elif dim == 2:
        weights = weights.reshape((1, 1, wshape[0], wshape[2]))
        outimg = cp.sum(weights * (inimg[:, :, indices].squeeze(axis=3)).astype(work_dtype), axis=3)

    if inimg.dtype == cp.uint8:
        outimg = cp.clip(outimg, 0, 255)
        return cp.around(outimg).astype(cp.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec", work_dtype=cp.float64):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim, work_dtype=work_dtype)
    else:
        out = imresizevec(A, weights, indices, dim, work_dtype=work_dtype)
    return out


def imresize(
    I, scalar_scale=None, method="bicubic", output_shape=None, mode="vec", work_dtype=None
):
    if method == "bicubic":
        kernel = cubic
    elif method == "bilinear":
        kernel = triangle
    else:
        raise ValueError("unidentified kernel method supplied")

    # Accumulation/gather dtype. Default float64 preserves the original
    # (MATLAB-matching) behaviour exactly for existing callers. Passing
    # work_dtype=cp.float32 halves the large intermediate gather buffers, which
    # is what lets the cross-resolution pipeline downsample full-size reference
    # volumes without exhausting GPU memory.
    if work_dtype is None:
        work_dtype = cp.float64

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError("either scalar_scale OR output_shape should be defined")
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError("either scalar_scale OR output_shape should be defined")

    scale_cp = cp.array(scale)
    order = cp.argsort(scale_cp)
    weights = []
    indices = []
    for k in range(3):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)

    B = cp.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = cp.expand_dims(B, axis=2)
        flag2D = True

    for k in range(3):
        dim = int(order[k])
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode, work_dtype=work_dtype)

    if flag2D:
        B = cp.squeeze(B, axis=2)
    return B


def convertDouble2Byte(I):
    B = cp.clip(I, 0.0, 1.0)
    B = 255 * B
    return cp.around(B).astype(cp.uint8)
