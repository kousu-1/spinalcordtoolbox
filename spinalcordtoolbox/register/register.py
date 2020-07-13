#!/usr/bin/env python
#########################################################################################
# Various modules for registration.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2020 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
#
# License: see the LICENSE.TXT
#########################################################################################

import logging
from math import asin, cos, sin, acos
import numpy as np

from scipy import ndimage
from scipy.signal import argrelmax, medfilt
from scipy.io import loadmat
from nibabel import load, Nifti1Image, save

from spinalcordtoolbox.image import Image, find_zmin_zmax, spatial_crop
from spinalcordtoolbox.utils import sct_progress_bar

import sct_utils as sct
import sct_apply_transfo
import sct_concat_transfo
from sct_convert import convert
from sct_image import split_data, concat_warp2d
from msct_register_landmarks import register_landmarks

logger = logging.getLogger(__name__)


def circular_conv(signal1, signal2):
    """
    Perform a circular convolution on two signals.

    :param signal1: np.array: 1D numpy array
    :param signal2: np.array: 1D numpy array, same length as signal1
    :return: resulting convoluted signal
    """

    if signal1.shape != signal2.shape:
        raise ValueError("The input signals need to have the same shape!")

    signal2_extended = np.concatenate((signal2, signal2, signal2))  # replicate signal at both ends
    signal_conv_extended = np.convolve(signal1, signal2_extended, mode="same")  # median filtering
    signal_conv = signal_conv_extended[len(signal1):2*len(signal1)]  # truncate back the signal

    return signal_conv


def circular_filter_1d(signal, window_size, kernel='gaussian'):
    """
    This function filters circularly the signal inputted with a median filter of inputted size, in this context
    circularly means that the signal is wrapped around and then filtered.

    :param signal: np.array: 1D numpy array
    :param window_size: int: size of the kernel
    :return: signal_smoothed: 1D numpy array, same size as signal
    """

    signal_extended = np.concatenate((signal, signal, signal))  # replicate signal at both ends
    if kernel == 'gaussian':
        signal_extended_smooth = ndimage.gaussian_filter(signal_extended, window_size)  # gaussian
    elif kernel == 'median':
        signal_extended_smooth = medfilt(signal_extended, window_size)  # median filtering
    else:
        raise ValueError("Unknow type of kernel")

    signal_smoothed = signal_extended_smooth[len(signal):2*len(signal)]  # truncate back the signal

    return signal_smoothed

def angle_between(a, b):
    """
    Compute angle in radian between a and b. Throws an exception if a or b has zero magnitude.

    :param a: tuple: x,y coordinates of point a
    :param b: tuple: x,y coordinates of point b
    :return: angle in rad between point a and b
    """
    arccosInput = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    sign_angle = np.sign(np.cross(a, b))

    return sign_angle * acos(arccosInput)

def compute_pca(data2d):
    """
    Compute PCA using sklearn

    :param data2d: 2d array. PCA will be computed on non-zeros values.
    :return:\
        coordsrc: 2d array: centered non-zero coordinates\
        pca: object: PCA result.\
        centermass: 2x1 array: 2d coordinates of the center of mass
    """
    # round it and make it int (otherwise end up with values like 10-7)
    data2d = data2d.round().astype(int)
    # get non-zero coordinates, and transpose to obtain nx2 dimensions
    coordsrc = np.array(data2d.nonzero()).T
    # get center of mass
    centermass = coordsrc.mean(0)
    # center data
    coordsrc = coordsrc - centermass
    # normalize data
    coordsrc /= coordsrc.std()
    # Performs PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, copy=False, whiten=False)
    pca.fit(coordsrc)
    return coordsrc, pca, centermass

def find_index_halfmax(data1d):
    """
    Find the two indices at half maximum for a bell-type curve (non-parametric). Uses center of mass calculation.

    :param data1d:
    :return: xmin, xmax
    """
    # normalize data between 0 and 1
    data1d = data1d / float(np.max(data1d))
    # loop across elements and stops when found 0.5
    for i in range(len(data1d)):
        if data1d[i] > 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmin = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])
    # continue for the descending slope
    for i in range(i, len(data1d)):
        if data1d[i] < 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmax = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])

    if 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(src1d)
        plt.plot(xmin, 0.5, 'o')
        plt.plot(xmax, 0.5, 'o')
        plt.savefig('./normalize1d.png')

    return xmin, xmax

def generate_warping_field(dest_img, warp_x, warp_y, verbose=1):
    """
    Generate an ITK warping field.

    :param dest_img: Image: Destination image
    :param warp_x: np.array:
    :param warp_y: np.array:
    :param verbose: int: verbosity level
    :return: warped image as Nifti1Image
    """
    if not isinstance(dest_img, Image):
        raise ValueError("Input image must be of Image type!")

    nx, ny, nz, nt, px, py, pz, pt = dest_img.dim

    # initialize
    data_warp = np.zeros((nx, ny, nz, 1, 3))

    # fill matrix
    data_warp[:, :, :, 0, 0] = -warp_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_y  # need to invert due to ITK conventions

    hdr_dest = dest_img.get_header()
    hdr_warp = hdr_dest.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')

    img = Nifti1Image(data_warp, None, hdr_warp)

    return img

