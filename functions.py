import glob
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage import color, morphology
import skimage
import sknw


def OCY_read_stack(mask):
    """
    Load all tif image files matching the mask and combinding them into a 3D image.

    Parameters:
        mask (string): only these images will be loaded (e.g. *.tif)

    returns:
        img (np.array): 3D image
    """
    files = glob.glob(mask)
    files = sorted(files, key=lambda s: s.lower())
    num_img = len(files)

    if (num_img > 0):
        size = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).shape
        img = np.empty([size[0], size[1], num_img])
        for i in range(0, num_img):
            img[:, :, i] = skimage.io.imread(files[i])  # , cv2.IMREAD_GRAYSCALE)
    return img


def OCY_thr_stack(img, THR):
    """
    Apply threshold and optional background filtering

    Parameters:
        img (np.array): 3D image
        THR (float): threshold value

    returns:
        bins (np.array): thresholded binary 3D image
    """

    num_img = img.shape[2]

    if np.median(img.reshape(-1, 1), axis=0) > 20:
        se = morphology.octagon(21, 14)
        for i in range(0, num_img):
            img[:, :, i] = morphology.white_tophat(img[:, :, i],
                                                   se)  # cv2.morphologyEx(img[:,:,i], cv2.MORPH_TOPHAT, se) #

        h, _ = np.histogram(np.divide(img[:, :, :], np.max(img)), 256)
        h[199:] = 0
        h[0] = 0

        max_idx = np.argmax(h)
        img = img - max_idx

        img[img < 0] = 0

    img = np.divide(img, max(img.reshape(-1, 1)))
    bins = cv2.threshold(img, THR, 1, cv2.THRESH_BINARY)[1]
    # bins = img > THR

    return bins


from skimage import morphology
from scipy import ndimage
import scipy


def OCY_fill_voids(bins, S):
    """
    Morphologically filter binary volume

    Parameters:
        bins (np.array): thresholded binary 3D image
        S (int): threshold value

    returns:
        cleaned (np.array): filtered binary 3D image
    """

    img = morphology.area_opening(bins, S).astype('uint8')
    # cleaned = morphology.remove_small_objects(bin_thr, min_size=S).astype('uint8') --> does same thing

    A6 = np.zeros([3, 3, 3], np.uint8)
    A6[0:3:2, 1, 1] = 1
    A6[1] = 1
    A6[1][0:3:2, 0:3:2] = 0
    kernel = A6.astype('uint8')

    # # ndimage.binary_closing == imclose
    # cv2.cvtColor(img, img, cv::CV_BGRA2GRAY)
    img = skimage.morphology.closing(img, kernel)  # scipy.ndimage.binary_closing(img, kernel)

    # put a box of 1's around the image to close cells
    padded = np.pad(img, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=[(1, 1), (1, 1), (1, 1)])
    not_pad = np.logical_not(padded)
    new = np.logical_not(morphology.area_opening(not_pad, round(0.1 * np.sum(not_pad)), 6))
    cleaned = new[1:-1, 1:-1, 1:-1]

    return cleaned


def OCY_get_cells(img, DIST_THR=7, CELL_SIZE=100, EXP_THR=4, EXP_FRAC=0.1):
    """
    Detect large objects in binary stack

    Parameters:
        img (np.array): 3D image

    returns:
        cells (np.array): cells in stack "img"
    """

    # distance transform of foreground
    Dst = scipy.ndimage.distance_transform_edt(np.logical_not(img == 0))

    # pad columes with zeros to avoid edge effects
    Dst = np.pad(Dst, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=[(0, 0), (0, 0), (0, 0)])
    img = np.pad(img, [(1, 1), (1, 1), (1, 1)], mode='constant', constant_values=[(0, 0), (0, 0), (0, 0)])

    # size
    x, y, z = img.shape

    # sphere as structuring element
    sp = skimage.morphology.ball(3)

    # start with detecting large objects
    cells = Dst > DIST_THR
    cells = morphology.area_opening(cells, CELL_SIZE)

    # grow cells to fill lacunae
    if np.max(cells):
        imd3 = cells
        df = np.sum(cells)
        while df >= (EXP_FRAC * np.sum(cells)):
            cells = imd3.astype('uint8')
            # imd3= cv2.dilate(cells, sp) & (Dst>EXP_THR)
            imd3 = skimage.morphology.dilation(cells, sp) & (Dst > EXP_THR)
            df = np.sum(np.subtract(imd3, cells))
        cells = imd3

    # remove objects that are too small
    cells = morphology.area_opening(cells, CELL_SIZE)

    # get rid of padded zeros
    cells = cells[1:-1, 1:-1, 1:-1]

    return cells


from skimage import feature
from skimage.segmentation import watershed
from skimage.morphology import opening


def OCY_extract_cells(img, FOOTPRINT_SIZE=160, KERNEL_SIZE=5):
    """
    Detect large objects in binary stack

    Parameters:
        img (np.array): 3D image
        FOOTPRINT_SIZE (int): size of local region within which to search (default: 160)
        KERNEL_SIZE (int): size of the kernel with which to perform the opening (default: 5)

    returns:
        cells (np.array): cells in stack "img"
    """

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndimage.distance_transform_edt(img)
    coords = feature.peak_local_max(distance, footprint=np.ones((FOOTPRINT_SIZE, FOOTPRINT_SIZE, FOOTPRINT_SIZE)),
                                    labels=img)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=img)

    kernel = np.ones([KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE])
    cells = opening(labels, kernel)

    return cells


def OCY_skeletonize(img, mask):
    """
    Parallel medial axis thinning of a 3D binary volume with mask

    Parameters:
        img (np.array): 3D image
        mask (np.array): foreground voxels in img

    returns:
        skel (np.array): returns the skeleton of the binary volume 'img', while preserving foreground voxels in 'mask'
    """
    skeleton = morphology.skeletonize_3d(img)
    skeleton = skeleton.astype(bool).astype('uint8')

    skel = np.add(skeleton, mask)
    skel = (skel >= 1).astype('uint8')

    return skel


def skel2Graph(skeleton, mask):
    """
    Converts a 3D binary voxel skeleton into a network graph described by nodes and edges

    Parameters:
        skeleton (np.array): 3D image
        mask (np.array): foreground voxels in skeleton

    returns:
        graph (Networkx graph): graph of the given skeleton
    """

    skel = np.subtract(skeleton, mask)

    graph = sknw.build_sknw(skel, multi=False)

    return graph