import glob
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage import color, morphology
import skimage
from functions import *
# from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from tifffile import imwrite
import numpy as np


def main_func(path, string='*ch00*.tif'):
    datapath = "/home/mkamphuis/public/10 Students/2022/Mirthe Kamphuis/Master Thesis/04_Raw Data/Data/"
    # datapath = "/home/mkamphuis/public/10 Students/2022/Mirthe Kamphuis/Master Thesis/05_Processed Data/3"
    # datapath = "/home/mkamphuis/public/10 Students/2022/Mirthe Kamphuis/Master Thesis/08_from XH/other data for PoC/"

    os.chdir(datapath + path)
    dirlist = os.listdir()
    N = 51
    THR = 0.13
    dirlist
    S = 10

    if (dirlist != 0):
        img = OCY_read_stack(string)

        # use only first N images
        if (img.shape[2] > N):
            img = img[:, :, 1:N]

        # Gaussian filter
        img = skimage.filters.gaussian(img, sigma=0.65)

        # apply threshold and save initial binary volume
        bin_thr = OCY_thr_stack(img, THR)

        # morphological filtering, then save processed binary
        bins = OCY_fill_voids(bin_thr, 10)

        # detect cells
        cells = OCY_extract_cells(bins)
        cells_wrong = OCY_get_cells(bins)
        imwrite(datapath + path + '/' + 'cells.tif', cells_wrong)

        # #calculate initia skeleton
        skeleton = OCY_skeletonize(bins, cells_wrong)
        imwrite(datapath + path + '/' + 'skeleton.tif', skeleton)

        graph = skel2Graph(skeleton, cells_wrong)

    return graph