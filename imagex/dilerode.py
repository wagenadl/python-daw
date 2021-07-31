#!/usr/bin/python

import skimage.morphology as morph
import numpy as np

def erode4(img):
    '''ERODE4 - Erode image with a 4-neighborhood
    img = ERODE4(img) erodes the input image using a 4-neighborhood.
    The input must be a 2-d image which is binarized if necessary.
    The output is always a binary image (in uint8 format).'''

    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    return morph.binary_erosion(img)

def erode8(img):
    '''ERODE8 - Erode image with a 4-neighborhood
    img = ERODE8(img) erodes the input image using an 8-neighborhood.
    The input must be a 2-d image which is binarized if necessary.
    The output is always a binary image (in uint8 format).'''

    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    return morph.binary_erosion(img, selem=np.ones((3,3)))


def dilate4(img):
    '''DILATE4 - Dilate image with a 4-neighborhood
    img = DILATE4(img) dilates the input image using a 4-neighborhood.
    The input must be a 2-d image which is binarized if necessary.
    The output is always a binary image (in uint8 format).'''

    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    return morph.binary_dilation(img)

def dilate8(img):
    '''DILATE8 - Dilate image with a 4-neighborhood
    img = DILATE8(img) dilates the input image using an 8-neighborhood.
    The input must be a 2-d image which is binarized if necessary.
    The output is always a binary image (in uint8 format).'''

    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    return morph.binary_dilation(img, selem=np.ones((3,3)))

def edge4(img):
    '''EDGE4 - Find edges by erosion with a 4-neighborhood.
    img = EDGE4(img) finds edges in image, i.e., pixels that are on in the
    image but that are turned off by ERODE4.'''
    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    im1 = erode4(img)
    return np.logical_and(img, np.logical_not(im1))

def edge8(img):
    '''EDGE8 - Find edges by erosion with a 8-neighborhood.
    img = EDGE8(img) finds edges in image, i.e., pixels that are on in the
    image but that are turned off by ERODE8.'''
    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    im1 = erode8(img)
    return np.logical_and(img, np.logical_not(im1))

def thin(img):
    '''THIN - Single-step image thinning
    img = THIN(img) retreats from edges using a 4-neighborhood but does not
    disconnect regions or shrink objects to nothing. Here, connectedness is 
    based on an 8-neighborhood, and so are edges. (But the erosion is done
    in 4-neighborhood.
    The input must be a 2-d image which is binarized if necessary.
    The output is always a binary image (in uint8 format).'''

    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    return morph.thin(img, 1)

def skel(img):
    '''SKEL - Skeletonize image
    img = SKEL(img) is more or less like calling THIN repeatedly.
    The input must be a 2-d image which is binarized if necessary.
    The output is always a binary image (in uint8 format).'''

    if img.dtype != np.uint8:
        img = (img>0).astype(np.uint8)
    return morph.skeletonize(img)
    
