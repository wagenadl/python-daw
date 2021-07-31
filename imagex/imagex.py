#!/usr/bin/python3

# From my octave library
# Imported so far:
# - convnorm
# - ace
# - imread and imwrite

import numpy as np
import cv2

from .. import basicx

def unpacktwople(v):
    '''UNPACKTWOPLE - Unpack (x,y) tuple, accepting it might just be x.
    Usage:
    x, y = UNPACKTWOPLE((x,y))
    x, y = UNPACKTWOPLE([x,y])
    x, y = UNPACKTWOPLE(z) # x = y = z'''
    vv = np.array([v]).flatten()
    if len(vv)==1:
        return vv[0], vv[0]
    elif len(vv)==2:
        return vv[0], vv[1]
    else:
        raise ValueError('Must be a scalar or pair of numbers')

def convnorm(img, kernel, dim=0):
    '''CONVNORM - Convolve with a kernel, normalizing at edges
    img = CONVNORM(img, kernel, dim) convolves the image IMG with the one-
    dimensional kernel KERNEL.
    DIM determines which dimension of IMG we operate on (default: 0).
    The convolution is normalized, even for image points near the edges.'''
    
    img, S = basicx.semiflatten(img, dim)
    K = len(kernel)
    N,L = img.shape

    if K>L:
        raise ValueError('Kernel must not be longer than image')
    if K%2 == 0:
        raise ValueError('Kernel must have odd length')
    
    kernel = kernel / np.sum(kernel)
    
    imo = img.copy()
    for n in range(N):
        imo[n,:] = np.convolve(img[n,:], kernel, 'same')

    K0 = K//2
    for k in range(K0):
        kern = kernel[K0-k:]
        kern /= np.sum(kern)
        kern = np.reshape(kern, [1, len(kern)])
        imo[:,k] = np.sum(img[:,:k+K0+1] * kern, 1)
    for k in range(L-K0, L):
        kern = kernel[:L-k+K0]
        kern /= np.sum(kern)
        kern = np.reshape(kern, [1, len(kern)])
        imo[:,k] = np.sum(img[:,k-K0:] * kern, 1)
        
    return basicx.semiunflatten(imo, S)

def ace(img, sig, r=None):
    '''ACE - Adaptive contrast enhancement
    res = ACE(img, sig) calculates an adaptive contrast enhanced image
    with given sigma.
    res = ACE(img, sig, r) also specifies a radius (default: 2.5*sig).
    res = ACE(img, (sigx, sigy), (rx, ry)) specifies sigma and radius 
    separately for X and Y directions.'''

    sigx, sigy = unpacktwople(sig)
    if r is None:
        rx, ry = 2.5*sigx, 2.5*sigy
    else:
        rx, ry = unpacktwople(r)

    rx = int(rx)
    ry = int(ry)
    xx = np.arange(-rx, rx+1)
    yy = np.arange(-ry, ry+1)
    gx = np.exp(-.5*xx**2/sigx**2)
    gy = np.exp(-.5*yy**2/sigy**2)
    dif = img - convnorm(convnorm(img, gx, 1), gy, 0)
    rms = np.sqrt(convnorm(convnorm(dif**2, gx, 1), gy, 0))
    rms = convnorm(convnorm(rms, gx, 1), gy, 0)
    return dif / (rms + np.mean(rms)/4)
    
def imread(ifn, dtype=None):
    '''IMREAD - Read an image file to a numpy array
    img = IMREAD(ifn) loads the named image file. The data may be
    Grayscale, RGB, or RGBA, depending on the file format.
    Optional argument DTYPE specifies conversion to the given type.
    If DTYPE is not an integer type, the image is scaled from [0, K]
    (where K=255 for 8-bit images, K=65535 for 16-bit images, etc)
    to [0, 1]. Try IMREAD(ifn, float).'''
    img = cv2.imread(ifn, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError('Could not open image file')
    if img.ndim==3:
        if img.shape[2]==3:
            img = img[:,:,[2,1,0]]
        elif img.shape[2]==4:
            img = img[:,:,[2,1,0,3]]
            
    if dtype is not None:
        mx = np.iinfo(img.dtype).max
        img = img.astype(dtype)
        if not np.issubdtype(dtype, np.integer):
            img /= mx
    return img

def imwrite(img, ofn, quality=None):
    '''IMWRITE - Write a numpy array as an image file
    IMWRITE(img, ofn) writes the image IMG to the named file. Most common
    file types are supported.
    Optional argument QUALITY specifies quality for jpeg.
    If the dtype of IMG is not any kind of integer,
    the assumption is that pixels should be scaled to [0, 255].'''
    if img.ndim==3:
        if img.shape[2]==3:
            img = img[:,:,[2,1,0]]
        elif img.shape[2]==4:
            img = img[:,:,[2,1,0,3]]
    if not np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.uint8)
    if quality is None:
        cv2.imwrite(ofn, img)
    else:
        cv2.imwrite(ofn, img, quality=quality)
