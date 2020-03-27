#!/usr/bin/python3

from . import basicx
import numpy as np
from scipy.signal import filtfilt

def boxcaravg(x, N, dim=0):
    '''BOXCARAVG - Apply box car averging to data
    y = BOXCARAVG(x, N) passes the vector through a 2N+1-point box-car
    averaging filter.
    If X is a matrix, works in first dimension. Optional argument DIM 
    overrides that default.'''
    x, s = basicx.semiflatten(x, dim)
    K, L = x.shape

    y = 0*x
    for n in range(N):
        y[:,n] = np.mean(x[:,:n+1], 1)
        y[:,L-n-1:] = np.mean(x[:,L-n-1:], 1)
    for n in range(-N, N+1):
        y[:, N:L-N] += x[:, N+n:L-N+n]

    y[:, N:L-N] /= 2*N + 1

    return basicx.semiunflatten(y, s)

def butterhigh1(f):
    '''BUTTERHIGH1 - Design first order high-pass Butterworth filter
    [b,a] = BUTTERHIGH1(f) creates a first order high-pass Butterworth filter
    with cutoff at f. (f=1 corresponds to the sample frequency, not half!)

    Filter coefficients lifted from http://www.apicsllc.com/apics/Sr_3/Sr_3.htm
    by Brian T. Boulter'''

    c = 1/np.tan(f*np.pi)

    n0 = c
    n1 = -c
    d0 = c+1
    d1 = -c+1

    a = [1, d1/d0]
    b = [n0/d0, n1/d0]
    return b, a

def butterhigh2(f):
    '''BUTTERHIGH2 - Design second order high-pass Butterworth filter
    [b,a] = BUTTERHIGH2(f) creates a second order high-pass Butterworth filter
    with cutoff at f. (f=1 corresponds to the sample frequency, not half!)

    Filter coefficients lifted from http://www.apicsllc.com/apics/Sr_3/Sr_3.htm
    by Brian T. Boulter'''

    c = 1/np.tan(f*np.pi)

    n0 = c**2
    n1 = -2*c**2
    n2 = c**2
    d0 = c**2+np.sqrt(2)*c+1
    d1 = -2*(c**2-1)
    d2 = c**2-np.sqrt(2)*c+1

    a = [1, d1/d0, d2/d0]
    b = [n0/d0, n1/d0, n2/d0]
    return b, a

def butterlow1(f):
    '''BUTTERLOW1 - Design first order low-pass Butterworth filter
    [b,a] = BUTTERLOW1(f) creates a first order low-pass Butterworth filter
    with cutoff at f. (f=1 corresponds to the sample frequency, not half!)
    
    Filter coefficients lifted from http://www.apicsllc.com/apics/Sr_3/Sr_3.htm
    by Brian T. Boulter'''

    c = 1/np.tan(f*np.pi)

    n0 = 1
    n1 = 1
    d0 = c+1
    d1 = -c+1

    a = [1, d1/d0]
    b = [n0/d0, n1/d0]
    return b, a

def butterlow2(f):
    '''BUTTERLOW2 - Design second order low-pass Butterworth filter
    [b,a] = BUTTERLOW2(f) creates a second order low-pass Butterworth filter
    with cutoff at F. (F=1 corresponds to the sample frequency, not half!)

    Filter coefficients lifted from http://www.apicsllc.com/apics/Sr_3/Sr_3.htm
    by Brian T. Boulter'''

    c = 1/np.tan(f*np.pi)

    n0 = 1
    n1 = 2
    n2 = 1
    d0 = c**2+np.sqrt(2)*c+1
    d1 = -2*(c**2-1)
    d2 = c**2-np.sqrt(2)*c+1

    a = [1, d1/d0, d2/d0]
    b = [n0/d0, n1/d0, n2/d0]
    return b, a

def fupsample(x, n, t=None, dim=0):
    '''FUPSAMPLE - Fourier upsampling 
    y, idx = FUPSAMPLE(x, n) uses Fourier upsampling to improve temporal
    resolution on signal X by a factor N.
    FUPSAMPLE operates on the first dimension of X, unless optional argument
    DIM overrides.
    On return, IDX is a real vector of pseudo-indices into X corresponding
    to the data in Y.
    y, tout = FUPSAMPLE(x, n, t), where T are time stamps, returns the time
    stamps of the output.
    CAUTION: if the length of X is odd, the last data point is dropped.'''

    x, s = basicx.semiflatten(x, dim)
    K, L = x.shape

    if n==1:
        if t is None:
            return basicx.semiunflatten(x, s), np.arange(L)
        else:
            return basicx.semiunflatten(x, s), t

    H = L//2
    f = np.fft.fft(x[:, :2*H])
    f = np.concatenate((f[:, :H],
                        np.zeros((K, H*2*(n-1)), f.dtype),
                        f[:, H:2*H]), 1)
    y = np.real(np.fft.ifft(f)) * n
    idx = np.arange(2*H)/n
    y = basicx.semiunflatten(y, s)
    if t is None:
        return y, idx
    else:
        return y, np.interp(idx, np.arange(len(t)), t)

def gaussianblur1d(img, rx, dim=0, normedge=False, radiusmul=4):
    '''GAUSSIANBLUR1D - One-dimensional Gaussian blurring
    img=GAUSSIANBLUR1D(img, r) performs a Gaussian blur on a 1d "image."
    More precisely, the input is convolved with a kernel

      exp[-½ (dx/rx)²]

    for dx is -4 rx to +4 rx. Optional argument RADIUSMUL overrides that
    factor 4.
    If IMG is multidimensional, operates on the first dimension, unless
    optional argument DIM overrides.
    Optional argument NORMEDGE normalizes the data near the edges taking
    into account that part of the Gaussian is outside the domain of the 
    data.'''
    L = int(np.ceil(radiusmul*rx))
    flt = np.exp(-.5*(np.arange(-L, L+1)/rx)**2)
    flt /= np.sum(flt)

    img, s = basicx.semiflatten(img, dim)

    K, N = img.shape
    res = np.zeros(img.shape, img.dtype)
    for k in range(K):
        res[k, :] = np.convolve(img[k, :], flt, 'same')
    if normedge:
        one = np.convolve(np.ones(N, img.dtype), flt, 'same')
        for k in range(K):
            res[k, :] /= one
    return basicx.semiunflatten(res, s)

def gaussianblur(img, rx, ry=None, normedge=False, radiusmul=4):
    '''GAUSSIANBLUR - Gaussian blurring on a 2-d image
    img = GAUSSIANBLUR(img, rx, ry) performs a Gaussian blur on the image
    IMG (which must be a 2d-matrix of size YxX.
    If ry is not specified, it defaults to rx.
    Optional arguments NORMEDGE and RADIUSMUL work as in GAUSSIANBLUR1D.'''

    if ry is None:
        ry = rx

    L = int(np.ceil(radiusmul*rx))
    fltx = np.exp(-.5*(np.arange(-L, L+1)/rx)**2)
    fltx /= np.sum(fltx)

    L = int(np.ceil(radiusmul*ry))
    flty = np.exp(-.5*(np.arange(-L, L+1)/ry)**2)
    flty /= np.sum(flty)
    
    Y, X = img.shape
    res = np.zeros(img.shape, img.dtype)
    for y in range(Y):
        res[y, :] = np.convolve(img[y, :], fltx, 'same')
    for x in range(X):
        res[:, x] = np.convolve(res[:, x], flty, 'same')

    if normedge:
        one = np.convolve(np.ones(X, img.dtype), fltx, 'same')
        for y in range(Y):
            res[y, :] /= one
        one = np.convolve(np.ones(Y, img.dtype), flty, 'same')
        for x in range(X):
            res[:, x] /= one
        
    return res

def gaussianinterp(xx, dat_x, dat_y, smo_x, err=False):
    '''GAUSSIANINTERP - Interpolate data using a Gaussian window
    yy = GAUSSIANINTERP(xx, dat_x, dat_y, smo_x) produces a smooth
    interpolation of the data: y(x) is estimated from all data points, 
    weighing them based on their distance to x:
                                                         
                     -½ (xᵢ ‎- x)² / smo²  
            sumᵢ yᵢ e
    y(x) = ------------------------------                
                      -½ (xᵢ ‎- x)² / smo²  
            sumᵢ    e

    where xᵢ are the elements of DAT_X and yᵢ are the elements of DAT_Y.
    Current implementation assumes data is 1D.
    Algorithm is not fast: Time is O(X*D) where X is the length of XX
    and D is the length of DAT_X.'''

    N = len(xx)
    yy = np.zeros(xx.shape, xx.dtype)
    for n in range(N):
        wei = np.exp(-.5*(dat_x - xx[n])**2 / smo_x**2)
        yy[n] = np.sum(dat_y*wei) / np.sum(wei)

    if not err:
        return yy

    sy = np.zeros(xx.shape, xx.dtype)
    y_i = np.interp(xx, yy, dat_x)
    s_i = dat_y - y_i
    bad = np.nonzero(np.isnan(s_i))
    s_i[bad] = 0
    for n in range(N):
        wei = np.exp(-.5*(dat_x - xx[n])**2 / smo_x**2)
        wei[bad] = 0
        wei /= np.sum(wei)
        eff_n = 1/np.max(wei)
        sy[n] = np.sqrt(np.sum(wei*s_i**2)) / np.sqrt(eff_n)
    return yy, sy
        
def hermiteinterp(xx, n, bias=0, tension=0):
    '''HERMITEINTERP - Upsampling using Hermite interpolation
    yy, tt = HERMITEINTERP(xx, n, bias, tension) performs equally spaced 
    Hermite interpolation of the data XX to N times the original frequency,
    returning the interpolated data and new time point vector.
    BIAS defaults to 0, useful range is -1 (toward 2nd segment) 
    to +1 (toward first segment).
    TENSION defaults to 0, useful range is -1 (low) to 1 (high).
    
    Algorithm lifted from 
    http://astronomy.swin.edu.au/~pbourke/other/interpolation/,
    by Paul Bourke, December 1999. See E&R p. 1671.'''

    xx, s = basicx.semiflatten(xx)
    K, L0 = xx.shape
    tt = np.arange(0, L0-.999, 1/n)
    Y = len(tt)
    base = np.floor(tt).astype(int)
    mu = tt - base
    mu2 = mu**2
    mu3 = mu**3
    yy = np.zeros((K,Y), xx.dtype)
    for k in range(K):
        x = np.hstack((xx[k,0], xx[k,:], xx[k,-1], xx[k, -1]))
        m0 = (x[base+1]-x[base+0]) * (1+bias)*(1-tension)/2 \
             + (x[base+2]-x[base+1]) * (1-bias)*(1-tension)/2
        m1 = (x[base+2]-x[base+1]) * (1+bias)*(1-tension)/2 \
             + (x[base+3]-x[base+2]) * (1-bias)*(1-tension)/2
        a0 =  2*mu3 - 3*mu2 + 1
        a1 =    mu3 - 2*mu2 + mu
        a2 =    mu3 -   mu2
        a3 = -2*mu3 + 3*mu2
        y = a0*x[base+1] + a1*m0 + a2*m1 + a3*x[base+2]
        yy[k,:] = y
    return basicx.semiunflatten(yy, s), tt

def medianflt(xx, dim=0):
    '''MEDIANFLT - Three-point median filter
    yy = MEDIANFLT(xx) passes the vector XX through a 3 point median filter.
    For multidimension data, works along the DIM-th axis (default: first).'''
    xx, s = basicx.semiflatten(xx)
    K,L = xx.shape
    yy = np.zeros(xx.shape, xx.dtype)
    for k in range(K):
        x1 = xx[k,:]
        x2 = np.hstack((xx[k,0], xx[k,:-1]))
        x3 = np.hstack((xx[k,1:], xx[k,-1]))
        yy[k,:] = np.median(np.stack((x1,x2,x3), 0), 0)
    return basicx.semiunflatten(yy, s)

def medianfltn(xx, N, dim=0):
    '''MEDIANFLT - Abitrary median filter
    yy = MEDIANFLT(xx, n) passes the vector XX through a 2n+1-point median 
    filter.
    For multidimension data, works along the DIM-th axis (default: first).'''
    
    xx, s = basicx.semiflatten(xx)
    K,L = xx.shape
    yy = np.zeros(xx.shape, xx.dtype)
    for k in range(K):
        x1 = xx[k,:]
        z = np.repeat(np.reshape(x1, (1,L)), 2*N+1, 0)
        for n in range(-N, N+1):
            z[n+N, N+n:L-N+n] = x1[N:L-N]
        yy[k,:] = np.median(z, 0)
    return basicx.semiunflatten(yy, s)

def templatefilter(xx, f_s, f_line=60, f_max=500, nperiods=50, dim=0,
                   xref=None):
    '''TEMPLATEFILTER - Remove 60 Hz line noise by template filtering
    yy = TEMPLATEFILTER(xx, f_s) removes (nearly) periodic noise (such as
    60 Hz line pickup) from the signal XX. 
    F_S must be the sample frequency of the signal (in Hertz).
    Optional arguments:
    F_LINE sets the frequency of the periodic noise; default is 60 Hz.
    F_MAX is the maximum frequency (in Hz) expected to exist in the 
    periodic noise; noise (or signal) above that frequency is not 
    treated. Default: F_MAX = 500.
    NPERIODS is the number of periods to use for estimation; default: 50.
    TEMPLATEFILTER works on vectors, or on the DIM-th axis of arrays.
    Instead of specifying F_LINE, you can also specify a reference signal 
    in XREF. In that case, the period is determined from the upward
    crossings of the reference.
    Ordinarilty, a Butterworth low-pass filter is used to create the 
    estimate to be subtracted. Set NPERIODS to a negative number to use
    a Gaussian blur filter with RX = |NPERIODS| instead. This has better
    behavior near the start and end of the signal, but is slower.'''

    xx, s = basicx.semiflatten(xx, dim)
    K, X = xx.shape
    yy = np.zeros(xx.shape, xx.dtype)

    # Step zero: convert a reference signal into a period.
    if xref is not None:
        from . import peakx
        m = np.mean(xref)
        s = np.std(xref)
        ion, iof = peakx.schmitt(xref, m+s/2, m-s/2)
        f_line = f_s / np.mean(np.diff(ion))

    if K>1:
        for k in range(K):
            yy[k,:] = templatefilter(xx[k,:], f_s, f_line, f_max, nperiods)
        return yy

    xx = xx.flatten() # Let's make this easy

    # Step one: resample the original signal to make period_sams be integer.
    period_sams = f_s / f_line
    int_sams = int(period_sams)
    rat = period_sams / int_sams
    zz = np.interp(np.arange(0, X, rat), np.arange(X), xx)

    # Step two: reshape into a matrix with one period per row (dropping
    # the final partial period).
    Z = len(zz)
    N = Z//int_sams
    zz = np.reshape(zz[:N*int_sams], [N, int_sams])
    
    # Step three: filter consecutive periods
    if nperiods>0:
        b, a = butterlow1(1/nperiods)
        zz = filtfilt(b, a, zz, axis=0)
    else:
        #zz = medianfltn(zz, -nperiods, dim=0)
        zz = gaussianblur1d(zz, -nperiods, dim=0, normedge=True, radiusmul=2)
        
    # Step four: Smooth the template by assuming there are no
    # high frequency components to the pickup.
    if f_max is not None:
        b,a = butterlow1(f_max/f_s)
        zz = filtfilt(b, a, zz, axis=1)

    # Step five: add an extra period at the end, based on the final period,
    # to compensate for data cut in step two
    zz = np.concatenate((zz, zz[-1:,:]), 0)

    # Step six: remove DC from the template
    zz -= np.mean(zz, 1, keepdims=True)

    # Step seven: reshape back to a vector, resample back to original f_s
    zz = zz.flatten()
    Z = len(zz)
    zz = np.interp(np.arange(0, Z, 1/rat), np.arange(Z), zz)
    
    # Step eight: subtract the template from the original signal
    yy = xx - zz[:X]

    return basicx.semiunflatten(yy, s)
