#!/usr/bin/python3

import numpy as np
from .. import basicx
# These are the "statx" functions from my octave library

# The following are not implemented:

# anova1x - Use numpy.stats.f_oneway
# anova2x - Use statsmodels with patsy (CNTC p. 1018)
# bestpart2 - Method seems poorly justified 
# cdf_norm --

def bootstrap(N, func, x, *args, **kargs):
    '''BOOTSTRAP - Calculate bootstrap statistics
    bs = BOOTSTRAP(N, func, x) calculates bootstrap statistics by calling
    the function FUN with subsamples from X. This is done N times, and
    the results are returned as an N-vector if FUNC returns scalar results.
    X is sliced into rows, that is, X must be shaped K, KxA, KxAxB, etc, where
    K is the number of data points.
    FUNC may return array results, in which case BST will be NxD1xD2x....
    bs = DBOOTSTRAP(N, func, x, y, ...) feeds extra arguments into FUNC.'''
    S = x.shape
    K = S[0]
    x = np.reshape(x, [K, np.prod(np.array(S[1:],int))]) # This is flipped wrt semiflatten
    res = None
    for n in range(N):
        idx = (np.random.rand(K)*K).astype(int)
        x1 = np.reshape(x[idx, :], S)
        res1 = func(x1, *args, **kargs)
        if res is None:
            isarray = type(res1)==np.ndarray
            if isarray:
                R = res1.shape
                res = np.zeros((N,R), dtype=res1.dtype)
            else:
                res = np.zeros(N, dtype=type(res))
        if isarray:
            res[n,:] = res1.flatten()
        else:
            res[n] = res1
    if isarray:
        R.insert(0, N)
        res = np.reshape(res, R)
    return res

def bs_resample(x, dim=0):
    '''BS_RESAMPLE - Random resampling from data for Bootstrap
    y = BS_RESAMPLE(x) returns a random resampling (with replacement) from
    the data X.
    This can be used for bootstrapping. For instance, if you want to know
    the uncertainty in some statistic f(X), you could do:

      ff=np.zeros(1000)
      for k in range(1000): ff[k] = f(bs_resample(x))
  
    Then, std(ff) will be the uncertainty in f, and, after ff = sort(ff),
    ff[25] and ff[975] represent the 95% confidence interval.

    This operates in the first dimension of X, even if X is 1xN.
    If X is NxD, this resamples the D-dimensional data as expected.
    Optional argument DIM specifies an alternative axis.

    Note that if ff[975]-ff[25] does not correspond to 4*std(ff), the
    output of the bootstrap is not normally distributed, which is a
    sign of potential trouble.
    It may be better to use "bootstrap bias-corrected accelerated" (BCa) 
    or "bootstrap tilting" methods to estimate the confidence intervals.'''

    y, s = basicx.semiflatten(x, dim)
    N = x.shape[1]
    nn = (np.random.rand(N)*N).astype(int)
    y = y[:, nn]
    return basicx.semiunflatten(y, s)

    
