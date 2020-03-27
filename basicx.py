#!/usr/bin/python3

# Notes on functions that I did not translate:
#
# argmax, argmin are trivial in python
# asize is irrelevant
# atoi is trivial
# autoipermute and autopermute have not been used in ages
# call is irrelevant
# candela2lumens and friends are in physx
# cellxfun is irrelevant
# chd is irrelevant
# clip01 became clip
# dbg is trivial in pytohn
# ddetrend has been renamed detrend
# div is trivial
# dpolyarea has been renamed polyarea
# drop is np.delete
# dshift is roll is np.roll
# dsqueeze is np.squeeze
# getopt is not needed in python
# identity, id is trivial in python
# num2thou is mainly implemented by python's f'{num:,}'.
# select doesn't seem so useful to me now
# savestruct is subsumed in ppersist
# structcat is a great idea but not as obvious in python
# same for structcut, subset
# swap is trivial in python
# ternary is now ifelse
# most of the useful functionality of uniq is in np.unique
# unshape is subsumed in the new semiflatten

import datetime
import numpy as np

# Sieve of Eratosthenes
# Code by David Eppstein, UC Irvine, 28 Feb 2002
# http://code.activestate.com/recipes/117119/
def gen_primes():
    """GEN_PRIMES - Generate an infinite sequence of prime numbers
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}
    
    # The running integer that's checked for primeness
    q = 2
    
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        
        q += 1
        
def primes(n):
    '''PRIMES - Return the first N primes
    pp = PRIMES(n) returns a list containing the first N primes.'''
    res = []
    if n<=0:
        return res
    for p in gen_primes():
        res.append(p)
        n -= 1
        if n<=0:
            return res

def semiflatten(x, d=0):
    '''SEMIFLATTEN - Permute and reshape an array to convenient matrix form
    y, s = SEMIFLATTEN(x, d) permutes and reshapes the arbitrary array X so 
    that input dimension D (default: 0) becomes the second dimension of the 
    output, and all other dimensions (if any) are combined into the first 
    dimension of the output. The output is always 2-D, even if the input is
    only 1-D.
    If D<0, dimensions are counted from the end.
    Return value S can be used to invert the operation using SEMIUNFLATTEN.
    This is useful to facilitate looping over arrays with unknown shape.'''
    x = np.array(x)
    shp = x.shape
    ndims = x.ndim
    if d<0:
        d = ndims + d
    perm = list(range(ndims))
    perm.pop(d)
    perm.append(d)
    y = np.transpose(x, perm)
    # Y has the original D-th axis first, followed by the other axes, in order
    rest = np.array(shp, int)[perm[:-1]]
    y = np.reshape(y, [np.prod(rest), y.shape[-1]])
    return y, (d, rest)

def semiunflatten(y, s):
    '''SEMIUNFLATTEN - Reverse the operation of SEMIFLATTEN
    x = SEMIUNFLATTEN(y, s), where Y, S are as returned from SEMIFLATTEN,
    reverses the reshaping and permutation.'''
    d, rest = s
    x = np.reshape(y, np.append(rest, y.shape[-1]))
    perm = list(range(x.ndim))
    perm.pop()
    perm.insert(d, x.ndim-1)
    x = np.transpose(x, perm)
    return x

def detrend(x, p=1, dim=0):
    '''DETREND - Remove polynomial trend from data
    y = DETREND(x) removes linear trend from the data in X.
    Optional argument P can be set to 0 to only remove constant baseline or
    to an arbitrary positive integer to remove a polynomial trend.
    If X is multidimensional, DETREND works the dimension specified by
    optional argument DIM, which defaults to 0.'''

    typ = x.dtype
    y, s = semiflatten(x.copy(), dim)
    N, L = y.shape
    rng = np.arange(L,dtype=typ) - L/2
    for n in range(N):
        fit = np.polynomial.polynomial.polyfit(rng, y[n,:], p).astype(typ)
        y[n,:] -= fit[0]
        if p>=1:
            y[n,:] -= fit[1]*rng
        for q in range(2, p+1):
            y[n,:] -= fit[q]*rng**q
    return semiunflatten(y, s)

def clip(x, min=0, max=1):
    '''CLIP - Clip values to a range
    y = CLIP(x) clips the values in X to the range [0, 1].
    Optional arguments MIN, MAX override limits.'''
    return np.clip(x, min, max)

def dateadd(yymmdd, days):
    '''DATEADD - Add a number of days to a YYMMDD-formatted date
    y = DATEADD(x, days), where X is a date in YYMMDD string format,
    adds the given number of DAYS to the date and returns the result
    as another YYMMDD-formatted string. DAYS may be negative.'''
    if len(yymmdd)!=6:
        return ValueError('YYMMDD must be 6 digits')
    yy = int(yymmdd[:2]) + 2000
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:])
    ymd = datetime.date(yy, mm, dd)
    ymd += datetime.timedelta(days=days)
    return f'{ymd.year-2000:02}{ymd.month:02}{ymd.day:02}'

def datesub(ymd1, ymd2):
    '''DATESUB  Subtract two dates in YYMMDD form to get number of days.
    days = DATESUB(yymmdd1, yymmdd2) subtracts YYMMDD2 from YYMMDD1 and
    returns the result expressed as an integer number of days.'''
    if len(ymd1)!=6:
        return ValueError('YMD1 must be 6 digits')
    if len(ymd2)!=6:
        return ValueError('YMD2 must be 6 digits')
    y1 = int(ymd1[:2]) + 2000
    m1 = int(ymd1[2:4])
    d1 = int(ymd1[4:])
    y2 = int(ymd2[:2]) + 2000
    m2 = int(ymd2[2:4])
    d2 = int(ymd2[4:])
    x1 = datetime.date(y1, m1, d1)
    x2 = datetime.date(y2, m2, d2)
    return (x1-x2).days

def timeadd(hhmmss, seconds):
    '''TIMEADD - Add a number of days to a HHMMSS-formatted time
    y = TIMEADD(x, seconds), where X is a time in HHMMSS string format,
    adds the given number of DAYS to the time and returns the result
    as another HHMMSS-formatted string. DAYS may be negative.'''
    if len(hhmmss)==4:
        hhmmss += '00'
    if len(hhmmss)!=6:
        return ValueError('HHMMSS must be 4 or 6 digits')
    yy = int(hhmmss[:2])
    mm = int(hhmmss[2:4])
    if len(hhmmss)==6:
        dd = int(hhmmss[4:])
    else:
        dd = 0
    ymd = datetime.datetime(2020, 1, 1, yy, mm, dd)
    ymd += datetime.timedelta(seconds=seconds)
    return f'{ymd.hour:02}{ymd.minute:02}{ymd.second:02}'

def timesub(hms1, hms2):
    '''TIMESUB  Subtract two times in HHMMSS form to get number of days.
    days = TIMESUB(hhmmss1, hhmmss2) subtracts HHMMSS2 from HHMMSS1 and
    returns the result expressed as an integer number of seconds.'''
    if len(hms1)==1:
        hms1 += '00'
    if len(hms2)==1:
        hms2 += '00'
    if len(hms1)!=6:
        return ValueError('HMS1 must be 4 or 6 digits')
    if len(hms2)!=6:
        return ValueError('HMS2 must be 4 or 6 digits')
    y1 = int(hms1[:2]) 
    m1 = int(hms1[2:4])
    d1 = int(hms1[4:])
    y2 = int(hms2[:2]) 
    m2 = int(hms2[2:4])
    d2 = int(hms2[4:])
    x1 = datetime.datetime(2020, 1, 1, y1, m1, d1)
    x2 = datetime.datetime(2020, 1, 1, y2, m2, d2)
    return (x1-x2).seconds

def polyarea(x,y):
    '''POLYAREA - Area inside a polygon in 2D space.
    a = POLYAREA(x,y), where X and Y are vectors, calculates the area 
    of the polygon with vertices (X_i,Y_i).
    Result is positive if vertices run counterclockwise around the polygon,
    otherwise negative.'''

    x = np.append(x, x[0])
    y = np.append(y, y[0])
    N=len(x)
    dx1 = x[1:-1] - x[0]
    dx2 = x[2:] - x[0];
    dy1 = y[1:-1] - y[0]
    dy2 = y[2:] - y[0]

    return np.sum(dx1*dy2 - dx2*dy1)/2

def equal(a, b=0, eps=1e-10):
    '''EQUAL - Test if two numbers are very nearly the same
    EQUAL(a, b) returns true if the difference between A and B is 
    less than 1e-10 or less than 1e-10*max(abs(A),abs(B)), whichever is more.
    EQUAL(a) tests A against zero.
    Optional argument EPS overrides the threshold.'''
    if type(a)==list:
        a = np.array(a)
        lst = True
    if type(b)==list:
        b = np.array(b)
    thr = eps * max(1, np.max(np.abs(a)), np.max(np.abs(b)))
    return np.abs(a-b) < thr

def inpoly(x, y, xx, yy):
    '''INPOLY - Does a point lie inside a polygon?
    INPOLY(x, y, xx, yy) returns True if the point (X, Y) lies inside the
    polygon defined by the points (XX_i, YY_i).'''

    xx = np.array(xx)
    yy = np.array(yy)
    xx1 = np.roll(xx, -1)
    yy1 = np.roll(yy, -1)
    dx = xx - x
    dy = yy - y
    dx1 = xx1 - x
    dy1 = yy1 - y
    sg = np.sign(dx*dy1 - dy*dx1)
    sg *= sg[0]
    return not np.any(sg<0)

def inrange(x, min, max, strict=False):
    '''INRANGE - Do numbers lie inside an interval?
    INRANGE(x, min, max) returns True if X lies in the interval [min, max].
    Also works for arrays.
    Optional argument STRICT specifies that the test should be whether
    X lies in the open interval (min, max).'''
    if strict:
        if type(x)==list or type(x)==np.ndarray:
            return np.logical_and(x>min, x<max)
        else:
            return x>min and x<max
    else:
        if type(x)==list or type(x)==np.ndarray:
            return np.logical_and(x>=min, x<=max)
        else:
            return x>=min and x<=max
    
def inrect(xy, xywh):
    '''INRECT - Is a point inside a rectangle?
    INRECT(xy,xywh) returns True if XY falls within XYWH, False otherwise.'''
    return xy[0] >= xywh[0] and xy[0] < xywh[0] + xywh[2] \
        and xy[1] >= xywh[1] and xy[1] < xywh[1] + xywh[3]

def invperm(x):
    '''INVPERM  Inverse permutation of a vector of numbers
    y = INVPERM(x), where X is a permutation of [0:N-1]], returns the
    inverse permutation, i.e. the vector Y s.t. y[x] = x[y] = [0:N-1].'''
    y = np.zeros(len(x), dtype=int)
    y[np.array(x)] = np.arange(len(x), dtype=int)
    if type(x)==list:
        return list(y)
    elif type(x)==tuple:
        return tuple(y)
    else:
        return y

def isnscalar(x):
    '''ISNSCALAR - True if array is a numeric scalar
    ISNSCALAR(x) returns True if X is a simple number (e.g., float, int, etc)
    or if X is a numpy array containing precisely one element.'''    
    import numbers
    if isinstance(x, numbers.Number):
        return True
    elif type(x)==np.ndarray:
        return isinstance(x.flat[0], numbers.Number) and np.prod(x.shape)==1
    else:
        return False
    
def asscalar(x):
    '''ASSCALAR - Returns X as a scalar
    ASSCALAR(x), where X is an object for which ISNSCALAR returns True,
    returns that scalar value. Warning: Garbage in, garbage out.'''
    if type(x)==np.ndarray:
        return x.flat[0]
    else:
        return x
    
def isnvector(x):
    '''ISNVECTOR - True if array is a numeric vector
    ISNVECTOR(x) returns True if X is a scalar, or a numpy array that
    (1) contains numbers and (2) has at most one non-singleton dimension.'''
    try:
        y = np.array(x)
        if len(y)==0:
            return True
        z = y.flat[0] + 0
        return np.prod(y.shape)==np.max(np.append(y.shape, 1))
    except:
        return False
    
def lin(x):
    '''LIN - Identity function.
    y=LIN(x) is the identity function.
    This is useful, e.g., in code like this:
    
      if LOGARITHMIC:
         foo = log
      else:
         foo = lin
      plot(foo(x))
    '''
    
    return x

def outer(v, w):
    '''OUTER  Cross product between vectors in 3D space.
    u = OUTER(v, w) computes the outer (hat) product between a pair of
    3D vectors. This also works for Nx3 arrays, which are treated row by
    row.'''

    v, sv = semiflatten(v, -1)
    w, sw = semiflatten(w, -1)
    res = np.stack((v[:,1]*w[:,2] - v[:,2]*w[:,1],
                    v[:,2]*w[:,0] - v[:,0]*w[:,2],
                    v[:,0]*w[:,1] - v[:,1]*w[:,0]), 1)
    return semiunflatten(res, sv)

def repmatto(x, s):
    '''REPMATTO - Replicate and tile an array to a given size
    y = REPMATTO(x, s), where S is a size vector, replicates the array X to
    give it the size S. Each component of S must be an integer multiple of 
    the current size of X in that dimension.'''
    dd = list(x.shape)
    while len(dd) < len(s):
        dd.append(1)
    x = np.reshape(x, dd)
    x = np.tile(x, [s[a]//dd[a] for a in range(len(dd))])
    return x

def ifelse(cond, x1, x2):
    '''IFELSE - Implements the C ternary operator
    y = IFELSE(cond, x1, x2), where COND is a boolean, returns X1 if COND
    is True, else X2.
    If COND is an array, it must have the same shape as X1 and X2, and
    the operation is done elementwise.'''
    if type(cond)==list or type(cond)==np.ndarray:
        y = x2
        nz = np.nonzero(cond.flat)
        y.flat[nz] = x1.flat[nz]
        return y
    else:
        if cond:
            return x1
        else:
            return x2
        
def xxyy(n, m=None, norm=False):
    '''XXYY - Return a pair of XX and YY matrices
    [xx, yy] = XXYY(N, M) returns a pair of NxM matrices where XX increases
    from 0 to M-1 from left to right and YY increases from 0 to N-1 from top to
    bottom.
    [xx, yy] = XXYY(N) returns square NxN matrices.
    Optional argument NORM scales the output to go from 0 to 1.'''
    if m is None:
        m = n
    x = np.arange(m)
    y = np.arange(n)
    if norm:
        x = x / (m-1)
        y = y / (n-1)
    xx = repmatto(np.reshape(x, (1,m)), (n,m))
    yy = repmatto(np.reshape(y, (n,1)), (n,m))
    return xx, yy
