#!/usr/bin/python3

import numpy as np
from . import basicx
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


def tukeyLetters(pp, means=None, alpha=0.05):
    '''TUKEYLETTERS - Produce list of group labels for TukeyHSD
    letters = TUKEYLETTERS(pp), where PP is a symmetric matrix of 
    probabilities from a Tukey test, returns alphabetic labels
    for each group to indicate clustering. PP may also be a vector
    from PAIRWISE_TUKEYHSD.
    Optional argument MEANS specifies group means, which is used for
    ordering the letters. ("a" gets assigned to the group with lowest
    mean.) Without this argument, ordering is arbitrary.
    Optional argument ALPHA specifies cutoff for treating groups as
    part of the same cluster.'''

    if len(pp.shape)==1:
        # vector
        G = int(3 + np.sqrt(9 - 4*(2-len(pp))))//2
        ppp = .5*np.eye(G)
        ppp[np.triu_indices(G,1)] = pp    
        pp = ppp + ppp.T
    conn = pp>alpha
    G = len(conn)
    if np.all(conn):
        return ['a' for g in range(G)]
    conns = []
    for g1 in range(G):
        for g2 in range(g1+1,G):
            if conn[g1,g2]:
                conns.append((g1,g2))

    letters = [ [] for g in range(G) ]
    nextletter = 0
    for g in range(G):
        if np.sum(conn[g,:])==1:
            letters[g].append(nextletter)
            nextletter += 1
    while len(conns):
        grp = set(conns.pop(0))
        for g in range(G):
            if all(conn[g, np.sort(list(grp))]):
                grp.add(g)
        for g in grp:
            letters[g].append(nextletter)
        for g in grp:
            for h in grp:
                if (g,h) in conns:
                    conns.remove((g,h))
        nextletter += 1

    if means is None:
        means = np.arange(G)
    means = np.array(means)
    groupmeans = []
    for k in range(nextletter):
        ingroup = [g for g in range(G) if k in letters[g]]
        groupmeans.append(means[np.array(ingroup)].mean())
    ordr = np.empty(nextletter, int)
    ordr[np.argsort(groupmeans)] = np.arange(nextletter)
    result = []
    for ltr in letters:
        lst = [chr(97 + ordr[x]) for x in ltr]
        lst.sort()
        result.append(''.join(lst))
    return result    


def tukeyHSD(groupdata, alpha=0.05):
    '''TUKEYHSD - Perform Tukey's Honestly Significant Differences test
    pp, letters = TUKEYHSD(groupdata), where GROUPDATA is a list of 
    data vectors calculates a probability matrix P_ij that indicates 
    the likelihood of obtaining the observed data according to the null
    model that groups I and J are no different. The data must be presented
    as a list of vectors, such that the k-th vector represents the data
    for the k-th group.
    LETTERS returns useful labels to attach to the groups: Groups that
    are (are not) significantly different from each other share (do not 
    share) a letter. These are assigned such that the group with least
    mean gets the letter 'a', etc.
    Normally, TUKEYHSD is used after an ANOVA.'''
    
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    data = []
    group = []
    for k, grp in enumerate(groupdata):
        for x in grp:
            data.append(x)
            group.append(k)
    tbl = pairwise_tukeyhsd(data, group)
    G = len(groupdata)
    pp = np.zeros((G,G)) + 1
    k = 0
    for g1 in range(G-1):
        for g2 in range(g1+1, G):
            pp[g1,g2] = tbl.pvalues[k]
            pp[g2,g1] = tbl.pvalues[k]
            k += 1

    means = [ np.mean(grp) for grp in groupdata ]
    letters = tukeyLetters(pp, means, alpha)
    return pp, letters

def games_howellHSD(groupdata, alpha=0.05):
    from scipy.stats import studentized_range
    means = [ np.mean(grp) for grp in groupdata ]
    varis = [ np.var(grp) for grp in groupdata ]
    nn = [ len(grp) for grp in groupdata ]
    G = len(means)
    qq = np.zeros((G,G))
    pp = np.zeros((G,G)) + 1
    for g1 in range(G-1):
        for g2 in range(g1+1, G):
            n1 = nn[g1]
            n2 = nn[g2]
            v1 = varis[g1] / n1
            v2 = varis[g2] / n2
            err = np.sqrt((v1 + v2) / 2)
            qq[g1,g2] = qq[g2,g1] = np.abs(means[g1] - means[g2]) / err
            df = (v1 + v2)**2 / (v1**2/(n1-1) + v2**2/(n2-1))
            pp[g1,g2] = pp[g2,g1] = 1 - studentized_range.cdf(qq[g1, g2],G,df)
    letters = tukeyLetters(pp, means, alpha)
    return pp, letters
    
