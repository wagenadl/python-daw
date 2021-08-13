#!/usr/bin/python3

import numpy as np
from .. import peakx

_corrfac = {}
def _estimatemuckfactor(chunksize, nchunks=1000, ipart=25):
    myid = (chunksize, nchunks, ipart)
    if myid in _corrfac:
        return _corrfac[myid]
    dat = np.reshape(np.random.randn(chunksize*nchunks), (chunksize, nchunks))
    rms = np.std(dat, 0)
    K = int(ipart * nchunks / 100)
    est = np.partition(rms, K)[K]
    _corrfac[myid] = est
    return est

def noiseest(dat, chunksize=300, percentile=25):
    '''NOISEEST - Estimate noise in a data trace
    
    rms = NOISEEST(dat) estimates the RMS noise in the given data (a
    1-d numpy array).  It is relatively insensitive to genuine spikes
    and simulus artifacts because it splits the data into small chunks
    and does the estimation within chunks. It then bases the final
    answer on the 25th percentile of estimates in chunks and corrects
    based on assumption of near Gaussianity. 
    Optional argument CHUNKSIZE specifies the length of chunks;
    PERCENTILE specifies an alternative percentile for estimation. '''
    L = len(dat)
    N = L//chunksize # number of chunks
    cf = _estimatemuckfactor(chunksize)
    dat = np.reshape(dat[:N*chunksize], [N, chunksize])
    rms = np.std(dat, 1)
    K = int(.25*N + .5)
    est = np.partition(rms, K)[K]
    return est / cf

def detectspikes(yy, threshold, polarity=0, tkill=50):
    '''DETECTSPIKES - Simple spike detection
    idx = DETECTSPIKES(yy, threshold) performs simple spike detection:
      (1) Find peaks YY > THRESHOLD;
      (2) Drop minor peaks within TKILL samples (default: 50) of major peaks;
      (3) Repeat for peaks YY < -THRESHOLD.
    Optional argument POLARITY limits to positive (or negative) peaks if
    POLARITY > 0 (or < 0).'''

    def droptoonear(ipk, hei, tkill):
        done = False
        while not done:
            done = True
            for k in range(len(ipk) - 1):
                if ipk[k+1] - ipk[k] < tkill:
                    done = False
                    if hei[k] < hei[k+1]:
                        hei[k] = 0
                    else:
                        hei[k+1] = 0
            idx = np.nonzero(hei)
            ipk = ipk[idx]
            hei = hei[idx]
        return ipk
            
    if polarity>=0:
        iup, idn = peakx.schmitt(yy, threshold, 0)
        ipk = peakx.schmittpeak(yy, iup, idn)
        if tkill is not None:
            ipk = droptoonear(ipk, yy[ipk], tkill)
    else:
        ipk = None   

    if polarity<=0:
        zz = -yy
        iup, idn = peakx.schmitt(zz, threshold, 0)
        itr = peakx.schmittpeak(zz, iup, idn)
        if tkill is not None:
            itr = droptoonear(itr, zz[itr], tkill)
    else:
        itr = None

    if ipk is None:
        return itr
    elif itr is None:
        return ipk
    else:
        return np.sort(np.append(ipk, itr))

def cleancontext(idx, dat, test=(np.arange(-25,-12), np.arange(12,25)),
                 testabs=(np.arange(-25,-4), np.arange(4,25)),
                 thr=.50, absthr=.90):
    '''CLEANCONTEXT - Drop spikes if their context is not clean
    idx = CLEANCONTEXT(idx, dat) treats the spikes at IDX (from DETECTSPIKES
    run on DAT) to the classic filtering operation in MEABench. That is,
    spikes are dropped if there are samples with voltage >50% of peak
    voltage at a distance of 12 to 25 samples from the main peak, or
    if there are samples with absolute voltage >90% of peak at a distance
    of 4 to 25 samples. 
    Optional argument TEST may specify a list of ranges to test at 50%;
    TESTABS may specify a list of ranges to test at 90% with abs. value.
    Optional arguments THR and ABSTHR override the 0.5 and 0.9 default
    test thresholds. Set THR or ABSTHR to None to avoid a test.
    Spikes too near the start or end of the recording are dropped 
    unconditionally.'''
    keep = np.zeros(idx.shape, dtype=idx.dtype)
    test = np.concatenate(test)
    testabs = np.concatenate(testabs)
    if thr is None:
        thr = 10000
    if absthr is None:
        absthr = 10000
    t0 = np.min([np.min(test), np.min(testabs)])
    t1 = np.max([np.max(test), np.max(testabs)])
    T = len(dat)
    hei = dat[idx]
    pol = np.sign(hei)
    for k in range(len(idx)):
        t = idx[k]
        if t+t0 < 0 or t+t1 >= T:
            continue
        if pol[k]>0:
            if any(dat[t+test] > thr*hei[k]):
                continue
        else:
            if any(dat[t+test] < thr*hei[k]):
                continue
        if any(np.abs(dat[t+testabs]) > absthr*np.abs(hei[k])):
            continue
        keep[k] = t
    return keep[keep > 0]
        
if __name__ == '__main__':
    import scipy.signal
    import pyqplot as qp
    b, a = scipy.signal.butter(3, .1)
    N = 1000
    tt = np.arange(N) / 1e3
    rnd = np.random.randn(N)
    flt = scipy.signal.filtfilt(b, a, rnd)
    qp.figure('/tmp/s1')
    qp.pen('b', 1)
    qp.plot(tt, flt)
    idx = detectspikes(flt, .1, tkill=0)
    qp.pen('r')
    qp.marker('o', 2)
    qp.mark(tt[idx], flt[idx])
    id2 = cleancontext(idx, flt)
    qp.marker('o', 4, fill='open')
    qp.pen('g', 1)
    qp.mark(tt[id2], flt[id2])
    qp.pen('k', .5)
    qp.xaxis()
    qp.yaxis()
    qp.shrink()

    
