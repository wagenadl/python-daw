# multitaper.py - multitaper spectral estimates
# This is DAW's translation into python of Adam Taylor's matlab code

import numpy as np
from scipy.signal.windows import dpss
import matplotlib.pyplot as plt
import scipy.stats


def _sum_pos_neg_freqs(Pxxs_ts):
    '''turns a two-sided PSD into a one-sided
    works on the the cols of Pxx_ts
    doesn't work for ndims>3'''
    
    # get the dims of Pxx_ts
    N, N_signals, K  = Pxxs_ts.shape
    
    # fold the positive and negative frequencies together
    # hpfi = 'highest positive frequency index'
    # also, generate frequency base
    hpfi = int(np.ceil(N/2))
    if N % 2 == 0: # if N_fft is even
        Pxxs_os1 = np.concatenate((Pxxs_ts[:hpfi,:,:],
                                   np.zeros((1,N_signals,K))), 0)
        Pxxs_os2 = np.concatenate((np.zeros((1,N_signals,K)),
                                   np.flip(Pxxs_ts[hpfi:,:,:], axis=0)), 0)
        Pxxs_os = Pxxs_os1 + Pxxs_os2
        f_os = np.arange(hpfi+1) / N
    else:
        Pxxs_os1 = Pxxs_ts[:hpfi,:,:]
        Pxxs_os2 = np.concatenate((np.zeros((1,N_signals,K)),
                                   np.flip(Pxxs_ts[hpfi:,:,:], axis=0)), 0)
        Pxxs_os =  Pxxs_os1 + Pxxs_os2
        f_os = np.arange(hpfi) / N
    return Pxxs_os, f_os


def mtpsd(x, f_s=1, f_res=None, nw=None, indiv=False):
    '''MTPSD - Multitaper power spectral density
    This is DW's adaptation of Adam Taylor's PDS_MTM code
    ff, Pxx = MTPSD(xx, f_s, f_res) calculates one-side multi-taper
    spectrogram.
    
      XX [TxN] is the data.
      F_S is the sampling rate.
      F_RES is the half-width of the transform of the tapers used.
    
      FF [Fx1] is the resulting (one-sided) frequency base.
      PXX [FxN] are the spectral estimates for the data XX at frequencies FF.

    If optional argument INDIV is true, PXX [FxNxK] are data for each 
    taper individually. Otherwise, those are averaged.

    Optional argument NW, given instead of F_RES, specifies the 
    standardized half bandwidth to DPSS directly.
    
    Note that the nature of the beast is that the output Pxx has a 
    full width of 2*F_RES even if the signal XX is perfectly sinusoidal.

    PXX is normalized such that the sum over all frequencies of PXX is T
    for a gaussian random noise signal with variance 1 (on average).
    This is different from the convention of WELCH, for which the output
    is scales with its NPERSEG parameter. As a further example, if
    the input signal is cos(ωt) for some frequency ω, the sum of the
    PXX values is T/2. This does not depend on sample frequency, as long
    as ω >> 1/T and ω << f_s.
    '''

    if len(x.shape)==1:
        isvec = True
        x = x.reshape(len(x), 1)
    else:
        isvec = False
        
    T, D = x.shape
    N_fft = int(2**np.ceil(np.log2(T)))
    dt = 1/f_s
    if nw is None:
        if f_res is None:
             raise ValueError("Either F_RES or NW should be given")
        else:
            nw = T*dt*f_res
    else:
        if f_res is None:
            f_res = nw / (T*dt)
        else:
            raise ValueError("Only one of F_RES or NW should be given")
    K = int(2*nw-1)
    tapers, ratios = dpss(T, nw, K, sym=False, return_ratios=True)
    tapers = tapers.T.reshape(T,1,K)

    x_tapered = x.reshape(T,D,1) * tapers
    X = np.fft.fft(x_tapered, N_fft, axis=0)
    
    Pxxs = np.abs(X)**2 / N_fft * T 
    Pxxs, f = _sum_pos_neg_freqs(Pxxs)
    f = f_s*f

    if indiv:
        if isvec:
            Pxxs = Pxxs[:,0,:]
        return f, Pxxs, (f_res, nw, K, ratios)
    else:
        Pxx = np.mean(Pxxs, 2)
        if isvec:
            Pxx = Pxx[:,0]
        return f, Pxx

def mtpsd_ci(x, f_s=1, f_res=None, nw=None, pp=[.16, .84]):
    try:
        iter(pp)
    except:
        if pp>.5:
            pp = 1 - pp
        pp = [pp/2, 1-pp/2]

    ff, pxxs, (f_res, nw, K, ratios) = mtpsd(x, f_s, f_res, nw, indiv=True)
    pxx = pxxs.mean(-1)
        
    if len(pxxs.shape)==2:
        isvec = True
        F = pxxs.shape[0]
        pxxs = pxxs.reshape(F,1,K)
    else:
        isvec = False

    F, R, K = pxxs.shape
    pdfss = [ [ scipy.state.gamma.fit(pxxs[f,r,:], floc=0) for r in range(R) ]
             for f in range(F) ]
    lims = [ [ scipy.stats.gamma.ppf(pp, a*K, 0, w/K) for a,_,w in pdfs ]
             for pdfs in pdfss ]
    lims = np.array(lims)
    if isvec:
        lims = lims[:,0,:]
    return ff, pxx, lims


def _mtcoh_core(sigs, ref, f_s=1, f_res=None, nw=None, detrend=True):
    if len(sigs.shape)==1:
        isvec = True
        sigs = sigs.reshape(len(sigs), 1)
    else:
        isvec = False
    T, D = sigs.shape

    if detrend:
        sigs = scipy.signal.detrend(sigs, axis=0)
        ref = scipy.signal.detrend(ref, axis=0)

    N_fft = int(2**np.ceil(np.log2(T)))
    F = N_fft // 2
    dt = 1/f_s
    if nw is None:
        if f_res is None:
             raise ValueError("Either F_RES or NW should be given")
        else:
            nw = T*dt*f_res
    else:
        if f_res is None:
            f_res = nw / (T*dt)
        else:
            raise ValueError("Only one of F_RES or NW should be given")
    K = int(2*nw-1)
    print(f_res, nw, K)
    tapers, ratios = dpss(T, nw, K, sym=False, return_ratios=True)
    tapers = tapers.T.reshape(T,1,K)

    sig_tapered = sigs.reshape(T,D,1) * tapers
    ref_tapered = ref.reshape(T,1,1) * tapers

    X = np.fft.fft(sig_tapered, N_fft, axis=0)
    Y = np.fft.fft(ref_tapered, N_fft, axis=0)
    X = X[:F] # drop negative frequencies; those are compl conj
    Y = Y[:F]

    ff = f_s * np.arange(F) / N_fft

    # Convert to PSDs
    Pxxs = (np.abs(X)**2) / f_s
    Pyys = (np.abs(Y)**2) / f_s
    Pxys = (X*np.conj(Y)) / f_s
    return ff, Pxxs, Pyys, Pxys, isvec

def mtcoh(sigs, ref, f_s, f_res=None, nw=None, detrend=True):
    '''MTCOH - Multitaper coherence
    This is DW's adaptation of Adam Taylor's COH_MTM code
    ff, coh = MTCOH(sigs, ref, f_s, f_res) calculates the coherence 
    of the signals SIGS wrt the reference signal REF.

      SIGS [T] or [TxN] contains the signals.
      REF [T] is the reference signal.
      F_S is the sampling rate.
      F_RES is the half-width of the transform of the tapers used, i.e.,
            the frequency resolution of the result.

    Results are:

      FF (Fx1): frequency base (one-sided).
      COH (FxN): complex coherence.

    The phase of COH is positive (i.e., COH has a positive imaginary
    component) if SIGS leads REF.

    Optional argument NW, given instead of F_RES, specifies the 
    standardized half bandwidth to DPSS directly.'''
    ff, Pxxs, Pyys, Pxys, isvec = _mtcoh_core(sigs, ref, f_s, f_res, nw,
                                              detrend=detrend)
    
    # Average across tapers
    Pxx = np.mean(Pxxs, -1)
    Pyy = np.mean(Pyys, -1)
    Pxy = np.mean(Pxys, -1)

    Cxy = Pxy/np.sqrt(Pxx*Pyy+1e-50)

    if isvec:
        Cxy = Cxy[:,0]
    return ff, Cxy


def _algcsqr(y):
    z = y**2
    z[z<1e-15] = 1e-15
    z[z>1-1e-15] = 1 - 1e-15
    return -np.log(1/z - 1)

def _sqrtlgc(x):
    return 1/np.sqrt(1+np.exp(-x))


def mtcoh_ci(sigs, ref, f_s=1, f_res=None, nw=None,
             alpha_ci=.05, detrend=True):
    '''MTCOH_CI - Multitaper coherence with confidence intervals
    This is DW's adaptation of Adam Taylor's COH_MTM code
    ff, mag, mag_lo, mag_hi, phase, phase_lo, phase_hi \
      = MTCOH_CI(sigs, ref, f_s, f_res, alpha_ci) calculates the 
    coherence of the signals SIGS wrt the reference signal REF along
    with confidence intervals.
    Arguments are as for MTCOH, except that ALPHA_CI specifies the
    α-value for the confidence interval.

    Results are:

      FF (F): frequency base (one-sided)
      MAG (FxN): magnitude of the coherence
      MAG_LO (FxN): lower bound on the magnitude
      MAG_HI (FxN): upper bound on the magnitude
      PHASE (FxN): phase of the coherence (-pi to +pi)
      PHASE_LO (FxN): lower bound on the phase
      PHASE_HI (FxN): upper bound on the phase

    '''

    ff, Pxxs, Pyys, Pxys, isvec = _mtcoh_core(sigs, ref, f_s, f_res, nw,
                                              detrend=detrend)
    F,D,K = Pxys.shape

    Pxx = np.sum(Pxxs, -1)
    Pyy = np.sum(Pyys, -1)
    Pxy = np.sum(Pxys, -1)

    Cxy = Pxy / np.sqrt(Pxx*Pyy+1e-50)

    mag = np.abs(Cxy)
    phase = np.angle(Cxy)

    # Transformed coherence
    mag_xf = _algcsqr(mag)

    # Take-away-one spectra
    Pxxs_tao = (Pxx.reshape(F,D,1) - Pxxs) / (K-1)
    Pyys_tao = (Pyy.reshape(F,D,1) - Pyys) / (K-1)
    Pxys_tao = (Pxy.reshape(F,D,1) - Pxys) / (K-1)

    # Take-away-one coherence
    Cxy_tao = Pxys_tao / np.sqrt(Pxxs_tao*Pyys_tao+1e-50)
    mag_tao = np.abs(Cxy_tao)
    mag_tao_xf = _algcsqr(mag_tao)
    phase_tao = np.angle(Cxy_tao)

    # Magnitude sigma
    mag_tao_xf_mean = mag_tao_xf.mean(-1, keepdims=True)
    mag_xf_sigma = np.sqrt((K-1)/K
                           * np.sum((mag_tao_xf - mag_tao_xf_mean)**2, -1))

    # Phase sigma
    Cxy_tao_hat = Cxy_tao / (mag_tao + 1e-50)
    Cxy_tao_hat_mean = Cxy_tao_hat.mean(-1)
    phase_sigma = np.sqrt(2*(K-1) * (1-np.abs(Cxy_tao_hat_mean)))

    ci_factor = scipy.stats.norm.ppf(1 - alpha_ci/2)
    mag_lo = _sqrtlgc(mag_xf - ci_factor*mag_xf_sigma)
    mag_hi = _sqrtlgc(mag_xf + ci_factor*mag_xf_sigma)

    phase_lo = phase - ci_factor*phase_sigma
    phase_hi = phase + ci_factor*phase_sigma

    if isvec:
        mag = mag[:,0]
        mag_lo = mag_lo[:,0]
        mag_hi = mag_hi[:,0]
        phase = phase[:,0]
        phase_lo = phase_lo[:,0]
        phase_hi = phase_hi[:,0]
    
    return ff, mag, mag_lo, mag_hi, phase, phase_lo, phase_hi
