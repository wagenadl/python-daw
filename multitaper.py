# multitaper.py - multitaper spectral estimates
# This is DAW's translation into python of Adam Taylor's matlab code

import numpy as np
from scipy.signal.windows import dpss

def sum_pos_neg_freqs_(Pxxs_ts):
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
        

def pds(x, f_s, f_res=None, nw=None, indiv=False):
    '''This is DW's adaptation of Adam Taylor's PDS_MTM code
    ff, Pxx = PDS(xx, f_s, f_res) calculates one-side multi-taper
    spectrogram.
    
      XX [TxD] is the (optical) data.
      F_S is the sampling rate.
      F_RES is the half-width of the transform of the tapers used.
    
      FF [Fx1] is the resulting (one-sided) frequency base.
      PXX [FxN] are the spectral estimates for the data XX at frequencies FF.

    If optional argument INDIV is true, PXX [FxNxK] are data for each 
    taper individually. Otherwise, those are averaged.

    Optional argument NW, given instead of F_RES, specifies the 
    standardized half bandwidth to DPSS directly.
    
    Note that the nature of the beast is that the output Pxx has a 
    full width of 2*F_RES even if the signal XX is perfectly sinusoidal.'''

    if len(x.shape)==1:
        isvec = True
        x = x.reshape(len(x), 1)
    else:
        isvec = False
        
    N, D = x.shape
    N_fft = int(2**np.ceil(np.log2(N)))
    dt = 1/f_s
    if nw is None:
        nw = N*dt*f_res
    else:
        if f_res is not None:
            raise ValueError("Either F_RES or NW (but not both) should be given")
    K = int(2*nw-1)
    tapers = dpss(N,nw,K, False).T.reshape(N,1,K)
    
    x_tapered = x.reshape(N,D,1) * tapers
    X = np.fft.fft(x_tapered, N_fft, axis=0)
    
    Pxxs = np.abs(X)**2 / f_s
    Pxxs, f = sum_pos_neg_freqs_(Pxxs)
    f = f_s*f

    if indiv:
        if isvec:
            Pxxs = Pxxs[:,0,:]
        return f, Pxxs
    else:
        Pxx = np.mean(Pxxs, 2)
        if isvec:
            Pxx = Pxx[:,0]
        return f, Pxx

def coh(x_ref, _sig, f_s, f_res=None, nw=None):
    '''Not yet implemented. See vscope_coherence'''
    pass

def coh_ci(x_ref, _sig, f_s, f_res=None, nw=None, ci=1):
    pass
