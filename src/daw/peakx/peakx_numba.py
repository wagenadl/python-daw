from numba import jit
import numpy as np


@jit
def schmittcore(data, thr_on, thr_off):
    trans = []
    isup = False
    for k in range(len(data)):
        if data[k]<=thr_off if isup else data[k]>=thr_on:
            trans.append(k)
            isup = not isup
    ion = trans[::2]
    ioff = trans[1::2]
    return ion, ioff


def schmitt(data, thr_on=None, thr_off=None, endtype=2, starttype=1):
    '''SCHMITT  Schmitt trigger of a continuous process.
    [ion, ioff] = SCHMITT(data, thr_on, thr_off) implements a Schmitt trigger:
    ION are the indices when DATA crosses up through THR_ON coming from 
    below THR_OFF;
    IOFF are the indices when DATA crosses down through THR_OFF coming from 
    above THR_ON.
    If DATA is high at the beginning, the first ION value will be 0, unless
    optional argument STARTTYPE = 0, in which case any such "partial peak"
    is dropped.
    If DATA is high at the end, the last downward crossing will be len(DATA).
    Optional argument ENDTYPE modifies this behavior:
      ENDTYPE = 0: Ignore last upward crossing if there is no following downward
                   crossing.  
      ENDTYPE = 1: Simply report last upward crossing without a corresponding
                   downward crossing; ION may be one longer than IOFF.
    If THR_OFF is not specified, it defaults to THR_ON/2.
    If neither THR_ON nor THR_OFF are specified, THR_ON=2/3 and THR_OFF=1/3.'''

    if thr_on is None:
        if data.dtype==bool:
            thr_on = True
        else:
            thr_on = 2./3
    if thr_off is None:
        if data.dtype==bool:
            thr_off = False
        else:
            thr_off = thr_on / 2.
    if data.ndim != 1:
        raise ValueError('Input must be 1-d array')

    iup, idn = schmittcore(data, thr_on, thr_off)

    if endtype==0:
        if len(iup)>len(idn):
            iup = iup[:len(idn)] # Drop last up transition
    elif endtype==1:
        pass # There may be an extra up transition
    elif endtype==2:
        if len(iup)>len(idn):
            idn.append(len(data))
    else:
        raise ValueError('Invalid end type')
    
    if starttype==0:
        if len(iup)>0 and iup[0]==0:
            iup = iup[1:]
            idn = idn[1:]
    elif starttype==1:
        pass
    else:
        raise ValueError('Invalid start type')

    return np.array(iup), np.array(idn)


def schmitt2(data, thr_a, thr_b):
    '''SCHMITT2 - Double Schmitt triggering
    [on_a,off_a,on_b,off_b] = SCHMITT2(data, thr_a, thr_b) Schmitt triggers
    twice. It is required that THR_B < THR_A.

    There are three equivalent ways to think about the result:
    
    (1) ON_A, OFF_A are the up and down crossings through THR_A;
        ON_B, OFF_B are the up and down crossings through THR_B.

    (2) ON_A, OFF_A describe the broadest possible peak above THR_A;
        ON_B, OFF_B describe the narrowest possible peak above THR_B.
        (But ON_B, OFF_B describe wider peaks than ON_A, OFF_A,
        since THR_B<THR_A.)
   
    (3) ON_A mark the first point where DATA >= THR_A.
        OFF_A mark the first point where DATA < THR_A.
        ON_B mark the first point where XX >= THR_B.
        OFF_B mark the first point where XX < THR_B.

    Note that a peak that exceeds THR_B but never exceeds THR_A is not
    reported.'''
   
    on_a, off_b = schmitt(data, thr_a, thr_b)
    off_a, on_b = schmitt(np.flip(data), thr_a, thr_b)
    off_a = np.flip(len(data) - off_a)
    on_b = np.flip(len(data) - on_b)
    return on_a, off_a, on_b, off_b

def schmittpeak(data, iup, idn):
    '''SCHMITTPEAK - Finds peaks in data after Schmitt triggering
    ipk = SCHMITTPEAK(data, iup, idn) returns the indices of the peaks 
    between each pair of threshold crossings. IUP and IDN must be from
    SCHMITT. Partial peaks at beginning and end of DATA _are_ considered,
    so use starttype=0 and/or endtype=0 when calling SCHMITT.'''

    ipk = np.zeros(iup.shape, dtype=iup.dtype)
    for k in range(len(iup)):
        ipk[k] = iup[k]+ np.argmax(data[iup[k]:idn[k]])
    return ipk

if __name__=='__main__':
    data = np.random.randn(100)
    iup, idn = schmitt(data, 1, -1)

    dat32 = data.astype(np.float32)
    iup1, idn1 = schmitt(dat32, 1, -1)

    iupa, idna, iupb, idnb = schmitt2(data, 1, -1)
    print(f'data = {data[:20]}')
    print(f'... {data[80:]}')
    print(f'iupa = {iupa}')
    print(f'idna = {idna}')
    print(f'iupb = {iupb}')
    print(f'idnb = {idnb}')
    ipk = schmittpeak(data, iupa, idna)
    print(f'ipk = {ipk}')
