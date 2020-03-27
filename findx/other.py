#!/usr/bin/python3

import numpy as np

def matchnearest(tt1, tt2, maxdt=np.inf, bidir=False):
    '''MATCHNEAREST - Find matching events in two point processes
    idx = MATCHNEAREST(tt1, tt2) returns a vector in which the k-th
    element indicates which event in point process TT2 occured most
    closely to the k-th event in point process TT1.
    idx = MATCHNEAREST(tt1, tt2, maxdt) specifies a maximum time interval
    beyond which matches cannot be declared.
    Events that do not have a match result in a (-1) entry in the IDX.
    
    Alternatively, idx1, idx2 = MATCHNEAREST(tt1, tt2, bidir=True) returns 
    two vectors, such that TT1[IDX1] and TT2[IDX2] are matching
    events. That means you can create a scatter plot of latencies with
    something like:

      qp.mark(tt2[idx2] - tt1[idx1])

    Notes:

    - MATCHNEAREST does not guarantee that the matching is
      one-to-one: Although at most one event in TT2 can be matched to a
      given even in TT1, it is possible that an event in TT2 is matched
      to multiple events in TT1. See MATCHNEAREST2 if this is
      undesirable.

    - MATCHNEAREST does not assume that TT1 and TT2 are presorted.

    - Caution: Current implementation is not very smart. Its time complexity
      is O(N*M) where N and M are the lengths of TT1 and TT2.'''
    
    N = len(tt1)
    idx = np.zeros(N, int) - 1
    for n in range(N):
        t0 = tt1[n]
        adt = np.abs(tt2 - t0)
        id1 = np.argmin(adt)
        if adt[id1] < maxdt:
            idx[n] = id1

    if bidir:
        idx1 = np.nonzero(idx>=0)
        idx2 = idx[idx1]
        return idx1, idx2
    else:
        return idx

def matchnearest2(tt1, tt2, maxdt, bidir=False):
    '''MATCHNEAREST2 - Find unique matching events in two point processes

    MATCHNEAREST2 is just like MATCHNEAREST, except that it
    guarantees that the matches are unique, that is, a given event in
    TT2 can be matched to at most one event in TT1: the closest match
    from the perspectives of both time series.

    Both the idx = MATCHNEAREST2(...) and idx1, idx2 = ... syntaxes
    are supported.

    Notes:

    - MATCHNEAREST2 does not assume that TT1 and TT2 are presorted.

    - Caution: Current implementation is not very smart. Its time complexity
      is O(N*M) where N and M are the lengths of TT1 and TT2.'''

    N = len(tt1)
    idx = np.zeros(N, int) - 1
    for n in range(N):
        t0 = tt1[n]
        adt = np.abs(tt2 - t0)
        id1 = np.argmin(np.abs(dt))
        if adt < maxdt:
            t0 = tt2[id1]
            adt = np.abs(tt1 - t0)
            id2 = np.argmin(np.abs(dt))
            if id2==n:
                idx[n] = id1

    if bidir:
        idx1 = np.nonzero(idx>=0)
        idx2 = idx[idx1]
        return idx1, idx2
    else:
        return idx
    
