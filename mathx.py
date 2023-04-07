#!/usr/bin/python3

import numpy as np

def primes(n):
    '''PRIMES - List of all primes up to given integer
    pp = PRIMES(n) returns an array containing all the primes up to (but
    not including) N.
    '''
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    if n<6:
        return np.array([x for x in [2,3,5] if x<n])
    sieve = np.ones(n//3 + (n%6==2), dtype=bool)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)//3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]
