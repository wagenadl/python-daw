from numba import jit
import numpy as np

@jit
def findfirst_ge(vec, cf):
    for k in range(len(vec)):
        if vec[k]>=cf:
            return k
    return None

@jit
def findfirst_gt(vec, cf):
    for k in range(len(vec)):
        if vec[k]>cf:
            return k
    return None


@jit
def findfirst_le(vec, cf):
    for k in range(len(vec)):
        if vec[k]<=cf:
            return k
    return None


@jit
def findfirst_lt(vec, cf):
    for k in range(len(vec)):
        if vec[k]<cf:
            return k
    return None


@jit
def findfirst_ne(vec, cf):
    for k in range(len(vec)):
        if vec[k]!=cf:
            return k
    return None


@jit
def findfirst_eq(vec, cf):
    for k in range(len(vec)):
        if vec[k]==cf:
            return k
    return None


@jit
def findfirst(vec):
    for k in range(len(vec)):
        if vec[k]:
            return k
    return None


######################################################################
@jit
def findlast_ge(vec, cf):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]>=cf:
            return k
    return None

@jit
def findlast_gt(vec, cf):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]>cf:
            return k
    return None


@jit
def findlast_le(vec, cf):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]<=cf:
            return k
    return None


@jit
def findlast_lt(vec, cf):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]<cf:
            return k
    return None


@jit
def findlast_ne(vec, cf):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]!=cf:
            return k
    return None


@jit
def findlast_eq(vec, cf):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]==cf:
            return k
    return None


@jit
def findlast(vec):
    for k in range(len(vec)-1,-1,-1):
        if vec[k]:
            return k
    return None

