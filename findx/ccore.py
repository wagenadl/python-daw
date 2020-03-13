#!/usr/bin/python3

import numpy as np
import ctypes as ct
import os

_ccore = np.ctypeslib.load_library('libccore', os.path.dirname(__file__))

ops = [ 'ge', 'gt', 'le', 'lt', 'eq', 'ne' ]
typs = [ 'float', 'double',
         'int64_t', 'uint64_t', 'int32_t', 'uint32_t', 'int8_t', 'uint8_t' ]

gg = globals()

for op in ops:
    for typ in typs:
        atyp = typ
        if atyp.endswith('_t'):
            atyp = atyp[:-2]
        func = _ccore.__getattr__(f'findfirst_{op}_{typ}')
        func.argtypes = [ ct.POINTER(eval(f'ct.c_{atyp}')),
                          ct.c_uint64, ct.c_int64,
                          eval(f'ct.c_{atyp}') ]
        func.restype = ct.c_uint64

def _findfirst_op(vec, op, cf):
    if vec.ndim != 1:
        raise ValueError('Input must be 1-d array')
    c = vec.ctypes
    count = vec.shape[0]
    stride_bytes = c.strides[0]
    if vec.dtype==np.float64:
        dat = c.data_as(ct.POINTER(ct.c_double))
        stride = stride_bytes // 8
        func = _ccore.__getattr__(f'findfirst_{op}_double')
    elif vec.dtype==np.float32:
        dat = c.data_as(ct.POINTER(ct.c_float))
        stride = stride_bytes // 4
        func = _ccore.__getattr__(f'findfirst_{op}_float')
    elif vec.dtype==np.int64:
        dat = c.data_as(ct.POINTER(ct.c_int64))
        stride = stride_bytes // 8
        func = _ccore.__getattr__(f'findfirst_{op}_int64_t')
    elif vec.dtype==np.uint64:
        dat = c.data_as(ct.POINTER(ct.c_uint64))
        stride = stride_bytes // 8
        func = _ccore.__getattr__(f'findfirst_{op}_uint64_t')
    elif vec.dtype==np.int32:
        dat = c.data_as(ct.POINTER(ct.c_int32))
        stride = stride_bytes // 4
        func = _ccore.__getattr__(f'findfirst_{op}_int32_t')
    elif vec.dtype==np.uint32:
        dat = c.data_as(ct.POINTER(ct.c_uint32))
        stride = stride_bytes // 4
        func = _ccore.__getattr__(f'findfirst_{op}_uint32_t')
    elif vec.dtype==np.int8:
        dat = c.data_as(ct.POINTER(ct.c_int8))
        stride = stride_bytes
        func = _ccore.__getattr__(f'findfirst_{op}_int8_t')
    elif vec.dtype==np.uint8:
        dat = c.data_as(ct.POINTER(ct.c_uint8))
        stride = stride_bytes
        func = _ccore.__getattr__(f'findfirst_{op}_uint8_t')
    elif vec.dtype==np.bool:
        dat = c.data_as(ct.POINTER(ct.c_bool))
        stride = stride_bytes
        func = _ccore.__getattr__(f'findfirst_{op}_bool')
    else:
        raise ValueError(f'Unsupported data type: {vec.dtype}')
    return func(dat, count, stride, cf)

def findfirst_ge(vec, cf):
    return _findfirst_op(vec, 'ge', cf)

def findfirst_gt(vec, cf):
    return _findfirst_op(vec, 'gt', cf)

def findfirst_le(vec, cf):
    return _findfirst_op(vec, 'le', cf)

def findfirst_lt(vec, cf):
    return _findfirst_op(vec, 'lt', cf)

def findfirst_ne(vec, cf):
    return _findfirst_op(vec, 'ne', cf)

def findfirst_eq(vec, cf):
    return _findfirst_op(vec, 'eq', cf)

def findfirst(vec):
    return _findfirst_op(vec, 'ne', 0)
