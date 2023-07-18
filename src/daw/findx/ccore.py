#!/usr/bin/python3

import numpy as np
import ctypes as ct
import os

if os.name=='posix':
    _pfx = 'lib'
else:
    _pfx =''
_libname = _pfx + 'ccore'

_ccore = np.ctypeslib.load_library(_libname, os.path.dirname(__file__))

_ops = [ 'ge', 'gt', 'le', 'lt', 'eq', 'ne' ]
_typs = [ 'float', 'double',
          'int64_t', 'uint64_t', 'int32_t',
          'uint32_t', 'int8_t', 'uint8_t',
          'bool' ]

#gg = globals()
_funcs = {}
for op in _ops:
    _funcs[op] = {}
    for typ in _typs:
        atyp = typ
        if atyp.endswith('_t'):
            atyp = atyp[:-2]
        name = f'findfirst_{op}_{typ}'
        func = _ccore.__getattr__(name)
        func.argtypes = [
            ct.POINTER(eval(f'ct.c_{atyp}')),
            ct.c_uint64, ct.c_int64,
            eval(f'ct.c_{atyp}') ]
        func.restype = ct.c_uint64
        _funcs[op][typ] = func

_typmap = { np.dtype(np.float64): (8, 'double', ct.c_double),
            np.dtype(np.float32): (4, 'float', ct.c_float),
            np.dtype(np.int64):   (8, 'int64_t', ct.c_int64),
            np.dtype(np.int32):   (4, 'int32_t', ct.c_int32),
            np.dtype(np.int16):   (2, 'int16_t', ct.c_int16),
            np.dtype(np.int8):    (1, 'int8_t', ct.c_int8),
            np.dtype(np.uint64):  (8, 'uint64_t', ct.c_uint64),
            np.dtype(np.uint32):  (4, 'uint32_t', ct.c_uint32),
            np.dtype(np.uint16):  (2, 'uint16_t', ct.c_uint16), 
            np.dtype(np.uint8):   (1, 'uint8_t', ct.c_uint8),
            np.dtype(np.bool):    (1, 'bool', ct.c_bool )}

def _findfirst_op(vec, op, cf):
    if vec.ndim != 1:
        raise ValueError('Input must be 1-d array')
    c = vec.ctypes
    count = vec.shape[0]
    stride_bytes = c.strides[0]
    sizeof, typname, ctyp = _typmap[vec.dtype]
    func = _funcs[op][typname]
    stride = stride_bytes // sizeof
    dat = c.data_as(ct.POINTER(ctyp))
    cf = ctyp(cf)
    '''if vec.dtype==np.float64:
        dat = c.data_as(ct.POINTER(ct.c_double))
        stride = stride_bytes // 8
        func = fncs['double']
    elif vec.dtype==np.float32:
        dat = c.data_as(ct.POINTER(ct.c_float))
        stride = stride_bytes // 4
        func = _fun_ccore.__getattr__(f'findfirst_{op}_float')
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
        raise ValueError(f'Unsupported data type: {vec.dtype}')'''
    idx = func(dat, count, stride, cf)
    if idx<len(vec):
        return idx
    else:
        return None


def _findlast_op(vec, op, cf):
    res = _findfirst_op(np.flip(vec), op, cf)
    if res is not None:
        return len(vec) - 1 - res
    else:
        return None

    
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


def findlast_ge(vec, cf):
    return _findlast_op(vec, 'ge', cf)

def findlast_gt(vec, cf):
    return _findlast_op(vec, 'gt', cf)

def findlast_le(vec, cf):
    return _findlast_op(vec, 'le', cf)

def findlast_lt(vec, cf):
    return _findlast_op(vec, 'lt', cf)

def findlast_ne(vec, cf):
    return _findlast_op(vec, 'ne', cf)

def findlast_eq(vec, cf):
    return _findlast_op(vec, 'eq', cf)

def findlast(vec):
    return _findlast_op(vec, 'ne', 0)

if __name__=='__main__':
    x = np.cos(np.arange(20))
    print('x = ', x, x.dtype)
    print('findfirst_eq(x, 1) = ', findfirst_eq(x, 1))
    print('findfirst_eq(x, 1.0) = ', findfirst_eq(x, 1.0))
    print('findfirst_le(x, 0) = ', findfirst_le(x, 0))
    print('findfirst_lt(x, -.50) = ', findfirst_lt(x, -.50))
    print('findfirst_gt(x, 10) = ', findfirst_gt(x, 10))
    print('findlast_ge(x, .9) = ', findlast_ge(x, .9))
    print('findlast_ge(x, 9) = ', findlast_ge(x, 9))
    print('findlast_lt(x, 0) = ', findlast_lt(x, 0))

    y = x > 0
    print('y = ', y, y.dtype)
    print('findfirst_eq(y, 0) = ', findfirst_eq(y, 0))
    
