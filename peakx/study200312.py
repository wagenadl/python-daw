import numpy as np
import ctypes as ct
import os

_ccore = np.ctypeslib.load_library('libccore', '.')
_ccore.schmitt_double.argtypes = [ct.POINTER(ct.c_double),
                                  ct.c_uint64, ct.c_uint64,
                                  ct.c_double, ct.c_double,
                                  ct.POINTER(ct.c_uint64)]
_ccore.schmitt_double.restype = ct.POINTER(ct.c_uint64)

data = np.random.randn(100)
upthr = 1
downthr = -1

c = data.ctypes
count = data.shape[0]
stride_bytes = c.strides[0]
nres = ct.c_uint64()
if data.dtype==np.float64: # including plain python float and np.float
    # this is c_double, 64-bits
    dat = c.data_as(ct.POINTER(ct.c_double))
    stride = stride_bytes // 8
    ptr = _ccore.schmitt_double(dat, count, stride,
                                upthr, downthr,
                                ct.pointer(nres))
elif data.dtype==np.float32: 
    # this is c_float, 32-bits
    dat = c.data_as(ct.POINTER(ct.c_float))
    stride = stride_bytes // 4
    ptr = _ccore.schmitt_float(dat, count, stride,
                               upthr, downthr,
                               ct.pointer(nres))
else:
    raise ValueError("data must be float64 or float32")

nres = nres.value
arr = np.ctypeslib.as_array(ptr, [nres])
iup = arr[0:nres:2].copy()
idn = arr[1:nres:2].copy()
_ccore.schmitt_free(ptr)

if 

