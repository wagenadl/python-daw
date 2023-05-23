#!/usr/bin/python3

import numpy as np
import os.path


def cell(dims):
    '''CELL - Construct a Matlab-style cell array
    Cell arrays are N-dimensional tensor-like structures with arbitrary 
    content in each cell. They are represented conveniently as numpy
    arrays with "object" dtype. Initial content is None in each cell.'''
    return np.ndarray(dims, object)


def iscell(obj):
    '''ISCELL - Test whether object is a cell array
    x = ISCELL(obj) returns True if OBJ is a cell array, i.e., something
    created by the CELL function.'''
    return type(obj) == np.ndarray and obj.dtype == object


class Struct:
    '''STRUCT - A Matlab-style struct or struct array
    A STRUCT is an entity with named fields that contain arbitrary data.
    A STRUCT ARRAY is an N-dimensional array of STRUCTS.
    In Matlab/Octave, indexing a scalar STRUCT is allowed, as long as 
    the index is 1. We do the same, but with index=0.
    The content of the struct can be accessed using dot notation (as in
    Octave) or using indexing notation (as in Python):
      s = Struct()
      s.foo = 3.14
      s['bar'] = 2.71
      print(s['foo'] + s.bar)
    Struct arrays can be indexed just like numpy arrays:
      cc = cell(3)
      cc[0] = 'one'
      cc[1] = 'two'
      cc[2] = 'three'
      s = struct(foo=cc, bar=1)
      s[1].bar = 3
      s[1:].bar = [5, 7] # This does *not* equal s[1].bar = 5; s[2].bar = 7 !
      print(s[0])        # {'foo': 'one', 'bar': 1}
      print(s.foo)       # ['one' 'two' 'three']
      print(s[:-1])      # a struct array with two members
      print(s[:-1].bar)  # [1 list([5,7])]'''

    def __init__(self, **args):
        '''Constructor - Like Matlab/Octave's STRUCT constructor
        s = STRUCT(field1=values1, field2=value2, ...) constructs a new
        scalar struct or struct array. An array is constructed if any of
        the values are CELL arrays, i.e., a numpy array with "object" dtype.
        In that case, all cell arrays must have the same shape. Non-cell
        array values are replicated across the struct array.
        s = STRUCT() creates an empty scalar structure.'''
        # First, find out if we are to create an array
        arrayshape = None
        for k, v in args.items():
            if iscell(v):
                if arrayshape is None:
                    arrayshape = v.shape
                else:
                    if v.shape != arrayshape:
                        raise ValueError('All cell arrays must match')
        if arrayshape is None:
            # Scalar struct
            arrayshape = []
        self.__dict__['_contents_'] = cell(arrayshape)
        N = len(self._contents_.flat)
        for n in range(N):
            self._contents_.flat[n] = {}

        # self.fields = [ k for k in args ]

        for k, v in args.items():
            if iscell(v):
                for n in range(N):
                    self._contents_.flat[n][k] = v.flat[n]
            else:
                for n in range(N):
                    self._contents_.flat[n][k] = v

    def keys(self):
        '''KEYS - Field names as a DICT_KEYS object'''
        return self._contents_.flat[0].keys()

    def fieldnames(self):
        '''FIELDNAMES - Field names as a list'''
        return [k for k in self.keys()]

    def shape(self):
        return self._contents_.shape

    def ndim(self):
        return self._contents_.ndim

    def __getattr__(self, key):
        # Dot notation is interpreted just like indexing notation
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getitem__(self, key):
        if type(key) == str:
            # String key gets field or cell array of fields
            if self._contents_.ndim == 0:
                return self._contents_.flat[0][key]
            else:
                res = cell(self._contents_.shape)
                N = len(self._contents_.flat)
                for n in range(N):
                    res.flat[n] = self._contents_.flat[n][key]
                return res
        else:
            # Proper indexing
            if self._contents_.ndim == 0:
                if np.issubdtype(type(key), np.integer):
                    if key != 0:
                        raise KeyError(key)
                elif np.issubdtype(type(key), np.floating):
                    raise KeyError(key)
                else:
                    if any(key != 0):
                        raise KeyError(key)
                return self
            else:
                ct = self._contents_[key]
                res = Struct()
                if type(ct) == dict:
                    res._contents_.flat[0] = ct
                else:
                    res.__dict__['_contents_'] = ct  # avoids calling setattr
                return res

        return key

    def __setitem__(self, key, value):
        if type(key) == str:
            # String key sets field or cell array of fields
            if self._contents_.ndim == 0:
                self._contents_.flat[0][key] = value
            else:
                N = len(self._contents_.flat)
                if type(value) == np.ndarray and value.dtype == object:
                    if value.shape == self._contents_.shape:
                        for n in range(N):
                            self._contents_.flat[n][key] = value.flat[n]
                    else:
                        raise ValueError('Mismatching shapes')
                else:
                    for n in range(N):
                        self._contents_.flat[n][key] = value

        else:
            # Proper indexing
            # Value must be a struct with fields that match ours.
            # or it may be a dict with fields that match ours
            if type(value) == Struct:
                if value.fieldnames() == self.fieldnames():
                    self._contents_[key] = value
                else:
                    raise ValueError('Mismatching fieldnames')

    def __repr__(self):
        if self._contents_.ndim == 0:
            return self._contents_.flat[0].__repr__()
        else:
            dims = 'x'.join([f'{x}' for x in self._contents_.shape])
            flds = '\n  '.join(self.fieldnames())
            return f'({dims}) struct array containing:\n  ' + flds

    def asdict(self, deep=False, keepscalarasarray=False):
        '''ASDICT - Convert Struct to dict
        ASDICT() returns the (scalar) struct as a dictionary. If the Struct
        is a struct array, the result is an array of dicts.
        Optional argument DEEP specifies that substructures are to be
        converted as well (inside cells as well).
        Optional argument KEEPSCALARASARRAY returns a (shapeless) array
        if the input is a scalar.'''
        if deep:
            def deStruct(x):
                if type(x) == type(self):
                    return x.asdict(deep=deep,
                                    keepscalarasarray=keepscalarasarray)
                elif type(x) == np.ndarray and x.dtype == np.object:
                    N = len(x.flat)
                    y = np.ndarray(x.shape, np.object)
                    for n in range(N):
                        y.flat[n] = deStruct(x.flat[n])
                    return y
                else:
                    return x
        else:
            def deStruct(x):
                return x

        res = np.ndarray(self.shape(), np.object)
        N = len(self._contents_.flat)
        for n in range(N):
            res.flat[n] = {k: deStruct(v)
                           for k, v in self._contents_.flat[n].items()
                           if k != '__class__'}
        if N == 1 and not keepscalarasarray:
            res = res.flat[0]
        return res


def _pythonify_matrix(v):
    if v.ndim == 2:
        # Convert 1x1 to scalar and 1xN, Nx1 to vector
        n, m = v.shape
        if n == 1 and m == 1:
            return v[0, 0]
        elif n == 1 or m == 1:
            return v.flatten()
    return v


def _pythonify_string(v):
    if len(v) == 1:
        return str(v[0])
    else:
        return [str(x) for x in v]


def _pythonify_cell(v):
    cc = cell(v.shape)
    N = len(v.flat)
    for n in range(N):
        cc.flat[n] = _pythonify(v.flat[n])
    return _pythonify_matrix(cc)  # Drop useless dimensions


def _pythonify_struct(v):
    dct = {}
    K = len(v.dtype.names)
    N = len(v.flat)
    for k in range(K):
        f = v.dtype.names[k]
        shp = v.shape
        if len(shp) == 2:
            n, m = shp
            if n == 1 and m == 1:
                shp = ()
            elif n == 1:
                shp = (m)
            elif m == 1:
                shp = (n)
            cc = cell(shp)
        for n in range(N):
            cc.flat[n] = _pythonify(v.flat[n][k])
        dct[f] = cc
    ss = Struct(**dct)
    return ss


def _pythonify(v):
    if type(v)==np.ndarray:
        if v.dtype == object:
            return _pythonify_cell(v)
        elif v.dtype.names is not None:
            return _pythonify_struct(v)
        elif v.dtype.kind == 'U':
            return _pythonify_string(v)
        else:
            return _pythonify_matrix(v)
    else:
        return v

def loadmat(ifn):
    '''LOAD - Load a .mat file
    x = LOAD(ifn) loads the variables in the .mat file as a STRUCT.
    This only works for Matlab v6 and v7 files. See also LOAD.'''
    import scipy.io
    if not os.path.exists(ifn):
        raise ValueError(f'File {ifn} does not exist')
    mat = scipy.io.loadmat(ifn)
    res = Struct()
    for k, v in mat.items():
        if k.startswith('__') and k.endswith('__'):
            continue
        res[k] = _pythonify(v)
    return res


def save(ofn, *args):
    '''SAVE - Save a .mat file
    SAVE(ofn, var1, var2, ...) saves the named variables into a .mat file.
    CAUTION: SAVE uses the INSPECT module to determine how it was called.
    That means that VARi must all be simple variables and that OFN, if given
    as a direct string, may not contain commas. Variable names must start
    with a letter and may only contain letters, numbers, and underscore.
    OK examples:
      x = 3
      y = 'Hello'
      z = np.eye(3)
      save('/tmp/test.mat', x, y, z)
      fn = '/tmp/test,1.mat'
      save(fn, x, y, z) # The fact that FN contains a comma is no problem
    Bad examples:
      save('/tmp/test,1.mat', x, y) # Comma is not allowed
      save('/tmp/test.mat', x+3) # "x+3" is not just a variable'''
    import scipy.io
    import inspect
    import re
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0]
    sol = string.find('(') + 1
    eol = string.find(')')
    names = [a.strip() for a in string[sol:eol].split(',')]
    names.pop(0)
    if len(names) != len(args):
        raise ValueError('Bad call to SAVE')
    nre = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')
    N = len(args)
    for k in range(N):
        if not nre.match(names[k]):
            raise ValueError('Bad variable name: ' + names[k])
    dct = {}
    for k in range(N):
        if type(args[k]) == Struct:
            dct[names[k]] = args[k]._contents_
        else:
            dct[names[k]] = args[k]
    scipy.io.savemat(ofn, dct, do_compression=True)


def load(ifn):
    '''LOAD - Load a .mat file
    x = LOAD(ifn) loads the variables in the given .mat file as a STRUCT.
    This works for anything that Octave can load. It is, however,
    significantly slower than LOADMAT, because it has to run Octave.
    IFN may not contain double quotes (").'''
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        mfile = tmpdir + '/convert.m'
        matfile = tmpdir + '/converted.mat'
        with open(mfile, 'w') as fh:
            fh.write(f'''
            clear
            load("{ifn}");
            save("-v7", "{matfile}");
            ''')
        opts = '-W -f --no-init-file --no-init-path --no-site-file'
        os.system(f'/usr/bin/octave {opts} {mfile}')
        res = loadmat(matfile)
    return res


if __name__ == '__main__':
    from drepr import d

    s = Struct(foo=1)
    c = cell((3))
    c[0] = 'x'
    c[1] = 'y'
    c[2] = 'z'
    d(c)
    t = Struct(foo=c)
    d(s)
    d(t)
    dir = '/home/wagenaar/python/daw/test'
    mat = load(f'{dir}/test-v7.mat')
    c = mat.c
    cc = mat.cc
    f = mat.f
    i16 = mat.i16
    u32 = mat.u32
    txt = mat.txt
    txts = mat.txts
    s = mat.s
    t = mat.t
    x = mat.x
    xx = mat.xx
    save('/tmp/test.mat', c, cc, f, i16, u32, txt, txts, s, t, x, xx)

