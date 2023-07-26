#!/usr/bin/python3

import numpy as np
import pandas as pd
import pickle
import inspect
import re
import collections

def cansave(v):
    '''CANSAVE - Can a given object be saved by PPERSIST?
    CANSAVE(v) returns True if V can be saved by PPERSIST.

    Currently, PPERSIST can save:

      - None
      - Numpy arrays (but not containing np.object)
      - Strings
      - Simple numbers (int, float, complex)
      - Lists, tuples, sets, and dicts containing those (even hierarchically).

    Importantly, PPERSIST cannot save objects of arbitrary class or
    function type. 

    PPERSIST will not attempt to load types it cannot save. This is
    intended to protect PPERSIST against the well-documented security
    problems of the underlying pickle module.'''
    if v is None:
        return True
    t = type(v)
    if t==np.ndarray and v.dtype!=np.object:
        return True
    if t==str:
        return True
    if t==int or t==np.int32 or t==np.int64 \
       or t==float or t==np.float64 or t==np.float32 \
       or t==complex or t==np.complex128 or t==np.complex64:
        return True
    if t==dict:
        for k,v1 in v.items():
            if not cansave(v1):
                return False
        return True
    if t==list or t==tuple or t==set:
        for v1 in v:
            if not cansave(v1):
                return False
        return True
    if t == pd.DataFrame or t == pd.Series:
        for v1 in v:
            if not cansave(v1):
                return False
        return True

    print(f'Cannot save {t}')
    return False

def save(fn, *args):
    '''SAVE - Save multiple variables in one go as in Octave/Matlab
    SAVE(filename, var1, var2, ..., varN) saves each of the variables
    VAR1...VARN into a single PICKLE file.
    The result can be loaded with
      LOAD(filename)
    or
      var1, var2, ..., varN = LOAD(filename)
    or
      vv = LOADDICT(filename)
    In the latter case, vv becomes a DICT with the original variable
    names as keys.
    Note that SAVE is a very hacked function: It uses the INSPECT module
    to determine how it was called. That means that VARi must all be simple
    variables and that the FILENAME, if given as a direct string, may not 
    contain commas. Variable names must start with a letter and may only
    contain letters, numbers, and underscore.
    OK examples:
      x = 3
      y = 'Hello'
      z = np.eye(3)
      save('/tmp/test.pkl', x, y, z)
      fn = '/tmp/test,1.pkl'
      save(fn, x, y, z)
    Bad examples:
      save('/tmp/test,1.pkl', x, y)
      save('/tmp/test.pkl', x+3)'''
    frame = inspect.currentframe().f_back
    string = inspect.getframeinfo(frame).code_context[0]
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
        if not cansave(args[k]):
            raise ValueError('Cannot save variable: ' + names[k])

    dct = {}
    for k in range(N):
        dct[names[k]] = args[k]
    dct['__names__'] = names
    savedict(fn, dct)

def savedict(fn, dct):
    '''SAVEDICT - Save data from a DICT
    SAVEDICT(filename, dct), where DCT is a DICT, saves the data contained
    therein as a PICKLE file. The result can be loaded with
      dct = LOADDICT(filename).'''
    nre = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')
    for k, v in dct.items():
        if not nre.match(k):
            raise ValueError('Bad variable name: ' + k)
        if not cansave(v):
            raise ValueError('Cannot save variable: ' + k)

    with open(fn, 'wb') as fd:  
        pickle.dump(dct, fd, pickle.HIGHEST_PROTOCOL)

    
        

_allowed = [
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy", "dtype"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("functools", "partial"),
    ("pandas.core.internals.blocks", "new_block"),
    ("builtins", "slice"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("builtins", "complex"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "range"),
    ("builtins", "slice"),
]

class SafeLoader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _allowed:
            return super().find_class(module, name)
        else:
            raise pickle.UnpicklingError(f"Not allowed “{module}”: “{name}”")

        
class UnsafeLoader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _allowed:
            print(f"Caution: Loading “{module}.{name}” from pickle.")
        return super().find_class(module, name)

    
def _load(fn, trusted=False):
    with open(fn, 'rb') as fd:
        if trusted:
            return UnsafeLoader(fd).load()
        else:
            return SafeLoader(fd).load()
    

def loaddict(fn, trusted=False):
    '''LOADDICT - Reload data saved with SAVE or SAVEDICT
    x = LOADDICT(fn) loads the file named FN, which should have been created
    by SAVE. The result is a dictionary with the original variable names
    as keys.
    Optional parameter TRUSTED may be used to turn off safety checks
    in the underlying pickle loading. With the default TRUSTED=False, 
    we only allow loading of very specific object types, even when nested
    inside Pandas dataframes or numpy arrays. TRUSTED=True enables loading
    any pickle. Only do this for files that you actually trust. '''
    # Pass unsafe=True only for debugging on trusted files!
    dct = _load(fn, trusted)
    del dct['__names__']
    return dct

def load(fn, trusted=False):
    '''LOAD - Reload data saved with SAVE or SAVEDICT
    x = LOAD(fn) loads the file named FN which should have been created
    by SAVE or SAVEDICT. 
    The result is a named tuple with the original variable names as keys.
    v1, v2, ..., vn = LOAD(fn) immediately unpacks the tuple.
    Optional parameter TRUSTED may be used to turn off safety checks
    in the underlying pickle loading. With the default TRUSTED=False, 
    we only allow loading of very specific object types, even when nested
    inside Pandas dataframes or numpy arrays. TRUSTED=True enables loading
    any pickle. Only do this for files that you actually trust. 
    '''
    dct = _load(fn, trusted)
    names = dct['__names__']
    class Tuple(collections.namedtuple('Tuple', names)):
        revmap = { name: num for num, name in enumerate(names) }
        def __getitem__(self, k):
            if type(k)==str:
                if k in Tuple.revmap:
                    k = Tuple.revmap[k]
                else:
                    raise KeyError(k)
            else:
                return super().__getitem__(k)
        def __str__(self):
            return "Tuple with fields:\n  " + "\n  ".join(Tuple.revmap.keys())
        def __repr__(self):
            return "<Tuple(" + ", ".join(["'"+n+"'" for n in Tuple.revmap.keys()]) + ")>"
        def keys(self):
            return Tuple.revmap.keys()
    lst = []
    for n in names:
        lst.append(dct[n])
    return Tuple(*lst)


def mload(fn, trusted=False):
    '''MLOAD - Reload data saved with SAVE 
    MLOAD(fn)  directly loads the variables saved by SAVE(fn, ...) 
    into the caller's namespace.
    This is a super ugly Matlab-style hack, but convenient for quick hacking.
    LOAD and LOADDICT are cleaner alternatives
    Optional parameter TRUSTED may be used to turn off safety checks
    in the underlying pickle loading. With the default TRUSTED=False, 
    we only allow loading of very specific object types, even when nested
    inside Pandas dataframes or numpy arrays. TRUSTED=True enables loading
    any pickle. Only do this for files that you actually trust. 
'''
    dct = _load(fn, trusted)
    names = dct['__names__']

    frame = inspect.currentframe().f_back
    # inject directly into calling frame
    for k in names:
        frame.f_locals[k] = dct[k]
    print(f'Loaded the following: {", ".join(names)}.')
