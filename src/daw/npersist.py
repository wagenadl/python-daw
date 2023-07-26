#!/usr/bin/python3

import numpy as np
import h5py
import inspect
import re
from collections import namedtuple

def cansave(v):
    '''CANSAVE - Can a given object be saved by NPERSIST?
    CANSAVE(v) returns True if V can be saved by NPERSIST.
    Currently, NPERSIST can save:
      - Numpy arrays
      - Strings
      - Simple numbers (int, float, complex)
      - Lists, tuples, and dicts containing those (even hierarchically).'''
    t = type(v)
    if v is None:
        return True
    if t==np.ndarray:
        return True
    if t==str:
        return True
    if t==int or t==np.int32 or t==np.int64 \
       or t==float or t==np.float64 or t==complex:
        return True
    if t==dict:
        for k,v1 in v.items():
            if not cansave(v1):
                return False
        return True
    if t==list or t==tuple:
        for v1 in v:
            if not cansave(v1):
                return False
        return True
    print(f'Cannot save {t}')
    return False

def saveto(fd, name, value):
    '''SAVETO - Add a named variable to a HDF5 file
    SAVETO(fd, name, value), where FD is a HDF5 file, saves VALUE as a
    new dataset with the given NAME.'''
    cre = re.compile("'(.*)'")
    typ = type(value)
    typstr = str(typ)
    m = cre.search(typstr)
    if m:
        typstr = m.group(1)
    print("Creating", name, typstr)
    if typ==tuple or typ==list:
        subtyp = None
        simple = True
        for v in value:
            if subtyp==None:
                subtyp=type(v)
            elif type(v)!=subtyp:
                simple = False
                break
        if simple and (subtyp==int or subtyp==float or subtyp==complex
                       or subtyp==np.int32 or subtyp==np.int64
                       or subtyp==np.float64):
            ds = fd.create_dataset(name, data=np.array(value))
        else:
            ds = fd.create_group(name)
            for k in range(len(value)):
                saveto(ds, str(k), value[k])
    elif typ==str:
        # This slightly ugly trick makes the result compatible with Octave
        ds = fd.create_dataset(name,
                               data=np.string_(bytes(value, encoding='utf8')))
    elif typ==dict:
        ds = fd.create_group(name)
        for k, v in value.items():
            saveto(ds, str(k), v)
    elif value is None:
        ds = fd.create_dataset(name, data=[])
    else:
        ds = fd.create_dataset(name, data=value)
    ds.attrs['origtype'] = typstr
    
def save(fn, *args):
    '''SAVE - Save multiple variables in one go as in Octave/Matlab
    SAVE(filename, var1, var2, ..., varN) saves each of the variables
    VAR1...VARN into a single HDF5 file.
    The result can be loaded with
      var1, var2, ..., varN = LOAD(filename)
    or with
      vv = LOADDICT(filename)
    In the latter case, vv becomes a DICT with the original variable
    names as keys.
    Note that SAVE is a very hacked function: It uses the INSPECT module
    to determine how it was called. That means that VARi must all be simple
    variables and the FILENAME, if given as a direct string, may not contain
    commas. Variable names must start with a letter and may only contain
    letters, numbers, and underscore.
    OK examples:
      x = 3
      y = 'Hello'
      z = np.eye(3)
      save('/tmp/test.h5', x, y, z)
      fn = '/tmp/test,1.h5'
      save(fn, x, y, z)
    Bad examples:
      save('/tmp/test,1.h5', x, y)
      save('/tmp/test.h5', x+3)'''
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
        if not cansave(args[k]):
            raise ValueError('Cannot save variable: ' + names[k])
    
    with h5py.File(fn, 'w') as fd:
        ds = fd.create_dataset('__core__', data=[])
        ds.attrs['names'] = names
        for k in range(N):
            saveto(fd, names[k], args[k])

def savedict(fn, dct):
    '''SAVEDICT - Save data from a DICT
    SAVEDICT(filename, dct), where DCT is a DICT, saves the data contained
    therein as a HDF5 file. The result can be loaded with
      dct = LOADDICT(filename).'''
    nre = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')
    for k, v in dct.items():
        if not nre.match(k):
            raise ValueError('Bad variable name: ' + k)
        if not cansave(v):
            raise ValueError('Cannot save variable: ' + k)
    
    with h5py.File(fn, 'w') as fd:
        ds = fd.create_dataset('__core__', data=[])
        ds.attrs['names'] = [k for k in dct.keys()]
        for k, v in dct.items():
            saveto(fd, k, v)
    
            
def loadfrom(fd, k):
    ds = fd[k]
    origtyp = ds.attrs['origtype']
    print("origtyp", origtyp)
    if origtyp=='NoneType':
        return None
    if type(ds) == h5py.Dataset:
        v = np.array(ds)
        if origtyp=='list' or origtyp=='tuple':
            v = list(v)
            for k1 in range(len(v)):
                if type(v[k1])==np.int64:
                    v[k1] = int(v[k1])
            if origtyp=='tuple':
                return tuple(v)
            else:
                return v
        elif origtyp=='str':
            return str(bytes(v), encoding='utf8')
        else:
            if origtyp=='int':
                return int(v)
            elif origtyp=='float':
                return float(v)
            elif origtyp=='complex':
                return complex(v)
            else:
                return v
    else:
        if origtyp=='None':
            return None
        elif origtyp=='list' or origtyp=='tuple':
            N = len(ds.keys())
            lst = []
            for n in range(N):
                lst.append(loadfrom(ds, str(n)))
            if origtyp=='tuple':
                lst = tuple(lst)
            return lst
        elif origtyp=='dict':
            dct = {}
            for k in ds:
                dct[k] = loadfrom(ds, k)
            return dct
        else:
            raise ValueError('Cannot load ' + k + ' of type ' + origtyp)
            
def loaddict(fn):
    '''LOADDICT - Reload data saved with SAVE
    x = LOADDICT(fn) loads the file named FN, which should have been created
    by SAVE. The result is a dictionary with the original variable names
    as keys.'''
    with h5py.File(fn, 'r') as fd:
        kk = fd['__core__'].attrs['names']
        x = {}
        for k in kk:
            x[k] = loadfrom(fd, k)
        return x
            
def load(fn):
    '''LOAD - Reload data saved with SAVE
    v1, v2, ..., vn = LOAD(fn) loads the file named FN, which should have
    been created by SAVE(fn, v1, v2, ..., vn).'''
    with h5py.File(fn, 'r') as fd:
        kk = fd['__core__'].attrs['names']
        res = []
        for k in kk:
            res.append(loadfrom(fd, k))
        ResType = namedtuple("npersist", kk)
        return ResType._make(res)
        #return tuple(res)

