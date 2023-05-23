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
      - Numpy arrays
      - Strings
      - Simple numbers (int, float, complex)
      - Lists, tuples, and dicts containing those (even hierarchically).'''
    if v is None:
        return True
    t = type(v)
    if t==np.ndarray:
        return True
    if t==str:
        return True
    if t==int or t==np.int32 or t==np.int64 \
       or t==float or t==np.float64 or t==complex \
       or t==np.bool_ or t==np.intc:
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

def loaddict(fn):
    '''LOADDICT - Reload data saved with SAVE or SAVEDICT
    x = LOADDICT(fn) loads the file named FN, which should have been created
    by SAVE. The result is a dictionary with the original variable names
    as keys.'''
    with open(fn, 'rb') as fd:
        dct = pickle.load(fd)
    del dct['__names__']
    return dct

def loadtuple(fn, typename='PPERSIST'):
    '''LOADTUPLE - Reload data saved with SAVE or SAVEDICT
    x = LOADTUPLE(fn) loads the file named FN which should have been created
    by SAVE. The result is a named tuple with the original variable names
    as keys.'''
    with open(fn, 'rb') as fd:
        dct = pickle.load(fd)
    names = dct['__names__']
    tpl = collections.namedtuple(typename, names)
    lst = []
    for n in names:
        lst.append(dct[n])
    return tpl(*lst)

def load(fn):
    '''LOAD - Reload data saved with SAVE
    vv = LOAD(fn) loads the file named FN, which should have been created
    by SAVE(fn, v1, v2, ..., vn) and returns a named tuple. Of course
    you can also say: v1, v2, ..., vn = LOAD(fn) to immediately unpack 
    the tuple.
    Simply calling LOAD(fn) without assignment to variables directly
    loads the variables saved by SAVE(fn, ...) into the caller's namespace.
    This is a super ugly Matlab-style hack, but really convenient.
    LOADDICT and LOADTUPLE are cleaner alternatives'''
    with open(fn, 'rb') as fd:
        dct = pickle.load(fd)
    names = dct['__names__']

    frame = inspect.currentframe().f_back
    string = inspect.getframeinfo(frame).code_context[0]
    if '=' in string:
        # Return a tuple
        bits = fn.split('/')
        leaf = bits[-1]
        bits = fn.split('.')
        base = bits[0]
        typename = ''
        for b in base:
            if (b>='A' and b<='Z') or (b>='a' and b<='z') \
               or (b>='0' and b<='9' and typename!=''):
                typename += b
        if typename=='':
            typename='anon'
        ResType = collections.namedtuple(typename, names)
        res = []
        for k in names:
            res.append(dct[k])
        return ResType._make(res)
    else:
        # Else inject directly into calling frame
        for k in names:
            frame.f_locals[k] = dct[k]
        print(f'Loaded the following: {", ".join(names)}.')
