#!/usr/bin/python3

import numpy as np
import oct2py
import inspect
import numbers

linelen = 70

def ol_summary(pfx, ss, bo, bc, sfx='', N=None):
    use = []
    luse = 0
    for s in ss:
        if luse + 2 + len(s) + len(pfx) + len(sfx) + 6 < linelen:
            use.append(s)
            luse += 2 + len(s) 
        else:
            break
    if len(use) < len(ss):
        if N is None:
            N = len(ss)
        use.append(f'... (+{N-len(use)})')
    print(f'{pfx} {bo} {", ".join(use)} {bc} {sfx}')

def ol_dict(pfx, x):
    lst = list(x.keys())
    lst.sort()
    ol_summary(pfx, lst, '{', '}')

def ol_tuple(pfx, x):
    ol_summary(pfx, [str(v) for v in x], '(', ')')

def ol_list(pfx, x):
    ol_summary(pfx, [str(v) for v in x], '[', ']')

def ol_nd_array(pfx, x):
    N = np.prod(x.shape)
    shp = '[' + '×'.join([str(v) for v in x.shape]) \
          + ' ' + str(x.dtype) + ']'
    if N==0 or N==np.max(x.shape):
        N = min(10, N)
        if x.dtype==np.float32 or x.dtype==np.float64 or x.dtype==np.complex:
            ol_summary(pfx, [f'{v:.4g}' for v in x.flat], '[', ']', shp)
        else:            
            ol_summary(pfx, [str(v) for v in x.flat], '[', ']', shp)
    else:
        print(shp)

def ol_string(pfx, x):
    if len(x) + len(pfx)> linelen:
        print(f'{pfx} "{x[:linelen-10-len(pfx)]}..." [{len(x)}]')
    else:
        print(f'{pfx} "{x}"')
        
def ol_number(pfx, x):
    print(f'{pfx} {x}')

def ol_other(pfx, x):
    print(f'{pfx} [{type(x)}]')
    
def oneline(pfx, x):
    t = type(x)
    if t == dict or t==oct2py.io.Struct:
        ol_dict(pfx, x)
    elif t == tuple:
        ol_tuple(pfx, x)
    elif t == list:
        ol_list(pfx, x)
    elif t == np.ndarray:
        ol_nd_array(pfx, x)
    elif t == str:
        ol_string(pfx, x)
    elif isinstance(x, numbers.Number):
        ol_number(pfx, x)
    else:
        ol_other(pfx, x)
    
def d_dict(name, x):
    print(f'{name} = [dict]:')
    for k,v in x.items():
        oneline(f'  {k}:', v)

def d_tuple(name, x):
    print(f'{name} = [tuple]:')
    for v in x:
        oneline(f'  ', v)

def d_list(name, x):
    print(f'{name} = [list]:')
    N = min((20, len(x)))
    for n in range(N):
        oneline(f'  ', v)
    if N<len(x):
            print(f'  ...+{len(x)-N}')

def d_string(name, x):
    if len(x) > linelen - len(name):
        print(f'{name} = "{x:linelen - len(name) - 10}..." [{len(x)}]')
    else:
        print(f'{name} = "{x}"')

def d_number(name, x):
    print(f'{name} = ', end='')
    ol_number(x)
    
def d_other(name, x):
    print(f'{name} = ', end='')
    ol_other(x)

def d_nd_array(name, x):
    N = np.prod(x.shape)
    shp = '[array, ' + '×'.join([str(v) for v in x.shape]) \
          + ' ' + str(x.dtype) + ']'
    print(f'{name} = {shp}:')
    if N==0 or N==np.max(x.shape):
        K = min(10, N)
        if x.dtype==np.float32 or x.dtype==np.float64 or x.dtype==np.complex:
            ol_summary('  ', [f'{v:.4g}' for v in x.flat[:K]], '[', ']', N=N)
        else:            
            ol_summary('  ', [str(v) for v in x.flat[:K]], '[', ']', N=N)
    elif x.ndim==2:
        N = min(10, x.shape[0])
        M = min(7, x.shape[1])
        for n in range(N):
            res = '  ['
            if x.dtype==np.float32 or x.dtype==np.float64:
                for m in range(M):
                    res += f' {x[n,m]:9.3g}'
            elif x.dtype==np.complex:
                M = min(4, M)
                for m in range(M):
                    res += f' {x[n,m]:16.3g}'
            else:
                for m in range(M):
                    res += f' {x[n,m]}'
            if M<x.shape[1]:
                res += ' ...'
            res += ' ]'
            print(res)
        if x.shape[0] > N:
            res = '  ['
            if x.dtype==np.float32 or x.dtype==np.float64:
                for m in range(M):
                    res += '       ...'
            elif x.dtype==np.complex:
                M = min(4, M)
                for m in range(M):
                    res += '              ...'
            else:
                res += ' ...'
            if M<x.shape[1]:
                res += ' ...'
            res += ' ]'
            print(res)
    
def d(x):
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    s = inspect.getframeinfo(frame[0]).code_context[0]
    sol = s.find('(') + 1
    eol = s.rfind(')')
    name = s[sol:eol]
    t = type(x)
    if t == dict or t==oct2py.io.Struct:
        d_dict(name, x)
    elif t == tuple:
        d_tuple(name, x)
    elif t == list:
        d_list(name, x)
    elif t == np.ndarray:
        d_nd_array(name, x)
    elif t == str:
        d_string(name, x)
    elif isinstance(x, numbers.Number):
        d_number(name, x)
    else:
        d_other(name, x)
        
def loadoct(fn):
    if fn.find("\\") >= 0:
        raise ValueError('Filename may not contain backslash')
    if fn.find("'") >= 0:
        if fn.find('"') >= 0:
            raise ValueError('Filename may not contain both kinds of quotes')
        code = f'x=load("{fn}");'
    else:
        code = f"x=load('{fn}');"
    oc = oct2py.Oct2Py()
    oc.eval(code)
    return oc.pull('x')
