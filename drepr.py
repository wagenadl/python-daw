#!/usr/bin/python3

import numpy as np
import inspect
import numbers

try:
    import oct2py
    haveo2p = True
except:
    print('Notice: Could not load oct2py. Loading octave files disabled. ')
    haveo2p = False

linelen = 70
maxlines = 15

def ol_summary(pfx, ss, bo, bc, sfx='', countrest=True):
    use = []
    luse = 0
    for s in ss:
        if luse==0 or (luse + 2 + len(s) + len(pfx) + len(sfx) + 6 < linelen):
            use.append(s)
            luse += 2 + len(s) 
        else:
            break
    if len(use) < len(ss):
        if countrest:
            use.append(f'... (+{len(ss)-len(use)})')
        else:
            use.append(f'...')
    return f'{pfx} {bo} {", ".join(use)} {bc} {sfx}'

def ol_dict(pfx, x):
    lst = list(x.keys())
    lst.sort()
    return ol_summary(pfx, lst, '{', '}')

def ol_tuple(pfx, x):
    return ol_summary(pfx, [str(v) for v in x], '(', ')')

def ol_list(pfx, x):
    return ol_summary(pfx, [str(v) for v in x], '[', ']')

def ol_nd_array(pfx, x):
    N = np.prod(x.shape)
    shp = '[' + '×'.join([str(v) for v in x.shape]) \
          + ' ' + str(x.dtype) + ']'
    if N==0 or N==np.max(x.shape):
        N = min(10, N)
        if x.dtype==np.float32 or x.dtype==np.float64 or x.dtype==np.complex:
            return ol_summary(pfx, [f'{v:.4g}' for v in x.flat], '[', ']', shp,
                              countrest=False)
        else:            
            return ol_summary(pfx, [str(v) for v in x.flat], '[', ']', shp,
                              countrest=False)
    else:
        return shp

def ol_string(pfx, x):
    if len(x) + len(pfx)> linelen:
        return f'{pfx} "{x[:linelen-10-len(pfx)]}..." [{len(x)}]'
    else:
        return f'{pfx} "{x}"'
        
def ol_number(pfx, x):
    try:
        # This fails for integers
        return f'{pfx} {x:.4}'
    except:
        return f'{pfx} {x}'

def ol_other(pfx, x):
    return f'{pfx} «{type(x)}»'
    
def oneline(pfx, x):
    t = type(x)
    if t == dict or (haveo2p and t==oct2py.io.Struct):
        return ol_dict(pfx, x)
    elif t == tuple:
        return ol_tuple(pfx, x)
    elif t == list:
        return ol_list(pfx, x)
    elif t == np.ndarray:
        return ol_nd_array(pfx, x)
    elif t == str:
        return ol_string(pfx, x)
    elif isinstance(x, numbers.Number):
        return ol_number(pfx, x)
    else:
        return ol_other(pfx, x)
    
def d_dict(name, x):
    res = [f'{name} = «dict»:']
    n = 0
    for k,v in x.items():
        res.append(oneline(f'  {k}:', v))
        n += 1
        if n >= maxlines:
            break
    if n < len(x):
        res.append(f'  ...')
        res.append(f'  ({len(x)} total items)')
    return "\n".join(res)

def d_tuple(name, x):
    res = [f'{name} = «tuple»:']
    n = 0
    for v in x:
        res.append(oneline(f'  ', v))
        n += 1
        if n >= maxlines:
            break
    if n < len(x):
        res.append(f'  ...')
        res.append(f'  ({len(x)} total items)')
    return "\n".join(res)

def d_list(name, x, typ='list'):
    res = [f'{name} = «{typ}»:']
    N = min((maxlines, len(x)))
    n = 0
    for v in x:
        res.append(oneline(f'  ', v))
        n += 1
        if n>=N:
            break
    if N<len(x):
        res.append(f'  ...')
        res.append(f'  ({len(x)} total items)')
    return "\n".join(res)

def d_string(name, x):
    if len(x) > linelen - len(name):
        return f'{name} = "{x:linelen - len(name) - 10}..." [{len(x)}]'
    else:
        return f'{name} = "{x}"'

def d_number(name, x):
    return ol_number(f'{name} =', x)
    
def d_other(name, x):
    return ol_other(f'{name} =', x)

def d_nd_array(name, x):
    N = np.prod(x.shape)
    shp = f"{'×'.join([str(v) for v in x.shape])} array of {x.dtype}"
    res = [f'{name} = «{shp}»:']
    if N==0 or N==np.max(x.shape):
        K = min(10, N)
        if x.dtype==np.float32 or x.dtype==np.float64 or x.dtype==np.complex:
            res.append(ol_summary('  ',
                                  [f'{v:.4g}' for v in x.flat[:K]],
                                  '[', ']'))
        else:            
            res.append(ol_summary('  ',
                                  [str(v) for v in x.flat[:K]],
                                  '[', ']'))
    elif x.ndim==2:
        N = min(10, x.shape[0])
        M = min(7, x.shape[1])
        for n in range(N):
            res1 = '  ['
            if x.dtype==np.float32 or x.dtype==np.float64:
                for m in range(M):
                    res1 += f' {x[n,m]:9.3g}'
            elif x.dtype==np.complex:
                M = min(4, M)
                for m in range(M):
                    res1 += f' {x[n,m]:16.3g}'
            else:
                for m in range(M):
                    res1 += f' {x[n,m]}'
            if M<x.shape[1]:
                res1 += ' ...'
            res1 += ' ]'
            res.append(res1)
        if x.shape[0] > N:
            res1 = '  ['
            if x.dtype==np.float32 or x.dtype==np.float64:
                for m in range(M):
                    res1 += '       ...'
            elif x.dtype==np.complex:
                M = min(4, M)
                for m in range(M):
                    res1 += '              ...'
            else:
                res1 += ' ...'
            if M<x.shape[1]:
                res1 += ' ...'
            res1 += ' ]'
            res.append(res1)
    return "\n".join(res)
    
def _d1(x):
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[2]
    #frame = inspect.getouterframes(frame)[1]
    s = inspect.getframeinfo(frame[0]).code_context[0]
    sol = s.find('(') + 1
    eol = s.rfind(')')
    name = s[sol:eol]
    t = type(x)
    if t == dict or (haveo2p and t==oct2py.io.Struct):
        return d_dict(name, x)
    elif t == tuple:
        return d_tuple(name, x)
    elif t == list:
        return d_list(name, x)
    elif t == type({}.keys()):
        return d_list(name, x, 'dict_keys')
    elif t == np.ndarray:
        return d_nd_array(name, x)
    elif t == str:
        return d_string(name, x)
    elif isinstance(x, numbers.Number):
        return d_number(name, x)
    else:
        return d_other(name, x)

def d(x):
    print(_d1(x))

def dvalue(x):
    return _d1(x)
        
def loadoct(fn):
    if not haveo2p:
        raise Exception('Oct2py not available. Cannot load octave file.')
    if fn.find("\\") >= 0:
        raise ValueError('Filename may not contain backslash. Use slash.')
    if fn.find("'") >= 0:
        if fn.find('"') >= 0:
            raise ValueError('Filename may not contain both kinds of quotes')
        code = f'x=load("{fn}");'
    else:
        code = f"x=load('{fn}');"
    oc = oct2py.Oct2Py()
    oc.eval(code)
    return oc.pull('x')
