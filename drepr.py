#!/usr/bin/python3

import numpy as np
import inspect
import numbers
try:
    import daw.octfile
    haveoct = True
except:
    haveoct = False
    # Quiet failure is OK; if we can't import octfile, neither can user.

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
            #use.append(f'... (+{len(ss)-len(use)})')
            use.append(f'... (𝘯={len(ss)})')
        else:
            use.append(f'...')
    return f'{pfx} {bo} {", ".join(use)} {bc} {sfx}'


def ol_dict(pfx, x):
    lst = list(x.keys())
    lst.sort()
    return ol_summary(pfx, lst, '{', '}')

def ol_struct(pfx, x):
    lst = list(x.keys())
    lst.sort()
    return ol_summary(pfx, lst, '{', '} «struct»')

def ol_tuple(pfx, x):
    return ol_summary(pfx, [str(v) for v in x], '(', ')')

def ol_list(pfx, x):
    return ol_summary(pfx, [str(v) for v in x], '[', ']')

def ol_nd_array(pfx, x):
    N = np.prod(x.shape)
    shp = '(' + '×'.join([str(v) for v in x.shape]) \
          + ' ' + str(x.dtype) + ')'
    if N==0 or N==np.max(x.shape):
        N = min(10, N)
        if x.dtype==np.float32 or x.dtype==np.float64 or x.dtype==np.complex:
            return ol_summary(pfx, [f'{v:.4g}' for v in x.flat], '[', ']', shp,
                              countrest=False)
        else:            
            return ol_summary(pfx, [str(v) for v in x.flat], '[', ']', shp,
                              countrest=False)
    else:
        return pfx + ' ' + shp

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
    if t==dict:
        return ol_dict(pfx, x)
    elif haveoct and t==daw.octfile.Struct:
        return ol_struct(pfx, x)
    elif t==tuple:
        return ol_tuple(pfx, x)
    elif t==list:
        return ol_list(pfx, x)
    elif t==np.ndarray:
        return ol_nd_array(pfx, x)
    elif t==str:
        return ol_string(pfx, x)
    elif isinstance(x, numbers.Number):
        return ol_number(pfx, x)
    else:
        return ol_other(pfx, x)

def d_struct(name, x):
    if x.ndim()==0:
        res = [f'{name} = «scalar Struct»:']
        x = x._contents_.flat[0]
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
    else:
        shp = '×'.join([f'{s}' for s in x.shape()])
        res = [f'{name} = «size {shp} Struct array». Field names:']
        res.append('  ' + ', '.join(x.fieldnames()))
        return "\n".join(res)
        
def d_dict(name, x):
    if len(x)==1:
        kk = 'one key'
    else:
        kk = f'{len(x)} keys'
    res = [f'{name} = «dict with {kk}»:']
    n = 0
    for k,v in x.items():
        res.append(oneline(f'  {k}:', v))
        n += 1
        if n >= maxlines:
            break
    if n < len(x):
        res.append(f'  ...')
    return "\n".join(res)

def d_tuple(name, x):
    res = [f'{name} = «tuple of length {len(x)}»:']
    n = 0
    for v in x:
        res.append(oneline(f'  ', v))
        n += 1
        if n >= maxlines:
            break
    if n < len(x):
        res.append(f'  ...')
    return "\n".join(res)

def d_list(name, x, typ='list'):
    res = [f'{name} = «{typ} of length {len(x)}»:']
    N = min((maxlines, len(x)))
    n = 0
    for v in x:
        res.append(oneline(f'  ', v))
        n += 1
        if n>=N:
            break
    if N<len(x):
        res.append(f'  ...')
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
    if x.ndim==0:
        N = 1
        shp = "scalar"
    elif x.ndim==1:
        N = x.shape[0]
        shp = f"vector of length {N}"
    else:
        N = np.prod(x.shape)
        shp = f"size {'×'.join([str(v) for v in x.shape])}"
    if x.ndim<3:
        colon = ':'
    else:
        colon = ''
    res = [f'{name} = «numpy array ({shp}) of {x.dtype}»{colon}']
    if N==0:
        res.append('  (empty)')
    elif x.ndim==0:
        if x.dtype==np.float32 or x.dtype==np.float64 or x.dtype==np.complex:
            res.append(ol_summary('  ',
                                  [f'{v:.4g}' for v in x.flat],
                                  '', ''))
        else:
            res.append(ol_summary('  ',
                                  [str(v) for v in x.flat],
                                  '', ''))
    elif x.ndim==1 or N==np.max(x.shape): # vector
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

def _getname():
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[2]
    #frame = inspect.getouterframes(frame)[1]
    s = inspect.getframeinfo(frame[0]).code_context[0]
    sol = s.find('(') + 1
    eol = s.rfind(')')
    name = s[sol:eol]
    return name

def d_any(x, name):
    t = type(x)
    if t==dict:
        return d_dict(name, x)
    elif haveoct and t==daw.octfile.Struct:
        return d_struct(name, x)
    elif t==tuple:
        return d_tuple(name, x)
    elif t==list:
        return d_list(name, x)
    elif t==type({}.keys()):
        return d_list(name, x, 'dict_keys')
    elif t==np.ndarray:
        return d_nd_array(name, x)
    elif t==str:
        return d_string(name, x)
    elif isinstance(x, numbers.Number):
        return d_number(name, x)
    else:
        return d_other(name, x)

def d(x):
    print(d_any(x, _getname()))

def dvalue(x):
    return d_any(x, _getname())
