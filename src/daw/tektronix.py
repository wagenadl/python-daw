#!/usr/bin/python3

import numpy as np
import glob
import csv

def readsettings(fn):
    '''READSETTINGS - Read an old-style Tektronix .SET file
    res = READSETTINGS(fn) reads a Tektronix ".SET" file and returns
    a dict with all the information contained in it.
    This works for the TDSxxxx oscilloscopes, not the MSO.
    See READSET.'''
    res = {}
    with open(fn, "r") as fd:
        txt = fd.read().strip()
        secs = txt[1:].split(";:")
        for sec in secs:
            ttl, recs = sec.split(":", 1)
            res[ttl] = {}
            recs = recs.split(";")
            for rec in recs:
                k, v = rec.split(" ", 1)
                try:
                    if '.' in v:
                        v = float(v)
                    else:
                        v = int(v)
                except:
                    pass
                res[ttl][k] = v
    return res

def _convert(v):
    try:
        if '.' in v:
            return float(v)
        else:
            return int(v)
    except:
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        else:
            return v


def _initialcaps(x):
    return x[0] + x[1:].lower()


def _insertintodict(dst, path, v):
    if len(path)==1:
        v = _initialcaps(v.strip())
        if ',' in v:
            v = [ _convert(x) for x in v.split(",")]
        else:
            v = _convert(v)
        dst[path[0]] = v
    else:
        if path[0] not in dst:
            dst[path[0]] = {}
        elif type(dst[path[0]])!=dict:
            dst[path[0]] = { '': dst[path[0]] }
        _insertintodict(dst[path[0]], path[1:], v)

def readset(fn):
    '''READSET - Read an new-style Tektronix .SET file
    res = READSETTINGS(fn) reads a Tektronix ".SET" file and returns
    a dict with all the information contained in it.
    This works for the MSO oscilloscope, not the TDSxxx.
    See READSETTINGS.'''
    res = {}
    with open(fn, "r") as fd:
        for line in fd.readlines():
            kv = line.split(" ", 1)
            if len(kv)<2:
                continue
            kk = [ _initialcaps(x) for x in kv[0].split(":")[1:]]
            v = kv[1]
            _insertintodict(res, kk, v)
    return res


def readisf(fn, resample=None):
    info = {}
    data = None
    with open(fn, "rt", encoding='latin-1') as fd:
        header = fd.read()
        idx = header.index(":CURV")
        stuff = header[:idx].split(":")
        curv = header[idx:idx+20]
    if curv.startswith(":CURV #"):
        l = 7
    elif curv.startswith(":CURVE #"):
        l = 8
    else:
        raise Exception("CURVE?")
    n = int(curv[l])
    nn = int(curv[l+1:l+n+1])
            
    with open(fn, "rb") as fd:
        data = fd.read()[idx+l+n+1:idx+l+n+1+nn]
    # There can be more than one trace in the file, and
    # it doesn't have to be int8. For now though:

    for s in stuff:
        for bit in s.split(";"):
            kv = bit.split(" ", 1)
            if len(kv)<2:
                continue
            info[_initialcaps(kv[0])] = _convert(kv[1])

    ttt = np.arange(nn) * info["Xin"] + info["Xze"]
    yyy = (np.frombuffer(data, np.int8) - info["Yof"]) * info["Ymu"] + info["Yze"]
    if resample is not None:
        kk = nn//resample
        nn = resample*kk
        ttt = ttt[:nn].reshape(kk, resample).mean(-1)
        yyy = yyy[:nn].reshape(kk, resample).mean(-1)

    return ttt, yyy, info

def loadmso(fn, resample=None):
    if fn.endswith("SET"):
        settings = readset(fn)
        fn = fn[:-4]
    else:
        settings = readset(fn + ".SET")
    ttt = {}
    yyy = {}
    chinfo = {}
    for k, v in settings['Select'].items():
        if k.startswith('Ch') and v>0:
            k = int(k[2:])
            ttt[k], yyy[k], chinfo[k] = readisf(f"{fn}CH{k}.ISF", resample)
    return ttt, yyy, chinfo, settings
                
def tekload(folder):
    '''TEKLOAD - Load data from Tektronix files
tt, vv, info = TEKLOAD(folder) loads data from a Tektronix oscilloscope.
FOLDER is typically something like "ALL0000", straight from the oscilloscope.
TT is a vector of time points, in seconds.
VV is a dict, indexed by channel numbers (1, 2, ...) of voltage (or current)
values, offset and scale applied.
INFO contains additional information from the .SET file.
For instance, INFO['CH2']['YUNIT'] is either "V" or "A".
In addition, INFO[k] is the header from the CHk.CSV file.
'''
    setfn = glob.glob(f"{folder}/*.SET")[0]
    info = readsettings(setfn)
    vv = {}
    tt = {}
    
    chfns = glob.glob(f"{folder}/*.CSV")
    for fn in chfns:
        with open(fn, "r") as fd:
            dct = {}
            xdat = []
            ydat = []
            rdr = csv.reader(fd)
            for row in rdr:
                if len(row):
                    k = row[0]
                    v = row[1]
                    x = row[3]
                    y = row[4]
                    if k:
                        try:
                            v = float(v)
                        except:
                            pass
                        dct[k] = v
                    xdat.append(float(x))
                    ydat.append(float(y))
        ch = int(fn.split(".")[-2][-1])
        info[ch] = dct
        tt[ch] = np.array(xdat)
        vv[ch] = np.array(ydat)

def demo():
    ttt, yyy, chinfo, settings = loadmso("T0049", 20)
    plt.figure(1)
    plt.clf()
    for k in ttt:
        plt.plot(ttt[k]*1e6, yyy[k])
        
