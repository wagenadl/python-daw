#!/usr/bin/python3

import numpy as np
import daw.filterx
import matplotlib.pyplot as plt

plt.interactive(True)

class Oggalyze:
    def __init__(self, fn):
        if fn.endswith('.ogg'):
            fn = fn[:-4]
        self.oggfn = fn + '.ogg'
        if not fn.endswith('.ana'):
            fn += '.ana'
        try:
            dat = np.loadtxt(fn)
        except OSError:
            fn = '/tmp/' + fn
            dat = np.loadtxt(fn)            
        self.fn = fn
        self.tt = dat[:,0]
        self.yy = dat[:,1:]
        self.zz = daw.filterx.medianfltn(self.yy.mean(-1), 2)
        self.replot()

    def replot(self, indiv=False):
        plt.figure(1)
        plt.clf()
        plt.plot(self.tt, self.zz)
        if indiv:
            plt.plot(self.tt, self.yy)

        plt.figure(2)
        plt.clf()
        hh, xx = np.histogram(self.zz, 100)
        xx = (xx[1:] + xx[:-1])/2
        plt.bar(xx, hh, np.mean(np.diff(xx)))

    def splits(self, thr=13, tmin=5):
        print("Split threshold:", thr)
        over = (self.zz > thr).astype(int)
        over[0] = 0
        over[-1] = 0
        up = np.nonzero(np.diff(over)>0)[0][1:] # Drop first break
        dn = np.nonzero(np.diff(over)<0)[0][:-1] # Drop final break
        tup = self.tt[up]
        tdn = self.tt[dn]
        t0 = (tdn + tup) / 2
        dt = tup - tdn
        plt.figure(1)
        plt.plot(tup, 0*tup + thr, 'r<')
        plt.plot(tdn, 0*tdn + thr, 'g>')

        plt.figure(3)
        plt.clf()
        plt.plot(t0, dt, '.')
        plt.plot(t0, 0*dt + tmin)

        tt = tup[dt>=tmin] - min(1, tmin/2)
        plt.figure(1)
        plt.plot(tt, 0*tt + np.min(self.zz), 'm^')
        times = " ".join([f"{t:.1f}" for t in tt])

        print("If you're happy, run:")
        print(f"  vmcut {self.oggfn} /tmp/out {times}")
        print("Otherwise:")
        print("  a.replot()")
        print("and retry splits")
