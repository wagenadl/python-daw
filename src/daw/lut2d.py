#!/usr/bin/python3

import numpy as np
import pyqplot as qp
import daw.colorx


def vonmises(xx, k):
    return np.exp(k*np.cos(xx))/np.exp(k)

def expo(xx, x0, r):
    return vonmises(xx - x0*np.pi/180, 1/(r*np.pi/180))

def shift(xx, x0, a, r):
    yy = vonmises(xx - x0*np.pi/180, 1/(r*np.pi/180))
    dydx = np.abs(np.max(np.diff(yy, axis=0) / np.diff(xx, axis=0)))
    return yy*a/dydx

def cexp(xx, x0, r):
    yy = np.exp(-.5*(xx-x0)**2/r**2)
    return yy

def z2rgb(zz):
    hh = np.angle(zz)
    hue = hh
    cc = np.abs(zz)
    chroma = cc**1.9*1.6
    lightness = 99 - 40*cc**1.5
    lightness += 20*cc**1.5*np.cos(hh-80*np.pi/180)

    hue = hue - .6*expo(hh, 250, 30)*(1-cc)
    hue = hue + .6*expo(hh, 120, 30)*(1-cc)
    chroma = chroma**(1 - .15*expo(hh, 80, 30))

    lch = np.stack((lightness, chroma, hue), 2)
    rgb = daw.colorx.colorconvert(lch, 'lshuv', 'srgb', clip=1)
    A,B,C = rgb.shape
    z0 = np.argmin(np.abs(zz.reshape(A*B)))
    rgb0 = rgb.reshape(A*B,C)[z0] # rgb.max((0,1), keepdims=1)
    rgb = rgb + .99 - rgb0
    rgb[rgb>1]=1
    rgb[rgb<0]=0
    return rgb


if __name__ == "__main__":
    qp.figure('s2', 8, 4)
    qp.subplot(1,2,0)
    xx = np.arange(-1,1.0001, .002)
    X = len(xx)
    xx = xx.reshape(1,X)
    yy = xx.T
    zz = xx + 1j*yy
    zz[np.abs(zz)>1] = 1e-9
    qp.image(z2rgb(zz), xx=xx, yy=-yy)
    phi = np.arange(0,2*np.pi+.0001,.001)
    qp.pen('k', 0, alpha=.2)
    for r in [.3, .4]:
        qp.plot(r*np.cos(phi), r*np.sin(phi))
    qp.marker('+',1)
    qp.mark(0,0)
    qp.shrink(1,1)

    qp.subplot(1,2,1)
    xx = np.arange(0,3*np.pi, .01)
    X = len(xx)
    yy = np.arange(0,1,.005)
    Y = len(yy)
    zz = yy.reshape(Y,1)*np.exp(1j*xx.reshape(1,X))
    qp.image(z2rgb(zz), xx=180*xx/np.pi, yy=-yy)
    qp.pen('k', 0, alpha=.2)
    for r in [.3, .4]:
        qp.plot(xx*180/np.pi, 0*xx-r)
    qp.shrink(1)
