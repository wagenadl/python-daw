#!/usr/bin/python3

import numpy as np
import pyqplot as qp
import daw.colorx


def lut2d(H=360, K=100):
    hue = np.arange(H).reshape(H,1)*np.pi/180
    chroma = np.arange(K).reshape(1,K)*1.5/K
    lightness = np.arange(K).reshape(1,K)*0 + 65 + 5*chroma*np.cos(hue-1.1)
    chroma = chroma + chroma*.1*np.cos(hue) + chroma*.15*np.cos(2*hue-3.5)
    lch = np.stack((lightness+0*hue, chroma+0*hue, hue+0*chroma), 2)
    rgb = daw.colorx.colorconvert(lch, 'lshuv', 'srgb', clip=1)
    return rgb

qp.figure('s1', 10, 3)

qp.image(np.concatenate((lut2d().transpose(1,0,2), lut2d().transpose(1,0,2)),1))

#%%


def lut2d(H=360, K=100):
    def vonmises(xx, k):
        return np.exp(k*np.cos(xx))/np.exp(k)

    def expo(xx, x0, r):
        return vonmises(xx - x0*np.pi/180, 1/(r*np.pi/180))

    def shift(xx, x0, a, r):
        yy = vonmises(xx - x0*np.pi/180, 1/(r*np.pi/180))
        dydx = np.abs(np.max(np.diff(yy, axis=0) / np.diff(xx, axis=0)))
        return yy*a/dydx

    def cexp(xx, x0, a, r):
        yy = a*np.exp(-.5*(xx-x0)**2/r**2)
        return yy
    
    hh = np.arange(H).reshape(H,1)*360/H * np.pi/180
    hue = hh
    cc = np.arange(K).reshape(1,K)/K
    chroma = cc**1.5
    chroma = chroma * (1 + .5*expo(hh, 240, 15))
    chroma = chroma ** (1 - .1*expo(hh, 235, 25))
    chroma = chroma ** (1 - .2*expo(hh, 205, 25))
    chroma = chroma ** (1 - .3*expo(hh, 475, 25))
    chroma = chroma ** (1 - .1*expo(hh, 375, 25))
    chroma = chroma ** (1 - .15*expo(hh, 430, 25))
    chroma = chroma ** (1 - .1*expo(hh, 300, 75))
    chroma = chroma * (1 + .4*expo(hh, 409, 25))
    chroma = chroma * (1 + .3*expo(hh, 469, 25))
    ll = np.zeros((1,K)) + 65
    lightness = 99 - 45*cc
    lightness = lightness * (1+.2*cc*expo(hh, 410, 25))
    lightness = lightness * (1-.2*cc**.5*expo(hh, 580, 25))
    #lightness = 100*(.01*lightness) ** (1+.02*expo(hh, 280, 25))
    hue = hue + shift(hh, 350, .1, 25)
    hue = hue + shift(hh, 230, .3, 20)
    hue = hue - shift(hh, 275, .2, 30)
    hue = hue + shift(hh, 390, .4, 15)
    hue = hue - shift(hh, 350, .05, 100)
    chroma = chroma * (1 + .2*expo(hh, 280, 15))
    chroma = chroma ** (1 - .3*expo(hh, 275, 25))
    chroma = chroma ** (1 - .2*expo(hh, 180, 25))
    chroma = chroma ** (1 - .2*expo(hh, 370, 25))
    hue = hue + cexp(cc, 1, 1, .2)*shift(hh, 270, .15, 20)
    hue = hue - cexp(cc, 1, 1, .2)*shift(hh, 460, .2, 20)
    chroma = chroma * (1 + .4*expo(hh, 267, 10))
    chroma = chroma ** (1 - .25*expo(hh, 262, 10))
    hue = hue - cexp(cc, 0, 1, .2)*shift(hh, 260, .1, 25)
    hue = hue - cexp(cc, .25, 1, .2)*shift(hh, 270, .1, 20)
    chroma = chroma ** (1 - .25*expo(hh, 275, 10))
    chroma = chroma ** (1 - .15*expo(hh, 207, 15))
    chroma = chroma ** (1 - .15*expo(hh, 490, 15))
    chroma = chroma ** (1 - .15*expo(hh, 377, 15))
    chroma = chroma ** (1 - .05*expo(hh, 460, 35))
    #chroma = chroma ** (1 + .25*expo(hh, 260, 15))
    lightness = 100*(.01*lightness) ** (1 - .3*np.cos(hh - 230*np.pi/180))
    lch = np.stack((lightness+0*hh, chroma+0*hh, hue+0*cc), 2)
    rgb = daw.colorx.colorconvert(lch, 'lshuv', 'srgb', clip=1)
    cc = cc.reshape(1,K,1)
    rgb = rgb + .99 - rgb[:,:1,:].mean(0, keepdims=1)
    #rgb += .1*(1-cc)
    rgb[rgb>1]=1
    rgb[rgb<0]=0
    #rgb = 1 - (.02+.98*cc.reshape(1,K,1))*(1-rgb)
    return rgb

qp.figure('s2', 10, 3)

qp.image(np.concatenate((lut2d().transpose(1,0,2), lut2d().transpose(1,0,2)),1))

