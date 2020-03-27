#!/usr/bin/python3

# colorx.py - converted from my octave library

# Not included here:
# colorconvert - part of my "color" library
# dhsv, hotpow, hotpow2 - see pyqplot's luts
# lchdemo, luvdemo - not critical

# Probably, all of the color maps should be moved into qp.luts

import numpy as np
import math

def applylut(xx, lut, clim=None, nanc=None):
    '''APPLYLUT - Apply a color lookup table
    cc = APPLYLUT(xx, lut) applies the lookup table LUT to the data XX
    and returns the resulting colors. XX may be any shape, LUT must be Nx3.
    Optional argument CLIM may be a 2-ple specifying colormap limits; if 
    not given, min/max of XX are used.
    Optional argument NANC specifies a separate color for NaNs in XX. NANC 
    must be a 3-vector.'''

    S = xx.shape
    fltx = xx.flat

    if clim is None:
        clim = (np.nanmin(fltx), np.nanmax(fltx))

    if nanc is None:
        nanc = lut[0,:]

    C = lut.shape[0]

    cc = np.floor((fltx - clim[0]) / (clim[1] - clim[0]) * C).astype(int)
    cc[cc<0] = 0
    cc[cc>=C] = C-1
    cc = lut[cc,:]
    idx = np.nonzero(np.isnan(fltx))
    cc[idx,:] = nanc
    return np.reshape(cc, np.append(S, 3))

def bluegrayred(n=100, f=.25):
    '''BLUEGRAYRED - Generate a colormap with grays, blues, and reds
    clr = BLUEGRAYRED() generates a colormap with white-to-black in the middle, 
    blues at the lower end, and reds at the upper end.
    clr = BLUEGRAYRED(n, f) specifies the total number of entries and the
    fraction at each of the ends. By default, f=0.25, so half the map is 
    gray scale.
    This is a pretty ugly LUT.'''

    n1 = int(np.ceil(f*n))
    n0 = n-2*n1
    c1 = np.arange(n1) / (n1-1)
    c0 = np.arange(n0) / (n0-1)

    o1=0*c1
    o0=0*c0
    i1=o1+1
    i0=o0+1

    clr = np.concatenate((np.stack((i1,c1,o1),1),
                          np.stack((c0,c0,c0),1),
                          np.stack((o1,c1,i1),1)), 0)
    return np.flipud(clr)

def unshape(rgb):
    S = rgb.shape
    return np.reshape(rgb, [np.prod(S[:-1]), 3]), S

def reshape(rgb, S):
    return np.reshape(rgb, S)

def clip01(xx):
    yy = xx.copy()
    yy[yy<0] = 0
    yy[yy>1] = 1
    return yy

def rotacol(rgb, theta):
    '''ROTACOL - Rotate colors in RGB space
    rgb_ = ROTACOL(rgb, theta), where RGB is an (anything x 3) array
    and THETA is an angle in degrees, rotates the colors in RGB.'''

    # Due to Jacob Eggers, https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color
    U = np.cos(theta * np.pi/180)
    W = np.sin(theta * np.pi/180)

    rgb, S = unshape(rgb)

    r = (.299+.701*U+.168*W)*rgb[:,0] \
        + (.587-.587*U+.330*W)*rgb[:,1] \
        + (.114-.114*U-.497*W)*rgb[:,2]
    g = (.299-.299*U-.328*W)*rgb[:,0] \
        + (.587+.413*U+.035*W)*rgb[:,1] \
        + (.114-.114*U+.292*W)*rgb[:,2]
    b = (.299-.3*U+1.25*W)*rgb[:,0] \
        + (.587-.588*U-1.05*W)*rgb[:,1] \
        + (.114+.886*U-.203*W)*rgb[:,2]
    return reshape(np.stack((r,g,b), 1), S)

def bluered(n=100, xt=0, th1=0, th2=0, th=0):
    '''BLUERED - Blue/red colormap
    clr = BLUERED() returns a graded flag colormap.
    Optional argument N specifies the number of gradations.
    Optional argument XT adds a fraction XT of darkening to both ends 
    of the spectrum. XT must be in [0, 1].
    Optional argument TH shifts the entire map by TH degrees.
    Optional arguments TH1, TH2 shift the "red" part of the map by TH1
    degrees and the blue part by TH2 degrees.'''
    th1 += th
    th2 += th

    up = np.arange(n) / (n-1)
    dn = np.flip(up)
    z0 = 0*up
    z1 = z0 + 1

    if xt>0:
        upa = up[max(0, int(n*(1-xt))):]
        dna = np.flip(upa)
        z0a = 0*upa;
        z1a = 0*upa+1
        clr = np.concatenate((np.stack((z0a, z0a, upa), 1),
                              np.stack((up, up, z1), 1),
                              np.stack((z1, dn, dn), 1),
                              np.stack((dna, z0a, z0a), 1)), 0)
    else:
        clr = np.concatenate((np.stack((up, up, z1), 1),
                              np.stack((z1, dn, dn), 1)), 0)

    if th1!=0 or th2!=0:
        N = clr.shape[0]//2
        clr = np.concatenate((rotacol(clr[:N,:], th2),
                              rotacol(clr[N:,:], th1)), 0)

    return clr

def cielabdist(lab1, lab2):
    '''CIELABDIST - Perceptual distance between a pair of CIE L*a*b* colors
    e = CIELABDIST(lab1, lab2) where LAB1 and LAB2 are CIE L*a*b* color 
    triplets calculates the perceptual distance between them according 
    to CIEDE2000.
    See http://en.wikipedia.org/wiki/Color_difference.'''

    kL = 1
    K1 = 0.045
    L2 = 0.015
    kC = 1
    kH = 1

    DLp = lab2[0] - lab1[0]
    Lbar = (lab1[0] + lab2[0])/2

    C1 = math.sqrt(lab1[1]**2 + lab1[2]**2)
    C2 = math.sqrt(lab2[1]**2 + lab2[2]**2)
    Cbar = (C1+C2)/2

    ap1 = lab1[1] + (lab1[1]/2)*(1-math.sqrt(Cbar**7/(Cbar**7+25**7)))
    ap2 = lab2[1] + (lab2[1]/2)*(1-math.sqrt(Cbar**7/(Cbar**7+25**7)))

    Cp1 = math.sqrt(ap1**2 + lab1[2]**2)
    Cp2 = math.sqrt(ap2**2 + lab2[2]**2)
    Cbarp = (Cp1 + Cp2)/2
    DCp = Cp2 - Cp1

    hp1 = math.atan2(lab1[2], ap1) % (2*math.pi)
    hp2 = math.atan2(lab2[2], ap2) % (2*math.pi)

    if Cp1==0 or Cp2==0:
      Dhp = 0
    elif abs(hp1 - hp2) <= math.pi:
      Dhp = hp2 - hp1
    elif hp2 <= hp1:
      Dhp = hp2 - hp1 + 2*math.pi
    else:
      Dhp = hp2 - hp1 - 2*math.pi

    DHp = 2*math.sqrt(Cp1*Cp2)*math.sin(Dhp/2)
    if Cp1==0 or Cp2==0:
      Hbarp = hp1 + hp2
    elif abs(hp1-hp2)>math.pi:
      Hbarp = (hp1 + hp2 + 2*math.pi)/2
    else:
      Hbarp = (hp1 + hp2)/2


    T = (1 - .17*math.cos(Hbarp-math.pi/6) + .24*math.cos(2*Hbarp) 
         + .32*math.cos(3*Hbarp + math.pi*6/180) 
         -.20*math.cos(4*Hbarp - math.pi*63/180))
    SL = 1 + (.015*(Lbar-50)**2) / math.sqrt(20+(Lbar-50)**2)
    SC = 1 + .045*Cbarp
    SH = 1 + .015*Cbarp*T
    RT = (-2*math.sqrt(Cbarp**7/(Cbarp**7+25**7)) 
          * math.sin(math.pi/3
                     * math.exp(-((Hbarp-math.pi*275/180)/(math.pi*25/180))**2)))
    DE = math.sqrt((DLp/(kL*SL))**2 + (DCp/(kC*SC))**2 + (DHp/(kH*SH))**2 
                   + RT * DCp/(kC*SC) * DHp/(kH*SH))
    return DE


def cielchdist(lch1, lch2, l=1, c=1):
    '''CIELCHDIST - Perceptual distance between a pair of CIE L*C*h colors
    e = CIELCHDIST(lch1, lch2), where LCH1 and LCH2 are CIE L*C*h color
    triplets calculates the perceptual distance between them according 
    to CMC l:c
    See http://en.wikipedia.org/wiki/Color_difference.'''

    if lch1[0]<16:
        SL = .511
    else:
        SL = .040975*lch1[0] / (1+.01765*lch1[0])

    SC = 0.0638*lch1[1] / (1+.0131*lch1[1]) + .638
    F = math.sqrt(lch1[1]**4 / (lch1[1]**4+1900))

    if lch1[2] % (2*math.pi) <= 345*math.pi/180:
        T = .56 + abs(.2*math.cos(lch1[2]+math.pi*168/180))
    else:
        T = .36 + abs(.4*math.cos(lch1[2]+math.pi*35/180))

    SH = SC*(F*T+1-F)

    a1 = lch1[1]*math.cos(lch1[2])
    a2 = lch2[1]*math.cos(lch2[2])
    b1 = lch1[1]*math.sin(lch1[2])
    b2 = lch2[1]*math.sin(lch2[2])
    DH = math.sqrt((a1-a2)**2 + (b1-b2)**2 + (lch1[1]-lch2[1])**2)

    DE = math.sqrt( ((lch2[0]-lch1[0])/(l*SL))**2 
                    + ((lch2[1]-lch1[1])/(c*SC))**2 
                    + (DH/SH)**2)
    
    return DE

def getgray(rgb):
    '''GETGRAY - Extract gray value from RGB colors
    gry = GETGRAY(rgb) extracts gray channel from RGB colors (arbitrary x 3 
    array), weighing R, G, B as 1/3, 1/2, 1/6.'''
    cc, S = unshape(rgb)
    gry = (cc[:,0]*2 + cc[:,1]*3 + cc[:,2]) / 6
    return reshape(gry, S)

def darkhsv(n, d=0.3):
    '''DARKHSV - A cyclic colormap with darkened HSV colors
    cc=DARKHSV(n, d) returns a hsv-like colormap, but darkened to D
    (default: D=0.3, useful range 0..1).'''

    phi=np.arange(n)*2*math.pi/n
    
    cc = np.stack((np.cos(phi),
                   np.cos(phi+2*np.pi/3),
                   np.cos(phi+4*np.pi/3)), 1)/2 + .5
    gry = getgray(cc)
    cc *= d / np.reshape(gry, (n,1))
    dc = cc / np.reshape(np.mean(cc, 1), (n,1))
    cc *= dc**.5
    return clip01(cc)

def jet(N=256):
    '''JET - Color lookup table akin to Matlab's JET
    JET(N) returns a color lookup table akin to Matlab's JET,
    but with better sampling of color space, especially for small N.
    N defaults to 256.'''
    phi = np.linspace(0, 1, N)
    B0 = .2
    G0 = .5
    R0 = .8
    SB = .2
    SG = .25
    SR = .2
    P=4
    
    blue = np.exp(-.5*(phi-B0)**P / SB**P)
    red = np.exp(-.5*(phi-R0)**P / SR**P)
    green = np.exp(-.5*(phi-G0)**P / SG**P)
    return np.column_stack((red, green, blue))

def darkjet(n, d=0.6):
    '''DARKJET - Color lookup table like JET, but darkened.
    cc=DARKJET(n, d) returns a JET-like colormap, but darkened to D
    (default: d=0.6; useful range: 0..2).'''

    cc = jet(n)
    gry = getgray(cc)
    cc *= np.reshape(np.tanh(d/gry), (n,1))
    dc = cc - np.reshape(np.mean(cc,1), (n,1))
    cc += dc
    return clip01(cc)

def grayred(n=100, f=0.25):
    '''GRAYRED - Generate a colormap with grays and reds
    clr = GRAYRED() generates a colormap with white-to-black at the
    lower end and reds at the upper end.
    clr = GRAYRED(n, f) specifies the total number of entries and the
    fraction at each of the ends. By default, f=0.25, so 3/4 of the map is 
    gray scale.'''
    

    n1 = int(f*n)
    n0 = n-n1
    c1 = np.arange(n1) / (n1 - 1)
    c0 = np.arange(n0) / (n0 - 1)

    o1=0*c1
    o0=0*c0
    i1=o1+1
    i0=o0+1
    clr = np.concatenate((np.stack((i1, c1, o1), 1),
                          np.stack((c0, c0, c0), 1)), 0)
    return np.flipud(clr)

def whitepointD(t):
    '''WHITEPOINTD - Return whitepoint for a given color temperature
    xyz = WHITEPOINTD(t), where T is a color temperature in Kelvin,
    returns the XYZ values corresponding to the white point at that
    temperature.'''

    # Source:
    # https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_series_D

    t = np.array(t)
    S = t.shape
    t = t.flatten()
    x = .244063 + .09911e3/t + 2.9678e6/t**2 - 4.6070e9/t**3
    x2 = .237040 + .24748e3/t + 1.9018e6/t**2 - 2.0064e9/t**3
    x[t>7000] = x2[t>7000]
    y = -3.000*x**2 + 2.870*x - 0.275

    Y = np.ones(t.shape)
    X = (Y/y) * x
    Z = (Y/y) * (1 - x - y)

    xyz = np.stack((X, Y, Z), 1)
    S = list(S)
    S.append(3)
    return np.reshape(xyz, S)

def whitepoints(s):
    '''WHITEPOINTS - Return XYZ values of standard white points
    xyz = WHITEPOINTS(s) returns XYZ values of one of several standard
    white points: d50, d55, d65, a, c'''
    # Source: Matlab R2012b's whitepoint function

    s = s.lower()
    if s=='d50':
        return np.array([0.9642,    1.0000,    0.8251])
    elif s=='d55':
        return np.array([0.9568,    1.0000,   0.9214])
    elif s=='d65':
        return np.array([0.9504,    1.0000,    1.0889])
    elif s=='a':
        return np.array([1.0985,    1.0000,    0.3558])
    elif s=='c':
        return np.array([0.9807,    1.0000,    1.1823])
    else:
        raise ValueError('Unknown white point')

def resistor(white=False, extra=0):
    '''RESISTOR - Resistor code color map
    rgb = RESISTOR() returns a color map with 9 colors following
    the standard resistor code. (The white at end is left out, unless 
    optional argument WHITE is passed as True.)
    Optional argument EXTRA can be 1 or 2 to add 2 or 4 extra colors.'''

    rgb = [[0, 0, 0], [ .7, .1, .1], [ 1, .1, 0], [ .9, .6, 0], [ .8, .8, 0],
           [ 0, .8, 0], [ 0, 0, 1], [ .7, 0, 1], [ .5, .5, .5]]
    if extra:
        rgb.insert(8, [1, .3, 1])
        rgb.insert(6, [0, 1, 1])
        if extra > 1:
            rgb.append([.45, .45, 1])
            rgb.append([0, .6, 0])
    if white:
        rgb.append([1, 1, 1])
    return np.array(rgb)

