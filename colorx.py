# color.py - collection of "COLORX" octave functions converted to python

import numpy as np

def unshape(cc):
    '''[cc, S] = unshape(cc) reshapes an AxBx...xL array to NxL, where
    N is the product of AxBx...'''
    S = cc.shape
    P = np.prod(np.array(S)[:-1])
    
    cc = np.reshape(cc.copy(), (P, S[-1]))
    return (cc,S)

def reshape(cc, S):
    '''Reverses the operation of UNSHAPE'''
    return np.reshape(cc,S)

def cliprgb(cc, clip):
    '''CLIPRGB - Clip linear RGB colors (helper for ciexyztolinearrgb)
    Operates in place.'''

    if clip==0:
        pass
    if clip==1:
        cc[cc<0] = 0
        cc[cc>1] = 1
    elif np.isnan(clip):
        nn = np.any(np.logical_or(cc<0, cc>1), 1)
        cc[nn, :] = np.nan
    elif clip==2:
        cc[cc<0] = 0
        mx = np.max(cc, 1)
        mx[mx<1] = 1
        cc = cc / np.repeat(mx, 3, 1)
    else:
        raise Exception('Illegal clip specification')



def cielchtocielab(cc):
    '''CIELCHTOCIELAB - Convert from CIE L*C*h to CIE L*a*b* colors
    cc = CIELCHTOCIELAB(cc) converts from CIE L*C*h to CIE L*a*b* color 
    space. CC must be AxBx...x3.
    In our representation, H ranges from 0 to 2 π.
    May also be used to convert from CIE L* C*_uv h_uv to CIE L*u*v* colors.'''
    (cc, S) = unshape(cc)
    a = cc[:,1] * np.cos(cc[:,2])
    b = cc[:,1] * np.sin(cc[:,2])
    cc[:,1] = a
    cc[:,2] = b
    return reshape(cc, S)

def lshuvtocieluv(cc):
    '''LSHUVTOCIELUV - Convert from L* s_uv h_uv colors to CIE L*u*v*
    cc = LSHUVTOCIELUV(cc) converts L* s_uv h_uv colors to CIE L*u*v*.
    CC must be AxBx...x3..
    May also be used to convert (unofficial) L* s h colors to CIE L*a*b*.
    Equations taken from http://en.wikipedia.org/wiki/CIELUV and
    http://en.wikipedia.org/wiki/Colorfulness#Saturation'''

    (cc, S) = unshape(cc)
    cc[:,1] *= cc[:,0]
    cc = reshape(cc, S)
    return cielchtocielab(cc)

def srgbtolinearrgb(cc):
    '''SRGBTOLINEARRGB - Convert from (gamma corrected) sRGB to linear RGB
    cc = SRGBTOLINEARRGB(cc) converts from sRGB to linear RGB. CC must
    be AxBx...x3 and have values in the range [0, 1].
    The conversion here is based on http://en.wikipedia.org/wiki/SRGB'''

    cc = cc.copy()
    big = cc>.04045;
    cc[big] = ((cc[big]+.055)/1.055)**2.4
    sml = np.logical_not(big)
    cc[sml] = cc[sml]/12.92;
    return cc

def linearrgbtosrgb(cc):
    '''LINEARRGBTOSRGB - Convert from linear RGB to (gamma corrected) sRGB
    cc = LINEARRGBTOSRGB(cc) converts from linear RGB to sRGB. CC must
    be AxBx...x3 and have values in the range [0, 1].'''
    # The conversion here is based on http://en.wikipedia.org/wiki/SRGB

    big = cc>.0031308;
    sml = np.logical_not(big)
    cc = cc.copy()
    cc[big] = 1.055*cc[big]**(1/2.4) - .055
    cc[sml] = cc[sml]*12.92
    return cc

def linearrgbtohcl(cc, gamma=10):
    '''LINEARRGBTOHCL - Convert linear RGB to HCL color space
    cc = LINEARRGBTOHCL(cc) converts linear RGB colors to HCL colors.
    CC must be AxBx...x3 and have values in the range [0, 1].
    The HCL color space was defined by Sarifuddin and Missaoui to be
    more perceptually uniform than other color spaces
    cc = LINEARRGBTOCHL(cc, gamma) specifies Missaoui's gamma parameter.'''

    cc, S = unshape(cc);
    
    Y0 = 100;
    minrgb = np.min(cc,  1)
    maxrgb = np.max(cc, 1)
    maxrgb[maxrgb==0] = 1e-99;
    alpha = minrgb/maxrgb / Y0;

    Q = np.exp(gamma*alpha)
    L = (Q*maxrgb + (Q-1)*minrgb) / 2;

    d_RG = cc[:,0]-cc[:,1]
    d_GB = cc[:,1]-cc[:,2]
    d_BR = cc[:,2]-cc[:,0]
    C = (Q/3) * (np.abs(d_RG) + np.abs(d_GB) + np.abs(d_BR))

    H0 = np.atan(d_GB/d_RG)

    H = (2/3) * H0

    idx = np.logical_and(d_RG>=0, d_GB<0)
    H[idx] = (4/3) * H0[idx];

    idx = np.logical_and(d_RG<0, d_GB>=0)
    H[idx] = (4/3) * H0[idx] + np.pi

    idx = np.logical_and(d_RG<0, d_GB<0)
    H[idx] = (2/3) * H0[idx] - np.pi
    cc[:,0] = H
    cc[:,1] = C
    cc[:,2] = L
    return reshape(cc, S)

#% See http://w3.uqo.ca/missaoui/Publications.html #60
#% Sarifuddin, M. & Missaoui, R. (2005). A New Perceptually Uniform Color Space with Associated Color Similarity Measure for Content-Based Image and Video Retrieval, ACM SIGIR Workshop on Multimedia Information Retrieval, Salvador, Brazil, August 2005. 
#% http://w3.uqo.ca/missaoui/Publications/TRColorSpace.zip

def hcltolinearrgb(cc, gamma=10):
    '''LINEARRGBTOHCL - Convert linear HCL to RGB color space
    cc = HCLTOLINEARRGB(cc) converts HCL colors to linear RGB.
    CC must be AxBx...x3.
    The HCL color space was defined by Sarifuddin and Missaoui to be
    more perceptually uniform than other color spaces
    cc = HCLTOLINEARRGB(cc, gamma) specifies Missaoui's gamma parameter.'''

    (cc, S) = unshape(cc)
    
    Y0 = 100
    Q = np.exp((1-(3*cc[:,1])/(4*cc[:,2])) * (gamma/Y0) )
    min_ = (4*cc[:,2] - 3*cc[:,1]) / (4*Q - 2)
    max_ = min_ + (3*cc[:,1])/(2*Q)
    
    H = np.mod(cc[:,0], 2*np.pi)
    H[H>=np.pi] = H[H>=np.pi] - 2*np.pi
    
    R = max_
    G = max_
    B = max_
    
    idx = np.logical_and(H>=0, H<=np.pi/3)
    t = np.tan(3*H[idx]/2)
    B[idx] = min_[idx] 
    G[idx] = (R[idx]*t + B[idx]) / (1+t)
    
    idx = np.logical_and(H>pi/3, H<=2*pi/3)
    B[idx] = min_[idx]
    t = tan(3*(H[idx]-pi)/4)
    R[idx] = (G[idx]*(1+t) - B[idx]) / t
    
    idx = H>2*pi/3
    R[idx] = min_[idx]
    t = tan(3*(H[idx]-pi)/4)
    B[idx] = G[idx]*(1+t) - R[idx]*t
    
    idx = np.logical_and(H>=-pi/3, H<0)
    G[idx] = min_[idx]
    t = tan(3*H[idx]/4)
    B[idx] = G[idx]*(1+t) - R[idx]*t
    
    idx = np.logical_and(H>=-2*pi/3, H<-pi/3)
    G[idx] = min_[idx]
    t = tan(3*H[idx]/4)
    R[idx] = (G[idx]*(1+t) - B[idx]) / t
    
    idx = H<-2*pi/3
    R[idx] = min_[idx]
    t = tan(3*(H[idx]+pi)/2)
    G[idx] = (R[idx]*t + B[idx]) / (1+t)
    cc[:,0] = R
    cc[:,1] = G
    cc[:,2] = B
    return reshape(cc, S)

rgbxyz_ = np.transpose(np.array([[.4124, .3576, .1805],
                                 [.2126, .7152, .0722],
                                 [.0193, .1192, .9505]]))

rgblms_ = np.transpose(np.array([[0.313899,   0.639496,   0.046612],
                                 [0.151644,   0.748242,   0.100047],
                                 [0.017725,   0.109473,   0.872939]]))
 

def linearrgbtociexyz(cc):
    '''LINEARRGBTOCIEXYZ - Convert from linear RGB to CIE XYZ
    cc = LINEARRGBTOCIEXYZ(cc) converts from linear RGB to XYZ. CC must
    be AxBx...x3 and have values in the range [0, 1].

    The conversion here is based on http://en.wikipedia.org/wiki/SRGB'''

    (cc, S) = unshape(cc)
    return reshape(np.matmul(cc, rgbxyz_), S)

whitepoints = { 'd50': np.array([0.9642,    1.0000,    0.8251]),
                'd55': np.array([0.9568,    1.0000,    0.9214]),
                'd65': np.array([0.9504,    1.0000,    1.0889]),
                'a':   np.array([1.0985,    1.0000,    0.3558]),
                'c':   np.array([0.9807,    1.0000,    1.1823]) }

def cielabtociexyz(cc, whitepoint='d65', clip=0):
    '''CIELABTOCIEXYZ - Convert CIE L*a*b* colors to CIE XYZ
    cc = CIELABTOCIEXYZ(cc) converts from CIE L*a*b* to CIE XYZ color
    space. CC must be AxBx...x3.

    Note: L*a*b* colors have unusual bounds: L* ranges from 0 to 100;
    a* between -500 and +500; b* between -200 and +200.

    By default, the D65 white point is used. (See WHITEPOINTS.)
    This can be overridden: cc = CIELABTOCIEXYZ(cc, whitepoint). The 
    white point may be given as an XYZ triplet or as one of several standard
    names: d50, d55, d65, a, or c.

    This function can potentially lead to out-of-range XYZ values. By default,
    these are left unclipped. cc = CIELABTOCIEXYZ(..., clip) changes this
    behavior:
      CLIP=0: no clipping (default)
      CLIP=1: hard clipping to [0, 1]
      CLIP=nan: set out of range values to NaN.
      CLIP=2: hard clip at black, proportional clip at white.

    The conversion here is based on 
    http://en.wikipedia.org/wiki/Lab_color_space
    White point information based on
    http://en.wikipedia.org/wiki/Illuminant_D65'''

    def finv(x):
        x = x.copy()
        big = x > 6/29
        sml = np.logical_not(big)
        x = x.copy()
        x[big] = x[big]**3
        x[sml] = 3 * (6/29)**2 * (x[sml] - (4/29))
        return x
    
    if type(whitepoint)==str:
        whitepoint = whitepoints[whitepoint]

    cc, S = unshape(cc)

    L0 = (1/116) * (cc[:,0] + 16)
    Y = whitepoint[1] * finv(L0);
    X = whitepoint[0] * finv(L0 + cc[:,1]/500);
    Z = whitepoint[2] * finv(L0 - cc[:,2]/200);
    cc[:,0] = X
    cc[:,1] = Y
    cc[:,2] = Z
    return reshape(clipxyz(cc, clip), S)

def lmstociexyz(cc):
    '''LMSTOCIEXYZ - Convert from LMS to CIE XYZ'''
    M = np.matmul(np.linalg.inv(rgblms_), rgbxyz_)
    cc, S = unshape(cc)
    return reshape(np.matmul(cc, M), S)

def cielabtocielch(cc):
    '''CIELABTOCIELCH - Convert from CIE L*a*b* to CIE L*C*h colors
    cc = CIELABTOCIELCH(cc) converts from CIE L*a*b* to CIE L*C*h colors
    space. CC must be AxBx...x3.
    In our representation, h ranges from 0 to 2 π.
    May also be used to convert from CIE L*u*v* to CIE L* C*_uv h_uv colors.'''

    cc, S = unshape(cc)
    c = np.sqrt(cc[:,1]**2 + cc[:,2]**2)
    h = np.atan2(cc[:,2], cc[:,1])
    cc[:,1] = c
    cc[:,2] = h
    return reshape(cc, S)

def cieluvtolshuv(cc):
    '''CIELUVTOLSHUV - Convert CIE L*u*v* to L* s_uv h_uv colors
    cc = CIELUVTOLSHUV(cc) converts CIE L*u*v* to L* s_uv h_uv colors.
    CC must be AxBx...x3..
    May also be used to convert CIE L*a*b* to (unofficial)  L* s h colors.'''

    #% Equations taken from http://en.wikipedia.org/wiki/CIELUV and
    #% http://en.wikipedia.org/wiki/Colorfulness#Saturation

    cc = cielabtocielsh(cc)
    cc, S = unshape(cc)
    cc[:,2] /= cc[:,1]
    return reshape(cc, S)

def ciexyztolinearrgb(cc, clip=1):
    '''CIEXYZTOLINEARRGB - Convert from CIE XYZ to linear RGB
    cc = CIEXYZTOLINEARRGB(cc) converts from XYZ to linear RGB. CC must
    be AxBx...x3 and have values in the range [0, 1].
    Not all XYZ values are valid RGB. Those that would fall outside of 
    the range get clipped to [0, 1]. This behavior can be refined:
    cc = CIEXYZTOLINEARRGB(cc, clip) specifies a clip mode:
      CLIP=0: no clipping (produces invalid rgb)
      CLIP=1: hard clipping to [0, 1] (default)
      CLIP=nan: set out of range values to NaN.
      CLIP=2: hard clip at black, proportional clip at white.'''

    cc, S = unshape(cc)

    M = np.linalg.inv(rgbxyz_)
    
    cc = np.matmul(cc, M)

    cliprgb(cc, clip)
    
    return reshape(cc, S)

def ciexyztocielab(cc, whitepoint='d65'):
    '''CIEXYZTOCIELAB - Convert from CIE XYZ to CIE L*a*b* color space
    cc = CIEXYZTOCIELAB(cc) converts from CIE XYZ to CIE L*a*b* color 
    space. CC must be AxBx...x3 and have values in the range [0, 1].

    Note: L*a*b* colors have unusual bounds: L* ranges from 0 to 100;
    a* between -500 and +500; b* between -200 and +200.

    By default, the D65 white point is used. (See WHITEPOINTS.)
    This can be overridden: cc = CIEXYZTOCIELAB(cc, whitepoint). The 
    white point may be given as an XYZ triplet or as one of several standard
    names: d50, d55, d65, a, or c.

    The conversion here is based on 
    http://en.wikipedia.org/wiki/Lab_color_space
    White point information based on
    http://en.wikipedia.org/wiki/Illuminant_D65'''
    
    if type(whitepoint)==str:
        whitepoint = whitepoints[whitepoint]
    
    cc, S = unshape(cc);

    def f(x):
        big = x > (6/29)**3
        sml = np.logical_not(big)
        x = x.copy()
        x[big] = x[big]**(1/3);
        x[sml] = (1/3) * (29/6)**2 * x[sml] + (4/29)
        return x
    
    L0 = f(cc[:,1]/whitepoint[1])
    Lstar = 116 * L0 - 16
    astar = 500 * (f(cc[:,0]/whitepoint[0]) - L0)
    bstar = 200 * (L0 - f(cc[:,2]/whitepoint[2]))
    cc[:,0] = Lstar
    cc[:,1] = astar
    cc[:,2] = bstar
    return reshape(cc, S)

def ciexyztocieluv(cc, whitepoint='d65'):
    '''CIEXYZTOCIELUV - Convert from CIE XYZ to CIE L*u*v* color space
    cc = CIEXYZTOCIELUV(cc) converts from CIE XYZ to CIE L*u*v* color 
    space. CC must be AxBx...x3 and have values in the range [0, 1].

    Note: L*u*v* colors have unusual bounds: L* ranges from 0 to 100;
    u* and v* "typically" between -100 and +100.

    By default, the D65 white point is used. (See WHITEPOINTS.)
    This can be overridden: cc = CIEXYZTOCIELUV(cc, whitepoint). The 
    white point may be given as an XYZ triplet or as one of several standard
    names: d50, d55, d65, a, or c.'''

    #% Equations taken from http://en.wikipedia.org/wiki/CIELUV

    if type(whitepoint)==str:
        whitepoint = whitepoints[whitepoint]

    cc, S = unshape(cc)
    
    nom = whitepoint[0] + 15*whitepoint[1] + 3*whitepoint[2]
    upn = 4*whitepoint[0] / nom
    vpn = 9*whitepoint[1] / nom

    nom = cc[:,0] + 15*cc[:,1] + 3*cc[:,2]
    up =  4*cc[:,0] / nom
    vp =  9*cc[:,1] / nom
    big = (cc[:,1]/whitepoint[1]) > (6/29)^3
    sml = no.logical_not(big)
    cc = cc.copy()
    cc[big, 0] = 116*(cc[big,1]/whitepoint[1])**(1/3.) - 16
    cc[sml, 0] = (29/3)*(cc[sml,1]/whitepoint[1])
    cc[:,1] = 13*cc[:,0] *(up - upn)
    cc[:,2] = 13*cc[:,0] *(vp - vpn)
    return reshape(cc, S)

def ciexyztolms(cc):
    '''CIEXYZTOLMS - Convert from CIE XYZ to  LMS'''

    M = np.matmul(np.linalg.inv(rgbxyz_), rgblms_)
    cc, S = unshape(cc)
    return reshape(np.matmul(cc, M), S)

def colorconvert(cc, fromspace, tospace, whitepoint='d65', clip=0):
    '''COLORCONVERT - Convert between various color spaces
    cc = COLORCONVERT(cc, fromspace, tospace) converts from
    color space FROMSPACE to color space TOSPACE.
    Recognized spaces are: 
      srgb -   The familiar gamma-corrected sRGB
               See http://en.wikipedia.org/wiki/SRGB
      linearrgb - Linear RGB simply sRGB with the gamma correction 
                  taken out
      ciexyz - CIE XYZ
               See http://en.wikipedia.org/wiki/CIE_1931_color_space
      cielab - CIE L*a*b* (L* is lightness a*, b* are chromaticities)
               L* ranges 0..100 a* ranges -500..+500 b* ranges -200..+200
               See http://en.wikipedia.org/wiki/Lab_color_space
      cielch - Cylindrical version of CIE L*a*b* (Lightness, Chroma, Hue)
               L* ranges 0..100, C* ranges 0..200 or so h ranges 0..2*pi.
      cieluv - CIE L*u*v* (L* is lightness u*, v* are chromaticities)
               L* ranges 0..100 u* and v* range approx -100..+100.
               See http://en.wikipedia.org/wiki/CIELUV
      cielchuv - Cylindrical version of CIE L*u*v*
               L* ranges 0..100, C* ranges 0..100 or so h ranges 0..2*pi.
      lshuv -  As cielchuv, but with C* replaced by saturation s.
               See http://en.wikipedia.org/wiki/Colorfulness#Saturation
      lshab -  As cielch, but with C* replaced by saturation s.
      hcl   -  Alternative to cielch proposed by Sarifuddin and Missaoui.
               See http://w3.uqo.ca/missaoui/Publications/TRColorSpace.zip
      lms -    Alternative to ciexyz more closely corresponding to human
               cone vision.

    Additional optional arguments:
      WHITEPOINT: whitepoint for cielab to/from ciexyz conversion. 
            (Either an XYZ triplet or one of 'd50', 'd55', 'd65', 'a', 'c'.)
      CLIP: how to deal white clipped values.
            (default: don't clip 1: hard clip; 2: proportional clip;
             nan: set to nan.)'''

    if fromspace==tospace:
        return cc
    
    # Work our way from source to XYZ
    if fromspace=='cielch':
        cc = cielchtocielab(cc)
        fromspace = 'cielab'
    elif fromspace=='cielchuv':
        cc = cielchtocielab(cc) # yes, really
        fromspace = 'cieluv'
    elif fromspace=='lshab':
        cc = lshuvtocieluv(cc)  # yes, really
        fromspace = 'cielab'
    elif fromspace=='lshuv':
        cc = lshuvtocieluv(cc)
        fromspace = 'cieluv'
    elif fromspace=='srgb':
        cc = srgbtolinearrgb(cc)
        fromspace = 'linearrgb'
    elif fromspace=='hcl':
        cc = hcltolinearrgb(cc)
        fromspace = 'linearrgb'
    
    if fromspace==tospace:
        return cc
    
    if fromspace=='linearrgb':
        cc = linearrgbtociexyz(cc)
        fromspace = 'ciexyz'
    elif fromspace=='cielab':
        cc = cielabtociexyz(cc, whitepoint, clip)
        fromspace = 'ciexyz'
    elif fromspace=='cieluv':
        cc = cieluvtociexyz(cc, whitepoint, clip)
        fromspace = 'ciexyz'  
    elif fromspace=='lms':
        cc = lmstociexyz(cc)
        fromspace = 'ciexyz'
    
    if fromspace != 'ciexyz':
        raise Exception('Unknown source color space')

    posts = { 'srgb': (linearrgbtosrgb, 'linearrgb'),
              'hcl': (linearrgbtohcl, 'linearrgb'),
              'cielch': (cielabtocielch, 'cielab'),
              'cielchuv': (cielabtocielch, 'cieluv'),
              'lshab': (cieluvtolshuv, 'lcielab'),
              'lshuv': (cieluvtolshuv, 'cieluv') }
    if tospace in posts:
        (post, tospace) = posts[tospace]
    else:
        post = None
        
    if tospace=='linearrgb':
        cc = ciexyztolinearrgb(cc, clip)
        fromspace = 'linearrgb'
    elif tospace=='cielab':
        cc = ciexyztocielab(cc, whitepoint)
        fromspace = 'cielab'
    elif tospace=='cieluv':
        cc = ciexyztocieluv(cc, kv.whitepoint)
        fromspace = 'cieluv'
    elif tospace=='lms':
        cc = ciexyztolms(cc)
        fromspace = 'lms'
    
    if fromspace!=tospace:
        raise Exception('Unknown destination color space')
    
    if post is None:
        return cc
    else:
        return post(cc)

def matlabcolor(s):
    rgbspec = { 'r': [1,0,0],
                'g': [0,1,0],
                'b': [0,0,1],
                'w': [1,1,1],
                'c': [0,1,1],
                'm': [1,0,1],
                'y': [1,1,0],
                'k': [0,0,0] }
    if s in rgbspec:
        return np.array(rgbspec[s])
    else:
        raise Exception(f'Not a valid color name {s}')
    
def distinctcolors(n_colors, bg='w', func=None):
    '''DISTINCTCOLORS - Return a set of distinct colors
    cc = DISTINCTCOLORS(n, bg) returns a set of N colors that will
    be maximally distinct to each other and to the given background 
    color(s) BG.
    
    This is unmodified from Tim Holy's "distinguishable_colors", except
    that I changed the name and shortened the help. More info in file!'''

    #%Copyright (c) 2010-2011, Tim Holy
    #%All rights reserved.
    #%
    #%Redistribution and use in source and binary forms, with or without
    #%modification, are permitted provided that the following conditions are
    #%met:
    #%
    #%    * Redistributions of source code must retain the above copyright
    #%      notice, this list of conditions and the following disclaimer.
    #%    * Redistributions in binary form must reproduce the above copyright
    #%      notice, this list of conditions and the following disclaimer in
    #%      the documentation and/or other materials provided with the distribution
    #%
    #%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    #%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    #%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    #%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    #%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    #%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    #%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    #%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    #%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    #%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    #%POSSIBILITY OF SUCH DAMAGE.
    #
    #% DISTINGUISHABLE_COLORS: pick colors that are maximally perceptually distinct
    #%
    #% When plotting a set of lines, you may want to distinguish them by color.
    #% By default, Matlab chooses a small set of colors and cycles among them,
    #% and so if you have more than a few lines there will be confusion about
    #% which line is which. To fix this problem, one would want to be able to
    #% pick a much larger set of distinct colors, where the number of colors
    #% equals or exceeds the number of lines you want to plot. Because our
    #% ability to distinguish among colors has limits, one should choose these
    #% colors to be "maximally perceptually distinguishable."
    #%
    #% This function generates a set of colors which are distinguishable
    #% by reference to the "Lab" color space, which more closely matches
    #% human color perception than RGB. Given an initial large list of possible
    #% colors, it iteratively chooses the entry in the list that is farthest (in
    #% Lab space) from all previously-chosen entries. While this "greedy"
    #% algorithm does not yield a global maximum, it is simple and efficient.
    #% Moreover, the sequence of colors is consistent no matter how many you
    #% request, which facilitates the users' ability to learn the color order
    #% and avoids major changes in the appearance of plots when adding or
    #% removing lines.
    #%
    #% Syntax:
    #%   colors = distinguishable_colors(n_colors)
    #% Specify the number of colors you want as a scalar, n_colors. This will
    #% generate an n_colors-by-3 matrix, each row representing an RGB
    #% color triple. If you don't precisely know how many you will need in
    #% advance, there is no harm (other than execution time) in specifying
    #% slightly more than you think you will need.
    #%
    #%   colors = distinguishable_colors(n_colors,bg)
    #% This syntax allows you to specify the background color, to make sure that
    #% your colors are also distinguishable from the background. Default value
    #% is white. bg may be specified as an RGB triple or as one of the standard
    #% "ColorSpec" strings. You can even specify multiple colors:
    #%     bg = {'w','k'}
    #% or
    #%     bg = [1 1 1; 0 0 0]
    #% will only produce colors that are distinguishable from both white and
    #% black.
    #%
    #%   colors = distinguishable_colors(n_colors,bg,rgb2labfunc)
    #% By default, distinguishable_colors uses the image processing toolbox's
    #% color conversion functions makecform and applycform. Alternatively, you
    #% can supply your own color conversion function.
    #%
    #% Example:
    #%   c = distinguishable_colors(25);
    #%   figure
    #%   image(reshape(c,[1 size(c)]))
    #%
    #% Example using the file exchange's 'colorspace':
    #%   func = @(x) colorspace('RGB->Lab',x);
    #%   c = distinguishable_colors(25,'w',func);
    
    # Copyright 2010-2011 by Timothy E. Holy
    
    # Parse the inputs
    if type(bg)==str:
        bg = matlabcolor(bg)
    bg = np.array(bg)
    if len(bg.shape)==1:
        bg = np.reshape(bg, (1, len(bg)))
      
    # Generate a sizable number of RGB triples. This represents our space of
    # possible choices. By starting in RGB space, we ensure that all of the
    # colors can be generated by the monitor.
    n_grid = 30  # number of grid divisions along each axis in RGB space
    x = np.reshape(np.linspace(0,1,n_grid), (n_grid, 1, 1))
    B = np.repeat(np.repeat(x, n_grid, 1), n_grid, 2)
    x = np.reshape(x, (1, n_grid, 1))
    G = np.repeat(np.repeat(x, n_grid, 0), n_grid, 2)
    x = np.reshape(x, (1, 1, n_grid))
    R = np.repeat(np.repeat(x, n_grid, 0), n_grid, 1)
    rgb = np.hstack((np.reshape(R, (n_grid**3,1)),
                     np.reshape(G, (n_grid**3,1)),
                     np.reshape(B, (n_grid**3,1))))
    if n_colors > rgb.shape[0]/3:
        raise Exception('You cannott readily distinguish that many colors')
      
    # Convert to Lab color space, which more closely represents human
    # perception
    if func is None:
        lab = colorconvert(rgb, 'srgb', 'cielab')
        bglab = colorconvert(bg, 'srgb', 'cielab')
    else:
        lab = func(rgb)
        bglab = func(bg)

    
    # If the user specified multiple background colors, compute distances
    # from the candidate colors to the background colors
    mindist2 = np.inf+np.zeros(rgb.shape[0])
    for i in range(bglab.shape[0]-1):
        dX = lab - bglab[i,:] # displacement all colors from bg
        dist2 = np.sum(dX**2,1)  # square distance
        mindist2 = np.minimum(dist2,mindist2)  # dist2 to closest previously-chosen color
        
    # Iteratively pick the color that maximizes the distance to the nearest
    # already-picked color
    colors = np.zeros((n_colors,3))
    lastlab = bglab[-1,:]   # initialize by making the "previous" color equal to background
    for i in range(n_colors):
        dX = lab - lastlab # displacement of last from all colors on list
        dist2 = np.sum(dX**2, 1)  # square distance
        mindist2 = np.minimum(dist2,mindist2)  # dist2 to closest previously-chosen color
        index = np.argmax(mindist2)  # find the entry farthest from all previously-chosen colors
        colors[i,:] = rgb[index,:]  # save for output
        lastlab = lab[index,:]  # prepare for next iteration

    return colors
