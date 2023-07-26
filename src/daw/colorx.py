# color.py - collection of "COLORX" octave functions converted to python

# colorconvert - part of my "color" library
# dhsv, hotpow, hotpow2 - see pyqplot's luts
# lchdemo, luvdemo - not critical

# Probably, all of the color maps should be moved into qp.luts

from . import basicx
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

def clipxyz(cc, clip):
    '''CLIPXYZ - Clip XYZ colors (helper for CIELUVTOCIEXYZ and friends)'''

    if clip==0:
        pass
    elif clip==1:
        cc = np.maximum(cc, 0)
        cc = np.minimum(cc, 1)
    elif np.isnan(clip):
        nn = np.any(np.logical_or(cc<0, cc>1), 1)
        cc[nn, :] = np.nan
    elif clip==2:
        cc = np.maximum(cc, 0)
        mx = np.max(cc, 1)
        mx = np.maximum(mx, 1)
        cc = cc / np.reshape(mx, (len(mx), 1))
    return cc

def cliprgb(cc, clip):
    '''CLIPRGB - Clip linear RGB colors (helper for ciexyztolinearrgb)
    Operates in place.'''

    if clip==0:
        pass
    elif clip==1:
        cc = np.maximum(cc, 0)
        cc = np.minimum(cc, 1)
    elif np.isnan(clip):
        nn = np.any(np.logical_or(cc<0, cc>1), 1)
        cc[nn, :] = np.nan
    elif clip==2:
        cc = np.maximum(cc, 0)
        mx = np.max(cc, 1)
        mx = np.maximum(mx, 1)
        cc = cc / np.reshape(mx, (mx.shape[0], 1))
    else:
        raise Exception(f'Illegal clip specification: {clip}')
    return cc


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

    isn = np.isnan(cc)
    cc[isn] = 0
    big = cc>.0031308
    sml = np.logical_not(big)
    cc = cc.copy()
    cc[big] = 1.055*cc[big]**(1/2.4) - .055
    cc[sml] = cc[sml]*12.92
    cc[isn] = np.nan
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

    H0 = np.arctan(d_GB/d_RG)

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
    
    idx = np.logical_and(H>np.pi/3, H<=2*np.pi/3)
    B[idx] = min_[idx]
    t = np.tan(3*(H[idx]-np.pi)/4)
    R[idx] = (G[idx]*(1+t) - B[idx]) / t
    
    idx = H>2*np.pi/3
    R[idx] = min_[idx]
    t = np.tan(3*(H[idx]-np.pi)/4)
    B[idx] = G[idx]*(1+t) - R[idx]*t
    
    idx = np.logical_and(H>=-np.pi/3, H<0)
    G[idx] = min_[idx]
    t = np.tan(3*H[idx]/4)
    B[idx] = G[idx]*(1+t) - R[idx]*t
    
    idx = np.logical_and(H>=-2*np.pi/3, H<-np.pi/3)
    G[idx] = min_[idx]
    t = np.tan(3*H[idx]/4)
    R[idx] = (G[idx]*(1+t) - B[idx]) / t
    
    idx = H<-2*np.pi/3
    R[idx] = min_[idx]
    t = np.tan(3*(H[idx]+np.pi)/2)
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
    cc1 = np.matmul(cc, rgbxyz_)
    return reshape(cc1, S)

whitepoints = { 'd50': np.array([0.9642,    1.0000,    0.8251]),
                 'd55': np.array([0.9568,    1.0000,    0.9214]),
                 'd65': np.array([0.9504,    1.0000,    1.0889]),
                 'a':   np.array([1.0985,    1.0000,    0.3558]),
                 'c':   np.array([0.9807,    1.0000,    1.1823]) }
def whitepoint(wh='d65'):
    '''WHITEPOINT - Return XYZ values of standard white points
    xyz = WHITEPOINT(s) returns XYZ values of one of several standard
    white points: d50, d55, d65, a, c'''
    # Source: Matlab R2012b's whitepoint function
    return whitepoints[wh]

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
    h = np.arctan2(cc[:,2], cc[:,1])
    cc[:,1] = c
    cc[:,2] = h
    return reshape(cc, S)

def cieluvtociexyz(cc, whitepoint='d65', clip=0):
    '''CIELUVTOCIEXYZ - Convert from CIE L*u*v to CIE XYZ color space
    cc = CIELUVTOCIEXYZ(cc) converts from CIE L*u*v to CIE XYZ color space.
    CC must be AxBx...x3 and have values in the range [0, 1].
    
    Note: L*u*v* colors have unusual bounds: L* ranges from 0 to 100;
    u* and b* "typicallay" between -100 and +100.
    
    By default, the D65 white point is used. (See WHITEPOINTS.)
    This can be overridden: cc = CIEXYZTOCIELUV(cc, whitepoint). The 
    white point may be given as an XYZ triplet or as one of several standard
    names: d50, d55, d65, a, or c.
    
    This function can potentially lead to out-of-range XYZ values. By default,
    these are left unclipped. cc = CIELUVTOCIEXYZ(..., clip) changes this
    behavior:
      CLIP=0: no clipping (default)
      CLIP=1: hard clipping to [0, 1]
      CLIP=nan: set out of range values to NaN.
      CLIP=2: hard clip at black, proportional clip at white.'''

    # Equations taken from http://en.wikipedia.org/wiki/CIELUV

    if type(whitepoint)==str:
        whitepoint = whitepoints[whitepoint]

    cc, S = unshape(cc)

    nom = whitepoint[0] + 15*whitepoint[1] + 3*whitepoint[2]
    upn = 4*whitepoint[0] / nom
    vpn = 9*whitepoint[1] / nom
    up = cc[:,1] / (13*cc[:,0]+1e-9) + upn
    vp = cc[:,2] / (13*cc[:,0]+1e-9) + vpn
    big = cc[:,0] > 8
    nbig = np.logical_not(big)
    cc[big, 1] = whitepoint[1] * ((cc[big, 0]+16)/116)**3
    cc[nbig, 1] = whitepoint[1] * cc[nbig, 0] * (3/29)**3
    cc[:,0] = cc[:,1] * (9*up)/(4*vp)
    cc[:,2] = cc[:,1] * (12 - 3*up - 20*vp) / (4*vp)

    cc = clipxyz(cc, clip)
    cc = reshape(cc, S)
    return cc
    

def cieluvtolshuv(cc):
    '''CIELUVTOLSHUV - Convert CIE L*u*v* to L* s_uv h_uv colors
    cc = CIELUVTOLSHUV(cc) converts CIE L*u*v* to L* s_uv h_uv colors.
    CC must be AxBx...x3..
    May also be used to convert CIE L*a*b* to (unofficial)  L* s h colors.'''

    #% Equations taken from http://en.wikipedia.org/wiki/CIELUV and
    #% http://en.wikipedia.org/wiki/Colorfulness#Saturation

    cc = cielabtocielch(cc)
    cc, S = unshape(cc)
    cc[:,1] /= cc[:,0]
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

    cc = cliprgb(cc, clip)
    
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
    big = (cc[:,1]/whitepoint[1]) > (6/29)**3
    sml = np.logical_not(big)
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
        cc = ciexyztocieluv(cc, whitepoint)
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

def clip01(xx):
    '''CLIP01 - Clip to range [0, 1]'''
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

def alphablend(base, above, coloraxis=-1):
    '''ALPHABLEND - Perform alpha-blending
    out = ALPHABLEND(base, over) alpha-blends the image OVER on top of
    the image BASE. BASE and OVER must have the same shape. It is assumed
    that the final axis of BASE and OVER is color and that the last
    color channel represents alpha. However, if the last axis of BASE
    has length 1 or 3, an implicit all-one alpha channel is assumed.
    The output has the shape of ABOVE.
    Inputs must be in the range [0, 1].'''
    base, Sbase = basicx.semiflatten(base, coloraxis)
    above, Sabove =  basicx.semiflatten(above, coloraxis)
    if base.shape[-1]==1 or base.shape[-1]==3:
        # gray or rgb: add alpha channel
        base = np.hstack((base, np.ones((base.shape[0], 1))))
    alphbase = base[:,-1]
    imgbase = base[:,:-1]
    alphabove = above[:,-1]
    imgabove = above[:,:-1]
    alphout = alphabove + (1-alphabove)*alphbase
    out = np.zeros(base.shape)
    for k in range(out.shape[-1]-1):
        out[:,k] = (alphabove*imgabove[:,k]
                    + (1-alphabove)*alphbase*imgbase[:,k]) / (alphout+1e-20)
    out[:,-1] = alphout
    out[alphout==0, :] = 0
    return  basicx.semiunflatten(out, Sabove)

def applyalpha(rgba, coloraxis=-1):
    '''ALPHAALPHA - Applies alpha channel to RGBA image
    rgb = APPLYALPHA(rgba) applies the alpha channel of an RGBA image,
    resulting in an RGB image. 
    Normally, the final axis of RGBA is assumed to be the color axis, but
    this can be overridden. ALPHA must be in the range [0, 1].'''
    rgba, Srgba = basicx.semiflatten(rgba, coloraxis)
    result = rgba[:,:-1] * rgba[:,-1:]
    return basicx.semiunflatten(result, Srgba)


def _adjusthue(ll, l0, dl, sc):
    dL = 1 - sc*(np.exp(-((ll-l0)/dl)**2)
                 +np.exp(-((ll-360-l0)/dl)**2)
                 +np.exp(-((ll+360-l0)/dl)**2))
    ll = np.cumsum(dL)
    ll = 360*ll/ll[-1]
    idx = np.argmax(ll>=l0)
    ll = ll - ll[idx]
    ll = ll + l0
    ll = np.mod(ll, 360)
    return ll


def _interp1(x, y, xi):
    y, S = basicx.semiflatten(y, 0)
    y = y.T
    N,C = y.shape
    K, = xi.shape
    y1 = np.zeros((K,C))
    for c in range(C):
        y1[:,c] = np.interp(xi, x, y[:,c])
    return basicx.semiunflatten(y1.T, S)


def dhsv(N=64):
    '''DHSV - Alternative HSV colormap with better perceptual uniformity
    cc = DHSV(N) returns a more perceptually uniform cyclic colormap
    with N entries. (N defaults to 64).'''

    on = np.ones(N)
    of = np.zeros(N)
    up = .5*(.5-.5*np.cos(np.arange(.5,N)/N * np.pi)) + .5*(np.arange(.5,N)/N)
    dn = np.flip(up)
    rgb0 = np.concatenate([np.stack([on,up,of],1),
                           np.stack([dn,on,of],1),
                           np.stack([of,on,up],1),
                           np.stack([of,dn,on],1),
                           np.stack([up,of,on],1),
                           np.stack([on,of,dn],1)], 0)
    L,C = rgb0.shape

    l0=np.arange(L)*360/L
    ll = l0
    ll = _adjusthue(ll, 320, 30, -0.8)
    rgb0 = _interp1(l0, rgb0, ll)
    ll = _adjusthue(ll, 240, 60, 0.45)
    rgb0 = _interp1(l0, rgb0, ll)
    ll = _adjusthue(ll, 245, 25, 0.35)
    rgb0 = _interp1(l0, rgb0, ll)
    ll = np.mod(_adjusthue(np.mod(ll-180, 360), 180, 30, 0.65), 360)
    rgb0 = _interp1(l0, rgb0, ll)

    rgb0 = _interp1(l0, rgb0, np.arange(.5,N)*360/N)

    cc = colorconvert(rgb0, 'linearrgb', 'srgb', clip=1)
    return cc


def _vonmises(xx, k):
    return np.exp(k*np.cos(xx))/np.exp(k)

def _expo(xx, x0, r):
    return _vonmises(xx - x0*np.pi/180, 1/(r*np.pi/180))

def _shift(xx, x0, a, r):
    yy = _vonmises(xx - x0*np.pi/180, 1/(r*np.pi/180))
    dydx = np.abs(np.max(np.diff(yy, axis=0) / np.diff(xx, axis=0)))
    return yy*a/dydx

def _cexp(xx, x0, a, r):
    yy = a*np.exp(-.5*(xx-x0)**2/r**2)
    return yy


def lut2d(H=360, C=100):
    '''LUT2D - Perceptually uniform 2d LUT for representing coherences
    lut = LUT2D(H, C) returns a HxCx3 color table for representing
    complex numbers |z| <= 1. The H axis (by default: 360 values) represents
    the angle of the number, the C axis (by default: 100 values) represents
    the radius. The radius is represented as chroma (saturation), whereas
    the angle is represented as hue.
    The map is acceptably uniform, though problems persist in the red portion
    of the color space.'''
    def vonmises(xx, k):
        return np.exp(k*np.cos(xx))/np.exp(k)
    def expo(xx, r):
        return vonmises(xx, 1/r)
    
    hue = np.arange(H).reshape(H,1)*np.pi/180
    chroma = np.arange(C).reshape(1,C)/C
    lightness = np.arange(C).reshape(1,C)*0 + 65 \
        + 10*chroma*expo(hue-1.3,.25) \
        - 10*chroma*expo(hue-4.5,.25)
    chroma = chroma ** (1. - .3*expo(hue-4.4,.7)
                        - .4*expo(hue-1.3,.7)
                        + .2*expo(hue-5.5,.5)
                        + .2*expo(hue-.2,.7)) * 1.5 \
                        * (1 - .3*expo(hue - 3.3, .7) -.2*expo(hue-5.5,.5))
    hue = hue + .6*expo(hue-3.7, .7)
    hue = hue - .6*expo(hue-5.3, .9)
    hue = hue + .6*expo(hue-.5, .8)
    chroma = chroma * (1+.9*expo(hue - 4.45, .35))
    chroma = chroma ** (1-.2*expo(hue - 4.45, .35))
    hue = hue - .2*expo(hue-5.3, .4)
    hue = hue + .3*expo(hue-3.5, .7)
    chroma = chroma ** (1 - .35*expo(hue - 3.3,.7))
    chroma = chroma * (1 - .15*expo(hue-5.4,.5))
    chroma = chroma ** (1 - .1*expo(hue-5.4,.5))
    hue = hue + .15*expo(hue-3.6, .5)
    lch = np.stack((lightness+0*hue, chroma+0*hue, hue+0*chroma), 2)
    
    rgb = colorconvert(lch, 'lshuv', 'srgb', clip=1)
    return rgb


def _z2rgb(zz):
    hh = np.angle(zz)
    hue = hh
    cc = np.abs(zz)
    chroma = cc**1.9*1.6
    lightness = 99 - 40*cc**1.5
    lightness += 20*cc**1.5*np.cos(hh-80*np.pi/180)

    hue = hue - .6*_expo(hh, 250, 30)*(1-cc)
    hue = hue + .6*_expo(hh, 120, 30)*(1-cc)
    chroma = chroma**(1 - .15*_expo(hh, 80, 30))

    lch = np.stack((lightness, chroma, hue), 2)
    rgb = colorconvert(lch, 'lshuv', 'srgb', clip=1)
    A,B,C = rgb.shape
    z0 = np.argmin(np.abs(zz.reshape(A*B)))
    rgb0 = rgb.reshape(A*B,C)[z0] # rgb.max((0,1), keepdims=1)
    rgb = rgb + .99 - rgb0
    rgb[rgb>1]=1
    rgb[rgb<0]=0
    return rgb


def lut2dlight(H=360, C=100):
    xx = np.arange(H).reshape(H,1)*np.pi*2/H
    yy = np.arange(C)/C
    zz = yy*np.exp(1j*xx)
    return _z2rgb(zz)
    

class Lut2D:
    def __init__(self, style=2, H=360, C=100, origin=1):
        '''Lut2D - Construct a 2D LUT for complex data
        lut = Lut2D(style, H, C)
        STYLE = 1: colormap with gray for |z| = 0
        STYLE = 2: colormap with white for |z| = 0
        In the future, STYLE = 0 may be implemented with black for |z| = 0.
        The LUT has H (360) values in the angular direction and C (100)
        in the radial direction.
        Optional ORIGIN rotates the LUT in one of two standard ways:
        ORIGIN = 1: phase 0 is purple, negative phase to yellow
        ORIGIN = 2: phase 0 is yellow, negative phase to red
        Both colormaps have been constructed to be as close to perceptually
        uniform as I could get them on my monitor. The result is still not
        perfect. In particular, there is a shade of pale blue that looks 
        brighter than its |z| value would warrant. See ST p. 1025.
        '''
        if style==1:
            self.data = lut2d(H, C)
        elif style==2:
            self.data = lut2dlight(H, C)
        else:
            raise ValueError("Unsupported style code")
        if origin==1:
            self.data = np.flip(self.data, axis=0)
            self.data = np.roll(self.data, -int(H*.15), axis=0)
        elif origin==2:
            self.data = np.roll(self.data, -int(H*.2), axis=0)

        
    def lut(self):
        '''LUT - Return the lookup table
        x = lut.LUT() returns the lookup table as an HxCx3 array of
        RGB values in [0,1].'''
        return self.data

    def lookup(self, z, gamma=1):
        '''LOOKUP - Look up a complex value in the table
        rgb = lut.LOOKUP(z), where z is a complex number with |z| ≤ 1,
        returns an RGB triplet.
        If Z is an array with shape AxBx...xC, the result is an array
        with shape AxBx...xCx3.
        Optional argument GAMMA applies gamma correction to |z|, resulting
        in lighter and less saturated colors if gamma>1.'''
        isarray = type(z)==np.ndarray
        if gamma != 1:
            z = z * (np.abs(z)+1e-99)**(gamma-1)
        H,C,_ = self.data.shape
        pha = (H*np.angle(z)/(2*np.pi)).astype(int) % H
        rad = (C*np.abs(z)).astype(int)
        if isarray:
            rad[rad>=C] = C - 1
        else:
            rad = min(rad, C-1)
        rgb = self.data[pha, rad, :]
        if isarray:
            return rgb
        else:
            return list(rgb.flatten())
        
        
