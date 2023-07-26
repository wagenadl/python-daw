#!/usr/bin/python3

#%   UNITS v. 0.10, Copyright (C) 2009, 2020 Daniel Wagenaar. 
#%   This software comes with ABSOLUTELY NO WARRANTY. See code for details.
#
#%   This program is free software; you can redistribute it and/or modify
#%   it under the terms of the GNU General Public License as published by
#%   the Free Software Foundation; version 2 of the License.
#%
#%   This program is distributed in the hope that it will be useful,
#%   but WITHOUT ANY WARRANTY; without even the implied warranty of
#%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#%   GNU General Public License for more details.
#%
#%   You should have received a copy of the GNU General Public License
#%   along with this program; if not, write to the Free Software
#%   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

import re
from collections import OrderedDict
import numpy as np

class Units:
    '''UNITS - Class for unit conversion

    Examples: Units('4 lbs').asunit('kg') -> 1.814
              Units('3 V / 200 mA').asunit('Ohm') -> 15.0
              Units('psi').definition() -> '6894.7573 kg m^-1 s^-2'

    The full syntax for unit specification is like this:

    BASEUNIT = m | s | g | A | mol
    PREFIX = m | u | n | p | f | k | M | G | T
    ALTUNIT = meter | meters | second | seconds | sec | secs |
              gram | grams | gm | amp | amps | ampere | amperes | 
              Amp | Ampere | Amperes
    ALTPREFIX = milli | micro | μ | nano | pico | femto | kilo |
                mega | Mega | giga | Giga | tera | Tera
    DERIVEDUNIT = in | inch | Hz | Hertz | hertz | cyc | cycles |
                  V | volt | Volt | volts | Volts |
                  N | newton | Newton | newtons | Newtons |
                  Pa | pascal | bar | atm | torr |
                  J | joule | joules | Joule | Joules |
                  barn | 
                  Ohm | Ohms | ohm | ohms | mho | Mho
    UNIT = (PREFIX | ALTPREFIX)? (BASEUNIT | ALTUNIT | DERIVEDUNIT)
    DIGITS = [0-9]
    INTEGER = ('-' | '+')? DIGIT+
    NUMBER = ('-' | '+')? DIGIT* ('.' DIGIT*)? ('e' ('+' | '-') DIGIT*)?
    POWFRAC = INTEGER ('|' INTEGER)?
    POWERED = UNIT ('^' POWFRAC)?
    FACTOR = POWERED | NUMBER
    MULTI = FACTOR (' ' MULTI)?
    FRACTION = MULTI ('/' MULTI)?

    Thus, the following would be understood:

      'kg m / s^2' - That's a newton
      'J / Hz^1|2' - Joules per root-Hertz

    NOTES:

    - Multiplication is implicit; do not attempt to write '*'.
    - Fractions in exponents are written with '|' rather than '/'. '|' binds
      more tightly than '^'. 
    - Division marked by '/' binds most loosely, e.g,

       'kg / m s' - kilogram per meter per second

    - Syntax checking is not overly rigorous. Some invalid expressions may
      return meaningless values without a reported error.'''
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Prepare unit database
    def mkunitcode():
        units = 'mol A g m s'.split(' ')
        uc = {}
        U = len(units)
        for u in range(U):
            vec = np.zeros(U)
            vec[u] = 1
            uc[units[u]] = vec
        uc[1] = np.zeros(U)
        return uc, units
        
    unitcode, unitvec = mkunitcode()
    def decodeunit(u):
        return Units.unitcode[u].copy()

    def mkprefix():
        prefix = OrderedDict({ 'd': -1, 'c': -2,
                               'm': -3, 'u': -6, 'n': '-9', 'p': -12, 'f': -15,
                               'k': 3, 'M': 6, 'G': 9, 'T': 12 })
        altprefix = [ 'deci=d',
                      'centi=c',
                      'milli=m',
                      'micro=μ=u',
                      'nano=n',
                      'pico=p',
                      'femto=f',
                      'kilo=k',
                      'mega=Mega=M',
                      'giga=Giga=G',
                      'tera=Tera=T' ]
        for ap in altprefix:
            bits = ap.split('=')
            val = bits.pop()
            for b in bits:
                prefix[b] = prefix[val]
        return prefix

    prefix = mkprefix()

    def mkunitmap():
        altunits = [ 'meter=meters=m', 
                     'second=seconds=sec=secs=s', 
                     'gram=grams=gm=g',
                     'lb=lbs=pound=453.59237 g',
                     'amp=amps=ampere=amperes=Amp=Ampere=Amperes=A',
                     'min=minute=60 s',
                     'h=hour=60 min',
                     'day=24 hour',
                     'in=inch=2.54 cm',
                     'l=L=liter=liters=1e-3 m^3',
                     'Hz=Hertz=hertz=cyc=cycles=s^-1',
                     'C=Coulomb=coulomb=Coulombs=coulombs=A s',
                     'N=newton=Newton=newtons=Newtons=kg m s^-2',
                     'lbf=4.4482216 kg m / s^2',
                     'J=joule=joules=Joule=Joules=N m',
                     'W=watt=Watt=watts=Watts=J s^-1',
                     'V=Volt=volt=Volts=volts=W A^-1',
                     'Pa=pascal=Pascal=N m^-2',
                     'bar=1e5 Pa',
                     'atm=101325 Pa',
                     'torr=133.32239 Pa',
                     'psi=6894.7573 kg / m s^2',
                     'Ohm=Ohms=ohm=ohms=V A^-1',
                     'mho=Mho=Ohm^-1',
                     'barn=1e-28 m^2',
                     'M=molar=mol l^-1',
        ]
        unitmap = {}
        for au in altunits:
            bits = au.split('=')
            val = bits.pop()
            for b in bits:
                unitmap[b] = val
        return unitmap

    unitmap = mkunitmap()

    def fracdecode(s):
        #return [mul,code]
        idx = s.find('/')
        if idx<0:
            numer = s
            denom = ''
        else:
            numer = s[:idx]
            denom = s[idx+1:].replace('/', ' ')

        multis = [ numer, denom ]
        mul = []
        code = []
        for q in range(2):
            mul.append(1)
            code.append(Units.decodeunit(1))
            factors = multis[q].split(' ')
            for fac in factors:
                mu, co = Units.factordecode(fac)
                mul[q] *= mu
                code[q] += co
        mul = mul[0]/mul[1]
        code = code[0] - code[1]
        return mul, code

    numre = re.compile('^[-0-9+.]')
    def factordecode(fac):
        if Units.numre.search(fac):
            # It's a number
            return float(fac), Units.decodeunit(1)

        idx = fac.find('^')
        if idx>=0:
            base = fac[:idx]
            powfrac = fac[idx+1:]
            if powfrac.find('^')>0:
                raise ValueError('Double exponentiation')
            idx = powfrac.find('|')
            if idx>=0:
                pw = float(powfrac[:idx]) / float(powfrac[idx+1:])
            else:
                pw = float(powfrac)
        else:
            base=fac
            pw = 1

        # Let's decode the UNIT
        if base=='':
            return 1., Units.decodeunit(1)
        elif base in Units.unitcode:
            # It's a base unit without a prefix
            mu = 1
            co = Units.decodeunit(base)*pw
            return mu, co
        elif base in Units.unitmap:
            mu, co = Units.fracdecode(Units.unitmap[base])
            mu = mu**pw
            co = co*pw
            return mu, co
        else:
            # So we must have a prefix
            for pf in reversed(Units.prefix):
                if base.startswith(pf):
                    L = len(pf)
                    mu, co = Units.fracdecode(base[L:])
                    mu *= 10**Units.prefix[pf]
                    mu = mu**pw
                    co = co*pw
                    return mu, co
        raise ValueError(f'I do not know any unit named “{fac}”')

    def __init__(self, value, unit=None):
        '''Constructor - Store a value with associated units
        UNITS(value, units), where VALUE is a number and UNITS a string,
        stores the given quantity. For instance, UNITS(9.81, 'm / s^2').
        For convenience, UNITS('9.81 m/s^2') also works.'''
        if unit is None:
            unit = value
            value = 1
        self.value = value
        self.oldmul, self.oldcode = Units.fracdecode(unit)

    def definition(self):
        '''DEFINITION - Definition of stored value in SI units
        DEFINITION() returns the definition of the stored unit in terms
        of SI base units.'''
        val = self.value * self.oldmul
        ss = []
        for u in range(len(Units.unitvec)):
            co = self.oldcode[u]
            un = Units.unitvec[u]
            if un=='g':
                val /= 1000**co
                un = 'kg'
            if co==0:
                pass
            elif co==1:
                ss.append(un)
            elif co==int(self.oldcode[u]):
                ss.append(f'{un}^{int(co)}')
            else:
                ss.append(f'{un}^{self.oldcode[u]}')
        ss.insert(0, f'{val}')
        return ' '.join(ss)
                
    def asunits(self, newunit, warn=False):
        '''ASUNITS - Convert to different units
        ASUNITS(newunits) converts the stored quantity to the given new
        units. An exception is raised if the units are incompatible.
        Optional argument WARN, if True, turns that into a warning.
        See the class documentation for unit syntax and note that
        addition or subtraction is not supported.'''
        newmul, newcode = Units.fracdecode(newunit)
        if np.any(self.oldcode != newcode):
            if warn:
                print(f'WARNING: Units {newunit} do not match {unit}')
            else:
                raise ValueError(f'Units {newunit} do not match {unit}')
        return self.value * self.oldmul / newmul
        
def convert(newunit, value, unit=None, warn=False):
    '''CONVERT - Unit conversion
    newval = CONVERT(newunit, value, unit) converts a VALUE expressed in 
    some UNIT to NEWUNIT using the UNITS class. It is simply a convenience
    function for UNITS(value, unit).asunits(newunit).
    For instance, CONVERT('cm', 2, 'inch') returns 5.08.
    As a convenience, CONVERT('cm', '2 inch') also works.'''

    return Units(value, unit).asunit(newunit, warn)

def candela2lumen(x_cd, twothetahalf_deg):
    '''CANDELA2LUMEN - Convert candelas to lumens for a given beam width
    lm = CANDELA2LUMEN(x_cd, twothetahalf_deg) calculates the full luminous
    flux (in lumens) in a beam with peak luminous intensity X_CD (in candelas)
    and full-width-at-half-max TWOTHETAHALF_DEG (in degrees).
    The answer is approximate as it depends on the precise beam profile.'''
    cdlm = 4182
    lm = x_cd * twothetahalf_deg**2 / cdlm
    return lm

def lumen2candela(lm, twothetahalf_deg):
    '''LUMEN2CANDELA - Convert lumens to candelas for a given beam width
    cd_ = LUMEN2CANDELA(x_lm, twothetahalf_deg) calculates the peak luminous
    intensite (in candelas) for a beam with  full luminous X_LM (in lumens)
    and full-width-at-half-max TWOTHETAHALF_DEG (in degrees).
    The answer is approximate as it depends on the precise beam profile.'''
    cdlm = 4182
    x_cd = lm / (twothetahalf_deg**2 / cdlm)

_vl1924e = np.array([
    [360,  0.000003917000],
    [365,  0.000006965000],
    [370,  0.000012390000],
    [375,  0.000022020000],
    [380,  0.000039000000],
    [385,  0.000064000000],
    [390,  0.000120000000],
    [395,  0.000217000000],
    [400,  0.000396000000],
    [405,  0.000640000000],
    [410,  0.001210000000],
    [415,  0.002180000000],
    [420,  0.004000000000],
    [425,  0.007300000000],
    [430,  0.011600000000],
    [435,  0.016840000000],
    [440,  0.023000000000],
    [445,  0.029800000000],
    [450,  0.038000000000],
    [455,  0.048000000000],
    [460,  0.060000000000],
    [465,  0.073900000000],
    [470,  0.090980000000],
    [475,  0.112600000000],
    [480,  0.139020000000],
    [485,  0.169300000000],
    [490,  0.208020000000],
    [495,  0.258600000000],
    [500,  0.323000000000],
    [505,  0.407300000000],
    [510,  0.503000000000],
    [515,  0.608200000000],
    [520,  0.710000000000],
    [525,  0.793200000000],
    [530,  0.862000000000],
    [535,  0.914850100000],
    [540,  0.954000000000],
    [545,  0.980300000000],
    [550,  0.994950100000],
    [555,  1.000000000000],
    [560,  0.995000000000],
    [565,  0.978600000000],
    [570,  0.952000000000],
    [575,  0.915400000000],
    [580,  0.870000000000],
    [585,  0.816300000000],
    [590,  0.757000000000],
    [595,  0.694900000000],
    [600,  0.631000000000],
    [605,  0.566800000000],
    [610,  0.503000000000],
    [615,  0.441200000000],
    [620,  0.381000000000],
    [625,  0.321000000000],
    [630,  0.265000000000],
    [635,  0.217000000000],
    [640,  0.175000000000],
    [645,  0.138200000000],
    [650,  0.107000000000],
    [655,  0.081600000000],
    [660,  0.061000000000],
    [665,  0.044580000000],
    [670,  0.032000000000],
    [675,  0.023200000000],
    [680,  0.017000000000],
    [685,  0.011920000000],
    [690,  0.008210000000],
    [695,  0.005723000000],
    [700,  0.004102000000],
    [705,  0.002929000000],
    [710,  0.002091000000],
    [715,  0.001484000000],
    [720,  0.001047000000],
    [725,  0.000740000000],
    [730,  0.000520000000],
    [735,  0.000361100000],
    [740,  0.000249200000],
    [745,  0.000171900000],
    [750,  0.000120000000],
    [755,  0.000084800000],
    [760,  0.000060000000],
    [765,  0.000042400000],
    [770,  0.000030000000],
    [775,  0.000021200000],
    [780,  0.000014990000],
    [785,  0.000010600000],
    [790,  0.000007465700],
    [795,  0.000005257800],
    [800,  0.000003702900],
    [805,  0.000002607800],
    [810,  0.000001836600],
    [815,  0.000001293400],
    [820,  0.000000910930],
    [825,  0.000000641530],
    [830,  0.000000451810],
])

def lumen2watt(lum_lm, wavelen_nm):
    '''LUMEN2WATT - Convert lumens to Watts at a given wavelength
    P_W = LUMEN2WATT(lum_lm, wavelen_nm) converts lumens to Watts at a
    given wavelength.'''
    rel = np.interp(wavelen_nm, _vl1924e[:,0], _vl1924e[:,1])
    bas = 683
    P_W = lum_lm / (bas*rel)
    return P_W

def candela2watt(br_cd, diam_deg, wavelength_nm=555):
    '''CANDELA2WATT - Convert brightness of light in candelas to power in watts
    P = CANDELA2WATT(br_cd, diam_deg, wavelength_nm) converts the 
    brightness of a lightsource BR_CD (measured in candelas) to the power P
    contained in the beam (in watts), given the angular divergence of the beam
    DIAM_DEG (in degrees) and the wavelength of the beam WAVELENGTH_NM
    (in nanometers; default: 555 nm).
    The diameter of the beam is nominally 2*theta_(1/2); i.e. the full angle
    at which light intensity has dropped to 50% from central peak.
    This function assumes that the light intensity drops of as
    exp(-1/2 theta^4/THETA0^4).'''

    # We assume that BR_CD is measured at 0 degrees, and that the intensity
    # drops of with exp(-alpha theta^4), where alpha is found by equating
    # exp(-alpha (radius_rad)^4) = 0.5, so:

    pwr=4
    radius_rad = .5*diam_deg * np.pi/180
    alpha = -np.log(.5) / radius_rad**pwr

    da_deg=.1
    meas_ang_deg = np.arange(da_deg/2, 90, da_deg)

    da_rad = da_deg * np.pi/180
    meas_ang_rad = meas_ang_deg * np.pi/180
    ringlets_st = da_rad*2*np.pi*np.sin(meas_ang_rad)

    br_local_cd = br_cd*np.exp(-alpha*meas_ang_rad**pwr)

    br_intg_lumen = np.sum(ringlets_st * br_local_cd)
    eff = np.interp(wavelength_nm, _vl1924e[:,0], _vl1924e[:,1])
    P_W = br_intg_lumen / eff / 683
    return P_W

if __name__=='__main__':
    u = Units('V')
    print(u.definition())
    u = Units('psi')
    print(u.definition())
    u = Units('4 lbs')
    print(u.asunit('kg'))
    
