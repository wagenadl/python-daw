import numpy as np

inch = 25.4
mm = 1
deg = np.pi/180
minute = 60
wcs = 'G54'

def X(x):
    '''Generate X parameter. Input is in mm.'''
    return f'X{x/inch:.4f}'

def Y(y):
    '''Generate Y parameter. Input is in mm.'''
    return f'Y{y/inch:.4f}'

def Z(z):
    '''Generate Z parameter. Input is in mm.'''
    return f'Z{z/inch:.4f}'

def A(a):
    '''Generate A parameter. Input is in degrees.'''
    return f'A{a:.3f}'

def C(c):
    '''Generate C parameter. Input is in degrees.'''
    return f'C{c:.3f}'

def F(f):
    '''Generate F parameter. Input is in mm/s.'''
    inch_min = minute*f/inch
    return f'F{inch_min:.3f}'

def preamble(tool=1):
    return f'''
G90 G94 G17
G20
G53 G0 Z0.
T{tool} M6
M1
{wcs}
G43 H{tool}
'''

def end():
    return f'''
M30
'''

def retractandrotate(a=0, c=0):
    return f'''
G53 G0 Z0.
{wcs}
M11
G0 A{a:.2f} C{c:.2f}
M10 
'''

def start(speed=1000):
    '''Start the spindle at given speed (rpm)'''
    return f'S{speed} M3\n'

def stop():
    '''Stop the spindle'''
    return 'M5\nM1\n'

def retractandstop():
    return '''
G53 G0 Z0.
M5
M1
'''

def probez(dz_mm=25):
    dz = abs(dz_mm)
    return f'''G65 P9995 W{wcs[1:]}. A20. H{-dz/25.4:.3f}
    '''

def probez_at(x_mm, y_mm, z_mm, zsafe_mm=100, dz_mm=25):
    dz = abs(dz_mm)
    s = preamble(10)
    s += f'''
G0 {X(x_mm)} {Y(y_mm)}    
G0 {Z(z_mm + zsafe_mm)}
G65 P9832
G65 P9810 {Z(z_mm + dz/2)} {F(20)}
'''
    s += probez(dz_mm)
    return s

def renumber(gcode):
    lines = gcode.split('\n')
    lines = [ f'N{n*10+10} {line}' for n,line in enumerate(lines) ]
    return '\n'.join(lines)
