import numpy as np

#deg = np.pi/180
wcs = 'G54'

def setwcs(w):
    global wcs
    wcs = w

def mm_to_machine(x):
    return x / 25.4 # convert mm to inch

def mm_s_to_machine(v):
    return 60*v/25.4 # convert mm/s to inch/min

def deg_to_machine(theta):
    return theta # leave degrees as degrees

def rpm_to_machine(rpm):
    return rpm # leave rpm as rpm

def X(x):
    '''Generate X parameter. Input is in mm.'''
    return f'X{mm_to_machine(x):.4f}'

def Y(y):
    '''Generate Y parameter. Input is in mm.'''
    return f'Y{mm_to_machine(y):.4f}'

def Z(z):
    '''Generate Z parameter. Input is in mm.'''
    return f'Z{mm_to_machine(z):.4f}'

def A(a):
    '''Generate A parameter. Input is in degrees.'''
    return f'A{deg_to_machine(a):.3f}'

def C(c):
    '''Generate C parameter. Input is in degrees.'''
    return f'C{deg_to_machine(c):.3f}'

def F(f):
    '''Generate F parameter. Input is in mm/s.'''
    return f'F{mm_s_to_machine(f):.3f}'

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
G0 A{deg_to_machine(a):.2f} C{deg_to_machine(c):.2f}
M10 
'''

def start(speed=1000):
    '''Start the spindle at given speed (rpm)'''
    return f'S{rpm_to_machine(speed)} M3\n'

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
    return f'''G65 P9995 W{wcs[1:]}. A20. H{mm_to_machine(-dz):.3f}
    '''

def probez_at(x_mm, y_mm, z_mm, zsafe_mm=100, dz_mm=25):
    dz = abs(dz_mm)
    s = preamble(10)
    s += f'''
G103 P1

    
G0 {X(x_mm)} {Y(y_mm)}    
G0 {Z(z_mm + zsafe_mm)}
G65 P9832
G65 P9810 {Z(z_mm + dz/2)} {F(20)}
G103

    
'''
    s += probez(dz_mm)
    return s

def renumber(gcode):
    lines = gcode.split('\n')
    lines = [ f'N{n*10+10} {line}' for n,line in enumerate(lines) ]
    return '\n'.join(lines)
