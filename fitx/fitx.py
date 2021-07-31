#!/usr/bin/python

import math

def circlefrom3(x1, y1, x2, y2, x3, y3):
    '''CIRCLEFROM3 - Find a circle given three points
    xc, yc, r = CIRCLEFROM3(x1, y1, x2, y2, x3, y3) finds the center
    and radius of a circle through the three given points.
    Translated from https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/'''
    
    x12 = x1 - x2
    x13 = x1 - x3
 
    y12 = y1 - y2
    y13 = y1 - y3
 
    y31 = y3 - y1
    y21 = y2 - y1
 
    x31 = x3 - x1
    x21 = x2 - x1
 
    sx13 = x1**2 - x3**2
    sy13 = y1**2 - y3**2
    sx21 = x2**2 - x1**2
    sy21 = y2**2 - y1**2
 
    f1 = sx13*x12 + sy13*x12 + sx21*x13 + sy21*x13
    f2 = 2 * (y31*x12 - y21*x13)
    f = f1/f2
    g1 = sx13*y12 + sy13*y12 + sx21*y13 + sy21*y13
    g2 = 2 * (x31*y12 - x21*y13)
    g = g1/g2
 
    c = -x1**2 - y1**2 - 2*g*x1 - 2*f*y1
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c
    r = sqr_of_r**.5
    return h, k, r
