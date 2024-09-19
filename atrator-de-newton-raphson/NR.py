#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:34:11 2019

@author: joaoguilherme
"""

from __future__ import division  
import numpy as np


def f(x,a):
    return input("Função": )
    #return np.exp(x) - x - 2 

def f_d(x):
    return 2*x - 3    
    #return np.exp(x) - 1
    
a=5.0#-2
 
TOL = 1e-10 
N = 20

def NR(f, f_d, a, TOL, N):    
    i = 1    
    fa = f(a)
    if fa == 0.0:
        x = a
        return np.array([x, i])        
    while (i<=N):    
        x = a - f(a)/f_d(a)
        if np.fabs(f(x)) < TOL: 
            return np.array([x, i])
        i += 1     
        a = x        
        
xr = NR(f, f_d, a, TOL, N)

print xr