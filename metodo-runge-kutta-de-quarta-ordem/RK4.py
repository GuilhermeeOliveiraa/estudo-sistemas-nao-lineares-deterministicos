# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:47:36 2019

@author: Gilson
"""

import numpy as np
import pylab as p
import matplotlib as mpt

def f(x,v):
    return v

def g(x,v):
    return -(k/m)*x    
#def RK4(f,x,h):
#    k1 = h*f(x)
#    k2 = h*f(x + k1/2)
#    k3 = h*f(x + k2/2)
#    k4 = h*f(x + k3)
#    x += (k1 + 2*k2 + 2*k3 + k4)/6    
#    return x

def RK4(f,g,x,v,h):
    k1x = h*f(x,v)
    k1v = h*g(x,v)
    k2x = h*f(x + k1x/2, v+ k1v/2)
    k2v = h*g(x + k1x/2, v+ k1v/2)
    k3x = h*f(x + k2x/2, v+ k2v/2)
    k3v = h*g(x + k2x/2, v+ k2v/2)    
    k4x = h*f(x + k3x, v+ k3v)
    k4v = h*g(x + k3x, v+ k3v)
    x += (k1x + 2*k2x + 2*k3x + k4x)/6    
    v += (k1v + 2*k2v + 2*k3v + k4v)/6    
    return np.array([x,v])


k = 1.0
m = 1.0    
t = np.linspace(0,20,20000)
h = t[1] - t[0]   
x = np.zeros_like(t)
v = np.zeros_like(t)

x[0] = 0.01
v[0] = 0.01

for i in range(0,len(t)-1,1):
    x[i+1],v[i+1] = RK4(f,g,x[i],v[i],h)
    
p.figure()
p.plot(t,x,'b')
p.plot(t,v,'r')
p.xlabel('Tempo')
p.ylabel('Velocdade e Espaco')
p.title('Eq Dif Ord com RK4')

p.figure()
p.plot(v,x)    
p.xlabel('Velocidade')
p.ylabel('Posicao')
p.title('Eq Dif Ord com RK4')