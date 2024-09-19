#!/usr/bin/env python2---------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:50:28 2019
    
@author: joaoguilherme
"""

import math as m
import numpy as np
from matplotlib import pylab as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Parâmetros do Sistema---------------------------------------------------------

t = np.linspace(0,200,20000)
h = t[1] - t[0]

R1 = 1.298
R2 = 3.440
R4 = 0.193
I_M = 22.5*10**-6
alpha_f = 11.60
alpha_r = 11.57

R1s = 1.298*1.01
R2s = 3.440*0.99
R4s = 0.193*1.01
I_Ms = 22.5*10**-6
alpha_fs = 11.60
alpha_rs = 11.57

c11 = 0.0
c22 = 4.0 

V1_0 = 0.01
V2_0 = 0.01
I_0 = 0.01
V1s_0 = 0.01
V2s_0 = 0.01
Is_0 = 0.01

#Funções do Sistema de Bienfang-Gauthier---------------------------------------

def g(v):
    return v/R2 + I_M*(m.exp(alpha_r*v) - m.exp(-alpha_r*v))

def dV1_dt(V1,V2,I):
    return V1/R1 - g(V1-V2)

def dV2_dt(V1,V2,I):
    return g(V1-V2) - I  

def dI_dt(V1,V2,I):
    return V2 - R4*I

def gs(v):
    return v/R2s + I_Ms*(m.exp(alpha_rs*v) - m.exp(-alpha_rs*v))

def dV1s_dt(V1,V2,I,V1m):
    return V1/R1s - gs(V1-V2) + c11*(V1m - V1)

def dV2s_dt(V1,V2,I,V2m):
    return gs(V1-V2) - I   + c22*(V2m - V2)

def dIs_dt(V1,V2,I):
    return V2 - R4s*I

#Método RK4--------------------------------------------------------------------

def RK4(dV1_dt,dV2_dt,dI_dt,dV1s_dt,dV2s_dt,dIs_dt,V1,V2,I,V1s,V2s,Is,h):
    k1V1 = h*dV1_dt(V1,V2,I)
    k1V2 = h*dV2_dt(V1,V2,I)
    k1I = h*dI_dt(V1,V2,I)
    k1V1s = h*dV1s_dt(V1s,V2s,Is,V1)
    k1V2s = h*dV2s_dt(V1s,V2s,Is,V2)
    k1Is = h*dIs_dt(V1s,V2s,Is)
    k2V1 = h*dV1_dt(V1 + k1V1/2, V2 + k1V2/2, I + k1I/2)
    k2V2 = h*dV2_dt(V1 + k1V1/2, V2 + k1V2/2,I + k1I/2)
    k2I = h*dI_dt(V1 + k1V1/2, V2 + k1V2/2, I + k1I/2)
    k2V1s = h*dV1s_dt(V1s + k1V1s/2,V2s + k1V2s/2,Is + k1Is/2,V1 + k1V1/2)
    k2V2s = h*dV2s_dt(V1s + k1V1s/2,V2s + k1V2s/2,Is + k1Is/2,V2 + k1V2/2)
    k2Is = h*dIs_dt(V1s + k1V1s/2,V2s + k1V2s/2,Is + k1Is/2)
    k3V1 = h*dV1_dt(V1 + k2V1/2, V2 + k2V2/2, I + k2I/2)
    k3V2 = h*dV2_dt(V1 + k2V1/2, V2 + k2V2/2, I + k2I/2)  
    k3I = h*dI_dt(V1 + k2V1/2, V2 + k2V2/2, I + k2I/2)
    k3V1s = h*dV1s_dt(V1s + k2V1s/2,V2s + k2V2s/2,Is + k2Is/2,V1 + k2V1/2)
    k3V2s = h*dV2s_dt(V1s + k2V1s/2,V2s + k2V2s/2,Is + k2Is/2,V2 + k2V2/2)
    k3Is = h*dIs_dt(V1s + k2V1s/2,V2s + k2V2s/2,Is + k2Is/2)
    k4V1 = h*dV1_dt(V1 + k3V1, V2 + k3V2, I + k3I)
    k4V2 = h*dV2_dt(V1 + k3V1, V2 + k3V2, I + k3I)
    k4I = h*dI_dt(V1 + k3V1, V2 + k3V2, I + k3I)
    k4V1s = h*dV1s_dt(V1s + k3V1s,V2s + k3V2s,Is + k3Is,V1 + k3V1)
    k4V2s = h*dV2s_dt(V1s + k3V1s,V2s + k3V2s,Is + k3Is,V2 + k3V2)
    k4Is = h*dIs_dt(V1s + k3V1s,V2s + k3V2s,Is + k3Is)    
    V1 += (k1V1 + 2*k2V1 + 2*k3V1 + k4V1)/6    
    V2 += (k1V2 + 2*k2V2 + 2*k3V2 + k4V2)/6 
    I += (k1I + 2*k2I + 2*k3I + k4I)/6  
    V1s += (k1V1s + 2*k2V1s + 2*k3V1s + k4V1s)/6    
    V2s += (k1V2s + 2*k2V2s + 2*k3V2s + k4V2s)/6 
    Is += (k1Is + 2*k2Is + 2*k3Is + k4Is)/6  
    return np.array([V1,V2,I,V1s,V2s,Is])

for i in range(0,len(t)-1,1):
    V1_0,V2_0,I_0,V1s_0,V2s_0,Is_0 = RK4(dV1_dt,dV2_dt,dI_dt,dV1s_dt,dV2s_dt,dIs_dt,V1_0,V2_0,I_0,V1s_0,V2s_0,Is_0,h)

t = np.linspace(0,10000,100000)
h = t[1] - t[0]

V1 = V1_0
V2 = V2_0
I = I_0
V1s = V1s_0
V2s = V2s_0
Is = Is_0

#for i in range(0,len(t)-1,1):
#    V1[i+1],V2[i+1],I[i+1],V1s[i+1],V2s[i+1],Is[i+1] = RK4(dV1_dt,dV2_dt,dI_dt,dV1s_dt,dV2s_dt,dIs_dt,V1[i],V2[i],I[i],V1s[i],V2s[i],Is[i],h)

n=1000000
x_max=np.zeros(n)
x_perp_0=1.0
x_perp_1=1.0
x_perp_2=1.0
count_max=0

while(count_max<n):
    V1,V2,I,V1s,V2s,Is = RK4(dV1_dt,dV2_dt,dI_dt,dV1s_dt,dV2s_dt,dIs_dt,V1,V2,I,V1s,V2s,Is,h)
    x_perp_2=np.abs(V1-V1s)+np.abs(V2-V2s)+np.abs(I-Is)
    if(x_perp_1>x_perp_2)&(x_perp_1>x_perp_0):
        x_max[count_max]=x_perp_1
        count_max += 1
    x_perp_0=x_perp_1
    x_perp_1=x_perp_2
    
#Figuras e Gráficos------------------------------------------------------------

p.figure()
P, bins = np.histogram(x_max[:count_max], bins=100, density=True)
bins=bins[:-1]
p.plot(np.log10(bins),np.log10(P),'o')
