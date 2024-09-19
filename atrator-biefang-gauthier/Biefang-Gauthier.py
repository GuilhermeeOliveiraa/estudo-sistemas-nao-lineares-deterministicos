#!/usr/bin/env python2---------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:50:28 2019

@author: joaoguilherme
"""

import math as m
import numpy as np
import pylab as p
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
 
V1_0 = 0.01
V2_0 = 0.01
I_0 = 0.01

#Funções do Sistema de Bienfang-Gauthier---------------------------------------

def g(v):
    return v/R2 + I_M*(m.exp(alpha_r*v) - m.exp(-alpha_r*v))

def dV1_dt(V1,V2,I):
    return V1/R1 - g(V1-V2)

def dV2_dt(V1,V2,I):
    return g(V1-V2) - I  

def dI_dt(V1,V2,I):
    return V2 - R4*I

def g(v):
    return v/R2 + I_M*(m.exp(alpha_r*v) - m.exp(-alpha_r*v))

def dV1_dt(V1,V2,I):
    return V1/R1 - g(V1-V2)

def dV2_dt(V1,V2,I):
    return g(V1-V2) - I  

def dI_dt(V1,V2,I):
    return V2 - R4*I

#Método RK4--------------------------------------------------------------------

def RK4(dV1_dt,dV2_dt,dI_dt,V1,V2,I,h):
    k1V1 = h*dV1_dt(V1,V2,I)
    k1V2 = h*dV2_dt(V1,V2,I)
    k1I = h*dI_dt(V1,V2,I)
    k2V1 = h*dV1_dt(V1 + k1V1/2, V2 + k1V2/2, I + k1I/2)
    k2V2 = h*dV2_dt(V1 + k1V1/2, V2 + k1V2/2,I + k1I/2)
    k2I = h*dI_dt(V1 + k1V1/2, V2 + k1V2/2, I + k1I/2)
    k3V1 = h*dV1_dt(V1 + k2V1/2, V2 + k2V2/2, I + k2I/2)
    k3V2 = h*dV2_dt(V1 + k2V1/2, V2 + k2V2/2, I + k2I/2)  
    k3I = h*dI_dt(V1 + k2V1/2, V2 + k2V2/2, I + k2I/2)
    k4V1 = h*dV1_dt(V1 + k3V1, V2 + k3V2, I + k3I)
    k4V2 = h*dV2_dt(V1 + k3V1, V2 + k3V2, I + k3I)
    k4I = h*dI_dt(V1 + k3V1, V2 + k3V2, I + k3I)
    V1 += (k1V1 + 2*k2V1 + 2*k3V1 + k4V1)/6    
    V2 += (k1V2 + 2*k2V2 + 2*k3V2 + k4V2)/6 
    I += (k1I + 2*k2I + 2*k3I + k4I)/6  
    return np.array([V1,V2,I])

for i in range(0,len(t)-1,1):
    V1_0,V2_0,I_0 = RK4(dV1_dt,dV2_dt,dI_dt,V1_0,V2_0,I_0,h)

t = np.linspace(0,500,100000)
h = t[1] - t[0]

V1 = np.zeros_like(t)
V2 = np.zeros_like(t)
I = np.zeros_like(t)

V1_ = np.zeros_like(t)
V2_ = np.zeros_like(t)
I_ = np.zeros_like(t)


V1[0] = V1_0
V2[0] = V2_0
I[0] = I_0

V1_[0] = V1_0*1.00000000001
V2_[0] = V2_0*1.00000000001
I_[0] = I_0*1.00000000001

for i in range(0,len(t)-1,1):
    V1[i+1],V2[i+1],I[i+1] = RK4(dV1_dt,dV2_dt,dI_dt,V1[i],V2[i],I[i],h)
    V1_[i+1],V2_[i+1],I_[i+1] = RK4(dV1_dt,dV2_dt,dI_dt,V1_[i],V2_[i],I_[i],h)

#Figuras e Gráficos------------------------------------------------------------

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(V1, V2, I, lw=0.5)
ax.set_xlabel("V1")
ax.set_ylabel("V2")
ax.set_zlabel("I")
ax.set_title("Bienfang-Gauthier")

p.figure()
p.plot(V1,V2,'r', lw=0.6)
p.xlabel('V1')
p.ylabel('V2')
p.title('Bienfang-Gauthier')

p.figure()
p.plot(V2,I,'g', lw=0.6)
p.xlabel('V2')
p.ylabel('I')
p.title('Bienfang-Gauthier')

p.figure()
p.plot(V1,I,'b', lw=0.6)
p.xlabel('V1')
p.ylabel('I')
p.title('Bienfang-Gauthier')

p.figure()
p.plot(t,V1,'r',lw=0.7)
p.plot(t,V2,'g', lw=0.7)
p.plot(t,I,'b', lw=0.7)
p.xlabel('Tempo')
p.ylabel('V1,V2,I')
p.title('Bienfang-Gauthier - Evolucao Temporal')

p.figure()
p.plot(t[:3000],V1[:3000],'r', lw=1.0)
#p.plot(t[:3000],x_[:3000],'b', lw=1.0)
p.xlabel('Tempo')
p.ylabel('X')

p.figure()
p.plot(t[:3000],V2[:3000],'g', lw=1.0)
#p.plot(t[:3000],y_[:3000],'b', lw=1.0)
p.xlabel('Tempo')
p.ylabel('Y')

p.figure()
p.plot(t[:3000],I[:3000],'b',  lw=1.0)
#p.plot(t[:3000],z_[:3000],'b', lw=1.0)
p.xlabel('Tempo')
p.ylabel('Z')
