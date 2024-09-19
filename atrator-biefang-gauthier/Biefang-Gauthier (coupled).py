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
c22 = 0.0 

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

V1 = np.zeros_like(t)
V2 = np.zeros_like(t)
I = np.zeros_like(t)
V1s = np.zeros_like(t)
V2s = np.zeros_like(t)
Is = np.zeros_like(t)


V1[0] = V1_0
V2[0] = V2_0
I[0] = I_0
V1s[0] = V1s_0
V2s[0] = V2s_0
Is[0] = Is_0

for i in range(0,len(t)-1,1):
    V1[i+1],V2[i+1],I[i+1],V1s[i+1],V2s[i+1],Is[i+1] = RK4(dV1_dt,dV2_dt,dI_dt,dV1s_dt,dV2s_dt,dIs_dt,V1[i],V2[i],I[i],V1s[i],V2s[i],Is[i],h)

x_perp = np.zeros_like(V1)
x_perp = np.abs(V1-V1s)+np.abs(V2-V2s)+np.abs(I-Is)

#Figuras e Gráficos------------------------------------------------------------

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(V1, V2, I, lw=0.5)
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Eixo Z")
ax.set_title("Bienfang-Gauthier - Mestre")

p.figure()
p.plot(V1,V2,'r', lw=0.6)
p.xlabel('X')
p.ylabel('Y')
p.title('Bienfang-Gauthier - Mestre')

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(V1s, V2s, Is, lw=0.5)
ax.set_xlabel("Eixo Xs")
ax.set_ylabel("Eixo Ys")
ax.set_zlabel("Eixo Zs")
ax.set_title("Bienfang-Gauthier - Escravo")

p.figure()
p.plot(V1s,V2s,'r', lw=0.6)
p.xlabel('Xs')
p.ylabel('Ys')
p.title('Bienfang-Gauthier - Escravo')

p.figure()
p.plot(V1,V1s,'b', lw=0.6)
p.xlabel('V1d')
p.ylabel('V1r')
p.title('Bienfang-Gauthier Acoplados')

p.figure()
p.plot(t,x_perp,'r',lw=0.1)
p.xlabel('Tempo')
p.ylabel(r'$x_\perp$')
p.title('Bienfang-Gauthier - Evolucao Temporal')


#p.figure()
#p.plot(V2,I,'g', lw=0.6)
#p.xlabel('Y')
#p.ylabel('Z')
#p.title('Bienfang-Gauthier')
#
#p.figure()
#p.plot(V1,I,'b', lw=0.6)
#p.xlabel('X')
#p.ylabel('Z')
#p.title('Bienfang-Gauthier')

#p.figure()
#p.plot(t,V1,'r',lw=0.7)
#p.plot(t,V2,'g', lw=0.7)
#p.plot(t,I,'b', lw=0.7)
#p.xlabel('Tempo')
#p.ylabel('X,Y,Z')
#p.title('Bienfang-Gauthier - Evolucao Temporal')
#
#p.figure()
#p.plot(t[:3000],V1[:3000],'r', lw=1.0)
##p.plot(t[:3000],x_[:3000],'b', lw=1.0)
#p.xlabel('Tempo')
#p.ylabel('X')
#
#p.figure()
#p.plot(t[:3000],V2[:3000],'g', lw=1.0)
##p.plot(t[:3000],y_[:3000],'b', lw=1.0)
#p.xlabel('Tempo')
#p.ylabel('Y')
#
#p.figure()
#p.plot(t[:3000],I[:3000],'b',  lw=1.0)
##p.plot(t[:3000],z_[:3000],'b', lw=1.0)
#p.xlabel('Tempo')
#p.ylabel('Z')
