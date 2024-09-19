#!/usr/bin/env python2---------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:50:28 2019

@author: joaoguilherme
"""

import numpy as np
import pylab as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Parâmetros do Sistema---------------------------------------------------------

t = np.linspace(0,200,20000)
h = t[1] - t[0]

sig = 10.0
rho = 126.52
bet = 8./3.   
 
x0 = 0.01
y0 = 0.01
z0 = 0.01

#Funções do Sistema de Lorenz--------------------------------------------------

def dx_dt(x,y,z):
    return sig * (y - x)

def dy_dt(x,y,z):
    return rho*x  - x*z - y  

def dz_dt(x,y,z):
    return x * y - bet * z

#Método RK4-------------------------------------------------------------------

def RK4(dx_dt,dy_dt,dz_dt,x,y,z,h):
    k1x = h*dx_dt(x,y,z)
    k1y = h*dy_dt(x,y,z)
    k1z = h*dz_dt(x,y,z)
    k2x = h*dx_dt(x + k1x/2, y + k1y/2, z + k1z/2)
    k2y = h*dy_dt(x + k1x/2, y + k1y/2, z + k1z/2)
    k2z = h*dz_dt(x + k1x/2, y + k1y/2, z + k1z/2)
    k3x = h*dx_dt(x + k2x/2, y + k2y/2, z + k2z/2)
    k3y = h*dy_dt(x + k2x/2, y + k2y/2, z + k2z/2)  
    k3z = h*dz_dt(x + k2x/2, y + k2y/2, z + k2z/2)
    k4x = h*dx_dt(x + k3x, y + k3y, z + k3z)
    k4y = h*dy_dt(x + k3x, y + k3y, z + k3z)
    k4z = h*dz_dt(x + k3x, y + k3y, z + k3z)
    x += (k1x + 2*k2x + 2*k3x + k4x)/6    
    y += (k1y + 2*k2y + 2*k3y + k4y)/6 
    z += (k1z + 2*k2z + 2*k3z + k4z)/6  
    return np.array([x,y,z])

for i in range(0,len(t)-1,1):
    x0,y0,z0 = RK4(dx_dt,dy_dt,dz_dt,x0,y0,z0,h)

x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)

x_ = np.zeros_like(t)
y_ = np.zeros_like(t)
z_ = np.zeros_like(t)


x[0] = x0
y[0] = y0
z[0] = z0

x_[0] = x0*1.00000000001
y_[0] = y0*1.00000000001
z_[0] = z0*1.00000000001

for i in range(0,len(t)-1,1):
    x[i+1],y[i+1],z[i+1] = RK4(dx_dt,dy_dt,dz_dt,x[i],y[i],z[i],h)
    x_[i+1],y_[i+1],z_[i+1] = RK4(dx_dt,dy_dt,dz_dt,x_[i],y_[i],z_[i],h)

#Figuras e Gráficos------------------------------------------------------------

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Eixo Z")
ax.set_title("Atractor de Lorenz")

p.figure()
p.plot(x,y,'r', lw=0.6)
p.xlabel('X')
p.ylabel('Y')
p.title('Atractor de Lorenz')

p.figure()
p.plot(y,z,'g', lw=0.6)
p.xlabel('Y')
p.ylabel('Z')
p.title('Atractor de Lorenz')

p.figure()
p.plot(x,z,'b', lw=0.6)
p.xlabel('X')
p.ylabel('Z')
p.title('Atractor de Lorenz')

p.figure()
p.plot(t[:1000],x[:1000],'r',lw=0.7)
p.plot(t[:1000],y[:1000],'g', lw=0.7)
p.plot(t[:1000],z[:1000],'b', lw=0.7)
p.xlabel('Tempo')
p.ylabel('X,Y,Z')
p.title('Atractor de Lorenz - Evolucao Temporal')

p.figure()
p.plot(t[:1000],x[:1000],'r', lw=1.0)
#p.plot(t[:2000],x_[:2000],'b', lw=1.0)
p.xlabel('Tempo')
p.ylabel('X')

p.figure()
p.plot(t[:1000],y[:1000],'g', lw=1.0)
#p.plot(t[:2000],y_[:2000],'b', lw=1.0)
p.xlabel('Tempo')
p.ylabel('Y')

p.figure()
p.plot(t[:1000],z[:1000],'b',  lw=1.0)
#p.plot(t[:2000],z_[:2000],'b', lw=1.0)
p.xlabel('Tempo')
p.ylabel('Z')
