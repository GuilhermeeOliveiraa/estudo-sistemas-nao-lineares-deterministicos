#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 02:00:46 2019

@author: joaoguilherme
"""

import numpy as np
import pylab as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Parâmetros do Sistema---------------------------------------------------------

t = np.linspace(0,200,20000)
h = t[1] - t[0]

a = 0.432
b = 2.
c = 4.   
 
x0 = 0.01
y0 = 0.01
z0 = 0.01

#Funções do Sistema de Rossler-------------------------------------------------

def dx_dt(x,y,z):
    return -y -z

def dy_dt(x,y,z):
    return x + (a * y)

def dz_dt(x,y,z):
    return b + z*(x - c)

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

x_[0] = x0*1.000001
y_[0] = y0*1.000001
z_[0] = z0*1.000001

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
ax.set_title("Atractor de Rossler")

p.figure()
p.plot(x,y,'r', lw=0.6)
p.xlabel('X')
p.ylabel('Y')
p.title('Atractor de Rossler')

p.figure()
p.plot(y,z,'g', lw=0.6)
p.xlabel('Y')
p.ylabel('Z')
p.title('Atractor de Rossler')

p.figure()
p.plot(x,z,'b', lw=0.6)
p.xlabel('X')
p.ylabel('Z')
p.title('Atractor de Rossler')

p.figure()
p.plot(t,x,'r',lw=0.3)
p.plot(t,y,'g', lw=0.3)
p.plot(t,z,'b', lw=0.3)
p.xlabel('Tempo')
p.ylabel('X,Y,Z')
p.title('Atractor de Rossler - Evolucao Temporal')

p.figure()
p.plot(t[:3000],x[:3000],'r', lw=1.0)
#p.plot(t[:3000],x_[:3000],'y', lw=1.0)
p.xlabel('Tempo')
p.ylabel('X')

p.figure()
p.plot(t[:3000],y[:3000],'g', lw=1.0)
#p.plot(t[:3000],y_[:3000],'y', lw=1.0)
p.xlabel('Tempo')
p.ylabel('Y')

p.figure()
p.plot(t[:3000],z[:3000],'b',  lw=1.0)
#p.plot(t[:3000],z_[:3000],'y', lw=1.0)
p.xlabel('Tempo')
p.ylabel('Z')
