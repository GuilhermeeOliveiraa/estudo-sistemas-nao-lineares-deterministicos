#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:48:38 2019

@author: joaoguilherme
"""

import numpy as np
import pylab as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

#Parâmetros do Sistema---------------------------------------------------------

t = np.linspace(0,200,20000)
h = t[1] - t[0]
sig = 10.0
#rho = 28
bet = 8./3.
x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)
x[0] = 15
y[0] = 15
z[0] = 15
N_rho = 50
rho_list = np.zeros(N_rho)
N_max = 40
x_max = np.zeros((N_max,N_rho))
t_max = np.zeros((N_max,N_rho))

for j in range(N_rho):
    rho = 120 + 1.0*j
    rho_list[j] = rho
    count = 0
    
    for i in range(0,len(t)-1,1):
        x[i+1],y[i+1],z[i+1] = RK4(dx_dt,dy_dt,dz_dt,x[i],y[i],z[i],h)
    x[0]=x[-1]
    y[0]=y[-1]
    z[0]=z[-1]
    for i in range(0,len(t)-1,1):
        x[i+1],y[i+1],z[i+1] = RK4(dx_dt,dy_dt,dz_dt,x[i],y[i],z[i],h)
    
    for i in range(0,len(t)-3,1):
        if (x[i+1]>x[i])&(x[i+1]>x[i+2])&(count < N_max):
            x_max[count,j] = x[i+1]
            t_max[count,j] = t[i+1]
            count += 1
#        if count == N_max:
#            i = len(t)
#    while (count<N_max):
#            if (x[-2]>x[0]) and (x[-2]>x[-1]):
#                x_max[j,count] = x[-2]
#                count +=1
#        while (i>0):
#            x[0]=x[i-1]
    
#p.figure()
#p.plot(rho_list,x_max,'.b')

p.figure()
for i in range(20):
    p.plot(x_max[:,i],'.b')
#    
#    
#p.figure()  
#p.plot(t,x)
#p.plot(t_max[:,j],x_max[:,j],'+r')  
#p.show()    