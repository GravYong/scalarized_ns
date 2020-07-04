import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy import log10 as lg
from numpy import pi as pi
from scipy.interpolate import interp1d as sp_interp1d
from scipy.integrate import odeint
from scipy.integrate import ode
import warnings
import timeit
import scipy.optimize as opt
from matplotlib import cm
from astropy import constants as const
from astropy import units as u
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
G=const.G.cgs.value
c=const.c.cgs.value
Ms=const.M_sun.cgs.value
hbar=const.hbar.cgs.value
m_n=const.m_n.cgs.value
km=10**5

import matplotlib.font_manager as font_manager

plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.labelpad'] = 8.0
plt.rcParams['figure.constrained_layout.h_pad'] = 0
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tick_params(axis='both', which='minor', labelsize=18)

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

data = np.genfromtxt('data/ex_boundedorbit.dat')
x1=data[:,1]*np.cos(data[:,2])*10
y1=data[:,1]*np.sin(data[:,2])*10
x2=data[:,3]*np.cos(data[:,4])*10
y2=data[:,3]*np.sin(data[:,4])*10


data1 = np.genfromtxt('data/ex_scatterorbit.dat')
x3=data1[:,1]*np.cos(data1[:,2])*10
y3=data1[:,1]*np.sin(data1[:,2])*10
x4=data1[:,3]*np.cos(data1[:,4])*10
y4=data1[:,3]*np.sin(data1[:,4])*10

r1 = 11.4
theta = np.linspace(0,2*pi,36)
x5 = r1*np.cos(theta)
y5 = r1*np.sin(theta)


fig, (ax1,ax2)=plt.subplots(2,1, figsize=(12,24))
plt.subplots_adjust(hspace=0.15)

ax1.plot(x1,y1, label=r'\rm GR',linewidth=2)
ax1.plot(x2,y2,label=r'\rm Scalar-tensor',linewidth=2)
ax1.fill(x5,y5,'b',alpha=0.6)
ax2.plot(x3,y3, label=r'\rm GR',linewidth=2)
ax2.plot(x4,y4,label=r'\rm Scalar-tensor',linewidth=2)
ax2.fill(x5,y5,'b',alpha=0.6)

ax1.tick_params(labelsize=30)
ax2.tick_params(labelsize=30)
ax1.minorticks_on()
ax2.minorticks_on()
ax1.grid()
ax1.set_xticks([-80,-60,-40,-20,0,20,40,60,80])
ax2.grid()
ax1.set_ylim(-84,84)
ax1.set_xlabel(r'$x_{\rm b}\,[\rm km]$',fontsize=35)
ax1.set_ylabel(r'$y_{\rm b}\,[\rm km]$',fontsize=35)
ax2.set_xlabel(r'$x_{\rm s}\,[\rm km]$',fontsize=35)
ax2.set_ylabel(r'$y_{\rm s}\,[\rm km]$',fontsize=35)
ax1.legend(fontsize=30,frameon=False)
ax2.legend(fontsize=30,frameon=False)
ax1.tick_params(labelsize=30)
ax2.tick_params(labelsize=30)
plt.savefig("orbitex.pdf", format='pdf', bbox_inches="tight")
plt.show()