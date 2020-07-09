import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy import log10 as lg
from numpy import pi as pi
from scipy.interpolate import interp1d as sp_interp1d
from scipy.interpolate import splrep,splev
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
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# 'y', 'grey','purple','b', 'g', 'b', 'm', 'orange'
names= ['m14_0','m14_5_001', 'm14_5_1','m14_10_001', 'm14_10_1', 'm20_0','m20_5_001','m20_10_001', 'm20_10_1']
colors = ['black', 'c', 'g', 'orange', 'red', 'black', 'c','orange','red']
linestyle=['-', ':', '-.', '-', '--' ,'-' ,'--' , '-.' ,':']
labels=[r'\rm GR',r'$\xi=5,\,\, a=0.01$', r'$\xi=5,\,\, a=1$',r'$\xi=10,\,\, a=0.01$',r'$\xi=10,\,\, a=1$',r'\rm GR',r'$\xi=5,\,\, a=0.01$',
        r'$\xi=10,\,\, a=0.01$',r'$\xi=10,\,\, a=1$']
fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(14,14),sharex=True, sharey=False)
plt.subplots_adjust(hspace=0.0)
import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=25)
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.02/5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))

for j in range(len(names)):

    data1 = np.genfromtxt('data/'+'geo_ap4_' +names[j]+ '.txt')
    alpha=data1[:,0]
    psi=data1[:,1]
    deltat=data1[:,2]*10*km/c*1000
    if j<=4:
        ax1.plot(alpha,psi, color=colors[j], linewidth=2,linestyle='--',label=labels[j])
        ax2.plot(alpha,deltat, color=colors[j], linewidth=2,linestyle='--')
    else:
        ax1.plot(alpha,psi, color=colors[j], linewidth=2,linestyle='-',label=labels[j])
        ax2.plot(alpha,deltat, color=colors[j], linewidth=2,linestyle='-')
    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    ax1.set_ylabel(r'$\rm \psi\, [\rm rad]$', fontsize=30)
    ax2.set_ylabel(r'$\rm \delta t\, [\rm ms]$', fontsize=30)
    ax2.set_xlabel(r'$\rm \alpha \,[\rm rad]$', fontsize=30)
    ax1.tick_params(labelsize=30)
    ax2.tick_params(labelsize=30)
    ax1.set_yticks([0,0.5, 1.0, 1.5, 2.0, 2.5,3])
    ax1.legend(fontsize=22,ncol=2,frameon=False, loc=(0.01,0.4))
    
    sub_axes = plt.axes([0.62, .53, .25, .12]) 
    if j<=4:
        sub_axes.plot(alpha,psi,linewidth=2, color=colors[j],linestyle='--') 
        sub_axes.set_ylim(1.83,2.02)
        sub_axes.set_xlim(1.39,1.51)
        sub_axes.set_yticks([1.85,1.9,1.95,2.0])
        sub_axes.grid(alpha=0.6)
        sub_axes.yaxis.set_minor_locator(MultipleLocator(0.05/2))
        sub_axes.xaxis.set_minor_locator(MultipleLocator(0.05/5))
        mark_inset(ax1, sub_axes, loc1=2, loc2=4)
        
    sub_axes = plt.axes([0.21, .25, .28, .21]) 
    if j<=4:
        sub_axes.plot(alpha,deltat,linewidth=2, color=colors[j],linestyle='--') 
        sub_axes.set_ylim(0.0284,0.0296)
        sub_axes.set_xlim(0.999,1.017)
        sub_axes.set_xticks([1,1.005,1.01,1.015])
#         sub_axes.set_yticks([1.85,1.9,1.95,2.0])
        sub_axes.grid(alpha=0.6)
        sub_axes.yaxis.set_minor_locator(MultipleLocator(0.0005/5))
        sub_axes.xaxis.set_minor_locator(MultipleLocator(0.005/5))
        mark_inset(ax2, sub_axes, loc1=1, loc2=4)
    fig.text(0.14, 0.83, '$M=1.4M_{\odot}$' ,fontsize=25)
    fig.text(0.4, 0.83, '$M=2M_{\odot}$' ,fontsize=25)
            
plt.savefig("geometry.pdf", format='pdf', bbox_inches="tight")
plt.show()
