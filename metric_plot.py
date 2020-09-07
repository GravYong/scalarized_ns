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


names1= ['m14','m14_5_001','m14_5_1', 'm14_10_001','m14_10_1']
names2=['m20','m20_5_001', 'm20_10_001','m20_10_1']
colors=['black', 'c', 'g', 'orange', 'red', 'black', 'c','orange','red']
linestyle=['-', ':', '-.', '-', '--' ,'-' ,'--' , '-.' ,':']
labels=[r'\rm GR',r'$\xi=5,\,\, a=0.01$', r'$\xi=5,\,\, a=1$',r'$\xi=10,\,\, a=0.01$',r'$\xi=10,\,\, a=1$',r'\rm GR',r'$\xi=5,\,\, a=0.01$',
        r'$\xi=10,\,\, a=0.01$',r'$\xi=10,\,\, a=1$']
fig, axs = plt.subplots(2, 2,figsize=(15,12),sharex=True, sharey='row')
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0)
axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.25/5))
axs[1,0].yaxis.set_minor_locator(MultipleLocator(0.2/5))
axs[0,0].xaxis.set_minor_locator(MultipleLocator(10/5))  



    
for i in range(len(names1)):
    
    data1 = np.genfromtxt('data/'+'sol_'+ 'ap4_'+names1[i]+'.txt')
    R, gtt, grr= data1[:,0]/10**5, data1[:,1], data1[:, 2]
    
    axs[1,0].plot(R,gtt,linewidth=2, color=colors[i],linestyle=linestyle[i])
    axs[1,0].grid(alpha=0.6)
    axs[1,0].set_ylabel(r'$ -g_{tt}$', fontsize=30)
    
    axs[0,0].plot(R,grr,linewidth=2, color=colors[i],linestyle=linestyle[i],label=labels[i])
    axs[0,0].grid(alpha=0.6)
    axs[0,0].set_ylabel(r'$ g_{rr}$', fontsize=30)
    axs[0,0].legend(fontsize=25, frameon=False,loc=(0.37,0.27))
    
    sub_axes = plt.axes([.3, .18, .20, .18]) 
    sub_axes.plot(R,gtt,linewidth=2, color=colors[i],linestyle=linestyle[i]) 
    sub_axes.set_ylim(0.67,0.725)
    sub_axes.set_xlim(13.4,14.6)
#     sub_axes.set_xticks([10,11,12])
#     sub_axes.grid(alpha=0.8)
    sub_axes.yaxis.set_minor_locator(MultipleLocator(0.02/5))
    sub_axes.xaxis.set_minor_locator(MultipleLocator(0.5/5))
    
    
    
for j in range(len(names2)):
    
    data2 = np.genfromtxt('data/'+'sol_'+ 'ap4_'+names2[j]+'.txt')
    R, gtt, grr= data2[:,0]/10**5, data2[:,1], data2[:, 2]
    axs[1,1].plot(R,gtt,linewidth=2, color=colors[j+5],linestyle=linestyle[j+5])
    axs[1,1].grid(alpha=0.6)
    axs[0,1].plot(R,grr,linewidth=2, color=colors[j+5],linestyle=linestyle[j+5],label=labels[j+5])
    axs[0,1].grid(alpha=0.6)
    axs[0,1].legend(fontsize=25, frameon=False,loc=(0.37,0.4))
    
    sub_axes = plt.axes([.69, .18, .19, .16]) 
    sub_axes.plot(R,gtt,linewidth=2, color=colors[j+5],linestyle=linestyle[j+5]) 
    sub_axes.set_xlim(13.4,14.6)
    sub_axes.set_ylim(0.53,0.59)
#     sub_axes.set_yticks([6,8,10])
    sub_axes.set_yticks([0.54,0.56,0.58])
#     sub_axes.grid(alpha=0.8)
    sub_axes.yaxis.set_minor_locator(MultipleLocator(0.02/5))
    sub_axes.xaxis.set_minor_locator(MultipleLocator(0.5/5))

fig.text(0.48, 0.04, r'$r\,[\rm km]$' ,fontsize=30)
# fig.text(0.7, 0.04, r'$r\,[\rm km]$' ,fontsize=30) 
axs[1,0].set_ylim(0.14,0.95)
axs[0,0].set_ylim(0.97,2.35)
axs[0,0].set_xlim(-1,43)

fig.text(0.28, 0.84, r'$M=1.4M_{\odot}$' ,fontsize=25)
fig.text(0.66, 0.84, r'$M=2M_{\odot}$' ,fontsize=25)  

plt.savefig("ap41.pdf", format='pdf', bbox_inches="tight")
plt.show()



