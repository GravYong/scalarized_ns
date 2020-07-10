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
q=0.5

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

names= ['m14_a','m14_5_001_a','m14_5_1_a', 'm14_10_001_a','m14_10_1_a','m20_a','m20_5_001_a', 'm20_10_001_a','m20_10_1_a']
names1= ['m14_b','m14_5_001_b','m14_5_1_b', 'm14_10_001_b','m14_10_1_b','m20_b','m20_5_001_b', 'm20_10_001_b','m20_10_1_b']
colors=['black', 'c', 'g', 'orange', 'red', 'black', 'c','orange','red']
linestyle=['-', ':', '-.', '-', '--' ,'-' ,'--' , '-.' ,':']
fig, axs = plt.subplots(2, 3,figsize=(15,10),sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0)
axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1/4))
axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.5/4))   
axs[0,0].set_xticks([ 0.5,1,1.5,  2.0])
axs[0,0].set_yticks([0,0.1,0.2,0.3,0.4])
axs[0,0].set_xlim(0,2.5)
labels=[r'\rm GR',r'$\xi=5,\,\, a=0.01$', r'$\xi=5,\,\, a=1$',r'$\xi=10,\,\, a=0.01$',r'$\xi=10,\,\, a=1$',r'\rm GR',r'$\xi=5,\,\, a=0.01$',
        r'$\xi=10,\,\, a=0.01$',r'$\xi=10,\,\, a=1$']
for i in range(len(names)):
    
    data1 = np.genfromtxt('data/'+'pp_ap4_' +names[i]+ '.dat')
    t=data1[:,0]*10**3
    flux1=data1[:,1]
    flux2=data1[:,2]
    flux3=data1[:,3]
    if i<=4:
        axs[0,0].plot( t, flux1,linewidth=2, color=colors[i],linestyle='--')
        axs[0,1].plot( t, flux2,linewidth=2, color=colors[i],linestyle='--',label=labels[i])
        axs[0,2].plot( t, flux3,linewidth=2, color=colors[i],linestyle='--')
    else:
        axs[0,0].plot( t, flux1,linewidth=2, color=colors[i],linestyle='-')
        axs[0,1].plot( t, flux2,linewidth=2, color=colors[i],linestyle='-',label=labels[i])
        axs[0,2].plot( t, flux3,linewidth=2, color=colors[i],linestyle='-')
    axs[0,1].legend(fontsize=12, ncol=2, frameon=False, loc=(0.04,0.5))
    axs[0,0].grid(alpha=q)
    axs[0,1].grid(alpha=q)
    axs[0,2].grid(alpha=q)
    
    data2 = np.genfromtxt('data/'+'pp_ap4_' +names1[i]+ '.dat')
    t1=data2[:,0]*10**3
    flux4=data2[:,1]
    flux5=data2[:,2]
    flux6=data2[:,3]
    if i<=4:
        axs[1,0].plot( t1, flux4,linewidth=2, color=colors[i],linestyle='--')
        axs[1,1].plot( t1, flux5,linewidth=2, color=colors[i],linestyle='--')
        axs[1,2].plot( t1, flux6,linewidth=2, color=colors[i],linestyle='--')
    else:
        axs[1,0].plot( t1, flux4,linewidth=2, color=colors[i],linestyle='-')
        axs[1,1].plot( t1, flux5,linewidth=2, color=colors[i],linestyle='-')
        axs[1,2].plot( t1, flux6,linewidth=2, color=colors[i],linestyle='-')
    axs[1,0].grid(alpha=q)
    axs[1,1].grid(alpha=q)
    axs[1,2].grid(alpha=q)

fig.text(0.4, 0.83, '$M=1.4M_{\odot}$' ,fontsize=15)
fig.text(0.53, 0.83, '$M=2M_{\odot}$' ,fontsize=15)  
fig.text(0.42, 0.03, r'$\rm{Observer\ time \,[\rm ms]}$' ,fontsize=30)   
fig.text(0.06, 0.5, r'${F}$',fontsize=30,rotation='vertical')
plt.savefig("pulse_profile.pdf", format='pdf', bbox_inches="tight")
plt.show()
