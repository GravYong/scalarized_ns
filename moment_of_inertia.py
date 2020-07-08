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

names= ['pal1','mpa1', 'ap4', 'sly4','wff1']
numbers=['1', '01', '001']
colors = ['purple', 'c', 'g', 'orange', 'r' , 'grey']
fig, axs = plt.subplots(3, 3,figsize=(15,15),sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0.0)

font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=25)
axs[0,0].yaxis.set_minor_locator(MultipleLocator(1/5))
axs[0,0].xaxis.set_minor_locator(MultipleLocator(1/5))
for j in range(len(names)):
    
    for i in range(len(numbers)):
        
        data1 = np.genfromtxt('data/'+ names[j]+ '_5_'+numbers[i]+'.txt')
        M1, I1 = data1[:,1]/Ms, data1[:, 5]/10**45
        index=np.where(M1==max(M1))[0][0]
        if i==0:
            
            axs[0,i].plot(M1[index::-1],I1[index::-1],linewidth=2, color=colors[j])
            
        else:
            axs[0,i].plot(M1[index::-1],I1[index::-1],linewidth=2, color=colors[j])
            
        axs[0,i].set_ylim(-0.3,4.3)
        axs[0,i].set_xlim(-0.3,2.8)
        axs[0,i].set_xticks([0, 1.0,  2.0])
        
        data11=np.genfromtxt('data/'+'TOV_4eqs_'+ names[j]+ '.txt')
        M2, I2 = data11[:,1]/Ms, data11[:, 4]/10**45
        index=np.where(M2==max(M2))[0][0]
        axs[0,i].plot(M2[index::-1],I2[index::-1],linewidth=2, color=colors[j],linestyle=':')
            
        axs[0,i].grid(alpha=0.8)

        
                
        data2 = np.genfromtxt('data/'+ names[j]+ '_7_'+numbers[i]+'.txt')
        M1, I1 = data2[:,1]/Ms, data2[:, 5]/10**45
        index=np.where(M1==max(M1))[0][0]
        axs[1,i].plot(M1[index::-1],I1[index::-1],linewidth=2, color=colors[j])
        
        data21=np.genfromtxt('data/'+'TOV_4eqs_'+ names[j]+ '.txt')
        M2, I2 = data21[:,1]/Ms, data21[:, 4]/10**45
        index=np.where(M2==max(M2))[0][0]
        axs[1,i].plot(M2[index::-1],I2[index::-1],linewidth=2, color=colors[j],linestyle=':')
        axs[1,i].grid(alpha=0.8)
        
        data3 = np.genfromtxt('data/'+ names[j]+ '_10_'+numbers[i]+'.txt')
        M1, I1 = data3[:,1]/Ms, data3[:, 5]/10**45
        index=np.where(M1==max(M1))[0][0]
        axs[2,i].plot(M1[index::-1],I1[index::-1],linewidth=2, color=colors[j])
        
        data31=np.genfromtxt('data/'+'TOV_4eqs_'+ names[j]+ '.txt')
        M2, I2 = data31[:,1]/Ms, data31[:, 4]/10**45
        index=np.where(M2==max(M2))[0][0]
        axs[2,i].plot(M2[index::-1],I2[index::-1],linewidth=2, color=colors[j],linestyle=':')
        axs[2,i].grid(alpha=0.8)
        
fig.text(0.06, 0.57, r'$I\,[\rm 10^{45}\,g\,cm^{2}]$', ha='center', fontsize=30,rotation='vertical')
fig.text(0.5, 0.06, r'$M\,[{\rm M_{\odot}}]$' ,fontsize=30)      
fig.text(0.92, 0.77, r'$\xi=5$' ,fontsize=30, rotation='90')     
fig.text(0.92, 0.52, r'$\xi=7$' ,fontsize=30, rotation='90')     
fig.text(0.92, 0.27, r'$\xi=10$' ,fontsize=30, rotation='90')     
fig.text(0.22, 0.9, r'$a=1$' ,fontsize=30)     
fig.text(0.48, 0.9, r'$a=0.1$' ,fontsize=30)     
fig.text(0.74, 0.9, r'$a=0.01$' ,fontsize=30)   
plt.savefig("moment.pdf", format='pdf', bbox_inches="tight")
plt.show()
