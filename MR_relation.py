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
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5
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

names= ['ap4','pal1','wff1','mpa1','sly4']
numbers=['1', '01', '001']
colors = ['c', 'g', 'r', 'm', 'orange',  'y', 'grey']
fig, axs = plt.subplots(3, 3,figsize=(15,15),sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0.0)
import matplotlib.font_manager as font_manager
font = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal', size=25)
axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.5/3))
axs[0,0].xaxis.set_minor_locator(MultipleLocator(2.5/4))

for j in range(len(names)):
    
    for i in range(len(numbers)):
        
        data1 = np.genfromtxt('data/'+ names[j]+ '_5_'+numbers[i]+'.txt')
        data11=np.genfromtxt('data/'+ names[j]+ '_5_'+numbers[i]+'.txt')
        M1, R1 = data1[:,1]/Ms, data1[:, 3]/10**5 * (1+ 5*data1[:,8]**2)**(-1/2)
        index=np.where(M1==max(M1))[0][0]
        if i==0:
            
            axs[0,i].plot(R1[index::-1],M1[index::-1],linewidth=3, color=colors[j])
            
        else:
            axs[0,i].plot(R1[index::-1],M1[index::-1],linewidth=2, color=colors[j])
            
        axs[0,i].set_ylim(0.1,2.6)
        axs[0,i].set_xlim(9.0,15.4)
        
        Ra=np.linspace(6,18,100)
        MB=Ra*10**5*c**2/3/G/Ms
        axs[0,i].fill_between(Ra, MB, 2.6,color='black',alpha=0.1)
#         axs[0,i].plot(Ra,MB, color='black',linewidth=2,alpha=0.1)
        axs[0,i].grid(alpha=0.8)
        
        data11=np.genfromtxt('data/'+'TOV_4eqs_'+ names[j]+ '.txt')
        M2, R2 = data11[:,1]/Ms, data11[:, 3]/10**5
        index=np.where(M2==max(M2))[0][0]
        axs[0,i].plot(R2[index::-1],M2[index::-1],linewidth=2, color=colors[j],linestyle='-.')
        axs[0,i].grid(alpha=0.8)

        
                
        data2 = np.genfromtxt('data/'+ names[j]+ '_7_'+numbers[i]+'.txt')
        M1, R1 = data2[:,1]/Ms, data2[:, 3]/10**5* (1+ 7*data2[:,8]**2)**(-1/2)
        index=np.where(M1==max(M1))[0][0]
        axs[1,i].plot(R1[index::-1],M1[index::-1],linewidth=2, color=colors[j])
        
        Ra=np.linspace(6,18,100)
        MB=Ra*10**5*c**2/3/G/Ms
        axs[1,i].fill_between(Ra, MB, 2.6,color='black',alpha=0.1)
#         axs[1,i].plot(Ra,MB, color='black',linewidth=2,alpha=0.6)?
        axs[1,i].grid(alpha=0.8)
        
        data21=np.genfromtxt('data/'+'TOV_4eqs_'+ names[j]+ '.txt')
        M2, R2 = data21[:,1]/Ms, data21[:, 3]/10**5
        index=np.where(M2==max(M2))[0][0]
        axs[1,i].plot(R2[index::-1],M2[index::-1],linewidth=2, color=colors[j],linestyle='-.')
        axs[1,i].grid(alpha=0.8)
        
        data3 = np.genfromtxt('data/'+ names[j]+ '_10_'+numbers[i]+'.txt')
        M1, R1 = data3[:,1]/Ms, data3[:, 3]/10**5* (1+ 10*data3[:,8]**2)**(-1/2)
        index=np.where(M1==max(M1))[0][0]
        axs[2,i].plot(R1[index::-1],M1[index::-1],linewidth=2, color=colors[j])
        
        Ra=np.linspace(6,18,100)
        MB=Ra*10**5*c**2/3/G/Ms
        axs[2,i].fill_between(Ra, MB, 2.6,color='black',alpha=0.1)
#         axs[2,i].plot(Ra,MB, color='black',linewidth=2,alpha=0.6)
        axs[2,i].grid(alpha=0.8)
        
        data31=np.genfromtxt('data/'+'TOV_4eqs_'+ names[j]+ '.txt')
        M2, R2 = data31[:,1]/Ms, data31[:, 3]/10**5
        index=np.where(M2==max(M2))[0][0]
        axs[2,i].plot(R2[index::-1],M2[index::-1],linewidth=2, color=colors[j],linestyle='-.')
        
        
fig.text(0.06, 0.53, r'$M\,[{\rm M_{\odot}}]$', ha='center', fontsize=30,rotation='vertical')
fig.text(0.5, 0.06, r'$R\,[\rm km]$' ,fontsize=30)      
fig.text(0.92, 0.77, r'$\xi=5$' ,fontsize=30, rotation='90')     
fig.text(0.92, 0.52, r'$\xi=7$' ,fontsize=30, rotation='90')     
fig.text(0.92, 0.27, r'$\xi=10$' ,fontsize=30, rotation='90')     
fig.text(0.22, 0.9, r'$a=1$' ,fontsize=30)     
fig.text(0.48, 0.9, r'$a=0.1$' ,fontsize=30)     
fig.text(0.74, 0.9, r'$a=0.01$' ,fontsize=30)   
plt.savefig("MR.pdf", format='pdf', bbox_inches="tight")
plt.show()
