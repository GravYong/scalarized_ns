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

data = np.genfromtxt('data/toyexterior.txt')
t=data[:,0]
ex001=data[:,1]
ex1=data[:,2]

data1 = np.genfromtxt('data/toyinterior.txt')
t1=data1[:,0]
in101=data1[:,1]
in11=data1[:,2]
in501=data1[:,3]
in51=data1[:,4]

plt.figure(figsize=(12,8))
ax=plt.subplot(111)

ax.plot(t,ex001, linewidth=2.5, color='#fc8d62', label=r'$\rm{a=0.01}$')
ax.plot(t1,in101, linewidth=2.5,color='#fc8d62',linestyle='--',label=r'$\rm{a=0.01\,,\xi=1}$')
ax.plot(t1,in501, linewidth=2.5,color='#fc8d62',linestyle='-.',label=r'$\rm{a=0.01\,,\xi=5}$')

ax.plot(t,ex1,linewidth=2.5,color='#8da0cb',label=r'$\rm{a=1}$')
ax.plot(t1,in11,linewidth=2.5, linestyle='--',color='#8da0cb',label=r'$\rm{a=1\,,\xi=1}$')
ax.plot(t1,in51,linewidth=2.5, linestyle='-.',color='#8da0cb',label=r'$\rm{a=1\,,\xi=5}$')

ax.set_xlim(0,0.51)
ax.set_ylim(-12.5,2.4)
ax.set_ylabel(r'$\Phi^{\prime}(x_{R})/\Phi(x_{R})$',fontsize=30)
ax.set_xlabel(r'$M/R$',fontsize=30)
ax.grid(alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.minorticks_on()
ax.legend(fontsize=25, frameon=False,ncol=2)
plt.savefig("linearized_phi_c.pdf", format='pdf', bbox_inches="tight")
plt.show()

data = np.genfromtxt('data/xia.txt')
a=data[:,0]
ap4=data[:,1]
pal1=data[:,2]
wff1=data[:,3]
mpa1=data[:,4]
sly4=data[:,5]

plt.figure(figsize=(12,8))
ax=plt.subplot(111)

ax.plot(a,ap4, linewidth=4, color='orange', label='{AP4}')
ax.plot(a,pal1, linewidth=2.5,color='c',label='PAL1')
ax.plot(a,wff1, linewidth=2.5,color='r',label='WFF1')
ax.plot(a,mpa1,linewidth=2.5,color='purple',label='MPA1')
ax.plot(a,sly4,linewidth=2.5 ,color='g',label='SLy4',alpha=1)

ax.set_xlim(0.0,3.05)
ax.set_ylim(1.55,18.5)
ax.set_ylabel(r'$ \xi_{\rm min}$',fontsize=30)
ax.set_xlabel(r'$a$',fontsize=30)
ax.grid(alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.minorticks_on()
ax.legend(fontsize=25, frameon=False,ncol=1)
plt.savefig("xi_a.pdf", format='pdf', bbox_inches="tight")
plt.show()