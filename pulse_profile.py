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

names= ['m14_a','m14_5_001_a', 'm14_10_001_a','m20_a','m20_5_001_a', 'm20_10_001_a']
colors = ['c', 'g', 'r', 'm', 'orange',  'y', 'grey']
plt.figure(figsize=(12,6))
for j in range(len(names)):
    data1 = np.genfromtxt('data/'+'pp_ap4_' +names[j]+ '.dat')
    t=data1[:,0]*10**3
    flux=data1[:,1]
    plt.plot(t,flux, color=colors[j], linewidth=1.5)
plt.ylim(0.001,0.5)
plt.xlim(0,2.5)
plt.ylabel(r'$\rm{Flux}$', fontsize=30)
plt.xlabel(r'$\rm{Time\,[10^{-3}\,s]}$', fontsize=30)
plt.grid(alpha=0.6)
plt.minorticks_on()
plt.show()

names= ['m14_b','m14_5_001_b', 'm14_10_001_b','m20_b','m20_5_001_b', 'm20_10_001_b']
colors = ['c', 'g', 'r', 'm', 'orange',  'y', 'grey']
plt.figure(figsize=(12,6))
for j in range(len(names)):
    data1 = np.genfromtxt('data/'+'pp_ap4_' +names[j]+ '.txt')
    t=data1[:,0]*10**3
    flux=data1[:,1]
    plt.plot(t,flux, color=colors[j], linewidth=2)
plt.ylim(0.001,0.5)
plt.xlim(0,2.5)
plt.ylabel(r'$\rm{Flux}$', fontsize=25)
plt.xlabel(r'$\rm{Time\,[10^{-3}\,s]}$', fontsize=25)
plt.grid(alpha=0.6)
plt.minorticks_on()
# plt.show()