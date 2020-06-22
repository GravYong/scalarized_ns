
""" Solve Tolman-Oppenheimer-Volkoff equations with ode from scipy

    TOV equations:

    dM/dr = 4 * pi * e * r^2
    dP/dr = - G / r^2 * [e + P / c^2] * [M + 4 * pi * r^3 * P / c^2] *
            1 / [1 - 2 * G * M / r / c^2]

    From: https://en.wikipedia.org/wiki/Tolman-Oppenheimer-Volkoff_equation
        - notice: symbol "rho" in wiki page is actually energy density

    units: cgs

"""

__author__ = "Lijing Shao"
__email__ = "Friendshao@gmail.com"
__license__ = "GPL"

import sys
import warnings
import timeit
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import ode as sp_ode

# sys.path.append('../EOS/')
from EOS import EOS

C = 29979245800.0  # cm/s
G = 6.67408e-8  # cm^3/g/s^2
MSUN = 1.98855e33  # g
KM = 1.0e5  # cm
E_NUCL = 2.0e14  # minimun energy density for NS core; g/cm^3
mB = 1.660538921e-24  # g
L2M_UNIT = C**2.0 / G  # convert [L] to [M]

def L2M(r):
    """ Convert length scale [L] to mass scale [M] """
    return r * L2M_UNIT

class TOV2(object):
    """ TOV equation solver with ode from scipy.integrate """

    def __init__(self, EOS_name='AP4'):
        """ Initialize the TOV solver """
        self.EOS_name = EOS_name
        self.EOS = EOS(EOS_name)

    def __str__(self):
        """ Print EOS """
        return ' TOV solver with EOS = %s' % self.EOS_name

    def dMdr(self, e, r):
        """ TOV equation 1 """
        return 4.0 * pi * e * r**2.0

    def dPdr(self, M, e, P, r):
        """ TOV equation 2 """
        f1 = e + P / C**2.0
        f2 = M + 4.0 * pi * r**3.0 * P / C**2.0
        f3 = 1.0 - 2.0 * G * M / r / C**2.0
        return - G / r**2.0 * f1 * f2 / f3

    def dMbardr(self, M, n, r):
        """ (3.6f) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        return 4.0*pi * mB * n * r**2.0 / np.sqrt(R2M / L2M(r)) # unit: g/cm

    def domdr(self, ombar, r):
        """ (3.6g) in Damour & Esposito-Farese 1996 """
        return ombar

    def dombardr(self, M, p, e, om, ombar, r):
        """ (3.6h) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        ep = e + p / C**2.0  # unit: g/cm^3
        return 4.0*pi * ep * (r*ombar+4.0*om) * r/R2M - 4.0/r*ombar

    def TOV_equation(self, r, y):
        """ TOV equations to feed ode solver """
        P, M, Mbar, om, ombar = y  # two quantities to be integrated
        if P < self.EOS.min_p:
            P = self.EOS.min_p
        if P > self.EOS.max_p:
            P = self.EOS.max_p
        e = self.EOS.p2e(P)
        n = self.EOS.p2n(P)
        dydr = [self.dPdr(M, e, P, r), self.dMdr(e, r), self.dMbardr(M, n, r), self.domdr(ombar, r), self.dombardr(M, P, e, om, ombar, r) ]
        return dydr

    def solout(self, r, y):
        """ Stop condition for ode """
        if y[0] <= self.EOS.min_p or y[0] >= self.EOS.max_p:
            return -1
        else:
            return 0

    def TOV_solver(self, ec, Rmax, verbose=False):
        """ Input ec, solve (M, R) """
        P0, M0, Mbar0, om0 = self.EOS.e2p(ec), 0.0, 0.0, 1.0
        Rini = Rmax * 1.0e-10
        ombar0 = 0.2*Rini*16.0*pi*G/C**4.0 * (ec*C**2.0+P0)
        #print(P0, self.dPdr(M0, ec, P0, Rini), self.dMdr(ec, Rini), Rini)
        solver = sp_ode(self.TOV_equation).set_integrator('dopri5')
        solver.set_solout(self.solout)  # stop condition
        solver.set_initial_value([P0, M0, Mbar0, om0, ombar0], Rini)
        warnings.filterwarnings("ignore", category=UserWarning)
        solver.integrate(Rmax)
        warnings.resetwarnings()

        if verbose:
          if solver.t==Rmax:
             print('\n (-_-) Integration is not ended with solout, ' 'for ec = %.2e \n' % ec)
        
        P_s, M_s, Mbar_s, om_s, ombar_s = solver.y
        R = solver.t 
        J_inf = R**4.0 * ombar_s * C**2.0 /G / 6.0
        om_inf = om_s + 2.0* G*J_inf/R**3.0/C**2.0
        I = 1.0 / (6*om_s*G/(R**4.0*ombar_s*C**2.0) + 2.0*G /(R**3.0*C**2.0) )
        #print( ( ombar_s * R**4.0 * C**2.0 ) / (6.0 * G ) )
        return {'P_s': P_s, 'M_s': M_s, 'Mbar_s': Mbar_s, 'om_s': om_s, 'ombar_s': ombar_s, 'R': R, 'I' : I, 'I_sph': 0.4*M_s*R**2.0, 'J_inf':J_inf, 'om_inf': om_inf, 'neggtt_s': 1.0 - 2.0* G*M_s/C**2.0/R }




t0 = timeit.time.time()
eos_name = 'SLy4'
x = TOV2(eos_name)
Rmax=2.0e6

switch = 0
""" 1 for single solution; 0 for M-R relation """

if switch:

  ec = 1.4*E_NUCL
  res = x.TOV_solver(ec, Rmax, verbose=0)
  print( '%.3e %.3e' % (ec, x.EOS.max_e) )
  print(res)
 
  if 0:
    """ Plot the solution """
    R = res['R']
    Rini = R * 1.0e-10
    solver1 = sp_ode(x.TOV_equation).set_integrator('dopri5')
    solver1.set_initial_value([x.EOS.e2p(ec), 0.0, 0.0, 1.0, 0.2*Rini*16*pi*G/C**4.0 * (ec*C**2.0+x.EOS.e2p(ec))], Rini)
    Npt = 100
    xR = np.linspace(Rini, R, Npt)
    xMbar, xM, xP, xe, xom = np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR)
    xMbar[0], xM[0], xP[0], xe[0], xom[0]= 0.0, 0.0, x.EOS.e2p(ec), ec, 1.0
    f = open("sol.txt", 'w+')
    f.write(('%.16e %.16e %.16e %.16e' % (xR[0], xM[0], xP[0], xe[0] ) ) + '\n')
    for i in range(1, Npt):
        xP[i], xM[i], xMbar[i], xom[i], xe[i] = solver1.integrate(xR[i])
        xe[i] = x.EOS.p2e(xP[i] )
        #print( xR[i], xM[i], xP[i] )
        f.write((' %.16e %.16e %.16e %.16e ' % (xR[i], xM[i], xP[i], xe[i] ) ) + '\n')
    f.close()

    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.set_xlabel(r'$R$', fontsize=20)
    ax1.set_ylabel(r'$M$', fontsize=20)
    ax2.set_xlabel(r'$R$', fontsize=20)
    ax2.set_ylabel(r'$\epsilon$', fontsize=20)
    ax3.set_xlabel(r'$R$', fontsize=20)
    ax3.set_ylabel(r'$\bar M$', fontsize=20)
    ax4.set_xlabel(r'$R$', fontsize=20)
    ax4.set_ylabel(r'$\omega$', fontsize=20)
    ax1.plot(xR/KM, xM )
    ax2.plot(xR/KM, xe )
    ax3.plot(xR/KM, xMbar )
    ax4.plot(xR/KM, xom )
         
     

if not switch:
    """ M-R relations & maximum mass """
    emin, emax = np.max([1.4*E_NUCL, 1.0001*x.EOS.min_e]), 0.9999*x.EOS.max_e
    N=1000
    #ecs = np.linspace(emin, emax, N)
    ecs = np.logspace(np.log10(emin), np.log10(emax), N)
    xM, xR, xMbar, xc, xI= np.zeros_like(ecs), np.zeros_like(ecs), np.zeros_like(ecs), np.zeros_like(ecs), np.zeros_like(ecs)
    f = open("data.txt", 'w+')
    for (i, ec) in enumerate(ecs):
        res = x.TOV_solver(ec, Rmax, verbose = 0)
        xM[i], xMbar[i], xR[i], xI[i] = res['M_s'], res['Mbar_s'], res['R'], res['I'] 
        xc[i] = xM[i]*G / xR[i] / C**2.0
        f.write(('%.6e %.6e %.6e %.6e %.6e' % (ecs[i], xM[i], xMbar[i], xR[i], xI[i]) ) + '\n')
    f.close()
    idxf = np.argmax(xM) 
    idxi = 0
    while xR[idxi]==Rmax:
       idxi = idxi + 1
    xM1, xR1, ecs1, xMbar1, xc1, xI1 = np.zeros(idxf-idxi+1), np.zeros(idxf-idxi+1), np.zeros(idxf-idxi+1), np.zeros(idxf-idxi+1), np.zeros(idxf-idxi+1), np.zeros(idxf-idxi+1)

    for i in range(0, idxf-idxi+1):
        xR1[i], xM1[i], ecs1[i], xMbar1[i], xc1[i], xI1[i] = xR[idxi+i], xM[idxi+i], ecs[idxi+i], xMbar[idxi+i], xc[idxi+i], xI[idxi+i]

    #print( xM[idxf] / MSUN, xM[idxf-1] / MSUN, np.log10(ecs[idxf] * C**2.0) , np.log10(ecs[idxf-1] * C**2.0) )
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.plot(xR/KM , xM/MSUN , marker='o')
    ax2.plot(ecs , xM/MSUN)
    ax3.plot(ecs , xc )
    ax4.plot(xM/MSUN , xMbar/MSUN)
    ax1.set_xlabel(r'$R$', fontsize=20)
    ax1.set_ylabel(r'$M$', fontsize=20)
    ax2.set_xlabel(r'$\epsilon\,[{\rm g\,cm}^{-3}]$', fontsize=20)
    ax2.set_ylabel(r'$M$', fontsize=20)
    ax3.set_xlabel(r'$\epsilon\,[{\rm g\,cm}^{-3}]$', fontsize=20)
    ax3.set_ylabel(r'$C$', fontsize=20)
    ax4.set_xlabel(r'$M$', fontsize=20)
    ax4.set_ylabel(r'$\bar M$', fontsize=20)
  
   
plt.show()

print( '\n *** STG_solver uses %.2f seconds\n' % (timeit.time.time() - t0))

