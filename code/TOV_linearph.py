
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
from scipy.interpolate import interp1d as sp_interp1d

# sys.path.append('../EOS/')
from EOS import EOS
from STG_massive import STG_quadratic as STGq

C = 29979245800.0  # cm/s
G = 6.67408e-8  # cm^3/g/s^2
MSUN = 1.98855e33  # g
KM = 1.0e5  # cm
E_NUCL = 2.0e14  # minimun energy density for NS core; g/cm^3
PI4 = 4.0 * pi
L2M_UNIT = C**2.0 / G  # convert [L] to [M]
lch = 10.0*KM

def L2M(r):
    """ Convert length scale [L] to mass scale [M] """
    return r * L2M_UNIT

class TOV2(object):
    """ TOV equation solver with ode from scipy.integrate """

    def __init__(self, EOS_name='AP4', th=STGq(xi=-4.4,msq=1) ):
        """ Initialize the TOV solver """
        self.EOS_name = EOS_name
        self.EOS = EOS(EOS_name)
        self.STGq = th 

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

    def TOV_equation(self, r, y):
        """ TOV equations to feed ode solver """
        P, M = y  # two quantities to be integrated
        e = self.EOS.p2e(P)
        dydr = [self.dPdr(M, e, P, r), self.dMdr(e, r)]
        return dydr

    def solout(self, r, y):
        """ Stop condition for ode """
        if y[0] <= self.EOS.min_p:
            return -1
        else:
            return 0

    def TOV_solver(self, ec, Rmax, verbose=False):
        """ Input ec, solve (M, R) """
        P0, M0 = self.EOS.e2p(ec), 0.0
        Rini = Rmax * 1.0e-10
        #print(P0, self.dPdr(M0, ec, P0, Rini), self.dMdr(ec, Rini), Rini)
        solver = sp_ode(self.TOV_equation).set_integrator('dopri5')
        solver.set_solout(self.solout)  # stop condition
        solver.set_initial_value([P0, M0], Rini)
        warnings.filterwarnings("ignore", category=UserWarning)
        solver.integrate(Rmax)
        warnings.resetwarnings()

        if verbose:
          if solver.t==Rmax:
             print('\n (-_-) Integration is not ended with solout, ' 'for ec = %.2e \n' % ec)

        return (solver.y[0], solver.y[1] , solver.t)


    def dvarphidr(self, r, psi):

        return psi

    def dpsidr(self, r, varphi, psi, M, p, e):

        R2M = L2M(r) - 2.0 * M  # unit: g
        dUdph = self.STGq.dUdph(varphi)

        e_3p = e - 3.0 * p / C**2.0  # unit: g/cm^3
        e_p = e - p / C**2.0  # unit: g/cm^3
        R1M = L2M(r) - M  # unit: g
        return PI4 * r / R2M * ( -2.0* xi*varphi * e_3p   + r * psi * e_p) - \
            2.0 * R1M / r / R2M * psi + r / (G*R2M/C**2.0) * msq*varphi/lch**2.0   # unit:cm^-2





t0 = timeit.time.time()
eos_name = 'MPA1'
xi = 3.0
msq = 0.5
x = TOV2(eos_name, th=STGq(xi,msq) )
Rmax=2.0e6


xiset=[3]
msqset=[0.5**2.0 ]

switch = 1
""" 1 for single solution; 0 for dphdr/ph - M/R relation """

if switch:

    ec = 1.4*E_NUCL

    R = x.TOV_solver(ec, Rmax, verbose=True)[2]
    Rini = R * 1.0e-10
    Npt = 1000
    xR = np.linspace(Rini, R, Npt)
    xM, xP, xe, xvarphi, xpsi = np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR)
    xM[0], xP[0], xe[0] = 0.0, x.EOS.e2p(ec), ec
    solver1 = sp_ode(x.TOV_equation).set_integrator('dopri5')
    solver1.set_initial_value( [xP[0], xM[0]], xR[0] )

    for i in range(1, Npt):
      xP[i], xM[i]= solver1.integrate(xR[i])
      xe[i] = x.EOS.p2e(xP[i] )
      #print( xR[i], xM[i], xP[i] )
    mr = sp_interp1d(xR, xM)
    pr = sp_interp1d(xR, xP)
    er = sp_interp1d(xR, xe)

    def ph_linear(r, y):
      M = mr(r)
      p = pr(r)
      e = er(r)
      dydr = [ x.dvarphidr(r, y[1]), x.dpsidr(r, y[0], y[1], M, p, e)]
      return dydr

    xvarphi[0] = 0.01
    xpsi[0] = Rini*xvarphi[0]*(msq - 8.0*pi* xi*lch**2.0 * ( er(Rini)-3.0*pr(Rini)/C**2.0 )*G/C**2.0 ) /3.0 / lch**2.0
    #print(xvarphi[0], xpsi[0], Rini)
    solver2 = sp_ode(ph_linear).set_integrator('dopri5')
    solver2.set_initial_value([ xvarphi[0], xpsi[0] ], Rini) 
    for i in range(1, Npt):
      xvarphi[i], xpsi[i]= solver2.integrate(xR[i])
      #print( solver2.integrate(xR[i]) )


    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(xR/KM, xM )
    ax2.plot(xR/KM, xvarphi )
    #print(  xpsi[Npt-1] /xvarphi[Npt-1] * R )

         

if not switch:
    dat = np.genfromtxt('TOV_4eqs_mpa1.txt')
    col0, col3 = dat[:, 0], dat[:, 3]
    N=len(col0)

    phpph = np.zeros( (len(xiset), len(msqset), N) )
   
    f = open("ypintdata.txt", 'w+')
    for i in range(0, N):
      Rini = col3[i] * 1.0e-10
      Npt = 100
      xR = np.linspace(Rini, col3[i], Npt)
      xM, xP, xe = np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR)

      solver1 = sp_ode(x.TOV_equation).set_integrator('dopri5')
      solver1.set_initial_value( [x.EOS.e2p(col0[i]), 0.0], Rini )

      for j in range(1, Npt):
        xP[j], xM[j]= solver1.integrate(xR[j])
        xe[j] = x.EOS.p2e(xP[j] )
      #print( Radi[i], Mass[i] )
      mr = sp_interp1d(xR, xM)
      pr = sp_interp1d(xR, xP)
      er = sp_interp1d(xR, xe)
     
      for it in range(0,len(xiset) ):
        xi = xiset[it]
        for jt in range(0, len(msqset) ):
          msq = msqset[jt]     
          x = TOV2(eos_name, th=STGq(xi,msq) )
          def ph_linear(r, y):
            M = mr(r)
            p = pr(r)
            e = er(r)
            dydr = [ x.dvarphidr(r, y[1]), x.dpsidr(r, y[0], y[1], M, p, e)]
            return dydr

          xvarphi0 = 0.1
          xpsi0 = Rini*xvarphi0*(msq - 8.0*pi* xi*lch**2.0 * ( er(Rini)-3.0*pr(Rini)/C**2.0 )*G/C**2.0 ) /3.0 / lch**2.0
          #print(xvarphi0, xpsi0, Rini)
          solver2 = sp_ode(ph_linear).set_integrator('dopri5')
          solver2.set_initial_value([ xvarphi0, xpsi0 ], Rini) 
          xvarphied, xpsied= solver2.integrate(col3[i])
          phpph[it, jt, i] = xpsied/xvarphied * lch
          #print( solver2.integrate(xR[i]) )
          f.write(('%.16e %.16e' % (phpph[it, jt, i] ) ) + '\n')
    
    f.close() 

   
plt.show()

print( '\n *** STG_solver uses %.2f seconds\n' % (timeit.time.time() - t0))



