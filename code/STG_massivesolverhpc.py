
""" Solve Tolman-Oppenheimer-Volkoff equations with ode from scipy

    TOV equations are (3.6a-f) in Damour & Esposito-Farese 1996

    * Note: this script is using cgs units -

      From "Einstein units" to cgs units, one needs the following replacements
        - tilde_epsilon * (G/C^4) -> e
        - tilde_p * (G/C^2) -> p
"""

__author__ = "Lijing Shao"
__email__ = "Friendshao@gmail.com"
__license__ = "GPL"

import sys
import warnings
import timeit
import numpy as np
import scipy.optimize
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import ode as sp_ode


from EOS import EOS
from STG_massive import STG_quadratic as STGq

PI4 = 4.0 * pi
PI8 = 8.0 * pi
C = 29979245800.0  # cm/s
G = 6.67408e-8  # cm^3/g/s^2
G_STAR = 6.67408e-8  # cm^3/g/s^2
L2M_UNIT = C**2.0 / G_STAR  # convert [L] to [M]
MSUN = 1.98855e33  # g
KM = 1.0e5  # cm
mB = 1.660538921e-24  # g
E_NUCL = 2.0e14  # minimun energy density for NS core; g/cm^3

lch = 10.0*KM

# e: g/cm^3, p: g/cm/s^2
def L2M(r):
    """ Convert length scale [L] to mass scale [M] """
    return r * L2M_UNIT


class STG_solver(object):
    """ Numerically solve TOV equations of scalar-tensor gravity in
        Damour & Esposito-Farese 1996
    """

    def __init__(self, EOS_name='AP4', th=STGq(xi=4.4,msq=1)):
        """ Initialize EOS and STG """
        self.EOS_name = EOS_name
        self.EOS = EOS(EOS_name)
        self.STGq = th  # scalar-tensor gravity theory

    def __str__(self):
        """ Print EOS name """
        return (' TOV solver for scalar-tensor gravity with EOS = %s' %
            self.EOS_name)

    def dMdr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6a) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        A = self.STGq.A(varphi)
        dphsq=self.STGq.dphsq(varphi)
        return PI4 * r**2.0 * A**4.0 * e + 0.5 * r * R2M * dphsq * psi**2.0 + (C**2.0/G) * r**2.0*A**4.0* (U/lch**2.0) / 4.0 # unit: g/cm

    def dnudr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6b) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        A = self.STGq.A(varphi)
        dphsq=self.STGq.dphsq(varphi)
        return PI8 * r**2.0 * A**4.0 * (p / C**2.0) / R2M + \
            r * dphsq * psi**2.0 + 2.0 * M / r / R2M - 0.5*r**2.0*A**4.0 * (U/lch**2.0) / (G*R2M/C**2.0) # unit:cm^-1

    def dvarphidr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6c) in Damour & Esposito-Farese 1996 """
        return psi

    def dpsidr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6d) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        dUdph = self.STGq.dUdph(varphi)
        A = self.STGq.A(varphi)
        alpha = self.STGq.alpha(varphi)
        dphsq=self.STGq.dphsq(varphi)
        ddphsq=self.STGq.ddphsq(varphi)
        e_3p = e - 3.0 * p / C**2.0  # unit: g/cm^3
        e_p = e - p / C**2.0  # unit: g/cm^3
        R1M = L2M(r) - M  # unit: g
        return PI4 * r * A**4.0 / R2M * ( alpha * e_3p / dphsq  + r * psi * e_p) - \
            2.0 * R1M / r / R2M * psi - psi**2.0 * ddphsq/2.0/dphsq + r * A**4.0 / (G*R2M/C**2.0) * (0.5*r*(U/lch**2.0)*psi + (alpha*(U/lch**2.0) + 0.25*(dUdph/lch**2.0) )/dphsq)  # unit:cm^-2

    def dpdr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6e) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        A = self.STGq.A(varphi)
        alpha = self.STGq.alpha(varphi)
        dphsq=self.STGq.dphsq(varphi)
        ep = e + p / C**2.0  # unit: g/cm^3
        return -(ep * C**2.0) * (PI4 * r**2.0 * A**4.0 * (p / C**2.0) / R2M + \
            0.5 * r * dphsq * psi**2.0 + M / r / R2M + alpha * psi - 0.25*r**2.0/(G*R2M/C**2.0)*A**4.0*(U/lch**2.0)  ) #unit: g/cm^2/s^2

    def dMbardr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6f) in Damour & Esposito-Farese 1996 """
        A = self.STGq.A(varphi)
        R2M = L2M(r) - 2.0 * M  # unit: g
        return PI4 * mB * n * A**3.0 * r**2.0 / np.sqrt(R2M / L2M(r)) # unit: g/cm

    def domdr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6g) in Damour & Esposito-Farese 1996 """
        return ombar

    def dombardr(self, r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar):
        """ (3.6h) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        A = self.STGq.A(varphi)
        dphsq=self.STGq.dphsq(varphi)
        ep = e + p / C**2.0  # unit: g/cm^3
        return PI4 * A**4 * ep * (r*ombar+4*om) * r/R2M + (r*dphsq*psi**2.0-4/r)*ombar # unit: cm^-2

    def TOV_equation(self, r, y):
        """ TOV equations to feed ode solver: with variable r """
        M, nu, varphi, psi, p, Mbar, om, ombar = y
        if p < self.EOS.min_p:
            p = self.EOS.min_p
        if p > self.EOS.max_p:
            p = self.EOS.max_p
        e = self.EOS.p2e(p)
        n = self.EOS.p2n(p)
        dydr = [self.dMdr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.dnudr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.dvarphidr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.dpsidr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.dpdr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.dMbardr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.domdr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar),
                self.dombardr(r, M, Mbar, nu, varphi, psi, p, e, n, om, ombar)]
        return dydr

    def solout(self, r, y):
        """ Stop condition for ode """
        if y[4] <= self.EOS.min_p or y[4] >= self.EOS.max_p:
            return -1
        else:
            return 0

    def extdMdr(self, r, M, nu, varphi, psi, om, ombar):
        """ (3.6a) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        A = self.STGq.A(varphi)
        dphsq=self.STGq.dphsq(varphi)
        return 0.5 * r * R2M * dphsq * psi**2.0 + (C**2.0/G) * r**2.0*A**4.0* (U/lch**2.0) / 4.0

    def extdnudr(self, r, M, nu, varphi, psi, om, ombar):
        """ (3.6b) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        A = self.STGq.A(varphi)
        dphsq=self.STGq.dphsq(varphi)
        return r * dphsq * psi**2.0 + 2.0 * M / r / R2M - 0.5*r**2.0*A**4.0 * (U/lch**2.0) / (G*R2M/C**2.0)

    def extdvarphidr(self, r, M, nu, varphi, psi, om, ombar):
        """ (3.6c) in Damour & Esposito-Farese 1996 """
        return psi

    def extdpsidr(self, r, M, nu, varphi, psi, om, ombar):
        """ (3.6d) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        U = self.STGq.U(varphi)
        dUdph = self.STGq.dUdph(varphi)
        A = self.STGq.A(varphi)
        alpha = self.STGq.alpha(varphi)
        dphsq=self.STGq.dphsq(varphi)
        ddphsq=self.STGq.ddphsq(varphi)
        R1M = L2M(r) - M  # unit: g
        return - 2.0 * R1M / r / R2M * psi - psi**2.0 * ddphsq/2.0/dphsq + r * A**4.0 / (G*R2M/C**2.0) * (0.5*r*(U/lch**2.0)*psi + (alpha*(U/lch**2.0) + 0.25*(dUdph/lch**2.0) )/dphsq)

    def extdomdr(self, r, M, nu, varphi, psi, om, ombar):
        """ (3.6g) in Damour & Esposito-Farese 1996 """
        return ombar

    def extdombardr(self, r, M, nu, varphi, psi, om, ombar):
        """ (3.6h) in Damour & Esposito-Farese 1996 """
        R2M = L2M(r) - 2.0 * M  # unit: g
        A = self.STGq.A(varphi)
        dphsq=self.STGq.dphsq(varphi)
        return (r*dphsq*psi**2.0-4/r)*ombar

    def ext_equation(self, r, y):
        """ TOV equations to feed ode solver: with variable r """
        M, nu, varphi, psi, om, ombar = y
        dydr = [self.extdMdr(r, M, nu, varphi, psi, om, ombar),
                self.extdnudr(r, M, nu, varphi, psi, om, ombar),
                self.extdvarphidr(r, M, nu, varphi, psi, om, ombar),
                self.extdpsidr(r, M, nu, varphi, psi, om, ombar),
                self.extdomdr(r, M, nu, varphi, psi, om, ombar),
                self.extdombardr(r, M, nu, varphi, psi, om, ombar)]
        return dydr

    def TOV_solver(self, e_c, varphi_c, Rmax=2.0e6, verbose=True):
        """ Input e_c, varphi_c, solve the TOV equation
            Initial conditions see (3.14) in DE96
            Surface quantities see (3.13) in DE96

            Return {m_A, m_A_bar, R, e_c, varphi_c, varphi_0, alpha_A}
              - if pathological problems happen, dict() is returned
        """

        R_ini = Rmax * 1.0e-10
        M0, nu0, varphi0, p0, Mbar0, om0= 0., 0., varphi_c, self.EOS.e2p(e_c), 0., 1.0
        U_c = self.STGq.U(varphi_c)
        dUdph_c = self.STGq.dUdph(varphi_c)
        A_c = self.STGq.A(varphi_c)
        alpha_c = self.STGq.alpha(varphi_c)
        dphsq_c = self.STGq.dphsq(varphi_c)
        e_3p_c = e_c - 3.0 * p0 / C**2.0
        ep_c = e_c + p0 / C**2.0  # unit: g/cm^3
        psi0 = R_ini /3.0 *( PI4 * A_c**4.0 * alpha_c * e_3p_c * G_STAR / C**2.0 + A_c**4.0*alpha_c* (U_c/lch**2.0) + A_c**4.0 * (dUdph_c/lch**2.0 ) / 4.0 ) / dphsq_c 
        ombar0 = 4*PI4 / 5.0 * R_ini * A_c**4.0 * ep_c * om0 * G_STAR/C**2.0
        # Use explicit Runge-Kutta method of order 4(5)
        solver = sp_ode(self.TOV_equation).set_integrator('dopri5')
        solver.set_solout(self.solout)  # stop condition
        solver.set_initial_value([M0, nu0, varphi0, psi0, p0, Mbar0, om0, ombar0], R_ini)
        warnings.filterwarnings("ignore", category=UserWarning)
        solver.integrate(Rmax)
        #print( solver.t )
        #warnings.resetwarnings()

        if not solver.successful():
            print('\n *** ERROR: integration fails for ec = %.6e, varphi_c = '
                    '%.6e\n' % (e_c, varphi_c))
            return dict()
        if solver.t >= Rmax * 0.999:  # not ended with solout
            return self.TOV_solver(e_c, varphi_c, Rmax=1.5 * Rmax,
                                   verbose=verbose)
        M_s, nu_s, varphi_s, psi_s, p_s, Mbar_s, om_s, ombar_s = solver.y
        R = solver.t
        if p_s >= self.EOS.min_p / 0.999:
            print('\n *** ERROR: p_s / min_p = %.2e, p_s / max_p = %.2e' % (
                  p_s / self.EOS.min_p, p_s / self.EOS.max_p))
            return dict()

        if L2M(R) < 2. * M_s:
            print(' *** Warning: black holes!')
            return dict()

        # Use explicit Runge-Kutta method of order 4(5)
        solver2 = sp_ode(self.ext_equation).set_integrator('dopri5')
        solver2.set_initial_value([M_s, nu_s, varphi_s, psi_s, om_s, ombar_s], R)
        inf=2000.0

        Nn = 1000
        xR = np.linspace(R, inf*R, Nn)
        yph, yM, ynu, yom, yombar = np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR), np.zeros_like(xR)
        yph[0], yM[0], ynu[0], yom[0], yombar[0] = varphi_s, M_s, nu_s, om_s, ombar_s
        for i in range(1, Nn):
            yM[i], ynu[i], yph[i], yphp, yom[i], yombar[i] = solver2.integrate(xR[i])
            
        idxm = np.argmin( abs(yph) )
        R_inf = xR[idxm]
        M_inf, nu_inf, varphi_inf, om_inf, ombar_inf = yM[idxm], ynu[idxm], yph[idxm], yom[idxm], yombar[idxm]
  
        #R_inf, nu_inf, om_inf, ombar_inf = xR[Nn-1], -ynu[Nn-1], yom[Nn-1], yombar[Nn-1]
        difnu = np.log(1.0 - 2.0 * G* M_inf/R_inf/C**2.0) - nu_inf
        J_s = ( ombar_s * R**4.0 * C**2.0 ) / (6.0 * G * np.sqrt( np.exp(nu_s - nu_inf) /(1-2.0*G*M_s/R/C**2.0) ) )
        J_inf = ( ombar_inf * R_inf**4.0 * C**2.0 ) / ( 6.0 * G )
        neggtt = ( self.STGq.A(varphi_s) )**2.0 * np.exp(nu_s+difnu)

        res = { 'm_A' : M_inf, 'm_A_bar' : Mbar_s, 'M_s' : M_s, 'R' : R, 'R_inf': R_inf, 'e_c' : e_c, 'varphi_c' : varphi_c , 'varphi_inf': varphi_inf, 'varphi_inftest' : yph[Nn-1],  'om_inf': om_inf, 'nu_inf':nu_inf, 'difnu': difnu, 'J_s': J_s, 'J_inf':J_inf, 'I': J_inf/om_inf, 'neggtt_s': neggtt, 'neggtt_s1': np.exp(nu_s+difnu) , 'varphi_s': varphi_s, 'om_s': om_s, 'ombar_s': ombar_s} 



        if verbose:
            print(('\n === Integration with e_c = %.2e g/cm3 and' % e_c) +
                  ' varphi_c = %.2e ===\n' % varphi_c)
            print('   * R  = %.9f Km' % (R / KM))
            print('   * surface pressure = %.9f ' % ( p_s ) )
            print('   * Mbar = %.9f Msun ' % (Mbar_s /MSUN ) )
            #print(M_s, psi_s * R / varphi_s )

            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
      
            solver = sp_ode(self.TOV_equation).set_integrator('dopri5')
            solver.set_initial_value([M0, nu0, varphi0, psi0, p0, Mbar0, om0, ombar0], R_ini)
            Nnin = 1000
            xRin = np.linspace(R_ini, R, Nnin)
            yMin, yphin, yphpin, ynuin, yomin, ypin, xRinj, neggttinj, grrinj = np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin), np.zeros_like(xRin)
            yMin[0], yphin[0], yphpin[0], ynuin[0], yomin[0], ypin[0] = M0, varphi0, psi0, nu0, om0, p0
            for i in range(1,Nnin):
              result = solver.integrate(xRin[i])
              yMin[i], ynuin[i], yphin[i], yphpin[i], ypin[i], yomin[i] = result[0], result[1], result[2], result[3], result[4], result[6]

            solver2 = sp_ode(self.ext_equation).set_integrator('dopri5')
            solver2.set_initial_value([M_s, nu_s, varphi_s, psi_s, om_s, ombar_s], R)
            Nnex = 1000
            inf2 = 20
            xRex = np.linspace(R, inf2*R, Nnex)
            yMex, ynuex, yphex, yphpex, yomex, xRexj, neggttexj, grrexj = np.zeros_like(xRex), np.zeros_like(xRex), np.zeros_like(xRex), np.zeros_like(xRex), np.zeros_like(xRex), np.zeros_like(xRex), np.zeros_like(xRex), np.zeros_like(xRex)
            yMex[0], ynuex[0], yphex[0], yomex[0], yphpex[0] = M_s, nu_s, varphi_s, om_s, psi_s
            for i in range(1, Nnex):
              result = solver2.integrate(xRex[i])
              yMex[i], ynuex[i], yphex[i], yphpex[i], yomex[i] = result[0], result[1], result[2], result[3], result[4]
              #print( result ) 

            f = open("sol.txt", 'w+')

            for i in range(0, Nnin):
              xRinj[i] = self.STGq.A(yphin[i]) * xRin[i]
              neggttinj[i] = ( self.STGq.A(yphin[i]) )**2.0 * np.exp(ynuin[i]+difnu)
              grrinj[i]=1.0/(1.0+xRin[i]*self.STGq.alpha(yphin[i])*yphpin[i] )**2.0 /(1.0-2.0*G*yMin[i]/xRin[i]/C**2.0)
              f.write(('%.16e %.16e %.16e %.16e %.16e %.16e' % (xRinj[i], neggttinj[i], grrinj[i], yomin[i], ypin[i], yphin[i] ) ) + '\n')
            for i in range(0, Nnex):
              xRexj[i] = self.STGq.A(yphex[i]) * xRex[i]
              neggttexj[i] = ( self.STGq.A(yphex[i]) )**2.0 * np.exp(ynuex[i]+difnu)
              grrexj[i] = 1.0/(1.0+xRex[i]*self.STGq.alpha(yphex[i])*yphpex[i] )**2.0 /(1.0-2.0*G*yMex[i]/xRex[i]/C**2.0)
              f.write(('%.16e %.16e %.16e %.16e %.16e %.16e' % (xRexj[i], neggttexj[i], grrexj[i], yomex[i], 0, yphex[i] ) ) + '\n')

            f.close()

            ax1.plot(xRin/lch, neggttinj, xRex/lch, neggttexj )
            ax2.plot(xRin/lch, grrinj, xRex/lch, grrexj)
            ax3.plot(xRin/lch, yphin, xRex/lch, yphex)
            ax4.plot(xRin/lch, yomin, xRex/lch, yomex)

        return res



    def TOV_iterator(self, e_c, tol=1.0e-9, max_iter=200):

        n_iter = 0
        low = 0.0001
        #while True:
        #    res = self.TOV_solver(e_c, low, verbose=verbose)
        #    n_iter += 1
        #    if len(res) == 0:  # not successful
        #        return dict()
        #    if res['varphi_0'] < varphi_0:
        #        break
        #    low /= 2.
            
        high = 1
        #while True:
        #    res = self.TOV_solver(e_c, high, verbose=verbose)
        #    n_iter += 1
        #    if len(res) == 0:
        #        return dict()
        #    if res['varphi_0'] > varphi_0:
        #        break
        #    high *= 2.

        varphi_c = low
        while abs( varphi_c - np.sqrt(high*low) ) > tol and n_iter < max_iter:
          varphi_c = np.sqrt(high*low)
          res = self.TOV_solver(e_c, varphi_c, verbose=False)
          tail = res['varphi_inftest']
          n_iter +=1
          #print(al, tail)
          if tail > 0.0:
            high = varphi_c
          else:
            low = varphi_c
          
        #print(n_iter, np.sqrt(high*low) )
        res = self.TOV_solver(e_c, np.sqrt(high*low), verbose=False)
        return res


t0 = timeit.time.time()
eos_name = 'SLy4'

switch = 0
""" 1 for e_c range; 0 for shooting """
zz = STG_solver(eos_name, th=STGq(xi=10.0,msq=0.01**2.0))


if switch:
 print( '%.3e' % (zz.EOS.max_e) )
 ec0 = 1.4*E_NUCL
 varphi_infa = zz.TOV_solver(ec0, 0.0001, verbose=0)['varphi_inftest']
 while varphi_infa > 0.0 and ec0 < 50.0e14:
    ec0 = ec0 + 0.05e14
    varphi_infa = zz.TOV_solver(ec0, 0.0001, verbose=0)['varphi_inftest']
 print( '%.3e' % (ec0) )
 
 ec1 = 0.9999*zz.EOS.max_e
 varphi_infa1 = zz.TOV_solver(ec1, 0.0001, verbose=0)['varphi_inftest']
 if varphi_infa1 > 0.0:
  while varphi_infa < 0.0:
     ec0 = np.min([ec0 + 1.0e14, 0.9999*zz.EOS.max_e])
     varphi_infa = zz.TOV_solver(ec0, 0.0001, verbose=0)['varphi_inftest']
  while varphi_infa >0.0:
     ec0 = ec0 - 0.1e14
     varphi_infa = zz.TOV_solver(ec0, 0.0001, verbose=0)['varphi_inftest']
  print( '%.3e' % (ec0-0.1e14) )
 else:
  print('upper value exceeds the maximum e in EOS file')

if not switch:
 Npt = 1000
 ecset = np.linspace(3.15e+14, 0.9999*zz.EOS.max_e, Npt )
 f = open("data2.txt", 'w+')
 for i in range(0, Npt):
   res = zz.TOV_iterator(ecset[i])
   f.write(('%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' % (res['e_c'], res['m_A'], res['m_A_bar'], res['R'], res['varphi_c'], res['I'], res['M_s'], res['neggtt_s'], res['varphi_s']) ) + '\n')  
 f.close()


plt.show()

print( '\n *** STG_solver uses %.2f seconds\n' % (timeit.time.time() - t0))



