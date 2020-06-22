
""" A quadratic scalar-tensor theory
"""

__author__ = "Lijing Shao"
__email__ = "Friendshao@gmail.com"
__license__ = "GPL"

import numpy as np

class STG_quadratic(object):
    """ The quadratic scalar-tensor theory defined in
        Damour & Esposito-Farese 1996
    """

    def __init__(self, xi=-4.4, msq = 1):
        """ The curvature parameter in the coupling function """
        self.xi = xi
        self.msq = msq

    def U(self, bphi):
        """ dimensionless potential """
        return self.msq*bphi**2.0

    def dUdph(self, bphi):
        """ dimensionless potential """
        return 2.0*self.msq*bphi

    def A(self, bphi):
        return 1.0/np.sqrt(1.0+self.xi*bphi**2.0)

    def alpha(self, bphi):
        """ d[ln(A)]/d[bphi] """
        return -( self.xi*bphi )/(1.0+self.xi*bphi**2.0) 

    def phtranbph(self, bph):
        """ph in terms of bph without the integral constant"""
        return ( np.sqrt(self.xi*(1 + 6*self.xi)) * np.log(1 + 2*np.sqrt(self.xi*(1 + 6*self.xi))* bph*(np.sqrt(1 + (np.sqrt(self.xi*(1 + 6*self.xi)))**2*bph**2) + np.sqrt(self.xi*(1 + 6*self.xi))*bph)) + np.sqrt(6)*self.xi*np.log( 1 - 2*np.sqrt(6)*self.xi*bph*(np.sqrt(1 + (np.sqrt(self.xi*(1 + 6*self.xi)))**2*bph**2) - np.sqrt(6)*self.xi*bph)/( 1 + self.xi*bph**2)) ) /2.0/np.sqrt(2)/self.xi

    def dphsq(self, bphi):
        """( d[phi]/d[bphi] )^2"""
        return ( 1.0 + (1.0+6.0*self.xi)*self.xi*bphi**2.0 )/( 2.0*(1.0+self.xi*bphi**2.0)**2.0 )

    def ddphsq(self, bphi):
        """ d/d[bphi] ( d[phi]/d[bphi] )^2"""
        return - ( self.xi*bphi*( 1.0 - 6.0*self.xi + (1.0+6.0*self.xi)*self.xi*bphi**2.0 ) )/( (1.0+self.xi*bphi**2.0)**3.0 )










