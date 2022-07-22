import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from scipy import integrate
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParabolaSpectralDensity
import hydra

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


class InverseFourierTransformKernelToolbox():

	def __init__(self, spectral_density, dim):

		self.dim = dim
		self.spectral_density = spectral_density

		# Get density and angle:
		omega_min = -40.
		omega_max = +40.
		Ndiv = 8001
		omegapred = np.linspace(omega_min,omega_max,Ndiv)
		self.Dw = omegapred[1] - omegapred[0]
		self.omegapred = np.reshape(omegapred,(-1,1))
		self.Sw_vec, self.phiw_vec = self.spectral_density.unnormalized_density(self.omegapred)

		self.volume_w = omega_max - omega_min
		self.volume_w = 1.0

		# self.Sw_vec = self.Sw_vec / np.amax(self.Sw_vec) * 2.0 # Kink
		# self.Sw_vec = self.Sw_vec / np.amax(self.Sw_vec) * 7.0 # Parabola
		# self.Sw_vec = self.Sw_vec / np.amax(self.Sw_vec) * (2.*math.pi) # Parabola
		# self.Sw_vec = self.Sw_vec / integrate.trapezoid(y=self.Sw_vec,dx=self.Dw) * 20.0

	def get_features_mat(self,x_in,const=0.0):
		"""

		x_in: [Npoints_x,self.dim]
		omega_vec: [Npoints_w,self.dim]
		Sw_vec: [Npoints_w,]
		phiw_vec: [Npoints_w,]
		"""

		feat_mat_x = np.cos(x_in @ self.omegapred.T + self.phiw_vec) + const # [Npoints_x,Npoints_w]

		return feat_mat_x

	def _inverse_fourier_transform(self,xpred,fomega_callable):
		"""

		xpred: [Npoints,self.dim]
		"""

		integrand_xpred = fomega_callable(xpred) * self.Sw_vec # [Npoints_x, Npoints_w]
		out_fun_x = integrate.trapezoid(y=integrand_xpred,dx=self.Dw,axis=1) / self.volume_w # [Npoints,]

		return out_fun_x

	def get_fx(self,xpred):

		fomega_callable = lambda x_in: self.get_features_mat(x_in,const=0.0)
		fx = self._inverse_fourier_transform(xpred,fomega_callable)

		return fx

	def get_kernel_diagonal(self,xpred):
		"""

		xpred: [Npoints,self.dim]
		"""

		fomega_callable = lambda x_in: self.get_features_mat(x_in,const=0.0)**2
		ker_x = self._inverse_fourier_transform(xpred,fomega_callable)

		return ker_x


	def get_covariance_diagonal(self,xpred):

		fomega_callable = lambda x_in: self.get_features_mat(x_in,const=1.0)**2

		var_x = self._inverse_fourier_transform(xpred,fomega_callable)

		return var_x


	def get_kernel_full(self,xpred):
		"""

		xpred: [Npoints,self.dim]
		"""
		raise NotImplementedError

	def get_cov_full(self,xpred):
		"""

		xpred: [Npoints,self.dim]
		"""
		raise NotImplementedError
