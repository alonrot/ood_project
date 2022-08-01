import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import tensorflow as tf
import scipy
from scipy import stats
from scipy import integrate
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
		assert self.dim == 1, "Not ready for dim > 1"
		self.spectral_density = spectral_density

		# Get density and angle:
		omega_min = -40.
		omega_max = +40.
		Ndiv = 8001
		self.Sw_vec, self.phiw_vec, self.omegapred = self.spectral_density.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,normalize_density_numerically=False)

		self.volume_w = omega_max - omega_min
		self.Dw = (omega_max-omega_min)/Ndiv
		self.volume_w = 1.0

		# self.Sw_vec = self.Sw_vec / np.amax(self.Sw_vec) * 2.0 # Kink
		# self.Sw_vec = self.Sw_vec / np.amax(self.Sw_vec) * 7.0 # Parabola
		# self.Sw_vec = self.Sw_vec / np.amax(self.Sw_vec) * (2.*math.pi) # Parabola
		# self.Sw_vec = self.Sw_vec / integrate.trapezoid(y=self.Sw_vec,dx=self.Dw) * 20.0

	def get_features_mat(self,X,const=0.0):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		# pdb.set_trace()
		WX = X @ tf.transpose(self.omegapred) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.phiw_vec)) + const # [Npoints, Nfeat]

		return harmonics_vec


	def _inverse_fourier_transform(self,xpred,fomega_callable):
		"""

		xpred: [Npoints,self.dim]
		"""

		integrand_xpred = fomega_callable(xpred) * tf.transpose(self.Sw_vec) # [Npoints_x, Npoints_w]
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
