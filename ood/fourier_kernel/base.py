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

	def __init__(self, spectral_density, dim, dim_out_ind=0):

		self.dim_in = dim
		self.dim_out_ind = dim_out_ind
		# assert self.dim_in == 1, "Not ready for dim > 1"
		self.spectral_density = spectral_density

		# Get density and angle:
		# omega_min = -40.
		# omega_max = +40.
		# Ndiv = 4001 # dim=1
		# self.Sw_vec, self.phiw_vec, self.omegapred = self.spectral_density.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,normalize_density_numerically=True)
		
		Ndiv = 101
		L = 10.0
		Sw_vec, phiw_vec, omegapred = self.spectral_density.get_Wpoints_discrete(L=L,Ndiv=Ndiv,normalize_density_numerically=False)

		self.omegapred = omegapred
		self.Dw = (self.omegapred[1,-1] - self.omegapred[0,-1])**self.dim_in # Equivalent to (math.pi/L)**self.dim for self.spectral_density.get_Wpoints_discrete()
		
		# Select channel:
		self.Sw_vec = Sw_vec[:,self.dim_out_ind:self.dim_out_ind+1]
		self.phiw_vec = phiw_vec[:,self.dim_out_ind:self.dim_out_ind+1]

		# Extarct normalization constant for selected channel dim_out_ind:
		Zs = self.spectral_density.get_normalization_constant_numerical(self.omegapred) # [self.dim_in,]
		self.Zs = Zs[self.dim_out_ind:self.dim_out_ind+1]

	def get_features_mat(self,X,const=0.0):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		WX = X @ tf.transpose(self.omegapred) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.phiw_vec)) + const # [Npoints, Nfeat]

		return harmonics_vec


	def _inverse_fourier_transform(self,xpred,fomega_callable):
		"""

		xpred: [Npoints,self.dim_in]
		"""

		integrand_xpred = fomega_callable(xpred) * tf.transpose(self.Sw_vec) # [Npoints_x, Npoints_w]		
		out_fun_x = tf.reduce_sum(integrand_xpred,axis=1) # [Npoints,]

		return out_fun_x

	def get_fx(self,xpred):

		fomega_callable = lambda x_in: self.get_features_mat(x_in,const=0.0) * self.Dw
		fx = self._inverse_fourier_transform(xpred,fomega_callable)

		return fx

	def get_kernel_diagonal(self,xpred):
		"""

		xpred: [Npoints,self.dim_in]
		"""

		fomega_callable = lambda x_in: self.get_features_mat(x_in,const=0.0)**2
		ker_x = self._inverse_fourier_transform(xpred,fomega_callable) / self.Zs

		return ker_x


	def get_covariance_diagonal(self,xpred):
		raise NotImplementedError

		# fomega_callable = lambda x_in: self.get_features_mat(x_in,const=1.0)**2

		# var_x = self._inverse_fourier_transform(xpred,fomega_callable)

		# return var_x


	def get_kernel_full(self,xpred):
		"""

		xpred: [Npoints,self.dim_in]
		"""
		raise NotImplementedError

	def get_cov_full(self,xpred):
		"""

		xpred: [Npoints,self.dim_in]
		"""
		raise NotImplementedError
