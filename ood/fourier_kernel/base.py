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
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

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

		self.dim_in = dim
		self.spectral_density = spectral_density

		self.spectral_values = None
		self.varphi_values = None
		self.dw_voxel_vec = None
		self.omega_locations = None
		
	def update_integration_parameters(self,omega_locations,dw_voxel_vec,dX_voxel_vec):

		assert omega_locations.shape[1] == self.dim_in
		assert omega_locations.shape[0] == dw_voxel_vec.shape[0]
		assert dw_voxel_vec.shape[1] == 1
		assert dX_voxel_vec.shape[1] == 1

		# Update integration dX voxels for computing S(w) and varphi(w)
		self.spectral_density.update_integration_dX_voxels(dX_voxel_new=dX_voxel_vec)
		
		# Compute S(w) varphi(w)
		self.spectral_values, self.varphi_values = self.spectral_density.unnormalized_density(omega_locations) # [Nomegas,1], [Nomegas,1]
		
		# Store:
		self.dw_voxel_vec = dw_voxel_vec # [Nomegas,1]
		self.omega_locations = omega_locations


	def get_fx_with_variable_integration_step(self,xpred):
		
		assert self.spectral_values is not None, "Call self.update_integration_parameters() first"
		assert self.varphi_values is not None, "Call self.update_integration_parameters() first"
		assert self.dw_voxel_vec is not None, "Call self.update_integration_parameters() first"
		assert self.omega_locations is not None, "Call self.update_integration_parameters() first"

		integrand_xpred = self.get_features_mat(xpred) * tf.transpose(self.spectral_values*self.dw_voxel_vec) # [Npoints_x, Npoints_w]
		fx_vec = tf.reduce_sum(integrand_xpred,axis=1,keepdims=True) # [Npoints_x, 1]

		# Reset to None to ensure that self.update_integration_parameters() is being called
		# NOTE: Change this by having a "allow_parameters_update" flag
		self.spectral_values = None
		self.varphi_values = None
		self.dw_voxel_vec = None
		self.omega_locations = None

		return fx_vec




	# def update_spectral_density_and_angle(self,omegapred,Dw=None):
	def update_spectral_density_and_angle(self,omegapred,Dw,dX=None):

		raise NotImplementedError("Deprecated")


		# logger.info("Storing grid of omegas ...")

		assert omegapred.shape[1] == self.dim_in

		self.omega_locations = omegapred
		self.Dw = Dw



		# Deprecated
		# ==========
		# # dbg_flag = False
		# if omegapred is None:

		# 	Ndiv = 101
		# 	L = 10.0

		# 	# Random grid using uniform/sobol randomization:
		# 	min_omega = -((Ndiv-1) //2) * (math.pi/L)
		# 	max_omega = +((Ndiv-1) //2) * (math.pi/L)
		# 	omegapred = min_omega + (max_omega - min_omega)*tf.math.sobol_sample(dim=self.dim_in,num_results=(Ndiv**self.dim_in),skip=2000 + 100*np.random.randint(0,100))
		# 	self.omega_locations = omegapred
		# 	self.Dw = (math.pi/L)**self.dim_in
		# elif omegapred.shape[0] > 1:
		# 	# pdb.set_trace()
			
		# 	self.omega_locations = omegapred
		# 	omega_min = tf.reduce_min(omegapred)
		# 	omega_max = tf.reduce_max(omegapred)
		# 	print("omega_min:",omega_min)
		# 	print("omega_max:",omega_max)
		# 	self.Dw = (omega_max - omega_min)**self.dim_in / omegapred.shape[0]

		# else:

		# 	# dbg_flag = True

		# 	# Only one omega (used only in reconstruction loss analysis)
		# 	assert omegapred.shape[1] == self.dim_in
		# 	self.omega_locations = omegapred
		# 	self.Dw = 1.0
		# 	# Note: for the reconstruction loss analysis, it doesn't matter the value of self.Zs, because it's not used when calling get_fx()
			

		if dX is not None:
			self.spectral_density.update_dX_voxels(dX_new=dX)

		# logger.info("Evaluating density ...")
		Sw_vec, phiw_vec = self.spectral_density.unnormalized_density(self.omega_locations) # [Nomegapred,1], [Nomegapred,1]

		
		# # Select channel:
		# self.spectral_values = Sw_vec[:,self.dim_out_ind:self.dim_out_ind+1]
		# self.varphi_values = phiw_vec[:,self.dim_out_ind:self.dim_out_ind+1]
		# # pdb.set_trace()

		self.spectral_values = Sw_vec
		self.varphi_values = phiw_vec
		

		"""
		
		# Make it stationary
		# ==================

		Sw_vec_np = self.spectral_values.numpy()[0:-1]
		ind_mid = Sw_vec_np.shape[0]//2
		Sw_vec_np[ind_mid::,0] = Sw_vec_np[0:ind_mid,0]

		if tf.math.reduce_all(self.varphi_values == 0.0):
			phiw_vec_np = np.zeros((self.spectral_values.shape[0]-1,1),dtype=np.float32)
		else:
			phiw_vec_np = self.varphi_values.numpy()[0:-1]
		phiw_vec_np[0:ind_mid,0] = 0.0
		phiw_vec_np[ind_mid::,0] = -math.pi/2.

		omegapred_np = self.omega_locations.numpy()[0:-1]
		omegapred_np[ind_mid::,0] = omegapred_np[0:ind_mid,0]

		self.spectral_values = Sw_vec_np
		self.varphi_values = phiw_vec_np
		self.omega_locations = omegapred_np

			
		"""



		"""
		
		# Compute Zs


		if dbg_flag ==True:
			self.Zs = None
		else:

			# Extarct normalization constant for selected channel dim_out_ind:
			logger.info("Getting normalizationconstat")
			Zs = self.spectral_density.get_normalization_constant_numerical(self.omega_locations) # [self.dim_in,]
			self.Zs = Zs[self.dim_out_ind:self.dim_out_ind+1]

			# self.prior_var_factor = 1./(tf.reduce_max(self.S_samples_vec)*((self.nu) / (self.nu - 2.) ) * self.Nfeat) * cfg.hyperpars.prior_variance
			self.Zs = self.Zs*(1./self.Dw)

		"""
		self.Zs = None

		# logger.info("Done!")


	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		WX = X @ tf.transpose(self.omega_locations) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.varphi_values)) # [Npoints, Nfeat]

		return harmonics_vec


	def _inverse_fourier_transform(self,xpred,fomega_callable):
		"""

		xpred: [Npoints,self.dim_in]
		"""

		raise NotImplementedError("Deprecated")

		if self.omega_locations is None or self.varphi_values is None or self.spectral_values is None:
			raise NotImplementedError("You need to call InverseFourierTransformKernelToolbox.update_spectral_density_and_angle() first!!!")


		integrand_xpred = fomega_callable(xpred) * tf.transpose(self.spectral_values) # [Npoints_x, Npoints_w]		
		out_fun_x = tf.reduce_sum(integrand_xpred,axis=1) # [Npoints,]

		return out_fun_x

	def get_fx(self,xpred):

		raise NotImplementedError("Deprecated")

		if self.omega_locations is None or self.varphi_values is None or self.spectral_values is None:
			raise NotImplementedError("You need to call InverseFourierTransformKernelToolbox.update_spectral_density_and_angle() first!!!")


		fomega_callable = lambda x_in: self.get_features_mat(x_in) * self.Dw
		fx = self._inverse_fourier_transform(xpred,fomega_callable)

		# pdb.set_trace()

		return fx

	def get_kernel_diagonal(self,xpred):
		"""

		xpred: [Npoints,self.dim_in]
		"""

		raise NotImplementedError("Deprecated")

		raise NotImplementedError("We are not computing Zs for now, so this method can't be called until we figure that out...")

		fomega_callable = lambda x_in: self.get_features_mat(x_in)**2
		ker_x = self._inverse_fourier_transform(xpred,fomega_callable) / self.Zs

		return ker_x


	def get_fx_integrand(self,xpred,Dw):
		"""
		xpred: [Npoints,self.dim_in]
		self.omega_locations: [Nomegas,self.dim_in]
		self.spectral_values: [Nomegas,1]
		self.get_features_mat(xpred): [Npoints, Nomegas]

		integrand_xpred: [Npoints, Nomegas]
		"""

		raise NotImplementedError("Deprecated")

		if self.omega_locations is None or self.varphi_values is None or self.spectral_values is None:
			raise NotImplementedError("You need to call InverseFourierTransformKernelToolbox.update_spectral_density_and_angle() first!!!")

		assert self.omega_locations.shape[1] == xpred.shape[1] == self.dim_in

		integrand_xpred = Dw * self.get_features_mat(xpred) * tf.transpose(self.spectral_values) # [Npoints_x, Npoints_w]
		# pdb.set_trace()
		return integrand_xpred

	def get_fx_integrand_variable_voxels(self,xpred,Dw_vec):
		"""
		xpred: [Npoints,self.dim_in]
		Dw_vec: [Nomegas,]
		self.omega_locations: [Nomegas,self.dim_in]
		self.spectral_values: [Nomegas,1]
		self.get_features_mat(xpred): [Npoints, Nomegas]

		integrand_xpred: [Npoints, Nomegas]
		"""

		raise NotImplementedError("Deprecated")

		assert self.Dw is None, "this assetion is not necessary, but requiring it enhances clarity, since we'll not use self.Dw"

		if self.omega_locations is None or self.varphi_values is None or self.spectral_values is None:
			raise NotImplementedError("You need to call InverseFourierTransformKernelToolbox.update_spectral_density_and_angle() first!!!")

		assert self.omega_locations.shape[1] == xpred.shape[1] == self.dim_in

		integrand_xpred = self.get_features_mat(xpred) * tf.transpose(self.spectral_values) * tf.expand_dims(Dw_vec,axis=0) # [Npoints_x, Npoints_w]
		# pdb.set_trace()
		return integrand_xpred

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
