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
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity, KinkSharpSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity
from lqrker.spectral_densities.base import SpectralDensityBase
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from lqrker.utils.common import CommonUtils
import hydra
import pickle
from ood.spectral_density_approximation.elliptical_slice_sampler import EllipticalSliceSampler
from ood.spectral_density_approximation.reconstruct_function_from_spectral_density import ReconstructFunctionFromSpectralDensity
import tensorflow as tf
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# plt.rc('legend',fontsize=fontsize_labels+2)
plt.rc('legend',fontsize=fontsize_labels//2)

@hydra.main(config_path="./config",config_name="config")
def reconstruct(cfg):

	# integration_method = "integrate_with_regular_grid"
	# integration_method = "integrate_with_irregular_grid"
	# integration_method = "integrate_with_bayesian_quadrature"
	integration_method = "integrate_with_data"

	dim_in = 2

	"""
	Create training dataset
	"""
	xmin_training = -10.0
	xmax_training = +10.0
	Ndiv_training = 201
	xpred_training = CommonUtils.create_Ndim_grid(xmin=xmin_training,xmax=xmax_training,Ndiv=Ndiv_training,dim=dim_in) # [Ndiv**dim_in,dim_in]
	fx_true_training = VanDerPolSpectralDensity._controlled_vanderpol_dynamics(state_vec=xpred_training,control_vec="gather_data_policy",use_nominal_model=True)
	# fx_true_training = fx_true_training[:,dim_out_ind:dim_out_ind+1] # [Ndiv,1]
	spectral_density = VanDerPolSpectralDensity(cfg=cfg.spectral_density.vanderpol,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,use_nominal_model=True,Xtrain=xpred_training,Ytrain=fx_true_training)

	"""
	Create testing dataset
	"""
	xmin_testing = -5.0
	xmax_testing = +5.0
	Ndiv_testing = 31
	xpred_testing = CommonUtils.create_Ndim_grid(xmin=xmin_testing,xmax=xmax_testing,Ndiv=Ndiv_testing,dim=dim_in) # [Ndiv**dim_in,dim_in]
	fx_true_testing = VanDerPolSpectralDensity._controlled_vanderpol_dynamics(state_vec=xpred_testing,control_vec="gather_data_policy",use_nominal_model=True)


	"""
	Discrete grid of omega, for plotting and analysis:
	"""

	Ndiv_omega_for_analysis = 71
	omega_lim = 2.0
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_in) # [Ndiv**dim_in,dim_in]
	Dw_analysis = (2.*omega_lim)**dim_in / omegapred_analysis.shape[0]
	Sw_vec, phiw_vec = spectral_density.unnormalized_density(omegapred_analysis)


	COLOR_MAP = "summer"
	hdl_fig, hdl_splots_omegas = plt.subplots(dim_in,2,figsize=(14,10),sharex=False)
	# hdl_fig.suptitle(r"Spectral density $S(\omega) = [S_1(\omega),S_2(\omega)]$ and spectral phase $\varphi(\omega) = [\varphi_1(\omega), \varphi_2(\omega)]$ for {0:s} kernel".format(labels[kk]),fontsize=fontsize_labels)
	
	extent_plot_omegas = [omegapred_analysis[0,0],omegapred_analysis[-1,0],omegapred_analysis[0,1],omegapred_analysis[-1,1]] #  scalars (left, right, bottom, top)
	for jj in range(dim_in):

		raise NotImplementedError("Show the integrand as well! Not just the spectral density...")

		S_vec_plotting = np.reshape(Sw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		hdl_splots_omegas[jj,0].imshow(S_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=S_vec_plotting.min(),vmax=S_vec_plotting.max(),interpolation='nearest')
		my_title = "S_{0:d}(\omega)".format(jj+1)
		hdl_splots_omegas[jj,0].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		if jj == dim_in-1: hdl_splots_omegas[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		hdl_splots_omegas[jj,0].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)

		if np.any(phiw_vec != 0.0):
			for jj in range(dim_in):

				phi_vec_plotting = np.reshape(phiw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
				hdl_splots_omegas[jj,1].imshow(phi_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=phi_vec_plotting.min(),vmax=phi_vec_plotting.max(),interpolation='nearest')
				my_title = "\\varphi_{0:d}(\omega)".format(jj+1)
				hdl_splots_omegas[jj,1].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
				if jj == dim_in-1: hdl_splots_omegas[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
				hdl_splots_omegas[jj,1].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)
		else:
			for jj in range(dim_in): hdl_splots_omegas[jj,1].set_xticks([],[]); hdl_splots_omegas[jj,1].set_yticks([],[])


	"""
	Figure out the scales AND omegas directly:
	"""

	hdl_fig, hdl_splots_statespace = plt.subplots(dim_in,3,figsize=(14,10),sharex=False)
	Nsamples_omega = 60
	Nepochs = 1000
	extent_plot_statespace = [xpred_testing[0,0],xpred_testing[-1,0],xpred_testing[0,1],xpred_testing[-1,1]] #  scalars (left, right, bottom, top)
	for jj in range(dim_in):

		inverse_fourier_toolbox_channel = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in,dim_out_ind=jj)

		reconstructor_fx_deltas_and_omegas = ReconstructFunctionFromSpectralDensity(dim_in=dim_in,omega_lim=omega_lim,Nomegas=Nsamples_omega,
																					inverse_fourier_toolbox=inverse_fourier_toolbox_channel,
																					Xtrain=xpred_testing,Ytrain=fx_true_testing[:,jj:jj+1],
																					omegas_weights=None)
		reconstructor_fx_deltas_and_omegas.train(Nepochs=Nepochs,learning_rate=1e-2,stop_loss_val=0.001)
		fx_optimized_omegas_and_voxels = reconstructor_fx_deltas_and_omegas.reconstruct_function_at(xpred=xpred_testing)
		omegas_trainedNN = reconstructor_fx_deltas_and_omegas.get_omegas_weights()
		fx_integrand_optimized_omegas_and_voxels = reconstructor_fx_deltas_and_omegas.get_integrand_for_pruning(xpred=xpred_testing)


		# True function:
		fx_true_testing_jj = np.reshape(fx_true_testing[:,jj:jj+1],(Ndiv_testing,Ndiv_testing),order="F")
		hdl_splots_statespace[jj,0].imshow(fx_true_testing_jj,extent=extent_plot_statespace,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=fx_true_testing_jj.min(),vmax=fx_true_testing_jj.max(),interpolation='nearest')
		if jj == 0: hdl_splots_statespace[jj,0].set_title(r"True function $f_1(x_t)$",fontsize=fontsize_labels)
		if jj == 1: hdl_splots_statespace[jj,0].set_title(r"True function $f_2(x_t)$",fontsize=fontsize_labels)

		# Reconstructed function:
		fx_optimized_omegas_and_voxels_jj = np.reshape(fx_optimized_omegas_and_voxels,(Ndiv_testing,Ndiv_testing),order="F")
		hdl_splots_statespace[jj,1].imshow(fx_optimized_omegas_and_voxels_jj,extent=extent_plot_statespace,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=fx_optimized_omegas_and_voxels_jj.min(),vmax=fx_optimized_omegas_and_voxels_jj.max(),interpolation='nearest')
		if jj == 0: hdl_splots_statespace[jj,1].set_title(r"Reconstructed function $\hat{f}_1(x_t)$",fontsize=fontsize_labels)
		if jj == 1: hdl_splots_statespace[jj,1].set_title(r"Reconstructed function $\hat{f}_2(x_t)$",fontsize=fontsize_labels)

		# Reconstruction error:
		error_reconstruction = abs(fx_true_testing[:,jj:jj+1] - fx_optimized_omegas_and_voxels)
		error_reconstruction_jj = np.reshape(error_reconstruction,(Ndiv_testing,Ndiv_testing),order="F")
		hdl_splots_statespace[jj,2].imshow(error_reconstruction_jj,extent=extent_plot_statespace,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=error_reconstruction_jj.min(),vmax=error_reconstruction_jj.max(),interpolation='nearest')
		mse = np.sqrt(np.mean(error_reconstruction**2))
		logger.info("Mean Squared Error: {0:f}".format(mse))
		if jj == 0: hdl_splots_statespace[jj,2].set_title("Reconstruction MSE {0:f}".format(mse),fontsize=fontsize_labels)
		if jj == 1: hdl_splots_statespace[jj,2].set_title("Reconstruction MSE {0:f}".format(mse),fontsize=fontsize_labels)


		if jj == 1:
			hdl_splots_statespace[jj,0].set_xlabel(r"$x_1$",fontsize=fontsize_labels)
			hdl_splots_statespace[jj,1].set_xlabel(r"$x_1$",fontsize=fontsize_labels)
			hdl_splots_statespace[jj,2].set_xlabel(r"$x_1$",fontsize=fontsize_labels)

		hdl_splots_statespace[jj,0].set_ylabel(r"$x_2$",fontsize=fontsize_labels)

		del inverse_fourier_toolbox_channel
		del reconstructor_fx_deltas_and_omegas

		hdl_splots_omegas[jj,0].plot(omegas_trainedNN[:,0],omegas_trainedNN[:,1],marker="*",color="indigo",markersize=5,linestyle="None")
		hdl_splots_omegas[jj,1].plot(omegas_trainedNN[:,0],omegas_trainedNN[:,1],marker="*",color="indigo",markersize=5,linestyle="None")


	plt.show(block=True)


if __name__ == "__main__":

	reconstruct()

