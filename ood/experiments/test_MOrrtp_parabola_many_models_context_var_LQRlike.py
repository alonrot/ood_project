import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
from lqrker.models import MultiObjectiveReducedRankProcess
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, ParaboloidSpectralDensity, KinkSpectralDensity
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from ood.spectral_density_approximation.elliptical_slice_sampler import EllipticalSliceSampler
from ood.spectral_density_approximation.reconstruct_function_from_spectral_density import ReconstructFunctionFromSpectralDensity
from lqrker.utils.common import CommonUtils
import numpy as np
import scipy
import hydra
from omegaconf import OmegaConf
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)
import pickle

import GPy

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


def ker_interp(fx_true_testing,xpred_testing,xx):
	"""
	fx_true_testing: [Nfunctions, Npoints, dim_in]
	xpred_testing: [Npoints, dim_in]
	xx: [Npoints_new, dim_in]

	return: [Nfunctions,Npoints,dim_in]
	"""

	dim_in = 1
	assert xx.shape[1] == dim_in
	assert xx.shape[0] > 1

	fx_interp_ii = np.zeros((fx_true_testing.shape[0],xx.shape[0],dim_in))
	for ii in range(fx_true_testing.shape[0]):
		fx_true_testing_ii = fx_true_testing[ii,:,0]
		fx_interp_ii[ii,:,0] = np.interp(xx[:,0],xpred_testing[:,0],fx_true_testing_ii)

	# ker_fx = np.mean(fx_true_testing**2,axis=0)
	# MO_std = np.sqrt(ker_fx)

	return fx_interp_ii


class MyNewKernel(GPy.kern.src.kern.Kern):

	def __init__(self,input_dim,variance=1.,lengthscale=1.,power=1.,active_dims=None,fx_true_testing=None,xpred_testing=None):
		super(MyNewKernel, self).__init__(input_dim, active_dims, 'my_new_ker')
		assert input_dim == 1, "For this kernel we assume input_dim=1"
		self.variance = GPy.core.parameterization.Param('variance', variance)
		self.lengthscale = GPy.core.parameterization.Param('lengtscale', lengthscale)
		self.power = GPy.core.parameterization.Param('power', power)
		self.link_parameters(self.variance, self.lengthscale, self.power)

		assert fx_true_testing is not None
		self.fx_true_testing = fx_true_testing
		self.xpred_testing = xpred_testing

	def K(self,X,X2=None):

		if X2 is None: X2 = X

		fX_allfuns = ker_interp(self.fx_true_testing,self.xpred_testing,X) # [Nfunctions,NpointsX,dim_in]
		fX2_allfuns = ker_interp(self.fx_true_testing,self.xpred_testing,X2) # [Nfunctions,NpointsX2,dim_in]

		fX_times_fX2_allfuns = self.variance*(fX_allfuns[...,0].T @ fX2_allfuns[...,0]) / self.fx_true_testing.shape[0] # [NpointsX,NpointsX2,dim_in]
		# fX_times_fX2_allfuns = self.variance*np.tensordot(fX_allfuns[...,0],fX2_allfuns[...,0],axes=0) / self.fx_true_testing.shape[0] # [NpointsX,NpointsX2,dim_in]

		return fX_times_fX2_allfuns

	def Kdiag(self,X):
		"""
		X: 	[Npoints,dim]
		X2: [Npoints,dim]

		return [Npoints,]
		"""

		fX_allfuns = ker_interp(self.fx_true_testing,self.xpred_testing,X) # [Nfunctions,NpointsX,dim_in]
		fX_times_fX_allfuns = self.variance*np.mean(fX_allfuns[...,0]**2,axis=0)
		return fX_times_fX_allfuns

	def update_gradients_full(self, dL_dK, X, X2):
		if X2 is None: X2 = X
		dist2 = np.square((X-X2.T)/self.lengthscale)

		dvar = (1 + dist2/2.)**(-self.power)
		dl = self.power * self.variance * dist2 * self.lengthscale**(-3) * (1 + dist2/2./self.power)**(-self.power-1)
		dp = - self.variance * np.log(1 + dist2/2.) * (1 + dist2/2.)**(-self.power)

		self.variance.gradient = np.sum(dvar*dL_dK)
		self.lengthscale.gradient = np.sum(dl*dL_dK)
		self.power.gradient = np.sum(dp*dL_dK)

	def update_gradients_diag(self,dL_dKdiag, X):
		self.variance.gradient = np.sum(dL_dKdiag)


def nonlinsys_true(xpred):
	return ParaboloidSpectralDensity._nonlinear_system_fun_static(xpred)

def nonlinsys_sampled_fixed(xpred):
	a0 = -0.8; a1 = -1.5; a2 = -3.0
	model_pars = dict(a0=a0,a1=a1,a2=a2)
	return ParaboloidSpectralDensity._nonlinear_system_fun_static(xpred,model_pars)

def nonlinsys_sampled(xpred):
	a0_min = -1.0; a0_max = +1.0; a0 = a0_min + (a0_max-a0_min)*np.random.rand(1)
	a1_min = -2.0; a1_max = +2.0; a1 = a1_min + (a1_max-a1_min)*np.random.rand(1)
	a2_min = -1; a2_max = +1; a2 = a2_min + (a2_max-a2_min)*np.random.rand(1)
	model_pars = dict(a0=a0,a1=a1,a2=a2)
	return ParaboloidSpectralDensity._nonlinear_system_fun_static(xpred,model_pars=model_pars)


def generate_data_from_multiple_parabola_systems(Nsamples_nominal_dynsys,xmin,xmax,Ndiv_per_dim,nonlin_fun):

	xpred_training_single = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv_per_dim,dim=1) # [Ndiv**dim_in,dim_in]
	yeval_training = np.zeros((Nsamples_nominal_dynsys,xpred_training_single.shape[0],1))
	for jj in range(Nsamples_nominal_dynsys):
		yeval_training[jj,...] = nonlin_fun(xpred_training_single)

	return xpred_training_single, yeval_training




def train_test_parabola(cfg: dict, block_plot: bool, which_kernel: str, which_nonlin_sys = "true", Nobs = 20, random_pars=None, my_seed = None, plotting = True, savefig = False) -> None:

	if my_seed is not None:
		np.random.seed(seed=my_seed)
		tf.random.set_seed(seed=my_seed)
	
	dim_in = 1; dim_out = 1

	"""
	Create training dataset from a variety of nominal dynamical systems, all sampled from an underlying distribution

	1) Create an input grid xpred_training
	2) Evalue the input grid at each sampled instance of the nominal model
	3) Create the training set by concatenating all inputs and outputs
	"""
	xmin_training = -10.0
	xmax_training = +10.0
	Nsamples_nominal_dynsys = 3
	Ndiv_per_dim = 201


	xpred_training, fx_true_training = generate_data_from_multiple_parabola_systems(Nsamples_nominal_dynsys=Nsamples_nominal_dynsys,xmin=xmin_training,
																				xmax=xmax_training,Ndiv_per_dim=Ndiv_per_dim,
																				nonlin_fun=nonlinsys_sampled)

	"""
	Create testing dataset
	"""
	xpred_testing = np.copy(xpred_training)
	fx_true_testing = np.copy(fx_true_training)
	xmax_testing = xmax_training
	xmin_testing = xmin_training
	Ndiv_testing = xpred_testing.shape[0]
	delta_statespace = (xmax_testing-xmin_testing)**dim_in / Ndiv_testing
	delta_statespace_vec = delta_statespace * np.ones((Ndiv_testing,1))



	# Hack the solution without going through the spectral density stuff:

	ker_new = MyNewKernel(input_dim=1,variance=1.,lengthscale=1.,power=1.,active_dims=None,fx_true_testing=fx_true_testing,xpred_testing=xpred_testing)
	# X_in = 0.0 + 2.*np.random.rand(3,1)
	# X2_in = 0.0 + 2.*np.random.rand(5,1)
	# kX1X2 = ker_new.K(X_in,X2_in)
	# kXX = ker_new.Kdiag(X_in)

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(16,8),sharex=False)
	hdl_fig.suptitle(r"This is exactly how the LQR kernel works ... ")
	hdl_splots[0].plot(xpred_testing[:,0],np.zeros(xpred_testing.shape[0]),linestyle="-",color="navy",lw=2,alpha=0.4)
	MO_var = np.reshape(ker_new.Kdiag(xpred_testing),(-1,1))
	MO_std = np.sqrt(MO_var)
	MO_mean = np.zeros((MO_std.shape[0],1))
	hdl_splots[0].fill_between(xpred_testing[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
	for ii in range(fx_true_testing.shape[0]):
		hdl_splots[0].plot(xpred_testing[:,0],fx_true_testing[ii,:,0],marker="None",linestyle="-",color="crimson",lw=2,alpha=0.4)


	Nevals0 = 3
	which_fun_ind = 0
	X0 = xmin_training + (xmax_training-xmin_training)*np.random.rand(Nevals0,1)
	Y0 = np.reshape(np.interp(X0[:,0],xpred_testing[:,0],fx_true_testing[which_fun_ind,:,0]),(Nevals0,1))
	lik = GPy.likelihoods.Gaussian(variance=0.05**2)
	gpy_instance = GPy.core.GP(X=X0,Y=Y0, kernel=ker_new, likelihood=lik) # Can't initialize GPy without samples, so passing one at zero
	MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred_testing)
	MO_std = np.sqrt(MO_var)
	hdl_splots[1].plot(xpred_testing[:,0],fx_true_testing[which_fun_ind,:,0],marker="None",linestyle="-",color="crimson",lw=2,alpha=0.4)
	hdl_splots[1].fill_between(xpred_testing[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
	hdl_splots[1].plot(X0[:,0],Y0[:,0],marker="o",linestyle="None",color="green",lw=0.5,markersize=7)

	plt.show(block=True)



	raise NotImplementedError("WARNING: The solution below is incorrect, i.e., ti doesn't resemble the actual LQR kernel construction...")




	"""
	Create omega landscape
	"""
	Nomegas_coarse = 31
	omega_lim_coarse = 3.0
	omegapred_coarse = CommonUtils.create_Ndim_grid(xmin=-omega_lim_coarse,xmax=omega_lim_coarse,Ndiv=Nomegas_coarse,dim=dim_in) # [Ndiv**dim_in,dim_in]
	Dw_coarse =  (2.*omega_lim_coarse)**dim_in / omegapred_coarse.shape[0]
	# Dw_coarse = 10.0*Dw_coarse
	# raise NotImplementedError("Do a line search to find the best Dw_coarse")
	# The training is very sensitive to how this parameter is initialized. It's worth doing a line search on it first.
	Dw_coarse_vec = Dw_coarse * np.ones((Nomegas_coarse,1))


	kernel_name_plot_label = "Parabola"
	integration_method = "integrate_with_data"
	spectral_density = []
	Nepochs = 3000
	# Nepochs = 1200
	spectral_density_optimized_list = []
	for jj in range(Nsamples_nominal_dynsys):

		str_banner = " << Reconstructing nominal system {0:d} / {1:d} >>".format(jj+1,Nsamples_nominal_dynsys)
		logger.info("="*len(str_banner)); logger.info("="*len(str_banner))
		logger.info(str_banner)
		logger.info("="*len(str_banner)); logger.info("="*len(str_banner))

		spectral_density = ParaboloidSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim_in=dim_in,integration_method=integration_method,Xtrain=xpred_training,Ytrain=fx_true_training[jj,...])
	
		"""
		Reconstruct the mean: and plot it
		"""
		inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in)
		reconstructor_fx = ReconstructFunctionFromSpectralDensity(	dim_in=dim_in,dw_voxel_init=Dw_coarse,dX_voxel_init=delta_statespace,
																	omega_lim=omega_lim_coarse,Nomegas=Nomegas_coarse,
																	inverse_fourier_toolbox=inverse_fourier_toolbox,
																	Xtest=xpred_testing,Ytest=fx_true_testing[jj,...])

		reconstructor_fx.train(Nepochs=Nepochs,learning_rate=1e-2,stop_loss_val=0.001)
		fx_optimized_voxels_coarse = reconstructor_fx.reconstruct_function_at(xpred=xpred_testing)
		spectral_density_optimized_list += [reconstructor_fx.update_internal_spectral_density_parameters()]
		omegapred_coarse_reconstr = reconstructor_fx.get_omegas_weights()
		Sw_coarse_reconstr = reconstructor_fx.inverse_fourier_toolbox.spectral_values
		phiw_coarse_reconstr = reconstructor_fx.inverse_fourier_toolbox.varphi_values



	# Fake spectral density:
	class FakeSpectralDensityWithAllReconstructedOnes():

		def __init__(self,dim_in,Nomegas,spectral_density_optimized_list):


			Nsys = len(spectral_density_optimized_list)
			self.Sw_all = np.zeros((Nsys,Nomegas,1))
			self.phiw_all = np.zeros((Nsys,Nomegas,1))
			self.omegas_all = np.zeros((Nsys,Nomegas,dim_in))
			self.dw_vec = np.zeros((Nsys,Nomegas,1))
			for jj in range(Nsys):
				self.Sw_all[jj,...] = spectral_density_optimized_list[jj].Sw_points
				self.phiw_all[jj,...] = spectral_density_optimized_list[jj].phiw_points
				self.omegas_all[jj,...] = spectral_density_optimized_list[jj].W_points
				self.dw_vec[jj,...] = spectral_density_optimized_list[jj].dw_vec


			# Now reshape:
			self.Sw_points = np.reshape(self.Sw_all,(-1,1))
			self.phiw_points = np.reshape(self.phiw_all,(-1,1))
			self.W_points = np.reshape(self.omegas_all,(-1,dim_in))
			self.dw_vec = np.reshape(self.dw_vec,(-1,1)) / Nsys


	# We don't need evaluations just yet:
	Xtrain_dummy = tf.random.uniform((1,dim_in))
	Ytrain_dummy = tf.random.uniform((1,1))
	Nsample_paths = 3
	fake_spectral_density = FakeSpectralDensityWithAllReconstructedOnes(dim_in,Nomegas_coarse,spectral_density_optimized_list)
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_in,cfg,fake_spectral_density,Xtrain_dummy,Ytrain_dummy)

	rrtp_MO.rrgpMO[0].hack_constant_variance = True

	Nobs = 10
	Xtrain_tot = xmin_training + (xmax_training-xmin_training)*tf.random.uniform((Nobs,1))
	Ytrain_tot = tf.convert_to_tensor(np.interp(x=Xtrain_tot[:,0].numpy(),xp=xpred_testing[:,0],fp=fx_true_testing[0,:,0]),dtype=tf.float32) # Takin the first chunk of fx_true_testing, which corrponsds to theta_cntxt_vals[0,0]
	Ytrain_tot = tf.reshape(Ytrain_tot,(-1,1))

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(16,8),sharex=False)
	logger.info("Close this window!!!!!!!!!!!")
	plt.show(block=True)

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(16,8),sharex=False)
	hdl_splots = [hdl_splots]
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/analysis/parabola_kernel_multiple_nominal_models/lqrker_approach/"
	for jj in range(Nobs):

		if jj > 0:
			Xtrain = Xtrain_tot[0:jj,:]
			Ytrain = Ytrain_tot[0:jj,:]
			rrtp_MO.update_model(X=Xtrain,Y=Ytrain)

		MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_testing,from_prior=jj==0)
		sample_paths = rrtp_MO.sample_path_from_predictive(xpred_testing,Nsamples=Nsample_paths,from_prior=jj==0)
		sample_paths = sample_paths[0] # It's a list with one element, so get the first element

		# Posterior mean and variance:
		hdl_splots[0].cla()
		hdl_splots[0].plot(xpred_testing,MO_mean,linestyle="-",color="navy",lw=2,alpha=0.4)
		hdl_splots[0].fill_between(xpred_testing[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
		# hdl_splots[0].plot(xpred_testing,yplot_true_fun_wrong,marker="None",linestyle="-",color="slategrey",lw=2)
		for ii in range(Nsamples_nominal_dynsys):
			if ii > 0: hdl_splots[0].plot(xpred_testing,fx_true_testing[ii,...],marker="None",linestyle="-",color="crimson",lw=2,alpha=0.3)
			if ii == 0: hdl_splots[0].plot(xpred_testing,fx_true_testing[ii,...],marker="None",linestyle="-",color="crimson",lw=2,alpha=0.9)

		# Sample paths:
		for ii in range(Nsample_paths):
			if sample_paths is not None: hdl_splots[0].plot(xpred_testing,sample_paths[:,ii],marker="None",linestyle="--",color="navy",lw=0.4)

		# Evaluations
		if jj > 0: 
			hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker="o",linestyle="None",color="green",lw=0.5,markersize=7)

		hdl_splots[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots[0].set_xlim([xmin_testing,xmax_testing])
		# hdl_splots[0].set_ylim([-12,3])
		hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)


		if savefig:
			path2save_full = "{0:s}/fitting_with_LQRkernel_like_approach_Nobs_{1:d}.png".format(path2save,jj)
			logger.info("Saving fig ...")
			hdl_fig.savefig(path2save_full,bbox_inches='tight',dpi=300,transparent=False)
			logger.info("Done saving fig!")
		else:
			if block_plot: # If we want to block the final plot we also want to show its progression; otherwise, we assume that we simply wanna return the log_evidence_loss_vec (if we are here, we definitely don't want to save the figs)
				plt.pause(1)
				plt.show(block=False)




		plt.show(block=False)
		plt.pause(2)



	plt.show(block=True)



	# Xtrain = tf.random.uniform((1,2)); Ytrain = tf.random.uniform((1,1))
	# rrtp_MO = MultiObjectiveReducedRankProcess(dim_in,cfg,spectral_density_optimized,Xtrain,Ytrain)
	# pdb.set_trace()

	# rrtp_MO.rrgpMO[0].get_prior_mean()
	# rrtp_MO.predict_at_locations(xpred_testing_cntxt[0:15,:],from_prior=True)[0]
	# 
	# reconstructor_fx.inverse_fourier_toolbox.get_fx_with_variable_integration_step(xpred_testing_cntxt[0:15,:])
	# reconstructor_fx.reconstruct_function_at(xpred_testing_cntxt[0:15,:])

	plotting_reconstruction = True
	if plotting_reconstruction:
		# hdl_fig, hdl_splots_reconstruct = plt.subplots(1,3,figsize=(30,10),sharex=False)
		hdl_fig, hdl_splots_reconstruct = plt.subplots(1,3,figsize=(20,5),sharex=False)
		hdl_fig.suptitle(r"Reconstruction of multiple nominal models (R = {0:d}) using the same GP (M = {1:d}) with a contextual variable $\theta_i$".format(3,Nomegas_coarse),fontsize=fontsize_labels)
		for ii in range(Nsamples_nominal_dynsys):
			xpred_testing_loc = xpred_testing[ii*Ndiv_per_dim:(ii+1)*Ndiv_per_dim,0]
			fx_true_testing_loc = fx_true_testing[ii*Ndiv_per_dim:(ii+1)*Ndiv_per_dim,0]
			fx_optimized_voxels_coarse_loc = fx_optimized_voxels_coarse[ii*Ndiv_per_dim:(ii+1)*Ndiv_per_dim,0] # The reconstructed function is the same for all Nsamples_nominal_dynsys
			
			hdl_splots_reconstruct[0].plot(xpred_testing_loc,fx_true_testing_loc,lw=2,color="navy",alpha=0.2,label="True")
			hdl_splots_reconstruct[0].plot(xpred_testing_loc,fx_optimized_voxels_coarse_loc,lw=2,color="navy",alpha=0.9,label="Reconstructed")
		
		hdl_splots_reconstruct[0].set_xlim([xmin_testing,xmax_testing])
		# hdl_splots_reconstruct[0].set_ylim([-45.,2.])
		hdl_splots_reconstruct[0].set_xticks([xmin_testing,0,xmax_testing])
		hdl_splots_reconstruct[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[0].set_ylabel(r"$f(x_t;\theta_i)$",fontsize=fontsize_labels)
		# hdl_splots_reconstruct[0].set_title(r"Reconstructed $f(x_t;\theta_i)$",fontsize=fontsize_labels)

		# """
		# Discrete grid of omega, for plotting and analysis:
		# """

		Ndiv_omega_for_analysis = 301
		omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim_coarse,xmax=omega_lim_coarse,Ndiv=Ndiv_omega_for_analysis,dim=dim_in) # [Ndiv**dim_in,dim_in]
		Sw_vec, phiw_vec = spectral_density_optimized.unnormalized_density(omegapred_analysis)
		extent_plot_omegas = [-omega_lim_coarse,omega_lim_coarse,-omega_lim_coarse,omega_lim_coarse] #  scalars (left, right, bottom, top)

		# Spectral density:
		COLOR_MAP = "copper"
		# S_vec_plotting = np.reshape(Sw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		S_vec_plotting = np.reshape(Sw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		hdl_splots_reconstruct[1].imshow(S_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=S_vec_plotting.min(),vmax=S_vec_plotting.max(),interpolation='nearest')
		hdl_splots_reconstruct[1].set_title(r"${0:s}$".format("S(\omega)"),fontsize=fontsize_labels)
		hdl_splots_reconstruct[1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[1].set_ylabel(r"$\omega_{\theta}$",fontsize=fontsize_labels)


		# Varphi:
		if np.any(phiw_vec != 0.0):
			# phi_vec_plotting = np.reshape(phiw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
			phi_vec_plotting = np.reshape(phiw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
			hdl_splots_reconstruct[2].imshow(phi_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=phi_vec_plotting.min(),vmax=phi_vec_plotting.max(),interpolation='nearest')
		else:
			hdl_splots_reconstruct[2].set_xticks([],[]); hdl_splots_reconstruct[2].set_yticks([],[])
		hdl_splots_reconstruct[2].set_title(r"${0:s}$".format("\\varphi(\omega)"),fontsize=fontsize_labels)
		hdl_splots_reconstruct[2].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[2].set_ylabel(r"$\omega_{\theta}$",fontsize=fontsize_labels)

		hdl_splots_reconstruct[1].plot(omegapred_coarse_reconstr[:,0],omegapred_coarse_reconstr[:,1],marker=".",color="navy",markersize=7,linestyle="None")
		hdl_splots_reconstruct[2].plot(omegapred_coarse_reconstr[:,0],omegapred_coarse_reconstr[:,1],marker=".",color="navy",markersize=7,linestyle="None")

		hdl_splots_reconstruct[1].set_xlim([-omega_lim_coarse,omega_lim_coarse])
		hdl_splots_reconstruct[1].set_ylim([-omega_lim_coarse,omega_lim_coarse])
		hdl_splots_reconstruct[2].set_xlim([-omega_lim_coarse,omega_lim_coarse])
		hdl_splots_reconstruct[2].set_ylim([-omega_lim_coarse,omega_lim_coarse])

		hdl_splots_reconstruct[1].set_xticks([-omega_lim_coarse,0,omega_lim_coarse])
		hdl_splots_reconstruct[1].set_yticks([-omega_lim_coarse,0,omega_lim_coarse])
		hdl_splots_reconstruct[2].set_xticks([-omega_lim_coarse,0,omega_lim_coarse])
		hdl_splots_reconstruct[2].set_yticks([-omega_lim_coarse,0,omega_lim_coarse])

		# plt.show(block=False)
		# plt.pause(1)

	# plt.ion()
	plt.show(block=True)
	# plt.pause(2)
	# input("Press return to continue")


	"""
	Fit a new function (could be true, fixed sample or new sample, depending on our choice for nonlinsys4GPfitting())
	"""

	# Create grid for predictions:
	# xmin = -6.0
	# xmax = +3.0
	xmin = -10.0
	xmax = +10.0
	xpred_plotting = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv_per_dim,dim=dim_in) # [Ndiv**dim_in,dim_in]
	# yplot_nonlin_sys = nonlinsys4GPfitting(xpred_plotting)

	# We don't need evaluations just yet:
	Xtrain = tf.random.uniform((1,2))
	Ytrain = tf.random.uniform((1,1))
	
	if which_kernel == "parabola":
		rrtp_MO = MultiObjectiveReducedRankProcess(dim_in,cfg,spectral_density_optimized,Xtrain,Ytrain)
		# rrtp_MO.train_model(verbosity=True)
	else:
		X0 = np.zeros((1,1))
		Y0 = np.zeros((1,1))
		# X0 = np.empty(shape=(1,0)) # Doesn't worj with GPy
		# Y0 = np.empty(shape=(1,0)) # Doesn't worj with GPy
		# gpy_instance = GPy.core.GP(X=Xtrain,Y=Ytrain, kernel=ker, likelihood=lik)
		gpy_instance = GPy.core.GP(X=X0,Y=Y0, kernel=ker, likelihood=lik) # Can't initialize GPy without samples, so passing one at zero



	"""
	See effect of added phase in the prior
	"""
	plotting_phase_influence = False
	logger.info("phase values: {0:s}".format(str(theta_cntxt_vals)))
	if plotting_phase_influence:
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		Ndiv_phase_added = 200
		Nsample_paths = 3
		var_context_vec = np.linspace(0.0,2.*np.pi,Ndiv_phase_added)

		ind_closest = np.argmin(abs(np.reshape(var_context_vec,(-1,1)) - np.reshape(theta_cntxt_vals,(1,-1))),axis=0)

		for jj in range(Ndiv_phase_added):

			logger.info("phase: {0:f}".format(var_context_vec[jj]))

			xpred_plotting_loc = np.concatenate([xpred_plotting,var_context_vec[jj]*np.ones(xpred_plotting.shape)],axis=1)

			MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_plotting_loc,from_prior=True)
			sample_paths = rrtp_MO.sample_path_from_predictive(xpred_plotting_loc,Nsamples=Nsample_paths,from_prior=True)
			sample_paths = sample_paths[0] # It's a list with one element, so get the first element
			hdl_splots.cla()
			hdl_splots.plot(xpred_plotting,MO_mean,linestyle="-",color="navy",lw=2,alpha=0.4)
			hdl_splots.fill_between(xpred_plotting[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
			# hdl_splots.plot(xpred_plotting,yplot_nonlin_sys,marker="None",linestyle="-",color="crimson",lw=3,alpha=0.6)

			for ii in range(Nsamples_nominal_dynsys):
				fx_true_testing_loc = fx_true_testing[ii*Ndiv_per_dim:(ii+1)*Ndiv_per_dim,0]
				hdl_splots.plot(xpred_plotting,fx_true_testing_loc,lw=2,color="crimson",alpha=0.4,label="Reconstructed")

			# Sample paths:
			for ii in range(Nsample_paths):
				if sample_paths is not None: hdl_splots.plot(xpred_plotting,sample_paths[:,ii],marker="None",linestyle="--",color="navy",lw=0.4)

			hdl_splots.set_xlabel(r"$x_t$",fontsize=fontsize_labels)
			hdl_splots.set_xlim([xmin,xmax])
			# hdl_splots.set_ylim([-12,3])
			hdl_splots.set_ylim([-20,20])
			hdl_splots.set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

			plt.show(block=False)
			plt.pause(0.01)
			if jj in ind_closest:
				logger.info("phase values: {0:s}".format(str(theta_cntxt_vals)))
				plt.pause(5.0)

		plt.show(block=False)

	# input("Press return to continue")



	if plotting: 
		# hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(24,10),sharex=True)
		hdl_fig, hdl_splots = plt.subplots(1,2,figsize=(16,8),sharex=False)

		# if which_nonlin_sys == "true":
		# 	hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta_{nom})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		# elif which_nonlin_sys == "wrong":
		# 	hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta_{rand})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		# elif which_nonlin_sys == "sampled":
		# 	hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta_{rand})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta)$, $\theta \sim p(\theta | Y,X)$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)


	# Prepare ESS:
	def target_fun_log_lik(cntxt_var,Xtrain,Ytrain,rrtp_MO):
		"""
		cntxt_var: [Npoints,dim_in=1]

		"""
		loss_val_vec = np.zeros(cntxt_var.shape[0])
		for lll in range(cntxt_var.shape[0]):
	
			Xtrain = Xtrain.numpy()
			Xtrain[:,1] = cntxt_var[lll] # Assume the evaluations come from cntxt_var
			Xtrain = tf.convert_to_tensor(Xtrain,dtype=tf.float32)

			mean_pred, cov_pred = rrtp_MO.rrgpMO[0].predict_at_locations(Xtrain,from_prior=True)
			loss_val_vec[lll] = scipy.stats.multivariate_normal.logpdf(Ytrain.numpy(),mean=mean_pred.numpy(),cov=cov_pred.numpy()) / Ytrain.shape[0]
		
		return loss_val_vec

	# Get evaluations from a pre-generated sobol grid:
	# Assume the evaluations come from theta_cntxt_vals[0,0]
	Nevals_tot = 100
	Xtrain_tot = xmin + (xmax - xmin)*tf.math.sobol_sample(dim=dim_in,num_results=(Nevals_tot),skip=10000)
	Ndiv_phase_added = 200
	cntxt_var_lims = [0.0,2.*np.pi]
	var_context_vec = np.linspace(cntxt_var_lims[0],cntxt_var_lims[1],Ndiv_phase_added)
	Nobs_analysis = 15
	Nsample_paths = 3
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_{0:s}/".format(which_nonlin_sys)
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/analysis/parabola_kernel_multiple_nominal_models/"
	log_evidence_loss_vec = np.zeros(Nobs_analysis)
	rmse_vec = np.zeros(Nobs_analysis)
	fx_true_testing_for_first_context_var = fx_true_testing[0:Ndiv_per_dim,0]
	for jj in range(Nobs_analysis-1):

		"""
		Sampling context variable from posterior, given data
		"""
		Xtrain = Xtrain_tot[0:jj+1,:]
		Ytrain = tf.convert_to_tensor(np.interp(x=Xtrain[:,0].numpy(),xp=xpred_plotting.numpy()[0:Ndiv_per_dim,0],fp=fx_true_testing_for_first_context_var),dtype=tf.float32) # Takin the first chunk of fx_true_testing, which corrponsds to theta_cntxt_vals[0,0]

		# Sample:
		# logger.info("Sampling!!")
		# target_fun_log_lik = construct_target_fun_log_lik(Xtrain,Ytrain,rrtp_MO)
		sampler = EllipticalSliceSampler(dim_in=1,target_log_lik=target_fun_log_lik,
										Nsamples=60,Nburning=100,
										Nrestarts=4,omega_lim_random_restarts=cntxt_var_lims,
										mean_w=np.array([np.pi]),var_w=np.array([2.]),
										kwargs_to_fun=(Xtrain,Ytrain,rrtp_MO))
		samples_vec, omega0_restarts = sampler.run_ess()


		log_evidence_vec = np.zeros(Ndiv_phase_added)
		for ll in range(Ndiv_phase_added):
			Xtrain = Xtrain.numpy()
			Xtrain[:,1] = var_context_vec[ll] # Assume the evaluations come from theta_opti
			Xtrain = tf.convert_to_tensor(Xtrain,dtype=tf.float32)

			mean_pred, cov_pred = rrtp_MO.rrgpMO[0].predict_at_locations(Xtrain,from_prior=True)
			log_evidence_vec[ll] = -scipy.stats.multivariate_normal.logpdf(Ytrain.numpy(),mean=mean_pred.numpy(),cov=cov_pred.numpy()) / Ytrain.shape[0]

		samples_vec_opti = np.mean(samples_vec,axis=0)

		# Make the data come from the optimal theta; needed when updating the MOrrp with the new datapoints:
		Xtrain = Xtrain.numpy()
		Xtrain[:,1] = samples_vec_opti # Assume the evaluations come from theta_opti
		Xtrain = tf.convert_to_tensor(Xtrain,dtype=tf.float32)


		hdl_splots[1].cla()
		hdl_splots[1].set_title(r"ESS Samples from $p(\theta|Y,X)$ with {0:d} observations".format(jj+1),fontsize=fontsize_labels)
		hdl_splots[1].plot(var_context_vec,log_evidence_vec,lw=3,alpha=0.1,color="crimson")
		hdl_splots[1].hist(samples_vec,bins=15,histtype="stepfilled",color="lightblue",alpha=0.3,ec="navy",lw=1.5)
		hdl_splots[1].set_xlim(cntxt_var_lims)
		# hdl_splots[1].set_ylim([log_evidence_vec.min(),log_evidence_vec.max()])
		ylims = [0,log_evidence_vec.min()+10.0]
		hdl_splots[1].set_ylim(ylims)
		# hdl_splots[1].vlines(x=theta_cntxt_vals[0,0],ymin=ylims[0],ymax=ylims[1],color="crimson",alpha=0.9)
		# hdl_splots[1].vlines(x=np.mean(samples_vec,axis=0),ymin=ylims[0],ymax=ylims[1],color="navy",alpha=0.9)
		hdl_splots[1].plot(theta_cntxt_vals[0,0],ylims[0] + 0.05*(ylims[1] - ylims[0]),color="crimson",alpha=0.4,markersize=15,marker="*")
		hdl_splots[1].plot(samples_vec_opti,ylims[0] + 0.05*(ylims[1] - ylims[0]),color="navy",alpha=0.4,markersize=15,marker="*")
		hdl_splots[1].set_xlabel(r"$\theta$",fontsize=fontsize_labels)
		hdl_splots[1].set_ylabel(r"$\log p(Y | X,\theta)$",fontsize=fontsize_labels)
		plt.show(block=False)
		plt.pause(1)

		"""
		Fit a GP, given the optimal context parameter
		"""

		xpred_plotting_loc = np.concatenate([xpred_plotting,samples_vec_opti*np.ones(xpred_plotting.shape)],axis=1)

		if jj == 0:

			if which_kernel == "parabola":
				MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_plotting_loc,from_prior=True)
				sample_paths = rrtp_MO.sample_path_from_predictive(xpred_plotting_loc,Nsamples=Nsample_paths,from_prior=True)
				sample_paths = sample_paths[0] # It's a list with one element, so get the first element
			else:
				MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred_plotting.numpy())
				MO_std = np.sqrt(MO_var)
				MO_std[:,:] = np.sqrt(variance_prior)
				sample_paths = gpy_instance.posterior_samples_f(X=xpred_plotting.numpy(),size=Nsample_paths)
				sample_paths = sample_paths[:,0,:]

		else:

			if which_kernel == "parabola":

				# Update posterior:
				Ytrain = tf.reshape(Ytrain,(-1,dim_out))
				rrtp_MO.update_model(X=Xtrain,Y=Ytrain)

				# Get moments:
				MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_plotting_loc)

				# Sample paths:
				sample_paths = rrtp_MO.sample_path_from_predictive(xpred_plotting_loc,Nsamples=Nsample_paths,from_prior=False)
				sample_paths = sample_paths[0] # It's a list with one element, so get the first element

			else:

				gpy_instance.set_XY(X=Xtrain.numpy(),Y=Ytrain.numpy())
				MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred_plotting.numpy())
				MO_std = np.sqrt(MO_var)
				sample_paths = gpy_instance.posterior_samples_f(X=xpred_plotting.numpy(),size=Nsample_paths)
				sample_paths = sample_paths[:,0,:]

		# log_evidence_loss_vec[jj] = np.mean(-scipy.stats.norm.logpdf(x=yplot_nonlin_sys[:,0],loc=MO_mean[:,0],scale=MO_std[:,0]))
		# rmse_vec[jj] = np.sqrt(np.mean((yplot_nonlin_sys[:,0]-MO_mean[:,0])**2))


		# Posterior mean and variance:
		hdl_splots[0].cla()
		hdl_splots[0].plot(xpred_plotting,MO_mean,linestyle="-",color="navy",lw=2,alpha=0.4)
		hdl_splots[0].fill_between(xpred_plotting[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
		# hdl_splots[0].plot(xpred_plotting,yplot_true_fun_wrong,marker="None",linestyle="-",color="slategrey",lw=2)
		hdl_splots[0].plot(xpred_plotting,fx_true_testing_for_first_context_var,marker="None",linestyle="-",color="crimson",lw=2,alpha=0.4)

		# Sample paths:
		for ii in range(Nsample_paths):
			if sample_paths is not None: hdl_splots[0].plot(xpred_plotting,sample_paths[:,ii],marker="None",linestyle="--",color="navy",lw=0.4)

		# Evaluations
		if jj > 0: hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker="o",linestyle="None",color="green",lw=0.5,markersize=7)

		hdl_splots[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots[0].set_xlim([xmin,xmax])
		# hdl_splots[0].set_ylim([-12,3])
		hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

		if savefig:
			path2save_full = "{0:s}/fitting_with_sampled_contextual_var_from_posterior_Nobs_{1:d}.png".format(path2save,jj)
			logger.info("Saving fig ...")
			hdl_fig.savefig(path2save_full,bbox_inches='tight',dpi=300,transparent=False)
			logger.info("Done saving fig!")
		else:
			if block_plot: # If we want to block the final plot we also want to show its progression; otherwise, we assume that we simply wanna return the log_evidence_loss_vec (if we are here, we definitely don't want to save the figs)
				plt.pause(1)
				plt.show(block=False)

	if plotting and block_plot:
		plt.show(block=block_plot)



	if plotting:
		plt.close(hdl_fig)
		hdl_fig.clf()
		del hdl_fig

	return log_evidence_loss_vec
	# return rmse_vec

@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict) -> None:

	# which_kernel = "gaussian"
	# which_kernel = "matern"
	which_kernel = "parabola"

	which_nonlin_sys = "true"
	# which_nonlin_sys = "wrong"
	# which_nonlin_sys = "sampled"
	
	train_test_parabola(cfg, block_plot=True, which_kernel=which_kernel, which_nonlin_sys=which_nonlin_sys, Nobs = 15, random_pars=None, my_seed=2, plotting=True, savefig=True)


if __name__ == "__main__":

	main()



