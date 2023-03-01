import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
from lqrker.models import MultiObjectiveReducedRankProcess
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
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

def simulate_nonlinsystem(Nsteps,x0,nonlinear_system_fun,visualize=False):

	dim = x0.shape[1]
	x_vec = np.zeros((Nsteps,dim))
	x_vec[0,:] = x0
	y_vec = np.zeros((Nsteps,dim))
	y_vec[0,:] = x0
	std_noise_process = 0.01
	std_noise_obs = np.sqrt(0.01)
	for ii in range(Nsteps-1):

		# True system evolution with process noise:
		x_vec[ii+1,:] = nonlinear_system_fun(x_vec[ii:ii+1,:]) + std_noise_process * np.random.randn()

		# Noisy observations:
		y_vec[ii+1,:] = x_vec[ii+1,:] + std_noise_obs * np.random.randn()

	# Get consecutive observations & latent states:
	Xlatent = x_vec[0:Nsteps-1,:] # [Nsteps-1,dim]
	Ylatent = x_vec[1:Nsteps,:] # [Nsteps-1,dim]
	Xobs = y_vec[0:Nsteps-1,:] # [Nsteps-1,dim]
	Yobs = y_vec[1:Nsteps,:] # [Nsteps-1,dim]

	if visualize:

		Ndiv = 201
		xplot_true_fun = np.linspace(-5.,2.,Ndiv)
		yplot_true_fun = nonlinear_system_fun(xplot_true_fun)

		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_splots.plot(xplot_true_fun,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
		hdl_splots.plot(Xtrain[:,0],Ytrain[:,0],marker=".",linestyle="--",color="gray",lw=0.5,markersize=4)
		hdl_splots.plot(Xobs[:,0],Yobs[:,0],marker=".",linestyle="--",color="steelblue",lw=0.5,markersize=4)
		plt.show(block=True)

	return Xlatent, Ylatent, Xobs, Yobs


def generate_data_from_multiple_kink_systems(Nsamples,xmin,xmax,Ndiv_per_dim,nonlin_fun):

	xpred_training_single = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv_per_dim,dim=1) # [Ndiv**dim_in,dim_in]
	yeval_samples = np.zeros((Nsamples,xpred_training_single.shape[0],1))
	for jj in range(Nsamples):
		yeval_samples[jj,...] = nonlin_fun(xpred_training_single)

	xpred_training = np.concatenate([xpred_training_single]*Nsamples,axis=0)
	ypred_training = np.reshape(yeval_samples,(-1,1),order="C")

	return xpred_training, ypred_training

def train_test_kink(cfg: dict, block_plot: bool, which_kernel: str, which_nonlin_sys = "true", Nobs = 20, random_pars=None, my_seed = None, plotting = True) -> None:
	"""

	Train the model to predict one-step ahead

	Then, see how well the model can predict long term trajectories.
	To this end, we evaluate the model at a training point, then sample a new point from the output distribution, 
	then evaluate the model at that sampled point, and then repeat the process
	"""


	"""
	TODO
	1) Most of the time, the samples are gathered at the high frequencies, which creates a lot of ripples in the prediction
		1.1) Sample from individual Gaussians placed at the modes?
	4) Include temporal dependendices in the model
	"""

	# print(OmegaConf.to_yaml(cfg))

	if my_seed is not None:
		np.random.seed(seed=my_seed)
		tf.random.set_seed(seed=my_seed)
	
	# x0 = np.array([[1.0]]) # Needed for rolling out the simulated system
	dim_in = 1
	dim_out = 1

	"""
	Create training dataset
	"""
	xmin_training = -10.0
	xmax_training = +10.0
	Nsamples = 400
	Ndiv_per_dim = 201


	


	# fx_true_training = KinkSpectralDensity._nonlinear_system_fun_static(xpred_training) # [Ndiv,1]

	def nonlin_fun_rand(xpred):
		a0_min = -1.0; a0_max = +1.0; a0 = a0_min + (a0_max-a0_min)*np.random.rand(1)
		a1_min = -2.0; a1_max = +2.0; a1 = a1_min + (a1_max-a1_min)*np.random.rand(1)
		a2_min = -4.0; a2_max = -1.0; a2 = a2_min + (a2_max-a2_min)*np.random.rand(1)
		random_pars = dict(a0=a0,a1=a1,a2=a2)
		return KinkSpectralDensity._nonlinear_system_fun_static(xpred,use_nominal_model=False,random_pars=random_pars)

	xpred_training, fx_true_training = generate_data_from_multiple_kink_systems(Nsamples=Nsamples,xmin=xmin_training,
																				xmax=xmax_training,Ndiv_per_dim=Ndiv_per_dim,
																				nonlin_fun=nonlin_fun_rand)

	if which_kernel == "kink":
		kernel_name_plot_label = "Kink"
		use_nominal_model = True
		# integration_method = "integrate_with_regular_grid"
		integration_method = "integrate_with_data"
		# spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,Xtrain=None,Ytrain=None,use_nominal_model=True)
		# spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,use_nominal_model=use_nominal_model)
		spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,Xtrain=xpred_training,Ytrain=fx_true_training,use_nominal_model=True)
	elif which_kernel == "kink_randomized":
		kernel_name_plot_label = "Kink (randomized)"
		integration_method = "integrate_with_regular_grid_randomized_parameters"
		spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,Xtrain=None,Ytrain=None,use_nominal_model=False)
		# spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,use_nominal_model=use_nominal_model)
	elif which_kernel == "gaussian":
		kernel_name_plot_label = "Gaussian"
		variance_prior = 2.0
		ker = GPy.kern.RBF(dim_in, variance=variance_prior, lengthscale=0.5)
		lik = GPy.likelihoods.Gaussian(variance=0.15**2)
	elif which_kernel == "matern":
		kernel_name_plot_label = "Matern"
		variance_prior = 2.0
		ker = GPy.kern.sde_Matern52(dim_in, variance=variance_prior, lengthscale=0.5)
		lik = GPy.likelihoods.Gaussian(variance=0.15**2)
		# spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_in)
		

	# omega_min = -10.
	# omega_max = +10.
	# Ndiv = 1001
	# cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_in
	# spectral_density.update_Wpoints_regular(omega_min,omega_max,Ndiv,normalize_density_numerically=False)

	def nonlinsys_true(xpred):
		return KinkSpectralDensity._nonlinear_system_fun_static(xpred)

	def nonlinsys_wrong(xpred):
		return KinkSpectralDensity._nonlinear_system_fun_static(xpred,use_nominal_model=False)


	if random_pars is None:
		# a0_min = -1.0; a0_max = +1.0; a0 = a0_min + (a0_max-a0_min)*np.random.rand(1)
		# a1_min = -2.0; a1_max = +2.0; a1 = a1_min + (a1_max-a1_min)*np.random.rand(1)
		# a2_min = -4.0; a2_max = -1.0; a2 = a2_min + (a2_max-a2_min)*np.random.rand(1)
		a0 = -0.8; a1 = -1.5; a2 = -3.0
		random_pars = dict(a0=a0,a1=a1,a2=a2)
	
	def nonlinsys_sampled(xpred):
		return KinkSpectralDensity._nonlinear_system_fun_static(xpred,use_nominal_model=False,random_pars=random_pars)


	if which_nonlin_sys == "true":
		nonlinsys2use = nonlinsys_true
	elif which_nonlin_sys == "wrong":
		nonlinsys2use = nonlinsys_wrong
	elif which_nonlin_sys == "sampled":
		nonlinsys2use = nonlinsys_sampled

	if which_kernel == "kink" or which_kernel == "kink_randomized":
		if integration_method == "integrate_with_regular_grid_randomized_parameters" or integration_method == "integrate_with_regular_grid":
			# L = 750.0
			# Ndiv = 1201
			L = 800.0
			Ndiv = 1001
			cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_in
			spectral_density.update_Wpoints_discrete(L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False)



	# # Generate training data:
	# Nsteps = 4
	# Xlatent, Ylatent, Xobs, Yobs = simulate_nonlinsystem(Nsteps,x0,nonlinsys_wrong,visualize=False)
	# 
	
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in,dim_out_ind=None)

	"""
	Create testing dataset
	"""
	xpred_testing = np.copy(xpred_training)
	fx_true_testing = np.copy(fx_true_training)
	xmax_testing = xmax_training
	xmin_testing = xmin_training
	Ndiv_testing = xpred_testing.shape[0] / 
	# Ndiv_testing = Ndiv_per_dim

	Nomegas_coarse = 21
	omega_lim_coarse = 4.0
	omegapred_coarse = CommonUtils.create_Ndim_grid(xmin=-omega_lim_coarse,xmax=omega_lim_coarse,Ndiv=Nomegas_coarse,dim=dim_in) # [Ndiv**dim_in,dim_in]
	Dw_coarse =  (2.*omega_lim_coarse)**dim_in / omegapred_coarse.shape[0]

	delta_statespace = (xmax_testing-xmin_testing)**dim_in / Ndiv_testing
	
	inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred=omegapred_coarse,Dw=None,dX=delta_statespace)
	fx_integrand = inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred=xpred_testing,Dw_vec=Dw_coarse) # [Npoints, Nomegas]
	fx_reconstructed = tf.math.reduce_sum(fx_integrand,axis=1,keepdims=True) # Integrate wrt omegas [Npoints, 1]
	reconstructor_fx = ReconstructFunctionFromSpectralDensity(	dim_in=dim_in,omega_lim=omega_lim_coarse,Nomegas=Nomegas_coarse,
																inverse_fourier_toolbox=inverse_fourier_toolbox,
																Xtrain=xpred_testing,Ytrain=fx_true_testing,omegas_weights=None)
	# print(reconstructor_fx.delta_omegas_pre_activation)
	# print(reconstructor_fx.delta_statespace_preactivation)

	reconstructor_fx.train(Nepochs=3000,learning_rate=1e-2,stop_loss_val=0.001)
	fx_optimized_voxels_coarse = reconstructor_fx.reconstruct_function_at(xpred=xpred_testing)
	omegapred_coarse_reconstr = reconstructor_fx.get_omegas_weights()
	dw_vec = reconstructor_fx.get_delta_omegas(reconstructor_fx.delta_omegas_pre_activation)

	dw_vec = tf.expand_dims(dw_vec,axis=1)

	dX_vec = reconstructor_fx.get_delta_statespace(reconstructor_fx.delta_statespace_preactivation)
	reconstructor_fx.inverse_fourier_toolbox.spectral_density.update_dX_voxels(dX_new=dX_vec)
	Sw_vec, phiw_vec = reconstructor_fx.inverse_fourier_toolbox.spectral_density.unnormalized_density(omegapred_coarse_reconstr)
	spectral_density.update_Wsamples_as(Sw_points=Sw_vec,phiw_points=phiw_vec,W_points=omegapred_coarse_reconstr,dw_vec=dw_vec,dX_vec=dX_vec)

	# Generate sobol grid for evaluations:
	xmin = -6.0
	xmax = +3.0
	Nevals_tot = 100
	Xtrain_tot = xmin + (xmax - xmin)*tf.math.sobol_sample(dim=dim_in,num_results=(Nevals_tot),skip=10000)
	Xtrain_tot = Xtrain_tot.numpy()
	Xtrain_tot[1,:] = 0.6
	Xtrain_tot = tf.convert_to_tensor(value=Xtrain_tot,dtype=tf.float32)

	Nevals = 1
	Xtrain = Xtrain_tot[0:Nevals,:]
	Ytrain = nonlinsys2use(Xtrain)

	Xtrain = tf.convert_to_tensor(value=Xtrain,dtype=tf.float32)
	Ytrain = tf.convert_to_tensor(value=Ytrain,dtype=tf.float32)

	if which_kernel == "kink" or which_kernel == "kink_randomized":
		rrtp_MO = MultiObjectiveReducedRankProcess(dim_in,cfg,spectral_density,Xtrain,Ytrain)
		# rrtp_MO.train_model(verbosity=True)
	else:
		X0 = np.zeros((1,1))
		Y0 = np.zeros((1,1))
		# X0 = np.empty(shape=(1,0))
		# Y0 = np.empty(shape=(1,0))
		# gpy_instance = GPy.core.GP(X=Xtrain,Y=Ytrain, kernel=ker, likelihood=lik)
		gpy_instance = GPy.core.GP(X=X0,Y=Y0, kernel=ker, likelihood=lik)

	# pdb.set_trace()

	# Create grid for predictions:
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=201,dim=dim_in) # [Ndiv**dim_in,dim_in]

	# # Get moments:
	# MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(xpred)
	# mean_prior, std_prior = rrtp_MO.predict_at_locations(xpred,from_prior=True)

	# # Sample paths:
	# Nsamples = 3
	# sample_paths_prior = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=Nsamples,from_prior=True)
	# sample_paths_predictive = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=Nsamples,from_prior=False)

	# # Get prior trajectory:
	# x0_sample = np.array([[0.9]])
	# Nsteps_sample = 2
	# Xlatent_sample, Ylatent_sample, _, _ = simulate_nonlinsystem(Nsteps_sample,x0_sample,KinkSpectralDensity._nonlinear_system_fun_static,visualize=False)
	# Xlatent_sample = tf.convert_to_tensor(value=Xlatent_sample,dtype=np.float32)
	# Ylatent_sample = tf.convert_to_tensor(value=Ylatent_sample,dtype=np.float32)
	# xsamples_X, _ = rrtp_MO.sample_state_space_from_prior_recursively(x0=Xlatent_sample,x1=Ylatent_sample,traj_length=20,Nsamples=2,sort=True,plotting=False) # [Npoints,self.dim_out,Nsamples]

	# Plot:
	assert dim_out == 1

	yplot_nonlin_sys = nonlinsys2use(xpred)

	# pdb.set_trace()

	# yplot_true_fun = nonlinsys_true(xpred)
	# yplot_true_fun_wrong = nonlinsys_wrong(xpred)
	# yplot_true_fun_sampled = nonlinsys_sampled(xpred)

	# True function wrong:
	# yplot_true_fun_wrong = spectral_density._nonlinear_system_fun(xpred)
	# yplot_true_fun_wrong = KinkSpectralDensity._nonlinear_system_fun_static(xpred,use_nominal_model=False)

	# spectral_density._nonlinear_system_fun(0.0)
	# pdb.set_trace()

	# hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	# hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format(which_kernel),fontsize=fontsize_labels)
	# hdl_splots[0].plot(xpred,MO_mean_pred,linestyle="-",color="navy",lw=2,alpha=0.4)
	# # pdb.set_trace()
	# hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred[:,0] - 2.*MO_std_pred[:,0],MO_mean_pred[:,0] + 2.*MO_std_pred[:,0],color="cornflowerblue",alpha=0.4)
	# hdl_splots[0].plot(xpred,yplot_true_fun_wrong,marker="None",linestyle="-",color="slategrey",lw=2)
	# hdl_splots[0].plot(xpred,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
	# for ii in range(len(sample_paths_predictive)):
	# 	hdl_splots[0].plot(xpred,sample_paths_predictive[ii],marker="None",linestyle="-",color="grey",lw=0.5)
	# # hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker=".",linestyle="--",color="grey",lw=0.5,markersize=5)
	# hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker="o",linestyle="None",color="green",lw=0.5,markersize=5)

	# # for ii in range(xsamples_X.shape[1]):
	# # 	hdl_splots[0].plot(xsamples_X[0:-1,ii],xsamples_X[1::,ii],marker=".",linestyle="--",lw=0.5,markersize=5)
	# # hdl_splots[0].set_xlim([xmin,xmax])
	# # hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

	# hdl_splots[1].plot(xpred,mean_prior,linestyle="-",color="navy",lw=2,alpha=0.4)
	# hdl_splots[1].fill_between(xpred[:,0],mean_prior[:,0] - 2.*std_prior[:,0],mean_prior[:,0] + 2.*std_prior[:,0],color="cornflowerblue",alpha=0.5)
	# hdl_splots[1].plot(xpred,yplot_true_fun_wrong,marker="None",linestyle="-",color="slategrey",lw=2)
	# hdl_splots[1].plot(xpred,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
	# for ii in range(len(sample_paths_prior)):
	# 	hdl_splots[1].plot(xpred,sample_paths_prior[ii],marker="None",linestyle="-",color="grey",lw=0.5)
	# hdl_splots[1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	# hdl_splots[1].set_xlim([xmin,xmax])
	# hdl_splots[1].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

	# plt.show(block=block_plot)
	# plt.pause(1)

	# del spectral_density
	# del rrtp_MO
	# 


	raise NotImplementedError("Plot always the reconstructed mean")


	if plotting: 
		# hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(24,10),sharex=True)
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_splots = [hdl_splots]

		if which_nonlin_sys == "true":
			hdl_fig.suptitle(r"Kink dynamical system $x_{t+1} = f(x_t;\theta_{nom})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		elif which_nonlin_sys == "wrong":
			hdl_fig.suptitle(r"Kink dynamical system $x_{t+1} = f(x_t;\theta_{rand})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		elif which_nonlin_sys == "sampled":
			hdl_fig.suptitle(r"Kink dynamical system $x_{t+1} = f(x_t;\theta_{rand})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)

	Nsample_paths = 3
	savefig = False
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_{0:s}/".format(which_nonlin_sys)
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/analysis/kink_kernel_constructed_with_nominal_model/nonlinsys_{0:s}/".format(which_nonlin_sys)
	log_evidence_loss_vec = np.zeros(Nobs)
	rmse_vec = np.zeros(Nobs)
	for jj in range(Nobs):

		if jj == 0:

			if which_kernel == "kink" or which_kernel == "kink_randomized":
				MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred,from_prior=True)
				sample_paths = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=Nsample_paths,from_prior=True)
				sample_paths = sample_paths[0] # It's a list with one element, so get the first element
			else:
				MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred.numpy())
				MO_std = np.sqrt(MO_var)
				MO_std[:,:] = np.sqrt(variance_prior)
				sample_paths = gpy_instance.posterior_samples_f(X=xpred.numpy(),size=Nsample_paths)
				sample_paths = sample_paths[:,0,:]

		else:

			# Get evaluations:
			Nevals = jj
			Xtrain = Xtrain_tot[0:Nevals,:]
			Ytrain = nonlinsys2use(Xtrain)

			Xtrain = tf.convert_to_tensor(value=Xtrain,dtype=tf.float32)
			Ytrain = tf.convert_to_tensor(value=Ytrain,dtype=tf.float32)

			if which_kernel == "kink" or which_kernel == "kink_randomized":

				# Update posterior:
				rrtp_MO.update_model(X=Xtrain,Y=Ytrain)

				# Get moments:
				MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred)

				# Sample paths:
				sample_paths = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=Nsample_paths,from_prior=False)
				sample_paths = sample_paths[0] # It's a list with one element, so get the first element

			else:

				gpy_instance.set_XY(X=Xtrain.numpy(),Y=Ytrain.numpy())
				MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred.numpy())
				MO_std = np.sqrt(MO_var)
				sample_paths = gpy_instance.posterior_samples_f(X=xpred.numpy(),size=Nsample_paths)
				sample_paths = sample_paths[:,0,:]

		log_evidence_loss_vec[jj] = np.mean(-scipy.stats.norm.logpdf(x=yplot_nonlin_sys[:,0],loc=MO_mean[:,0],scale=MO_std[:,0]))
		rmse_vec[jj] = np.sqrt(np.mean((yplot_nonlin_sys[:,0]-MO_mean[:,0])**2))

		if plotting: 

			# Posterior mean and variance:
			hdl_splots[0].cla()
			hdl_splots[0].plot(xpred,MO_mean,linestyle="-",color="navy",lw=2,alpha=0.4)
			hdl_splots[0].fill_between(xpred[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
			# hdl_splots[0].plot(xpred,yplot_true_fun_wrong,marker="None",linestyle="-",color="slategrey",lw=2)
			hdl_splots[0].plot(xpred,yplot_nonlin_sys,marker="None",linestyle="-",color="crimson",lw=3,alpha=0.6)

			# Sample paths:
			for ii in range(Nsample_paths):
				if sample_paths is not None: hdl_splots[0].plot(xpred,sample_paths[:,ii],marker="None",linestyle="--",color="navy",lw=0.4)

			# Evaluations
			if jj > 0: hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker="o",linestyle="None",color="green",lw=0.5,markersize=7)

			hdl_splots[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
			hdl_splots[0].set_xlim([xmin,xmax])
			hdl_splots[0].set_ylim([-12,3])
			hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

			if savefig:
				# figtitle = "kink_example_Nevals{0:d}_kernel_{1:s}_nonlinsys_{2:s}.png".format(jj+1,which_kernel,which_nonlin_sys)
				figtitle = "kink_example_Nevals{0:d}_kernel_{1:s}_nonlinsys_{2:s}.png".format(jj+1,which_kernel,which_nonlin_sys)
				path2save_tot = path2save + figtitle
				logger.info("Saving fig ...")
				hdl_fig.savefig(path2save_tot,bbox_inches='tight',dpi=300,transparent=True)
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
def test_single(cfg: dict) -> None:

	# which_kernel = "gaussian"
	# which_kernel = "matern"
	# which_kernel = "kink_randomized"
	which_kernel = "kink"

	# which_nonlin_sys = "true"
	which_nonlin_sys = "wrong"
	
	train_test_kink(cfg, block_plot=True, which_kernel=which_kernel, which_nonlin_sys=which_nonlin_sys, Nobs = 15, random_pars=None, my_seed=1, plotting=True)


@hydra.main(config_path="./config",config_name="config")
def test_statistical(cfg: dict) -> None:
	
	which_nonlin_sys = "sampled" # Do not change

	# which_kernel = "gaussian"
	which_kernel = "matern"
	# which_kernel = "kink_randomized"

	my_seed = 1
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	Nsampled_systems = 100
	Nobs = 20 # Collect this amount of observations
	log_evidence_loss_mat = np.zeros((Nsampled_systems,Nobs))
	rmse_mat = np.zeros((Nsampled_systems,Nobs))
	for jj in range(Nsampled_systems):

		logger.info("Computing log-evidence for sampled system {0:d} / {1:d} ...".format(jj+1,Nsampled_systems))

		a0_min = -1.0; a0_max = +1.0; a0 = a0_min + (a0_max-a0_min)*np.random.rand(1)
		a1_min = -2.0; a1_max = +2.0; a1 = a1_min + (a1_max-a1_min)*np.random.rand(1)
		a2_min = -4.0; a2_max = -1.0; a2 = a2_min + (a2_max-a2_min)*np.random.rand(1)
		random_pars = dict(a0=a0,a1=a1,a2=a2)

		log_evidence_loss_mat[jj,...] = train_test_kink(cfg, block_plot=False, which_kernel=which_kernel, which_nonlin_sys=which_nonlin_sys, Nobs=Nobs, random_pars=random_pars, my_seed=None, plotting=False)
		# rmse_mat[jj,...] = train_test_kink(cfg, block_plot=False, which_kernel=which_kernel, which_nonlin_sys=which_nonlin_sys, Nobs=Nobs, random_pars=random_pars, my_seed=None, plotting=False)


	# Save log loss:
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_{0:s}/log_loss_statistical_{1:s}.pickle".format(which_nonlin_sys,which_kernel)
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_{0:s}/rmse_statistical_{1:s}.pickle".format(which_nonlin_sys,which_kernel)
	logger.info("Saving regression fit error at: {0:s} ...".format(path2save))
	file = open(path2save, 'wb')
	pickle.dump(log_evidence_loss_mat,file)
	# pickle.dump(rmse_mat,file)
	file.close()
	logger.info("Done!")


def test_plot_log_loss_evolution():

	which_kernel_list = ["gaussian","matern","kink_randomized"]
	which_label_list = ["Gaussian kernel", "Matern kernel","Kink kernel"]

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(24,10),sharex=True)
	hdl_fig.suptitle(r"RMSE($f(x_t;\theta_{rand}) - E[f(x_t) | \mathcal{D}]$)",fontsize=fontsize_labels)
	ii = 0
	alpha = 0.3
	color_list = ["crimson","darkgreen","navy"]
	for which_kernel in which_kernel_list:
		# path2load = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_sampled/log_loss_statistical_{0:s}.pickle".format(which_kernel)
		path2load = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_sampled/rmse_statistical_{0:s}.pickle".format(which_kernel)
		file = open(path2load, 'rb')
		log_evidence_loss_vec = pickle.load(file)
		file.close()

		Nobs = log_evidence_loss_vec.shape[1]
		Nobs_vec = np.arange(1,Nobs+1)
		log_evidence_loss_vec_mean = np.mean(log_evidence_loss_vec,axis=0)
		log_evidence_loss_vec_std = np.std(log_evidence_loss_vec,axis=0)
		hdl_splots.plot(Nobs_vec,log_evidence_loss_vec_mean,linestyle="-",color=color_list[ii],lw=2,alpha=alpha,label=which_label_list[ii])
		hdl_splots.fill_between(Nobs_vec,log_evidence_loss_vec_mean - log_evidence_loss_vec_std,log_evidence_loss_vec_mean + log_evidence_loss_vec_std,color=color_list[ii],alpha=alpha)

		ii += 1

	hdl_splots.set_title("")
	hdl_splots.set_xlim([Nobs_vec[0],Nobs_vec[-1]])
	# hdl_splots.set_ylabel(r"$-\log p(x_{t+1} | x_t, D)$",fontsize=fontsize_labels)
	hdl_splots.set_ylabel(r"RMSE",fontsize=fontsize_labels)
	hdl_splots.set_xlabel(r"Nr. Observations",fontsize=fontsize_labels)
	hdl_splots.set_xticks([1,5,10,15,20])
	hdl_splots.legend(loc="best",fontsize=fontsize_labels)

	plt.show(block=True)


if __name__ == "__main__":

	test_single()

	# test_statistical()

	# test_plot_log_loss_evolution()



