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


def construct_reconstruction_loss(inverse_fourier_toolbox,xpred,fx_true,sigma_noise_stddev):

	def loss_integrand(omega_in,Dw_vec):
		"""
		To be maximized

		"""
		omega_in = tf.convert_to_tensor(omega_in,dtype=tf.float32)

		if omega_in.ndim == 1:
			omega_in = tf.expand_dims(omega_in,axis=0)

		if omega_in.shape[1] != 2 and omega_in.shape[0] == 2:
			omega_in = tf.transpose(omega_in)

		inverse_fourier_toolbox.update_spectral_density_and_angle(omega_in,None)
		fx_integrand = inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred,Dw_vec) # [Npoints, Nomegas]
		fx = tf.math.reduce_sum(fx_integrand,axis=1,keepdims=True) # Integrate wrt omegas [Npoints, 1]
		loss_val = tf.reduce_mean(0.5*((fx_true - fx)/sigma_noise_stddev)**2,axis=0,keepdims=True) # [1, 1]

		# Offset:
		loss_out = loss_val # [1, 1]

		return loss_out # [1, 1]

	return loss_integrand

def construct_log_integrand_with_offset(inverse_fourier_toolbox,xpred,fx_true,sigma_noise_stddev,loss_min,use_rme_of_integrand_as_loss=False):

	def log_integrand_with_offset(omega_in,Dw):
		"""
		To be maximized


		"""
		omega_in = tf.convert_to_tensor(omega_in,dtype=tf.float32)

		if omega_in.ndim == 1:
			omega_in = tf.expand_dims(omega_in,axis=0)

		if omega_in.shape[1] != 2 and omega_in.shape[0] == 2:
			omega_in = tf.transpose(omega_in)

		inverse_fourier_toolbox.update_spectral_density_and_angle(omega_in,Dw)
		fx = inverse_fourier_toolbox.get_fx_integrand(xpred,Dw) # [Npoints, Nomegas]
		loss_val = -tf.reduce_mean(0.5*((fx_true - fx)/sigma_noise_stddev)**2,axis=0,keepdims=True) # [1, Nomegas]

		# We amplify the integrand. We collect samples were there's more going on:
		if use_rme_of_integrand_as_loss: 
			loss_val = np.sqrt(tf.reduce_mean(fx**2,axis=0,keepdims=True))
			loss_min_local = 0.0 # [1, Nomegas]
		else:
			loss_min_local = loss_min
		# pdb.set_trace()

		# Offset:
		loss_out = loss_val - loss_min_local # [1, Nomegas]
		# loss_out = loss_val # [1, Nomegas]

		loss_out = loss_out[0,:]

		return loss_out # [Nomegas,]

	return log_integrand_with_offset


def construct_fun_elbo_loss(log_integrand_with_offset,dim_in,Nomegas,Nsamples_per_eval,return_std=False):

	def fun_elbo_loss(omega_lim,Dw):
		"""
		To be minimized


		"""

		loss_vec = np.zeros(Nsamples_per_eval)
		for jj in range(Nsamples_per_eval):

			# logger.info(" * sample: {0:d}".format(jj+1))

			# Sample from the hypercube, for the queried value of omega_lim:
			omegapred = -omega_lim + 2.*omega_lim*tf.math.sobol_sample(dim=dim_in,num_results=Nomegas,skip=2000 + jj*1000)

			# Compute -E_w[log p(y|x,w)], w ~ q(w)
			log_loss_vec = -log_integrand_with_offset(omegapred,Dw=1.0) # [Nomegas,]
			loss_vec[jj] = tf.reduce_mean(log_loss_vec) # This is an approximate integral


		# Compute entropy H(q):
		loss_entropy = -dim_in * np.log(omega_lim)

		loss_mean = tf.reduce_mean(loss_vec).numpy()
		loss_std = tf.math.reduce_std(loss_vec).numpy()

		loss_tot = loss_mean + loss_entropy

		if return_std:
			return loss_tot, loss_std
		else:
			return loss_tot

	return fun_elbo_loss







@hydra.main(config_path="./config",config_name="config")
def reconstruct(cfg):





	"""
	Remarks:
	*) It looks like for each Ndiv there's an optimal L that makes the reconstruction loss go to zero
	*) Neither MCMC nor VarInf seem to yield good results. It obviously has to do with the integration in irregular grids
	*) Using the RMS of the integrand itself helps; and maybe that's our likelihood, although without observations, which is weird. Maybe use it as utility?
	
	Steps:
	1) In the Dubins car example, select a suepr large grid and do line search to find the right L. 
	2) Heuristic approach: For pruning the grid of omegas we have 2 options:
		1) Remove those omegas such that the integrand (or better, its RMS) is close to zero after averaging over xt
		2) Remove those omegas such that the reconstruction loss remains approximately the same
	3) To test, compute the reconstruction loss before and after pruning and compare
	4) Plot in xt

	Actually, should be done first for the parabola case

	Another option:
	a) Select wlim by looking at were the RMS of the integrand is above a certain value
	b) Out of wlim, compute L and Ndiv. Or maybe optimize for Ndiv here directly...

	"""








	"""
	1) Make sure the sign of the entropy is correct in ELBO
	2) Figure out why changing Dw is so disruptive
	3) Seems that Dw is very important. Maybe pass it as an input argument to inverse_fourier_toolbox.update_spectral_density_and_angle()


	"""
	# integration_method = "integrate_with_regular_grid"
	# integration_method = "integrate_with_irregular_grid"
	# integration_method = "integrate_with_bayesian_quadrature"
	integration_method = "integrate_with_data"

	dim_in = 1
	dim_out_ind = 0

	"""
	Create training dataset
	"""
	xmin_training = -10.0
	xmax_training = +10.0
	Ndiv_training = 4001
	xpred_training = CommonUtils.create_Ndim_grid(xmin=xmin_training,xmax=xmax_training,Ndiv=Ndiv_training,dim=dim_in) # [Ndiv**dim_in,dim_in]
	
	# fx_true_training = KinkSpectralDensity._nonlinear_system_fun_static(xpred_training)
	fx_true_training = ParaboloidSpectralDensity._nonlinear_system_fun_static(xpred_training)
	
	fx_true_training = fx_true_training[:,dim_out_ind:dim_out_ind+1] # [Ndiv,1]
	
	spectral_density = SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_in)
	# spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_in)
	# spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,Xtrain=xpred_training,Ytrain=fx_true_training,use_nominal_model=True)
	# spectral_density = ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,Xtrain=xpred_training,Ytrain=fx_true_training)

	# Initialize spectral density and class to reconstruct the original function:
	# spectral_density = ParaboloidSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,Xtrain=xpred_training,Ytrain=fx_true_training)
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in,dim_out_ind=None)



	"""
	Create testing dataset
	"""
	# xmin_testing = -5.0
	# xmax_testing = +2.0
	# Ndiv_testing = 201
	xmin_testing = -10.0
	xmax_testing = +10.0
	Ndiv_testing = 4001
	xpred_testing = CommonUtils.create_Ndim_grid(xmin=xmin_testing,xmax=xmax_testing,Ndiv=Ndiv_testing,dim=dim_in) # [Ndiv**dim_in,dim_in]
	fx_true_testing = spectral_density._nonlinear_system_fun(xpred_testing)
	fx_true_testing = fx_true_testing[:,dim_out_ind:dim_out_ind+1] # [Ndiv,1]




	"""
	Discrete grid of omega, for plotting and analysis:
	"""

	Ndiv_omega_for_analysis = 301
	omega_lim = 3.0
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_in) # [Ndiv**dim_in,dim_in]
	Dw_analysis = (2.*omega_lim)**dim_in / omegapred_analysis.shape[0]
	Sw_vec, phiw_vec = spectral_density.unnormalized_density(omegapred_analysis)

	if np.all(phiw_vec == 0.0):
		phiw_vec = np.zeros(Sw_vec.shape)

	"""
	Number of required sampled frequencies
	"""

	# Dw = 1.0
	sigma_noise_stddev = 0.5 # Set it to small values in order to enlarge the loss output
	Nsamples_omega = 20
	# Nsamples_omega = 80
	Dw_samples_omega = (2.*omega_lim)**dim_in / Nsamples_omega

	# Loss minimum value (will be reached when the integrand goes to zero)
	loss_min = -tf.reduce_mean(0.5*fx_true_testing**2/sigma_noise_stddev**2,axis=0,keepdims=True) # [1, Nomegas]


	hdl_fig, hdl_splots_reconstruct = plt.subplots(2,2,figsize=(12,8),sharex=False)
	hdl_splots_reconstruct[0,0].plot(xpred_testing,fx_true_testing,lw=1)
	hdl_splots_reconstruct[0,0].set_xlim([xmin_testing,xmax_testing])
	hdl_splots_reconstruct[0,0].set_xticks([])
	hdl_splots_reconstruct[0,0].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[0,0].set_title("Reconstruction | MCMC Samples",fontsize=fontsize_labels)

	hdl_splots_reconstruct[1,0].plot(xpred_testing,fx_true_testing,lw=1,label="True function")
	hdl_splots_reconstruct[1,0].set_xlim([xmin_testing,xmax_testing])
	hdl_splots_reconstruct[1,0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1,0].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1,0].set_title("Reconstruction | VarInf Samples",fontsize=fontsize_labels)

	hdl_splots_reconstruct[0,1].plot(omegapred_analysis,Sw_vec,lw=1)
	hdl_splots_reconstruct[0,1].set_xlim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[0,1].set_xticks([])
	hdl_splots_reconstruct[0,1].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1,1].plot(omegapred_analysis,phiw_vec,lw=1)
	hdl_splots_reconstruct[1,1].set_xlim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[1,1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1,1].set_ylabel(r"$\varphi(\omega)$",fontsize=fontsize_labels)



	# As sanity check, plot already here the reconstructed function:
	delta_statespace = (xmax_testing-xmin_testing)**dim_in / Ndiv_testing

	
	# Nomegas_coarse = 201
	# L = 100.0
	# assert Nomegas_coarse % 2 != 0 and Nomegas_coarse > 2, "Nomegas_coarse must be an odd positive integer"
	# j_indices = CommonUtils.create_Ndim_grid(xmin=-(Nomegas_coarse-1)//2,xmax=(Nomegas_coarse-1)//2,Ndiv=Nomegas_coarse,dim=dim_in) # [Ndiv**dim_in,dim_in]
	# omegapred_coarse = tf.cast((math.pi/L) * j_indices,dtype=tf.float32)
	# Dw_coarse = (math.pi/L)**dim_in
	# inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred_coarse,Dw=Dw_coarse) # Here we are using a different Dw; Seems that Dw is very important. Maybe pass it as an input argument to inverse_fourier_toolbox.update_spectral_density_and_angle()
	# fx_discrete_grid = inverse_fourier_toolbox.get_fx(xpred_testing)


	"""
	Plots for presentation


	"""

	xpred_testing = np.copy(xpred_training)
	fx_true_testing = np.copy(fx_true_training)

	Nomegas_coarse = 21
	omega_lim_coarse = 4.0
	omegapred_coarse = CommonUtils.create_Ndim_grid(xmin=-omega_lim_coarse,xmax=omega_lim_coarse,Ndiv=Nomegas_coarse,dim=dim_in) # [Ndiv**dim_in,dim_in]
	Dw_coarse =  (2.*omega_lim_coarse)**dim_in / omegapred_coarse.shape[0]
	
	inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred=omegapred_coarse,Dw=None,dX=delta_statespace)
	fx_integrand = inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred=xpred_testing,Dw_vec=Dw_coarse) # [Npoints, Nomegas]
	fx_reconstructed = tf.math.reduce_sum(fx_integrand,axis=1,keepdims=True) # Integrate wrt omegas [Npoints, 1]
	reconstructor_fx_deltas_only = ReconstructFunctionFromSpectralDensity(	dim_in=dim_in,omega_lim=omega_lim_coarse,Nomegas=Nomegas_coarse,
																inverse_fourier_toolbox=inverse_fourier_toolbox,
																Xtrain=xpred_testing,Ytrain=fx_true_testing,omegas_weights=None)
	reconstructor_fx_deltas_only.train(Nepochs=10,learning_rate=1e-2,stop_loss_val=0.001)
	fx_optimized_voxels_coarse = reconstructor_fx_deltas_only.reconstruct_function_at(xpred=xpred_testing)
	omegapred_coarse_reconstr = reconstructor_fx_deltas_only.get_omegas_weights()

	hdl_fig, hdl_splots_reconstruct = plt.subplots(1,3,figsize=(30,10),sharex=False)
	hdl_splots_reconstruct[0].plot(xpred_testing,fx_true_testing,lw=2,color="navy",alpha=0.35,label="True")
	# hdl_splots_reconstruct[0].plot(xpred_testing,fx_reconstructed,lw=2,color="navy",alpha=0.5)
	hdl_splots_reconstruct[0].plot(xpred_testing,fx_optimized_voxels_coarse,lw=2,color="navy",alpha=0.7,label="Reconstructed")
	# hdl_splots_reconstruct[0].plot(xpred_testing,fx_optimized_voxels_coarse,lw=1)
	# hdl_splots_reconstruct[0].plot(xpred_testing,fx_discrete_grid,lw=1)
	# hdl_splots_reconstruct[0].set_xlim([-5,2])
	hdl_splots_reconstruct[0].set_xlim([xmin_testing,xmax_testing])
	# hdl_splots_reconstruct[0].set_ylim([-45.,2.])
	hdl_splots_reconstruct[0].set_xticks([xmin_testing,0,xmax_testing])
	hdl_splots_reconstruct[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[0].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[0].set_title(r"Reconstruction; $M=20$",fontsize=fontsize_labels)
	
	hdl_splots_reconstruct[1].plot(omegapred_analysis,Sw_vec,lw=2,color="crimson",alpha=0.6)
	hdl_splots_reconstruct[1].set_xlim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[1].set_xticks([-omega_lim,0,omega_lim])
	hdl_splots_reconstruct[1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1].set_title(r"Spectral density $S(\omega)$",fontsize=fontsize_labels)
	# Sw_coarse_reconstr, phiw_coarse_reconstr = spectral_density.unnormalized_density(omegapred_coarse_reconstr)
	# hdl_splots_reconstruct[1].stem(omegapred_coarse_reconstr[:,0],Sw_coarse_reconstr[:,0],linefmt="crimson",markerfmt=".")
	Sw_coarse_reconstr_interp = np.interp(x=omegapred_coarse_reconstr[:,0],xp=omegapred_analysis[:,0],fp=Sw_vec[:,0])
	phiw_coarse_reconstr_interp = np.interp(x=omegapred_coarse_reconstr[:,0],xp=omegapred_analysis[:,0],fp=phiw_vec[:,0])
	hdl_splots_reconstruct[1].plot(omegapred_coarse_reconstr[:,0],Sw_coarse_reconstr_interp,linestyle="None",marker=".",color="crimson",markersize=8)

	hdl_splots_reconstruct[2].plot(omegapred_analysis,phiw_vec,lw=2,color="crimson",alpha=0.6)
	hdl_splots_reconstruct[2].set_xlim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[2].set_xticks([-omega_lim,0,omega_lim])
	hdl_splots_reconstruct[2].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[2].set_ylabel(r"$\varphi(\omega)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[2].set_title(r"Phase $\varphi(\omega)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[2].plot(omegapred_coarse_reconstr[:,0],phiw_coarse_reconstr_interp,linestyle="None",marker=".",color="crimson",markersize=8)

	plt.show(block=True)


	"""
	Plot the integrand
	"""
	
	inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred=omegapred_analysis,Dw=Dw_analysis)
	fx_integrand_for_xpred_and_omegapred_analisys = inverse_fourier_toolbox.get_fx_integrand(xpred_testing,Dw_analysis).numpy() # [Npoints, Nomegas]
	fx_selected_for_plotting = fx_integrand_for_xpred_and_omegapred_analisys[np.random.randint(0,fx_integrand_for_xpred_and_omegapred_analisys.shape[0],size=(3)),:]
	fx_integral_for_each_omega = np.mean(fx_integrand_for_xpred_and_omegapred_analisys,axis=0)

	hdl_fig, hdl_splots_integrand = plt.subplots(4,1,figsize=(14,9),sharex=False)

	for jj in range(fx_selected_for_plotting.shape[0]):
		hdl_splots_integrand[0].plot(omegapred_analysis,fx_selected_for_plotting[jj,:],lw=0.5,color="grey",alpha=0.5)
		hdl_splots_integrand[0].plot(omegapred_analysis,fx_selected_for_plotting[jj,:],lw=0.5,color="grey",alpha=0.5)
	hdl_splots_integrand[0].plot(omegapred_analysis,fx_integral_for_each_omega,lw=1.0,color="crimson",alpha=0.75)
	hdl_splots_integrand[0].set_xlim([-omega_lim,omega_lim])
	hdl_splots_integrand[0].set_xticks([])
	hdl_splots_integrand[0].set_ylabel(r"$g(x_t,\omega)$",fontsize=fontsize_labels)
	hdl_splots_integrand[0].set_title(r"Integrand $g(x_t;\omega) = S(\omega)\cos(\omega^\top x_t + \varphi(\omega))\Delta \omega$ for different $x_t$ and average $(1/T) \sum_t g(x_t,\omega)$",fontsize=fontsize_labels)


	# plt.show(block=True)


	# Log-evidence:
	log_integrand_with_offset = construct_log_integrand_with_offset(inverse_fourier_toolbox,xpred_testing,fx_true_testing,sigma_noise_stddev,loss_min,use_rme_of_integrand_as_loss=False)
	log_loss_vec = log_integrand_with_offset(omegapred_analysis,Dw=1.0) # [Nomegas,]; Why Dw_analysis=1.0? Because here we're formulating a proxy likelihood that compares the original with the observations, but without integrating anything. Just comparing. Hence the integrand should not include the integration step.
	# log_loss_vec = log_integrand_with_offset(omegapred_analysis,Dw=Dw_analysis) # [Nomegas,]

	log_loss_exp_normconst = np.mean(np.exp(log_loss_vec))
	loss_prob_vec = np.exp(log_loss_vec) / log_loss_exp_normconst

	hdl_splots_integrand[1].plot(omegapred_analysis,log_loss_vec,lw=1)
	hdl_splots_integrand[1].set_xlim([-omega_lim,omega_lim])
	hdl_splots_integrand[1].set_xticks([])
	hdl_splots_integrand[1].set_ylabel(r"$\log p(y | x,\omega)$",fontsize=fontsize_labels)
	hdl_splots_integrand[1].set_title(r"Log evidence",fontsize=fontsize_labels)
	# print("loss_min:",loss_min)
	# plt.show(block=True)

	hdl_splots_integrand[2].plot(omegapred_analysis,loss_prob_vec,lw=1)
	hdl_splots_integrand[2].set_xlim([-omega_lim,omega_lim])
	hdl_splots_integrand[2].set_ylabel(r"$p(y | x,\omega)$",fontsize=fontsize_labels)
	hdl_splots_integrand[2].set_title(r"Evidence - normalized",fontsize=fontsize_labels)
	hdl_splots_integrand[2].set_xticks([])



	# Prior:
	mean_w = 0.0*np.ones(dim_in)
	var_w = 1.0*np.ones(dim_in)



	Dw_log_posterior = (2.*omega_lim)**dim_in / omegapred_analysis.shape[0]
	log_prior = scipy.stats.multivariate_normal.logpdf(x=omegapred_analysis,mean=mean_w,cov=np.diag(var_w)) # [omegapred_analysis.shape[0]]
	log_posterior = log_loss_vec + log_prior - np.log(np.sum(np.exp(log_loss_vec + log_prior))*Dw_log_posterior)
	hdl_splots_integrand[3].plot(omegapred_analysis,log_posterior,lw=1)
	hdl_splots_integrand[3].set_xlim([-omega_lim,omega_lim])
	hdl_splots_integrand[3].set_ylabel(r"$\propto \log p(\omega | y,x)$",fontsize=fontsize_labels)
	hdl_splots_integrand[3].set_title(r"Proportional log-Posterior",fontsize=fontsize_labels)
	hdl_splots_integrand[3].set_xlabel(r"$\omega$",fontsize=fontsize_labels)


	# plt.show(block=True)
	# 
	

	"""
	Sample from posterior using MCMC
	================================
	"""

	Nburning = 20
	Nrestarts = 5

	def target_log_lik(omegapred_in):
		return log_integrand_with_offset(omega_in=omegapred_in,Dw=1.0)
		# return log_integrand_with_offset(omega_in=omegapred_in,Dw=Dw_analysis)

	sampler = EllipticalSliceSampler(dim_in=dim_in,target_log_lik=target_log_lik,
									Nsamples=Nsamples_omega,Nburning=Nburning,
									Nrestarts=Nrestarts,omega_lim_random_restarts=omega_lim,
									mean_w=mean_w,var_w=var_w)
	samples_omega_mcmc_vec, omega0_restarts = sampler.run_ess() # samples_omega_mcmc_vec: [Nrestarts*Nsamples,dim_in]

	hdl_splots_integrand[3].plot(samples_omega_mcmc_vec[:,0],np.amin(log_posterior)*np.ones(samples_omega_mcmc_vec.shape[0]),linestyle="None",marker="x",markersize=8,color="crimson")


	"""
	Approximate posterior using variational inference
	=================================================
	"""




	Nsamples_per_eval_elbo = 4
	fun_elbo_loss = construct_fun_elbo_loss(log_integrand_with_offset,dim_in,Nsamples_omega,Nsamples_per_eval_elbo,return_std=True)

	omega_lim_analysis_upper_lim = 20.0
	omega_lim_Ndiv = 31

	fun_elbo_loss_mean_vec = np.zeros(omega_lim_Ndiv)
	fun_elbo_loss_std_vec = np.zeros(omega_lim_Ndiv)
	omega_lim_vec = np.linspace(0.1,omega_lim_analysis_upper_lim,omega_lim_Ndiv)
	for ii in range(omega_lim_Ndiv):
		# Dw_uniform_for_optimizing_omega_lim = (2.*omega_lim_vec[ii]) / Nsamples_omega
		fun_elbo_loss_mean_vec[ii], fun_elbo_loss_std_vec[ii] = fun_elbo_loss(omega_lim_vec[ii],Dw=1.0)

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,9),sharex=False)
	hdl_splots.errorbar(omega_lim_vec,fun_elbo_loss_mean_vec,yerr=fun_elbo_loss_std_vec)
	hdl_splots.set_xlim([0.1,omega_lim_analysis_upper_lim])
	hdl_splots.set_xlabel(r"$\omega_{lim}$",fontsize=fontsize_labels)
	hdl_splots.set_ylabel(r"$\mathcal{L}(\omega_{lim})$",fontsize=fontsize_labels)
	hdl_splots.set_title(r"ELBO Loss",fontsize=fontsize_labels)
	
	fun_elbo_loss = construct_fun_elbo_loss(log_integrand_with_offset,dim_in,Nsamples_omega,Nsamples_per_eval_elbo,return_std=False)
	def fun_elbo_loss_for_scipy(omega_in):
		return fun_elbo_loss(omega_in,Dw=1.0)
	omega_lim0 = np.array([2.0])
	result = scipy.optimize.minimize(fun_elbo_loss_for_scipy,omega_lim0,method="L-BFGS-B",bounds=[(0.1,+omega_lim)],options=dict(maxiter=20))

	omega_lim_opti = result.x
	fun_elbo_loss_opti = fun_elbo_loss(omega_lim_opti,Dw=1.0)
	hdl_splots.plot(omega_lim_opti,fun_elbo_loss_opti,markersize=10,marker="*",markerfacecolor="crimson")


	# Now, given the optimal parameter, sample a bunch of sets of omegas and get the best:
	Nsamples_irregular_omegapred_varinf = 20
	Dw_omega_var_inf_vec_this_sample = (2.*omega_lim_opti)/Nsamples_omega
	samples_omega_var_inf_vec = 0.0
	loss_varinf_best_sample = +np.inf
	for ii in range(Nsamples_irregular_omegapred_varinf):
		samples_omega_var_inf_vec_this_sample = -omega_lim_opti + 2.*omega_lim_opti*tf.math.sobol_sample(dim=dim_in,num_results=Nsamples_omega,skip=2000 + np.random.randint(0,100)*100)
		inverse_fourier_toolbox.update_spectral_density_and_angle(samples_omega_var_inf_vec_this_sample,Dw=Dw_omega_var_inf_vec_this_sample) # Here we are using a different Dw; Seems that Dw is very important. Maybe pass it as an input argument to inverse_fourier_toolbox.update_spectral_density_and_angle()
		fx_samples_omega_var_inf_vec_this_sample = inverse_fourier_toolbox.get_fx(xpred_testing)
		loss_varinf_best_sample_new = tf.reduce_mean(0.5*((fx_true_testing[:,0] - fx_samples_omega_var_inf_vec_this_sample)/1.0)**2) # [1,]
		if loss_varinf_best_sample_new < loss_varinf_best_sample:
			loss_varinf_best_sample = loss_varinf_best_sample_new
			samples_omega_var_inf_vec = samples_omega_var_inf_vec_this_sample

	# samples_omega_var_inf_vec = -omega_lim_opti + 2.*omega_lim_opti*tf.math.sobol_sample(dim=dim_in,num_results=Nsamples_omega,skip=2000 + np.random.randint(0,100)*100)
	hdl_splots_integrand[3].plot(samples_omega_var_inf_vec[:,0],np.amin(log_posterior)*np.ones(samples_omega_var_inf_vec.shape[0]),linestyle="None",marker="o",markersize=5,color="darkgreen")

	

	# # DBG: replace these samples with samples from a super wide domain:
	# # 
	# samples_omega_var_inf_vec = -3.0 + 2.*3.0*tf.math.sobol_sample(dim=dim_in,num_results=Nsamples_omega,skip=2000 + np.random.randint(0,100)*100)

	"""
	Figure out the scales for the VarInf solution using NN training:
	"""

	# loss_integrand = construct_reconstruction_loss(inverse_fourier_toolbox,xpred,fx_true_testing,sigma_noise_stddev)
	# voxel_val = (2.*omega_lim_opti)/samples_omega_var_inf_vec.shape[0]
	# nn4integrand = NNforIntegrand(dim_in=dim_in,voxel_val=voxel_val,samples_omega=samples_omega_mcmc_vec,target_loss=loss_integrand) # With MCMC

	reconstructor_fx_deltas_only = ReconstructFunctionFromSpectralDensity(	dim_in=dim_in,omega_lim=omega_lim_opti,Nomegas=Nsamples_omega,
																inverse_fourier_toolbox=inverse_fourier_toolbox,
																Xtrain=xpred_testing,Ytrain=fx_true_testing,omegas_weights=samples_omega_var_inf_vec) # With VarInf
	# nn4integrand = NNforIntegrand(dim_in=dim_in,voxel_val=voxel_val,samples_omega=samples_omega_var_inf_vec,target_loss=loss_integrand,omega_lim=None,Nepochs=Nepochs) # With VarInf
	reconstructor_fx_deltas_only.train(Nepochs=1000,learning_rate=1e-2,stop_loss_val=0.001)

	fx_varinf_optimized_voxels = reconstructor_fx_deltas_only.reconstruct_function_at(xpred=xpred_testing)
	# delta_omegas_trainedNN = reconstructor_fx_deltas_only.get_delta_omegas(reconstructor_fx_deltas_only.delta_omegas_pre_activation)
	# inverse_fourier_toolbox.update_spectral_density_and_angle(samples_omega_var_inf_vec,Dw=None)
	# fx_integrand_varinf_optimized_voxels = inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred,delta_omegas_trainedNN)
	# fx_varinf_optimized_voxels = tf.math.reduce_sum(fx_integrand_varinf_optimized_voxels,axis=1,keepdims=True) # Integrate wrt omegas [Npoints, 1]
	
	hdl_splots_reconstruct[1,0].plot(xpred_testing,fx_varinf_optimized_voxels,lw=1,label=r"$\omega$ from VarInf, $\Delta \omega$ from NN")

	


	"""
	Figure out the scales AND omegas directly:
	"""


	reconstructor_fx_deltas_and_omegas = ReconstructFunctionFromSpectralDensity(	dim_in=dim_in,omega_lim=omega_lim_opti,Nomegas=Nsamples_omega,
																inverse_fourier_toolbox=inverse_fourier_toolbox,
																Xtrain=xpred_testing,Ytrain=fx_true_testing,omegas_weights=None)
	reconstructor_fx_deltas_and_omegas.train(Nepochs=1000,learning_rate=1e-2,stop_loss_val=0.001)
	fx_optimized_omegas_and_voxels = reconstructor_fx_deltas_and_omegas.reconstruct_function_at(xpred=xpred_testing)
	omegas_trainedNN = reconstructor_fx_deltas_and_omegas.get_omegas_weights()
	fx_integrand_optimized_omegas_and_voxels = reconstructor_fx_deltas_and_omegas.get_integrand_for_pruning(xpred=xpred_testing)
	# ind_irrelevant_omegas = abs(tf.reduce_mean(fx_integrand_optimized_omegas_and_voxels,axis=0)) < 0.001
	# np.sum(ind_irrelevant_omegas)
	# omegas_trainedNN.numpy()[ind_irrelevant_omegas.numpy(),:]
	# pdb.set_trace()

	# loss_integrand = construct_reconstruction_loss(inverse_fourier_toolbox,xpred,fx_true_testing,sigma_noise_stddev)
	# voxel_val = (2.*omega_lim_opti)/samples_omega_var_inf_vec.shape[0]
	# # del nn4integrand.delta_omegas_pre_activation
	# # del nn4integrand.sam
	# # del nn4integrand
	# # nn4integrand = NNforIntegrand(dim_in=dim_in,voxel_val=voxel_val,samples_omega=samples_omega_mcmc_vec,target_loss=loss_integrand) # With MCMC
	# Nepochs = 1500
	# nn4integrand = NNforIntegrand(dim_in=dim_in,voxel_val=voxel_val,samples_omega=samples_omega_var_inf_vec,target_loss=loss_integrand,omega_lim=omega_lim_opti,Nepochs=Nepochs) # With VarInf
	# nn4integrand.train()

	# delta_omegas_trainedNN = nn4integrand.get_delta_omegas(nn4integrand.delta_omegas_pre_activation)
	# omegas_trainedNN = nn4integrand.get_omegas_weights()

	# inverse_fourier_toolbox.update_spectral_density_and_angle(omegas_trainedNN,Dw=None)
	# fx_integrand_varinf_optimized_voxels = inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred,delta_omegas_trainedNN)
	# fx_optimized_omegas_and_voxels = tf.math.reduce_sum(fx_integrand_varinf_optimized_voxels,axis=1,keepdims=True) # Integrate wrt omegas [Npoints, 1]
	
	hdl_splots_reconstruct[1,0].plot(xpred_testing,fx_optimized_omegas_and_voxels,lw=1,label=r"optimized $\omega$, $\Delta \omega$")

	hdl_splots_integrand[3].plot(omegas_trainedNN[:,0],np.amin(log_posterior)*np.ones(omegas_trainedNN.shape[0]),linestyle="None",marker="*",markersize=8,color="navy")


	"""
	Brute force optimize parameter L
	====================================
	"""
	Ndiv_omegaperd_discrete_grid_L_analysis_vec = np.array([5,21,51,101])
	hdl_fig, hdl_splots = plt.subplots(len(Ndiv_omegaperd_discrete_grid_L_analysis_vec),1,figsize=(14,9),sharex=True)
	for jj in range(len(Ndiv_omegaperd_discrete_grid_L_analysis_vec)):
		
		Ndiv_omegaperd_discrete_grid_L_analysis = Ndiv_omegaperd_discrete_grid_L_analysis_vec[jj]
		assert Ndiv_omegaperd_discrete_grid_L_analysis % 2 != 0 and Ndiv_omegaperd_discrete_grid_L_analysis > 2, "Ndiv_omegaperd_discrete_grid_L_analysis must be an odd positive integer"
		Ndiv_L = 101
		L_vec = np.linspace(1.0,100.0,Ndiv_L)
		loss_L_analysis_vec = np.zeros(Ndiv_L)
		for ii in range(Ndiv_L):

			if (ii+1) % 100 == 0: logger.info("Computing reconstruction log-loss for different L values, with {0:d} discrete points; iteration {1:d} / {2:d}".format(Ndiv_omegaperd_discrete_grid_L_analysis,ii+1,Ndiv_L))

			L = L_vec[ii]
			j_indices = CommonUtils.create_Ndim_grid(xmin=-(Ndiv_omegaperd_discrete_grid_L_analysis-1)//2,xmax=(Ndiv_omegaperd_discrete_grid_L_analysis-1)//2,Ndiv=Ndiv_omegaperd_discrete_grid_L_analysis,dim=dim_in) # [Ndiv_omegaperd_discrete_grid_L_analysis**dim_in,dim_in]
			omegapred_discrete_grid_L_analysis = tf.cast((math.pi/L) * j_indices,dtype=tf.float32)
			Dw_discrete_L_analysis = (math.pi/L)**dim_in
			inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred_discrete_grid_L_analysis,Dw=Dw_discrete_L_analysis) # Here we are using a different Dw; Seems that Dw is very important. Maybe pass it as an input argument to inverse_fourier_toolbox.update_spectral_density_and_angle()
			fx_discrete_grid_L_analysis = inverse_fourier_toolbox.get_fx(xpred_testing)
			loss_L_analysis = tf.reduce_mean(0.5*((fx_true_testing[:,0] - fx_discrete_grid_L_analysis)/1.0)**2) # [1,]
			loss_L_analysis_vec[ii] = np.log(loss_L_analysis)


		hdl_splots[jj].axhline(y=0.0,color="grey",linestyle='-',lw=0.75)
		hdl_splots[jj].plot(L_vec,loss_L_analysis_vec,marker="*",linestyle="--",lw=1.0)
		hdl_splots[jj].set_xlim([L_vec[0],L_vec[-1]])
		hdl_splots[jj].set_ylabel(r"$\log\mathcal{L}(L)$",fontsize=fontsize_labels)
		str_title = "Log-Reconstruction Loss for varying L; Discrete grid of {0:d} points; D = {1:d}".format(Ndiv_omegaperd_discrete_grid_L_analysis,dim_in)
		hdl_splots[jj].set_title(r"{0:s}".format(str_title),fontsize=fontsize_labels)
		if jj < len(Ndiv_omegaperd_discrete_grid_L_analysis_vec)-1: hdl_splots[jj].set_xticks([])

	hdl_splots[-1].set_xlabel(r"$L$",fontsize=fontsize_labels)
	# plt.show(block=True)



	"""
	Reconstruct functions with acquired samples
	===========================================
	"""

	# MCMC Samples
	samples_omega_mcmc_vec = tf.convert_to_tensor(samples_omega_mcmc_vec,dtype=tf.float32)
	Dw_omega_mcmc = (np.amax(samples_omega_mcmc_vec) - np.amin(samples_omega_mcmc_vec)) / samples_omega_mcmc_vec.shape[0] # NOTE: this is a heuristic, and it's only valid in 1D
	inverse_fourier_toolbox.update_spectral_density_and_angle(samples_omega_mcmc_vec,Dw=Dw_omega_mcmc)
	fx_mcmc = inverse_fourier_toolbox.get_fx(xpred_testing)

	# Variational inference samples
	Dw_uniform_for_omega_lim_opti = (2.*omega_lim_opti) / Nsamples_omega
	inverse_fourier_toolbox.update_spectral_density_and_angle(samples_omega_var_inf_vec,Dw=Dw_uniform_for_omega_lim_opti)
	fx_varinf = inverse_fourier_toolbox.get_fx(xpred_testing)

	inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred_analysis,Dw=Dw_analysis) # Here we are using a different Dw; Seems that Dw is very important. Maybe pass it as an input argument to inverse_fourier_toolbox.update_spectral_density_and_angle()
	fx_regular_grid = inverse_fourier_toolbox.get_fx(xpred_testing)


	# L = 500.
	# L = 100.
	# Ndiv = 2001
	Ndiv_omega_discrete_grid = 21
	L = 14.0

	assert Ndiv_omega_discrete_grid % 2 != 0 and Ndiv_omega_discrete_grid > 2, "Ndiv_omega_discrete_grid must be an odd positive integer"
	j_indices = CommonUtils.create_Ndim_grid(xmin=-(Ndiv_omega_discrete_grid-1)//2,xmax=(Ndiv_omega_discrete_grid-1)//2,Ndiv=Ndiv_omega_discrete_grid,dim=dim_in) # [Ndiv**dim_in,dim_in]
	omegapred_discrete_grid = tf.cast((math.pi/L) * j_indices,dtype=tf.float32)

	Dw_discrete = (math.pi/L)**dim_in
	inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred_discrete_grid,Dw=Dw_discrete) # Here we are using a different Dw; Seems that Dw is very important. Maybe pass it as an input argument to inverse_fourier_toolbox.update_spectral_density_and_angle()
	fx_discrete_grid = inverse_fourier_toolbox.get_fx(xpred_testing)

	hdl_splots_reconstruct[0,0].plot(xpred_testing,fx_mcmc,lw=1)
	hdl_splots_reconstruct[1,0].plot(xpred_testing,fx_varinf,lw=1,label="$\omega$ from VarInf, No NN")

	# hdl_splots_reconstruct[0,0].plot(xpred_testing,fx_regular_grid,lw=1)
	# hdl_splots_reconstruct[1,0].plot(xpred_testing,fx_regular_grid,lw=1)

	# hdl_splots_reconstruct[0,0].plot(xpred_testing,fx_discrete_grid,lw=1)
	# hdl_splots_reconstruct[1,0].plot(xpred_testing,fx_discrete_grid,lw=1)
	# 
	hdl_splots_reconstruct[1,0].legend()

	plt.show(block=True)





if __name__ == "__main__":

	reconstruct()

