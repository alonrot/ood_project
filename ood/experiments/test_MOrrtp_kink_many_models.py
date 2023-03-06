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


def nonlinsys_true(xpred):
	return KinkSpectralDensity._nonlinear_system_fun_static(xpred)

def nonlinsys_sampled_fixed(xpred):
	a0 = -0.8; a1 = -1.5; a2 = -3.0
	model_pars = dict(a0=a0,a1=a1,a2=a2)
	return KinkSpectralDensity._nonlinear_system_fun_static(xpred,model_pars)

def nonlinsys_sampled(xpred):
	a0_min = -1.0; a0_max = +1.0; a0 = a0_min + (a0_max-a0_min)*np.random.rand(1)
	a1_min = -2.0; a1_max = +2.0; a1 = a1_min + (a1_max-a1_min)*np.random.rand(1)
	a2_min = -4.0; a2_max = -1.0; a2 = a2_min + (a2_max-a2_min)*np.random.rand(1)
	model_pars = dict(a0=a0,a1=a1,a2=a2)
	return KinkSpectralDensity._nonlinear_system_fun_static(xpred,model_pars=model_pars)


def generate_data_from_multiple_kink_systems(Nsamples_nominal_dynsys,xmin,xmax,Ndiv_per_dim,nonlin_fun):

	xpred_training_single = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv_per_dim,dim=1) # [Ndiv**dim_in,dim_in]
	yeval_samples = np.zeros((Nsamples_nominal_dynsys,xpred_training_single.shape[0],1))
	for jj in range(Nsamples_nominal_dynsys):
		yeval_samples[jj,...] = nonlin_fun(xpred_training_single)

	xpred_training = np.concatenate([xpred_training_single]*Nsamples_nominal_dynsys,axis=0)
	ypred_training = np.reshape(yeval_samples,(-1,1),order="C")

	return xpred_training, ypred_training




def train_test_kink(cfg: dict, block_plot: bool, which_kernel: str, which_nonlin_sys = "true", Nobs = 20, random_pars=None, my_seed = None, plotting = True, savefig = False) -> None:

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
	# Nsamples_nominal_dynsys = 20
	Nsamples_nominal_dynsys = 1
	Ndiv_per_dim = 2001
	nonlinsys_for_data_generation = nonlinsys_true
	# nonlinsys_for_data_generation = nonlinsys_sampled

	xpred_training, fx_true_training = generate_data_from_multiple_kink_systems(Nsamples_nominal_dynsys=Nsamples_nominal_dynsys,xmin=xmin_training,
																				xmax=xmax_training,Ndiv_per_dim=Ndiv_per_dim,
																				nonlin_fun=nonlinsys_for_data_generation)
	if which_kernel == "kink":
		kernel_name_plot_label = "Elbow"
		use_nominal_model = True
		integration_method = "integrate_with_data"
		spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim_in=dim_in,integration_method=integration_method,Xtrain=xpred_training,Ytrain=fx_true_training)
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


	if which_nonlin_sys == "true":
		nonlinsys2use = nonlinsys_true
	elif which_nonlin_sys == "wrong":
		nonlinsys2use = nonlinsys_sampled_fixed
	elif which_nonlin_sys == "sampled":
		nonlinsys2use = nonlinsys_sampled
	


	"""
	Create testing dataset
	"""
	xpred_testing = np.copy(xpred_training)
	fx_true_testing = np.copy(fx_true_training)
	xmax_testing = xmax_training
	xmin_testing = xmin_training
	Ndiv_testing = xpred_testing.shape[0]
	delta_statespace = (xmax_testing-xmin_testing)**dim_in / Ndiv_testing
	# delta_statespace_vec = delta_statespace * np.ones((Ndiv_testing,1))


	Nomegas_coarse = 31
	omega_lim_coarse = 3.0
	omegapred_coarse = CommonUtils.create_Ndim_grid(xmin=-omega_lim_coarse,xmax=omega_lim_coarse,Ndiv=Nomegas_coarse,dim=dim_in) # [Ndiv**dim_in,dim_in]
	Dw_coarse =  (2.*omega_lim_coarse)**dim_in / omegapred_coarse.shape[0]
	# Dw_coarse_vec = Dw_coarse * np.ones((Nomegas_coarse,1))



	"""
	Reconstruct the mean: and plot it
	"""
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in)
	reconstructor_fx = ReconstructFunctionFromSpectralDensity(	dim_in=dim_in,dw_voxel_init=Dw_coarse,dX_voxel_init=delta_statespace,
																omega_lim=omega_lim_coarse,Nomegas=Nomegas_coarse,
																inverse_fourier_toolbox=inverse_fourier_toolbox,
																Xtest=xpred_testing,Ytest=fx_true_testing)

	Nepochs = 6000
	reconstructor_fx.train(Nepochs=Nepochs,learning_rate=1e-2,stop_loss_val=0.001)
	fx_optimized_voxels_coarse = reconstructor_fx.reconstruct_function_at(xpred=xpred_testing)
	spectral_density_optimized = reconstructor_fx.update_internal_spectral_density_parameters()
	omegapred_coarse_reconstr = reconstructor_fx.get_omegas_weights()
	Sw_coarse_reconstr = reconstructor_fx.inverse_fourier_toolbox.spectral_values
	phiw_coarse_reconstr = reconstructor_fx.inverse_fourier_toolbox.varphi_values


	if plotting:
		# hdl_fig, hdl_splots_reconstruct = plt.subplots(1,3,figsize=(30,10),sharex=False)
		hdl_fig, hdl_splots_reconstruct = plt.subplots(1,3,figsize=(12,8),sharex=False)
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
		hdl_splots_reconstruct[0].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[0].set_title(r"Reconstruction; $M=20$",fontsize=fontsize_labels)

		"""
		Discrete grid of omega, for plotting and analysis:
		"""

		Ndiv_omega_for_analysis = 301
		omega_lim = 3.0
		omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_in) # [Ndiv**dim_in,dim_in]
		Sw_vec, phiw_vec = spectral_density_optimized.unnormalized_density(omegapred_analysis)

		hdl_splots_reconstruct[1].plot(omegapred_analysis,Sw_vec,lw=2,color="crimson",alpha=0.6)
		hdl_splots_reconstruct[1].set_xlim([-omega_lim,omega_lim])
		hdl_splots_reconstruct[1].set_xticks([-omega_lim,0,omega_lim])
		hdl_splots_reconstruct[1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[1].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[1].set_title(r"Spectral density $S(\omega)$",fontsize=fontsize_labels)
		# Sw_coarse_reconstr, phiw_coarse_reconstr = spectral_density_optimized.unnormalized_density(omegapred_coarse_reconstr)
		# hdl_splots_reconstruct[1].stem(omegapred_coarse_reconstr[:,0],Sw_coarse_reconstr[:,0],linefmt="crimson",markerfmt=".")
		
		# Sw_coarse_reconstr_interp = np.interp(x=omegapred_coarse_reconstr[:,0],xp=omegapred_analysis[:,0],fp=Sw_vec[:,0])
		# phiw_coarse_reconstr_interp = np.interp(x=omegapred_coarse_reconstr[:,0],xp=omegapred_analysis[:,0],fp=phiw_vec[:,0])
		# hdl_splots_reconstruct[1].plot(omegapred_coarse_reconstr[:,0],Sw_coarse_reconstr_interp,linestyle="None",marker=".",color="crimson",markersize=8)

		Sw_coarse_reconstr_interp = np.interp(x=omegapred_coarse_reconstr[:,0],xp=omegapred_analysis[:,0],fp=Sw_vec[:,0])
		phiw_coarse_reconstr_interp = np.interp(x=omegapred_coarse_reconstr[:,0],xp=omegapred_analysis[:,0],fp=phiw_vec[:,0])
		hdl_splots_reconstruct[1].plot(omegapred_coarse_reconstr[:,0],Sw_coarse_reconstr,linestyle="None",marker=".",color="crimson",markersize=8)

		hdl_splots_reconstruct[2].plot(omegapred_analysis,phiw_vec,lw=2,color="crimson",alpha=0.6)
		hdl_splots_reconstruct[2].set_xlim([-omega_lim,omega_lim])
		hdl_splots_reconstruct[2].set_xticks([-omega_lim,0,omega_lim])
		hdl_splots_reconstruct[2].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[2].set_ylabel(r"$\varphi(\omega)$",fontsize=fontsize_labels)
		hdl_splots_reconstruct[2].set_title(r"Phase $\varphi(\omega)$",fontsize=fontsize_labels)
		# hdl_splots_reconstruct[2].plot(omegapred_coarse_reconstr[:,0],phiw_coarse_reconstr_interp,linestyle="None",marker=".",color="crimson",markersize=8)
		hdl_splots_reconstruct[2].plot(omegapred_coarse_reconstr[:,0],phiw_coarse_reconstr,linestyle="None",marker=".",color="crimson",markersize=8)

		plt.show(block=True)
		# plt.pause(1)


	"""
	Fit a new function (could be true, fixed sample or new sample, depending on our choice for nonlinsys2use())
	"""

	# Create grid for predictions:
	xmin = -6.0
	xmax = +3.0
	xpred_plotting = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=201,dim=dim_in) # [Ndiv**dim_in,dim_in]
	yplot_nonlin_sys = nonlinsys2use(xpred_plotting)

	# Get evaluations from a pre-generated sobol grid:
	Nevals_tot = 100
	Xtrain_tot = xmin + (xmax - xmin)*tf.math.sobol_sample(dim=dim_in,num_results=(Nevals_tot),skip=10000)
	Xtrain_tot = Xtrain_tot.numpy()
	Xtrain_tot = tf.convert_to_tensor(value=Xtrain_tot,dtype=tf.float32)

	Nevals = 1
	Xtrain = Xtrain_tot[0:Nevals,:]
	Ytrain = nonlinsys2use(Xtrain)

	Xtrain = tf.convert_to_tensor(value=Xtrain,dtype=tf.float32)
	Ytrain = tf.convert_to_tensor(value=Ytrain,dtype=tf.float32)

	if which_kernel == "kink":
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
	plotting_phase_added = False
	if plotting_phase_added:
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		Ndiv_phase_added = 100
		Nsample_paths = 3
		phase_added_vec = np.linspace(0.0,2.*np.pi,Ndiv_phase_added)
		for jj in range(Ndiv_phase_added):

			logger.info("phase: {0:f}".format(phase_added_vec[jj]))

			rrtp_MO.rrgpMO[0].dbg_phase_added_to_features = phase_added_vec[jj]

			MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_plotting,from_prior=True)
			sample_paths = rrtp_MO.sample_path_from_predictive(xpred_plotting,Nsamples=Nsample_paths,from_prior=True)
			sample_paths = sample_paths[0] # It's a list with one element, so get the first element
			hdl_splots.cla()
			hdl_splots.plot(xpred_plotting,MO_mean,linestyle="-",color="navy",lw=2,alpha=0.4)
			hdl_splots.fill_between(xpred_plotting[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
			hdl_splots.plot(xpred_plotting,yplot_nonlin_sys,marker="None",linestyle="-",color="crimson",lw=3,alpha=0.6)

			# Sample paths:
			for ii in range(Nsample_paths):
				if sample_paths is not None: hdl_splots.plot(xpred_plotting,sample_paths[:,ii],marker="None",linestyle="--",color="navy",lw=0.4)

			hdl_splots.set_xlabel(r"$x_t$",fontsize=fontsize_labels)
			hdl_splots.set_xlim([xmin,xmax])
			# hdl_splots.set_ylim([-12,3])
			hdl_splots.set_ylim([-20,20])
			hdl_splots.set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

			plt.show(block=False)
			plt.pause(0.1)

		rrtp_MO.rrgpMO[0].dbg_phase_added_to_features = 0.0 # reset to zero!
		plt.show(block=True)



	if plotting: 
		# hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(24,10),sharex=True)
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_splots = [hdl_splots]

		if which_nonlin_sys == "true":
			hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta_{nom})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		elif which_nonlin_sys == "wrong":
			hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta_{rand})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)
		elif which_nonlin_sys == "sampled":
			hdl_fig.suptitle(r"Elbow dynamical system $x_{t+1} = f(x_t;\theta_{rand})$ "+"| Kernel: {0}".format(kernel_name_plot_label),fontsize=fontsize_labels)

	Nsample_paths = 3
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/kink/nonlinsys_{0:s}/".format(which_nonlin_sys)
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/analysis/kink_kernel_constructed_with_nominal_model/nonlinsys_{0:s}/".format(which_nonlin_sys)
	log_evidence_loss_vec = np.zeros(Nobs)
	rmse_vec = np.zeros(Nobs)
	for jj in range(Nobs):

		if jj == 0:

			if which_kernel == "kink":
				MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_plotting,from_prior=True)
				sample_paths = rrtp_MO.sample_path_from_predictive(xpred_plotting,Nsamples=Nsample_paths,from_prior=True)
				sample_paths = sample_paths[0] # It's a list with one element, so get the first element
			else:
				MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred_plotting.numpy())
				MO_std = np.sqrt(MO_var)
				MO_std[:,:] = np.sqrt(variance_prior)
				sample_paths = gpy_instance.posterior_samples_f(X=xpred_plotting.numpy(),size=Nsample_paths)
				sample_paths = sample_paths[:,0,:]

		else:

			# Get evaluations:
			Nevals = jj
			Xtrain = Xtrain_tot[0:Nevals,:]
			Ytrain = nonlinsys2use(Xtrain)

			Xtrain = tf.convert_to_tensor(value=Xtrain,dtype=tf.float32)
			Ytrain = tf.convert_to_tensor(value=Ytrain,dtype=tf.float32)

			if which_kernel == "kink":

				# Update posterior:
				rrtp_MO.update_model(X=Xtrain,Y=Ytrain)

				# Get moments:
				MO_mean, MO_std = rrtp_MO.predict_at_locations(xpred_plotting)

				# Sample paths:
				sample_paths = rrtp_MO.sample_path_from_predictive(xpred_plotting,Nsamples=Nsample_paths,from_prior=False)
				sample_paths = sample_paths[0] # It's a list with one element, so get the first element

			else:

				gpy_instance.set_XY(X=Xtrain.numpy(),Y=Ytrain.numpy())
				MO_mean, MO_var = gpy_instance.predict_noiseless(Xnew=xpred_plotting.numpy())
				MO_std = np.sqrt(MO_var)
				sample_paths = gpy_instance.posterior_samples_f(X=xpred_plotting.numpy(),size=Nsample_paths)
				sample_paths = sample_paths[:,0,:]

		log_evidence_loss_vec[jj] = np.mean(-scipy.stats.norm.logpdf(x=yplot_nonlin_sys[:,0],loc=MO_mean[:,0],scale=MO_std[:,0]))
		rmse_vec[jj] = np.sqrt(np.mean((yplot_nonlin_sys[:,0]-MO_mean[:,0])**2))

		if plotting: 

			# Posterior mean and variance:
			hdl_splots[0].cla()
			hdl_splots[0].plot(xpred_plotting,MO_mean,linestyle="-",color="navy",lw=2,alpha=0.4)
			hdl_splots[0].fill_between(xpred_plotting[:,0],MO_mean[:,0] - 2.*MO_std[:,0],MO_mean[:,0] + 2.*MO_std[:,0],color="cornflowerblue",alpha=0.7)
			# hdl_splots[0].plot(xpred_plotting,yplot_true_fun_wrong,marker="None",linestyle="-",color="slategrey",lw=2)
			hdl_splots[0].plot(xpred_plotting,yplot_nonlin_sys,marker="None",linestyle="-",color="crimson",lw=3,alpha=0.6)

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
def main(cfg: dict) -> None:

	# which_kernel = "gaussian"
	# which_kernel = "matern"
	which_kernel = "kink"

	# which_nonlin_sys = "true"
	which_nonlin_sys = "wrong"
	# which_nonlin_sys = "sampled"
	
	Nobs = 4
	train_test_kink(cfg, block_plot=True, which_kernel=which_kernel, which_nonlin_sys=which_nonlin_sys, Nobs=Nobs, random_pars=None, my_seed=1, plotting=True, savefig=False)


if __name__ == "__main__":

	main()



