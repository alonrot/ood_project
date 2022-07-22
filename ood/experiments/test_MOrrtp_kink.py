import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
from lqrker.models.rr_features import RRTPRegularFourierFeatures
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity
import numpy as np
import scipy
import hydra
from omegaconf import OmegaConf
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

def kink_fun(x):
	return 0.8 + (x + 0.2)*(1. - 5./(1 + np.exp(-2.*x)) )

def simulate_nonlinsystem(Nsteps,x0,nonlinear_system_fun,visualize=False):

	dim = x0.shape[1]
	x_vec = np.zeros((Nsteps,dim))
	x_vec[0,:] = x0
	y_vec = np.zeros((Nsteps,dim))
	y_vec[0,:] = x0
	std_noise_process = 0.05
	std_noise_obs = np.sqrt(0.8)
	for ii in range(Nsteps-1):

		# True system evolution with process noise:
		# x_vec[ii+1,:] = kink_fun(x_vec[ii:ii+1,:]) + std_noise_process * np.random.randn()
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


def train_test_kink(cfg: dict, block_plot: bool, which_kernel: str) -> None:
	"""

	Train the model to predict one-step ahead

	Then, see how well the model can predict long term trajectories.
	To this end, we evaluate the model at a trining point, then sample a new point from the output distribution, 
	then evaluate the model at that sampled point, and then repeat the process
	"""


	"""
	TODO
	1) Most of the time, the samples are gathered at the high frequencies, which creates a lot of ripples in the prediction
		1.1) Sample from individual Gaussians placed at the modes?
	4) Intricude temporal dependendices in the model
	"""

	print(OmegaConf.to_yaml(cfg))

	my_seed = 3
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)
	
	dim_y = 1
	x0 = np.array([[1.0]])
	dim_x = x0.shape[1]

	if which_kernel == "kink":
		spectral_densities = [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x)]*dim_y
	elif which_kernel == "matern":
		spectral_densities = [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]*dim_y

	# Generate training data:
	# Nsteps = 120
	Nsteps = 4
	Xlatent, Ylatent, Xobs, Yobs = simulate_nonlinsystem(Nsteps,x0,spectral_densities[0]._nonlinear_system_fun,visualize=False)
	rrtp_MO = [None]*dim_y
	MO_mean_pred = [None]*dim_y
	MO_std_pred = [None]*dim_y
	# xmin = cfg.config.spectral_density.spectral_density_pars.x_lim_min
	# xmax = cfg.config.spectral_density.spectral_density_pars.x_lim_max
	xmin = -6.
	xmax = +3.
	Ndiv = 201
	xpred = tf.linspace(xmin,xmax,Ndiv)
	xpred = tf.reshape(xpred,(-1,1))

	Xtrain = tf.convert_to_tensor(value=Xlatent,dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=Ylatent,dtype=np.float32)

	for ii in range(dim_y):

		rrtp_MO[ii] = RRTPRegularFourierFeatures(dim=dim_x,cfg=cfg.gpmodel,spectral_density=spectral_densities[ii])
		rrtp_MO[ii].update_spectral_density(None,None)

		rrtp_MO[ii].update_model(Xtrain,Ytrain) # Update model indexing the target outputs at the corresponding dimension
		rrtp_MO[ii].train_model()

		# Compute predictive moments:
		MO_mean_pred[ii], cov_pred = rrtp_MO[ii].predict_at_locations(xpred)
		MO_std_pred[ii] = tf.sqrt(tf.linalg.diag_part(cov_pred))

	# Sample paths:
	sample_paths_prior = rrtp_MO[ii].sample_path_from_predictive(xpred,Nsamples=15,from_prior=True)
	sample_paths_predictive = rrtp_MO[ii].sample_path_from_predictive(xpred,Nsamples=3,from_prior=False)

	# Get moments:
	mean_prior, cov_prior = rrtp_MO[ii].predict_at_locations(xpred,from_prior=True)
	std_prior = tf.sqrt(tf.linalg.diag_part(cov_prior))

	# # Sample from beta:
	# sample_paths_predictive_from_beta = rrtp_MO[ii].sample_path_from_predictive_given_beta_moments(xpred,Nsamples=15)

	# Get prior trajectory:
	x0_sample = np.array([[0.9]])
	Nsteps_sample = 2
	Xlatent_sample, Ylatent_sample, _, _ = simulate_nonlinsystem(Nsteps_sample,x0_sample,spectral_densities[0]._nonlinear_system_fun,visualize=False)
	Xlatent_sample = tf.convert_to_tensor(value=Xlatent_sample,dtype=np.float32)
	Ylatent_sample = tf.convert_to_tensor(value=Ylatent_sample,dtype=np.float32)
	xsamples_X, xsamples_Y = rrtp_MO[ii].sample_state_space_from_prior_recursively(x0=Xlatent_sample,x1=Ylatent_sample,traj_length=5,sort=True)
	
	# print("xsamples_X:",xsamples_X)
	# print("xsamples_Y:",xsamples_Y)

	# print("rrtp_MO[ii].sample_mvt0:",rrtp_MO[ii].sample_mvt0[0:3,:])

	# Plot:
	assert dim_y == 1
	ii = 0

	Ndiv = 201
	xplot_true_fun = np.linspace(-5.,2.,Ndiv)
	yplot_true_fun = spectral_densities[0]._nonlinear_system_fun(xplot_true_fun)

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format(which_kernel),fontsize=fontsize_labels)
	hdl_splots[0].plot(xpred,MO_mean_pred[ii],linestyle="-",color="b",lw=3)
	hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred[ii] - 2.*MO_std_pred[ii],MO_mean_pred[ii] + 2.*MO_std_pred[ii],color="cornflowerblue",alpha=0.5)
	hdl_splots[0].plot(xplot_true_fun,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
	for ii in range(sample_paths_predictive.shape[1]):
		hdl_splots[0].plot(xpred,sample_paths_predictive[:,ii],marker="None",linestyle="--",color="r",lw=0.5)
	# for ii in range(sample_paths_predictive_from_beta.shape[1]):
	# 	hdl_splots[0].plot(xpred,sample_paths_predictive_from_beta[:,ii],marker="None",linestyle="--",color="r",lw=0.5)
	hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker=".",linestyle="--",color="gray",lw=0.5,markersize=5)
	hdl_splots[0].set_xlim([xmin,xmax])
	hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

	hdl_splots[1].plot(xpred,mean_prior,linestyle="-",color="b",lw=3)
	hdl_splots[1].fill_between(xpred[:,0],mean_prior - 2.*std_prior,mean_prior + 2.*std_prior,color="cornflowerblue",alpha=0.5)
	hdl_splots[1].plot(xplot_true_fun,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
	for ii in range(sample_paths_prior.shape[1]):
		hdl_splots[1].plot(xpred,sample_paths_prior[:,ii],marker="None",linestyle="--",color="k",lw=0.5)
	hdl_splots[1].plot(xsamples_X,xsamples_Y,marker="o",linestyle="--",color="r",lw=0.5,markersize=5)
	hdl_splots[1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots[1].set_xlim([xmin,xmax])
	hdl_splots[1].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)


	plt.show(block=block_plot)
	plt.pause(1)

	del spectral_densities
	del rrtp_MO



@hydra.main(config_path="./config",config_name="config")
def test(cfg: dict) -> None:
	

	train_test_kink(cfg, block_plot=False, which_kernel="kink")
	train_test_kink(cfg, block_plot=True, which_kernel="matern")



if __name__ == "__main__":

	test()


