import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
from lqrker.models import MultiObjectiveRRTPRegularFourierFeatures
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity
from ood.utils.common import CommonUtils
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
	4) Include temporal dependendices in the model
	"""

	print(OmegaConf.to_yaml(cfg))

	my_seed = 3
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)
	
	x0 = np.array([[1.0]])
	dim_x = x0.shape[1]
	dim_y = dim_x

	if which_kernel == "kink":
		spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x)
	elif which_kernel == "matern":
		spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)

	# omega_min = -10.
	# omega_max = +10.
	# Ndiv = 1001
	# cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_x
	# spectral_density.update_Wpoints_regular(omega_min,omega_max,Ndiv,normalize_density_numerically=False)

	L = 750.0
	Ndiv = 2001
	cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_x
	spectral_density.update_Wpoints_discrete(L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False)

	# Generate training data:
	Nsteps = 4
	Xlatent, Ylatent, Xobs, Yobs = simulate_nonlinsystem(Nsteps,x0,KinkSpectralDensity.kink_dynamics,visualize=False)

	Xtrain = tf.convert_to_tensor(value=Xlatent,dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=Ylatent,dtype=np.float32)

	rrtp_MO = MultiObjectiveRRTPRegularFourierFeatures(dim_x,cfg,spectral_density,Xtrain,Ytrain)

	# Create grid for predictions:
	xmin = -6.0
	xmax = +3.0
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=201,dim=dim_x) # [Ndiv**dim_x,dim_x]

	# Get moments:
	MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(xpred)
	mean_prior, std_prior = rrtp_MO.predict_at_locations(xpred,from_prior=True)

	# Sample paths:
	sample_paths_prior = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=15,from_prior=True)
	sample_paths_predictive = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=15,from_prior=False)

	# Get prior trajectory:
	x0_sample = np.array([[0.9]])
	Nsteps_sample = 2
	Xlatent_sample, Ylatent_sample, _, _ = simulate_nonlinsystem(Nsteps_sample,x0_sample,KinkSpectralDensity.kink_dynamics,visualize=False)
	Xlatent_sample = tf.convert_to_tensor(value=Xlatent_sample,dtype=np.float32)
	Ylatent_sample = tf.convert_to_tensor(value=Ylatent_sample,dtype=np.float32)
	xsamples_X, _ = rrtp_MO.sample_state_space_from_prior_recursively(x0=Xlatent_sample,x1=Ylatent_sample,traj_length=30,Nsamples=4,sort=True) # [Npoints,self.dim_out,Nsamples]

	# Plot:
	assert dim_y == 1

	Ndiv = 201
	xplot_true_fun = np.linspace(-5.,2.,Ndiv)
	yplot_true_fun = KinkSpectralDensity.kink_dynamics(xplot_true_fun)

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format(which_kernel),fontsize=fontsize_labels)
	hdl_splots[0].plot(xpred,MO_mean_pred,linestyle="-",color="b",lw=3)
	# pdb.set_trace()
	hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred[:,0] - 2.*MO_std_pred[:,0],MO_mean_pred[:,0] + 2.*MO_std_pred[:,0],color="cornflowerblue",alpha=0.5)
	hdl_splots[0].plot(xplot_true_fun,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
	for ii in range(len(sample_paths_predictive)):
		hdl_splots[0].plot(xpred,sample_paths_predictive[ii],marker="None",linestyle="--",color="r",lw=0.5)
	hdl_splots[0].plot(Xtrain[:,0],Ytrain[:,0],marker=".",linestyle="--",color="gray",lw=0.5,markersize=5)
	hdl_splots[0].set_xlim([xmin,xmax])
	hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

	hdl_splots[1].plot(xpred,mean_prior,linestyle="-",color="b",lw=3)
	hdl_splots[1].fill_between(xpred[:,0],mean_prior[:,0] - 2.*std_prior[:,0],mean_prior[:,0] + 2.*std_prior[:,0],color="cornflowerblue",alpha=0.5)
	hdl_splots[1].plot(xplot_true_fun,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
	for ii in range(len(sample_paths_prior)):
		hdl_splots[1].plot(xpred,sample_paths_prior[ii],marker="None",linestyle="--",color="k",lw=0.5)
	for ii in range(xsamples_X.shape[2]):
		hdl_splots[1].plot(xsamples_X[0:-1,0,ii],xsamples_X[1::,0,ii],marker=".",linestyle="--",lw=0.5,markersize=5)
	hdl_splots[1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots[1].set_xlim([xmin,xmax])
	hdl_splots[1].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)

	plt.show(block=block_plot)
	plt.pause(1)

	del spectral_density
	del rrtp_MO



@hydra.main(config_path="./config",config_name="config")
def test(cfg: dict) -> None:
	

	train_test_kink(cfg, block_plot=False, which_kernel="kink")
	train_test_kink(cfg, block_plot=True, which_kernel="matern")



if __name__ == "__main__":

	test()


