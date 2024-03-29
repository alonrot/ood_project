import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from lqrker.models import MultiObjectiveReducedRankProcess
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity
from lqrker.utils.common import CommonUtils
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



def simulate_nonlinsystem(Nsteps,x0,nonlinear_system_fun,std_noise_process=0.001,std_noise_obs=0.001,visualize=False,use_nominal_model=True):

	dim = x0.shape[1]
	x_vec = np.zeros((Nsteps,dim))
	x_vec[0,:] = x0
	y_vec = np.zeros((Nsteps,dim))
	y_vec[0,:] = x0
	for ii in range(Nsteps-1):

		# True system evolution with process noise:
		# x_vec[ii+1,:] = nonlinear_system_fun(x=x_vec[ii:ii+1,0:1],y=x_vec[ii:ii+1,1::],u1=0.,u2=0.,use_nominal_model=use_nominal_model) + std_noise_process * np.random.randn()
		x_vec[ii+1,:] = nonlinear_system_fun(x_vec[ii:ii+1,:],control_vec="gather_data_policy",use_nominal_model=use_nominal_model) + std_noise_process * np.random.randn()

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


def generate_training_data(Nsteps,dim_x,Nx0_training,nonlinear_system_fun):
	"""
	<<< Generate training data >>>
	"""

	# Nsteps = 500
	# dim_x = 2
	# Nx0_training = 2
	x0 = np.random.rand(Nx0_training,dim_x)
	Xlatent_list = []; Ylatent_list = []
	for ii in range(Nx0_training):

		Xlatent, Ylatent, Xobs, Yobs = simulate_nonlinsystem(Nsteps,x0[ii:ii+1,:],nonlinear_system_fun,std_noise_process=0.0,visualize=False)

		Xlatent_list += [Xlatent]
		Ylatent_list += [Ylatent]

	# return Xtrain, Ytrain, Xlatent_list
	return Xlatent_list, Ylatent_list




def initialize_GPmodel_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,use_nominal_model_for_spectral_density=True):
	"""
	<<< Initialize GP model >>>
	"""
	
	# dim_y = dim_x

	print("Initializing Spectral density ...")
	if which_kernel == "vanderpol":
		spectral_density = VanDerPolSpectralDensity(cfg.spectral_density.vanderpol,cfg.sampler.hmc,dim=dim_X)
	elif which_kernel == "matern":
		spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_X)
	elif which_kernel == "dubinscar":
		spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.dubinscar,cfg.sampler.hmc,dim=dim_X,use_nominal_model=use_nominal_model_for_spectral_density)
	else:
		raise ValueError("Possibilities: [vanderpol,matern,dubinscar]")

	# omega_min = -6.
	# omega_max = +6.
	# Ndiv = 31
	# cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_x
	# spectral_density.update_Wpoints_regular(omega_min,omega_max,Ndiv)

	L = 200.0
	# L = 30.0
	# Ndiv = 61
	Ndiv = 5 # 5**5=3125
	cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_X
	spectral_density.update_Wpoints_discrete(L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False)

	print("Initializing GP model ...")
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_X,cfg,spectral_density,Xtrain,Ytrain)
	# rrtp_MO.train_model()

	return rrtp_MO, Ndiv


def analyze_predictive_distribution(rrtp_MO,dim_x,Xlatent_list):
	"""
	<<< Quick analysis >>>
	"""

	xmin = -3.
	xmax = +3.
	Ndiv = 21
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]

	# Get moments:
	mean_prior, std_prior = rrtp_MO.predict_at_locations(xpred,from_prior=True)

	# pdb.set_trace()
	# xpred_plotting = tf.reshape(xpred,(Ndiv,Ndiv))
	mean_prior_plotting_0 = tf.reshape(mean_prior[:,0],(Ndiv,Ndiv))
	mean_prior_plotting_1 = tf.reshape(mean_prior[:,1],(Ndiv,Ndiv))
	std_prior_plotting_0 = tf.reshape(std_prior[:,0],(Ndiv,Ndiv))
	std_prior_plotting_1 = tf.reshape(std_prior[:,1],(Ndiv,Ndiv))

	hdl_fig, hdl_splots = plt.subplots(2,2,figsize=(12,8),sharex=True)
	hdl_splots[0,0].imshow(mean_prior_plotting_0, origin='lower', cmap=cm.winter, interpolation='spline36', extent=([xmin, xmax, xmin, xmax]))
	hdl_splots[0,0].set_title("Mean 0")
	hdl_splots[0,1].imshow(mean_prior_plotting_1, origin='lower', cmap=cm.winter, interpolation='spline36', extent=([xmin, xmax, xmin, xmax]))
	hdl_splots[0,1].set_title("Mean 1")
	hdl_splots[1,0].imshow(std_prior_plotting_0, origin='lower', cmap=cm.winter, interpolation='spline36', extent=([xmin, xmax, xmin, xmax]))
	hdl_splots[1,0].set_title("std 0")
	hdl_splots[1,1].imshow(std_prior_plotting_1, origin='lower', cmap=cm.winter, interpolation='spline36', extent=([xmin, xmax, xmin, xmax]))
	hdl_splots[1,1].set_title("std 1")

	for jj in range(len(Xlatent_list)):

		Xlatent = Xlatent_list[jj]

		hdl_splots[0,0].plot(Xlatent[:,0],Xlatent[:,1],marker=".",linestyle="None",color="grey",alpha=0.6)
		hdl_splots[0,1].plot(Xlatent[:,0],Xlatent[:,1],marker=".",linestyle="None",color="grey",alpha=0.6)
		hdl_splots[1,0].plot(Xlatent[:,0],Xlatent[:,1],marker=".",linestyle="None",color="grey",alpha=0.6)
		hdl_splots[1,1].plot(Xlatent[:,0],Xlatent[:,1],marker=".",linestyle="None",color="grey",alpha=0.6)

	# if not block_plot:
	# 	return
	# else:
	# 	plt.show(block=True)

	return xpred, Ndiv


def predict_long_trajectories(rrtp_MO,dim_x,Xlatent_list,block_plot,which_kernel,xpred,Ndiv):
	"""
	<<< Sample trajectory >>>
	"""

	xmin = -3.
	xmax = +3.

	nonlinear_system_fun_vanderpol = VanDerPolSpectralDensity._controlled_vanderpol_dynamics

	# pdb.set_trace()

	# # Sample paths:
	# sample_paths_prior = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=15,from_prior=True)
	# sample_paths_predictive = rrtp_MO.sample_path_from_predictive(xpred,Nsamples=3,from_prior=False)

	# fx = rrtp_MO.get_sample_path_callable(Nsamples=3,from_prior=False)

	# Get prior trajectory:
	# x0_sample = np.array([[0.9,0.8]])
	xsamples_X_list = []
	Nsamples_x0 = 5
	x0_offset = np.array([1.0,1.0])
	x0_sample_vec = x0_offset + np.random.rand(Nsamples_x0,dim_x)

	Nsteps_sample = 2
	traj_length = 500
	traj_length_true = 500
	for ii in range(Nsamples_x0):
		logger.info("Generating data for sample x0: {0:d}".format(ii+1))
		# x0_sample_vec[...,ii] = x0[ii:ii+1,:] + 0.05*np.random.normal((1,2))
		# Xlatent_sample, Ylatent_sample, _, _ = simulate_nonlinsystem(Nsteps_sample,x0_sample_vec[...,ii],nonlinear_system_fun_vanderpol,visualize=False)
		Xlatent_sample, Ylatent_sample, _, _ = simulate_nonlinsystem(Nsteps_sample,x0_sample_vec[ii:ii+1,:],nonlinear_system_fun_vanderpol,visualize=False)
		Xlatent_sample = tf.convert_to_tensor(value=Xlatent_sample,dtype=np.float32)
		Ylatent_sample = tf.convert_to_tensor(value=Ylatent_sample,dtype=np.float32)
		xsamples_X, _ = rrtp_MO.sample_state_space_from_prior_recursively(x0=Xlatent_sample,x1=Ylatent_sample,traj_length=traj_length,Nsamples=4)

		xsamples_X_list += [xsamples_X]


	# Xlatent_true, _, _, _ = simulate_nonlinsystem(traj_length_true,x0,nonlinear_system_fun_vanderpol,std_noise_process=0.0,visualize=False)

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots = [hdl_splots]
	hdl_fig.suptitle(r"Van Der Pol function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format(which_kernel),fontsize=fontsize_labels)

	for ii in range(len(Xlatent_list)):
		hdl_splots[0].plot(Xlatent_list[ii][:,0],Xlatent_list[ii][:,1],marker=".",linestyle="--",color="red",lw=0.5,markersize=5)
	
	for jj in range(Nsamples_x0):
		xsamples_X_el = xsamples_X_list[jj]
		for ii in range(xsamples_X.shape[2]):
			hdl_splots[0].plot(xsamples_X_el[:,0,ii],xsamples_X_el[:,1,ii],marker=None,linestyle="-",lw=0.5,markersize=5,color="gray")
		# hdl_splots[0].plot(Xlatent_true[:,0],Xlatent_true[:,1],marker=".",linestyle="--",color="red",lw=0.5,markersize=5)
		# plt.show(block=True)

		hdl_splots[0].plot(x0_sample_vec[jj:jj+1,0],x0_sample_vec[jj:jj+1,0],color="black",marker="o")

	# Plot true system:
	xpred_next = nonlinear_system_fun_vanderpol(x=xpred[:,0:1],y=xpred[:,1::],u1=0.,u2=0.)
	xpred_next_X1 = tf.reshape(xpred_next[:,0],(Ndiv,Ndiv))
	xpred_next_X2 = tf.reshape(xpred_next[:,1],(Ndiv,Ndiv))

	hdl_fig, hdl_splots = plt.subplots(1,2,figsize=(12,8),sharex=True)
	hdl_fig.suptitle(r"True system Van Der Pol function simulation $x_{t+1} = f(x_t)$",fontsize=fontsize_labels)
	hdl_splots[0].imshow(xpred_next_X1, origin='lower', cmap=cm.winter, interpolation='spline36', extent=([xmin, xmax, xmin, xmax]))
	hdl_splots[1].imshow(xpred_next_X2, origin='lower', cmap=cm.winter, interpolation='spline36', extent=([xmin, xmax, xmin, xmax]))
	plt.show(block=block_plot)


	

def test_vanderpol(cfg: dict, block_plot: bool, which_kernel: str) -> None:
	"""


	NOTES
	------------------------------------------------------------------------
	Train the model to predict one-step ahead

	Then, see how well the model can predict long term trajectories.
	To this end, we evaluate the model at a training point, then sample a new point from the output distribution, 
	then evaluate the model at that sampled point, and then repeat the process

	TODO
	1) Most of the time, the samples are gathered at the high frequencies, which creates a lot of ripples in the prediction
		1.1) Sample from individual Gaussians placed at the modes?
	4) Intricude temporal dependendices in the model

	# Basiclaly, repeat the above operation and collect data for 5-10 different initial conditions. 
	# Then, see how the GP reacts to that and predicts
	# Once that's done, see how to handle the OOD case
		# Posbility 1) Lateral wind disturbance -> trigger re-learning... include disturbances in the spectral density?
		# 2) Obstacles

	"""

	print(OmegaConf.to_yaml(cfg))

	my_seed = 4
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	Nsteps = 500
	dim_x = 2
	Nx0_training = 2
	nonlinear_system_fun_vanderpol = VanDerPolSpectralDensity._controlled_vanderpol_dynamics
	Xlatent_list, Ylatent_list = generate_training_data(Nsteps,dim_x,Nx0_training,nonlinear_system_fun_vanderpol)
	Xtrain = tf.convert_to_tensor(value=np.concatenate(Xlatent_list,axis=0),dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=np.concatenate(Ylatent_list,axis=0),dtype=np.float32)

	rrtp_MO, Ndiv = initialize_GPmodel_with_existing_data(cfg,dim_x,Xtrain,Ytrain,which_kernel)

	# xpred, Ndiv = analyze_predictive_distribution(rrtp_MO,dim_x,Xlatent_list)

	predict_long_trajectories(rrtp_MO,dim_x,Xlatent_list,block_plot,which_kernel,xpred,Ndiv)


@hydra.main(config_path="./config",config_name="config")
def test(cfg: dict) -> None:
	

	test_vanderpol(cfg, block_plot=True, which_kernel="vanderpol")
	test_vanderpol(cfg, block_plot=True, which_kernel="matern")



if __name__ == "__main__":

	test()


