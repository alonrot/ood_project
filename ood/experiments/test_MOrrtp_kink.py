import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
from lqrker.models.rrtp import RRTPSarkkaFeatures, RRTPRandomFourierFeatures
from lqrker.utils.spectral_densities import MultiDimensionalFourierTransformQuadratureFromData, KinkSpectralDensity, MaternSpectralDensity
import numpy as np
import numpy.random as npr
import scipy
import hydra
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

def simulate_kink(Nsteps,x0,visualize=False):

	dim = x0.shape[1]
	x_vec = np.zeros((Nsteps,dim))
	x_vec[0,:] = x0
	y_vec = np.zeros((Nsteps,dim))
	y_vec[0,:] = x0
	std_noise_process = 0.05
	std_noise_obs = np.sqrt(0.8)
	for ii in range(Nsteps-1):

		# True system evolution with process noise:
		x_vec[ii+1,:] = kink_fun(x_vec[ii:ii+1,:]) + std_noise_process * np.random.randn()

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
		yplot_true_fun = kink_fun(xplot_true_fun)

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
	2) Use the function "kink"itself as a feature (same as in the LQR kernel) and compare the regression, no idea if the features have to be PSD
	3) So, does a customized spectral density help or not?
	4) Intricude temporal dependendices in the model
	"""

	np.random.seed(seed=0)

	# pdb.set_trace()

	# Generate training data:
	# Nsteps = 120
	Nsteps = 3
	x0 = np.array([[1.0]])
	Xlatent, Ylatent, Xobs, Yobs = simulate_kink(Nsteps,x0,visualize=False)
	dim_x = x0.shape[1]
	dim_y = 1
	spectral_density = [None]*dim_y
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

	# Xtrain = tf.convert_to_tensor(value=Xobs,dtype=np.float32)
	# Ytrain = tf.convert_to_tensor(value=Yobs,dtype=np.float32)

	# Xtrain_subset_np = np.array([[-3.0,-2.0,0.5,1.2]])
	# Ytrain_subset_np = kink_fun(Xtrain_subset_np)
	# Xtrain = tf.convert_to_tensor(value=Xtrain_subset_np,dtype=np.float32)
	# Ytrain = tf.convert_to_tensor(value=Ytrain_subset_np + np.sqrt(0.8)*np.random.randn(*Xtrain_subset_np.shape),dtype=np.float32)

	# pdb.set_trace()

	for ii in range(dim_y):

		if which_kernel == "kink":
		
			cfg.config.spectral_density.spectral_density_pars = dict(name="kink",x_lim_min=-5.0,x_lim_max=+2.0,prior_var=1.0,Nsteps_integration=401)
			raise NotImplementedError("These params are being overwritten")
			spectral_density[ii] = KinkSpectralDensity(cfg.config.spectral_density.spectral_density_pars,dim=dim_x)
		
		elif which_kernel == "matern":

			cfg.config.spectral_density.spectral_density_pars = dict(name="matern",nu=2.5,ls=0.5,prior_var=1.0)
			spectral_density[ii] = MaternSpectralDensity(cfg.config.spectral_density.spectral_density_pars,dim=dim_x)

		rrtp_MO[ii] = RRTPRandomFourierFeatures(dim=dim_x,cfg=cfg.RRTPRandomFourierFeatures,spectral_density=spectral_density[ii])
		rrtp_MO[ii].update_spectral_density(None,None)

		rrtp_MO[ii].update_model(Xtrain,Ytrain) # Update model indexing the target outputs at the corresponding dimension
		rrtp_MO[ii].train_model()

		# Compute predictive moments:
		MO_mean_pred[ii], cov_pred = rrtp_MO[ii].get_predictive_moments(xpred)
		MO_std_pred[ii] = tf.sqrt(tf.linalg.diag_part(cov_pred))

		sample_paths = rrtp_MO[ii].sample_path(mean_pred=MO_mean_pred[ii],cov_pred=cov_pred,Nsamples=3)

	# pdb.set_trace()
	Nfeat = cfg.RRTPRandomFourierFeatures.hyperpars.weights_features.Nfeat
	feat_mat = rrtp_MO[ii].get_features_mat(xpred)
	# noise_vec = np.random.rand(feat_mat.shape[1],5) / cfg.RRTPRandomFourierFeatures.hyperpars.weights_features.Nfeat
	noise_vec = (np.ones((feat_mat.shape[1],5)) + 2.*np.random.rand(feat_mat.shape[1],5)) / Nfeat * 10.0
	# noise_vec = np.ones((feat_mat.shape[1],5)) / cfg.RRTPRandomFourierFeatures.hyperpars.weights_features.Nfeat
	fun_prior = feat_mat @ noise_vec


	# pdb.set_trace()
	cov_prior = feat_mat @ tf.eye(feat_mat.shape[1])/Nfeat @ tf.transpose(feat_mat)
	var_prior = tf.linalg.diag_part(cov_prior)
	std_prior = tf.sqrt(var_prior)
	mean_prior = tf.ones(var_prior.shape[0])/Nfeat*10

	sample_paths_prior = rrtp_MO[ii].sample_path(mean_pred=mean_prior,cov_pred=(cov_prior+1e-6*tf.eye(cov_prior.shape[0])),Nsamples=10)*10.


	# Plot:
	hdl_fig, hdl_splots = plt.subplots(dim_y,1,figsize=(12,8),sharex=True)
	if dim_y == 1:
		hdl_splots = [hdl_splots]
	hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format(which_kernel),fontsize=fontsize_labels)
	# hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$",fontsize=fontsize_labels))
	for ii in range(dim_y):
		
		Ndiv = 201
		xplot_true_fun = np.linspace(-5.,2.,Ndiv)
		yplot_true_fun = kink_fun(xplot_true_fun)

		hdl_splots[ii].plot(xpred,MO_mean_pred[ii],linestyle="-",color="b",lw=3)
		hdl_splots[ii].fill_between(xpred[:,0],MO_mean_pred[ii] - 2.*MO_std_pred[ii],MO_mean_pred[ii] + 2.*MO_std_pred[ii],color="cornflowerblue",alpha=0.5)
		hdl_splots[ii].fill_between(xpred[:,0],mean_prior - 2.*std_prior,mean_prior + 2.*std_prior,color="red",alpha=0.5)
		# hdl_splots[ii].plot(xpred,MO_mean_pred[ii] + 2.*MO_std_pred[ii],linestyle="-",color="b")
		# hdl_splots[ii].plot(xpred,MO_mean_pred[ii] - 2.*MO_std_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(xplot_true_fun,yplot_true_fun,marker="None",linestyle="-",color="k",lw=2)
		hdl_splots[ii].plot(xpred,sample_paths[:,0],marker="None",linestyle="--",color="k",lw=0.5)
		hdl_splots[ii].plot(xpred,sample_paths[:,1],marker="None",linestyle="--",color="k",lw=0.5)
		hdl_splots[ii].plot(xpred,sample_paths[:,2],marker="None",linestyle="--",color="k",lw=0.5)
		# for jj in range(fun_prior.shape[1]):
		# 	hdl_splots[ii].plot(xpred,fun_prior[:,jj],marker="None",linestyle="--",color="r",lw=0.5)
		# for jj in range(sample_paths_prior.shape[1]):
		# 	hdl_splots[ii].plot(xpred,sample_paths_prior[:,jj],marker="None",linestyle="--",color="r",lw=0.5)
		hdl_splots[ii].plot(Xtrain[:,0],Ytrain[:,0],marker=".",linestyle="--",color="gray",lw=0.5,markersize=5)
		hdl_splots[ii].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots[ii].set_xlim([xmin,xmax])
		hdl_splots[ii].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)
		# hdl_splots[ii].plot(Xobs[:,0],Yobs[:,0],marker="o",linestyle="--",color="steelblue",lw=0.5,markersize=6)


	plt.show(block=block_plot)
	plt.pause(1)



@hydra.main(config_path=".",config_name="config/config.yaml")
def test(cfg: dict) -> None:
	

	train_test_kink(cfg, block_plot=True, which_kernel="kink")
	# train_test_kink(cfg, block_plot=True, which_kernel="matern")



if __name__ == "__main__":

	test()


