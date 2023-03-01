import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
from lqrker.models import MultiObjectiveReducedRankProcess
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity
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


def get_kernel_values(cfg: dict, which_kernel: str) -> None:

	my_seed = 1
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)
	
	x0 = np.array([[1.0]])
	dim_x = x0.shape[1]
	dim_y = dim_x

	Xtrain = tf.zeros((1,1))
	Ytrain = tf.zeros((1,1))

	# Create grid for predictions:
	xmin = -5.0
	xmax = +5.0
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=201,dim=dim_x) # [Ndiv**dim_x,dim_x]

	# Create grid for omegas:
	omega_lim = 3.0
	Ndiv_omega_for_analysis = 201
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=1) # [Ndiv**dim_in,dim_in]

	if which_kernel == "kink_randomized":
		
		kernel_name_plot_label = "Kink (randomized)"
		integration_method = "integrate_with_regular_grid_randomized_parameters"
		spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x,integration_method=integration_method,Xtrain=None,Ytrain=None,use_nominal_model=False)
		# spectral_density = KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x,use_nominal_model=use_nominal_model)

		# L = 750.0
		# Ndiv = 1201
		L = 800.0
		Ndiv = 1001
		cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_x
		spectral_density.update_Wpoints_discrete(L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False)

		rrtp_MO = MultiObjectiveReducedRankProcess(dim_x,cfg,spectral_density,Xtrain,Ytrain)
		MO_mean_pred, cov_pred = rrtp_MO.rrgpMO[0].predict_at_locations(xpred,from_prior=True)
		cov_pred = cov_pred.numpy()

		Sw_vec, phiw_vec = spectral_density.unnormalized_density(omegapred_analysis)

	elif which_kernel == "gaussian":
	
		kernel_name_plot_label = "Gaussian"
		variance_prior = 2.0
		ker = GPy.kern.RBF(dim_x, variance=variance_prior, lengthscale=1.0)
		lik = GPy.likelihoods.Gaussian(variance=0.15**2)

		gpy_instance = GPy.core.GP(X=Xtrain.numpy(),Y=Ytrain.numpy(), kernel=ker, likelihood=lik)

		cov_pred = ker.K(X=xpred.numpy(),X2=xpred.numpy())

		spectral_density = SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)

		Sw_vec, phiw_vec = spectral_density.unnormalized_density(omegapred_analysis)
	
	elif which_kernel == "matern":
		kernel_name_plot_label = "Matern"
		variance_prior = 2.0
		ker = GPy.kern.sde_Matern52(dim_x, variance=variance_prior, lengthscale=1.0)
		lik = GPy.likelihoods.Gaussian(variance=0.15**2)
		# spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)
		# 
		gpy_instance = GPy.core.GP(X=Xtrain.numpy(),Y=Ytrain.numpy(), kernel=ker, likelihood=lik)

		cov_pred = ker.K(X=xpred.numpy(),X2=xpred.numpy())

		spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)

		Sw_vec, phiw_vec = spectral_density.unnormalized_density(omegapred_analysis)

	return cov_pred, xpred, Sw_vec, phiw_vec, omegapred_analysis


	

@hydra.main(config_path="./config",config_name="config")
def plotting_stuff(cfg):


	which_kernel_list = ["kink_randomized","matern","gaussian"]
	which_kernel_label_list = ["Kink","Matern","Gaussian"]

	hdl_fig, hdl_splots = plt.subplots(3,len(which_kernel_list),figsize=(24,10),sharex=False)
	# COLOR_MAP = "cividis"
	# COLOR_MAP = "bone"
	# COLOR_MAP = "gist_heat"
	COLOR_MAP = "copper"
	cc = 0
	for which_kernel in which_kernel_list:

		cov_pred, xpred, Sw_vec, phiw_vec, omegapred_analysis = get_kernel_values(cfg,which_kernel)

		xmin = xpred[0,0]
		xmax = xpred[-1,0]
		extent_plot_xpred = [xmin,xmax,xmin,xmax] #  scalars (left, right, bottom, top)
		omega_min = omegapred_analysis[0,0]
		omega_max = omegapred_analysis[-1,0]

		hdl_splots[0,cc].imshow(cov_pred,extent=extent_plot_xpred,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=cov_pred.min(),vmax=cov_pred.max(),interpolation='nearest')
		hdl_splots[0,cc].set_xlim([xmin,xmax])
		hdl_splots[0,cc].set_ylim([xmin,xmax])
		hdl_splots[0,cc].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots[0,cc].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
		hdl_splots[0,cc].set_title(r"$k(x_t,x^\prime_t)$ {0:s}".format(which_kernel_label_list[cc]),fontsize=fontsize_labels)
		hdl_splots[0,cc].set_xticks([])
		hdl_splots[0,cc].set_yticks([])


		hdl_splots[1,cc].plot(omegapred_analysis,Sw_vec)
		hdl_splots[1,cc].set_xlim([omega_min,omega_max])
		hdl_splots[1,cc].set_xticks([])
		hdl_splots[1,cc].set_yticks([])
		hdl_splots[1,cc].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
		# hdl_splots[1,cc].set_title("Spectral density for {0:s} kernel".format(which_kernel_label_list[cc]),fontsize=fontsize_labels)

		if np.all(phiw_vec == 0.0): phiw_vec = np.zeros(omegapred_analysis.shape[0])
		hdl_splots[2,cc].plot(omegapred_analysis,phiw_vec)
		hdl_splots[2,cc].set_xlim([omega_min,omega_max])
		hdl_splots[2,cc].set_xticks([])
		hdl_splots[2,cc].set_yticks([])
		hdl_splots[2,cc].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
		# hdl_splots[2,cc].set_title("Phase for {0:s} kernel".format(which_kernel_label_list[cc]),fontsize=fontsize_labels)
		# 
		
		cc += 1

	hdl_splots[1,0].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	hdl_splots[2,0].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	hdl_splots[2,0].set_ylabel(r"$\varphi(\omega)$",fontsize=fontsize_labels)

	plt.show(block=True)


@hydra.main(config_path="./config",config_name="config")
def plotting_stationarity(cfg):


	which_kernel_list = ["kink_randomized","matern"]
	# which_kernel_list = ["matern","kink_randomized"]
	which_kernel_label_list = ["Kink","Matern"]

	hdl_fig, hdl_splots = plt.subplots(2,len(which_kernel_list),figsize=(24,10),sharex=False)
	# COLOR_MAP = "cividis"
	# COLOR_MAP = "bone"
	# COLOR_MAP = "gist_heat"
	COLOR_MAP = "copper"
	cc = 0
	for which_kernel in which_kernel_list:

		cov_pred, xpred, _, _, _ = get_kernel_values(cfg,which_kernel)

		xmin = xpred[0,0]
		xmax = xpred[-1,0]
		extent_plot_xpred = [xmin,xmax,xmin,xmax] #  scalars (left, right, bottom, top)

		ind_mid = xpred.shape[0]//2
		ind_3quart = int(xpred.shape[0]//4*3)

		hdl_splots[0,cc].imshow(cov_pred,extent=extent_plot_xpred,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=cov_pred.min(),vmax=cov_pred.max(),interpolation='nearest')
		hdl_splots[0,cc].set_xlim([xmin,xmax])
		hdl_splots[0,cc].set_ylim([xmin,xmax])
		hdl_splots[0,cc].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots[0,cc].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
		hdl_splots[0,cc].set_title(r"$k(x_t,x^\prime_t)$ {0:s}".format(which_kernel_label_list[cc]),fontsize=fontsize_labels)
		hdl_splots[0,cc].set_xticks([])
		hdl_splots[0,cc].set_yticks([])

		# Covariance:
		# hdl_splots[0,cc].hlines(y=xpred[ind_mid,0],xmin=xmin,xmax=xmax,color="blue",alpha=0.5,lw=2.0)
		# hdl_splots[0,cc].hlines(y=xpred[ind_3quart,0],xmin=xmin,xmax=xmax,color="red",alpha=0.5,lw=2.0)

		# hdl_splots[1,cc].plot(xpred,cov_pred[:,ind_mid],lw=2.0,alpha=0.5,color="navy")
		# hdl_splots[1,cc].plot(xpred,cov_pred[:,ind_3quart],lw=2.0,alpha=0.5,color="crimson")
		# hdl_splots[1,cc].set_xlim([xmin,xmax])
		# hdl_splots[1,cc].set_xticks([])
		# hdl_splots[1,cc].set_yticks([])
		# hdl_splots[1,cc].set_ylabel(r"$k(x_t,x_*^\prime)$",fontsize=fontsize_labels)
		# hdl_splots[1,cc].set_title(r"Cov$[f(x_t),f(x_t^\prime = x_*)]$",fontsize=fontsize_labels)

		# # Variance:
		# hdl_splots[0,cc].plot([xmin,xmax],[xmin,xmax],color="darkgreen",alpha=0.5,lw=2.0)

		# hdl_splots[1,cc].plot(xpred,np.diag(cov_pred),lw=2.0,alpha=0.5,color="darkgreen")
		# hdl_splots[1,cc].set_xlim([xmin,xmax])
		# hdl_splots[1,cc].set_xticks([])
		# hdl_splots[1,cc].set_yticks([])
		# hdl_splots[1,cc].set_ylabel(r"$k(x_t,x_t)$",fontsize=fontsize_labels)
		# hdl_splots[1,cc].set_title(r"Var$[f(x_t)]$",fontsize=fontsize_labels)

		# Mutual information:
		hdl_splots[0,cc].hlines(y=xpred[ind_mid,0],xmin=xmin,xmax=xmax,color="blue",alpha=0.5,lw=2.0)
		hdl_splots[0,cc].hlines(y=xpred[ind_3quart,0],xmin=xmin,xmax=xmax,color="red",alpha=0.5,lw=2.0)

		MI_mid = np.log(np.diag(cov_pred)) - np.log(cov_pred[:,ind_mid])
		MI_3quart = np.log(np.diag(cov_pred)) - np.log(cov_pred[:,ind_3quart])
		hdl_splots[1,cc].plot(xpred,MI_mid,lw=2.0,alpha=0.5,color="navy")
		hdl_splots[1,cc].plot(xpred,MI_3quart,lw=2.0,alpha=0.5,color="crimson")
		hdl_splots[1,cc].set_xlim([xmin,xmax])
		hdl_splots[1,cc].set_xticks([])
		# hdl_splots[1,cc].set_yticks([])
		hdl_splots[1,cc].set_ylabel(r"MI$(x_t,x_*^\prime)$",fontsize=fontsize_labels)
		hdl_splots[1,cc].set_title(r"MI$[f(x_t),f(x_t^\prime = x_*)]$",fontsize=fontsize_labels)


		# hdl_splots[2,cc].plot(xpred,cov_pred[:,int(cov_pred.shape[1]//4*3)],lw=2.0,alpha=0.5,color="navy")
		# hdl_splots[2,cc].set_xlim([xmin,xmax])
		# hdl_splots[2,cc].set_xticks([])
		# hdl_splots[2,cc].set_yticks([])
		# hdl_splots[2,cc].set_ylabel(r"$k(x_*,x_t^\prime)$",fontsize=fontsize_labels)
		# hdl_splots[2,cc].set_title(r"Cov$[f(x_t = x_*),f(x_t^\prime)]$",fontsize=fontsize_labels)
		

		hdl_splots[-1,cc].set_xlabel(r"$x_t$",fontsize=fontsize_labels)

		cc += 1


	# hdl_splots[1,0].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	# hdl_splots[2,0].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	# hdl_splots[2,0].set_ylabel(r"$\varphi(\omega)$",fontsize=fontsize_labels)

	plt.show(block=True)



@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict) -> None:

	# which_kernel = "gaussian"
	# which_kernel = "matern"
	which_kernel = "kink_randomized"

	get_kernel_values(cfg,which_kernel)



if __name__ == "__main__":

	# main()

	# plotting_stuff()
	# 
	
	plotting_stationarity()


