import tensorflow as tf
# import tensorflow_probability as tfp
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
import hydra
from omegaconf import OmegaConf
import pickle
from lqrker.spectral_densities import ExponentiallySuppressedPolynomialsFromData
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from ood.spectral_density_approximation.reconstruct_function_from_spectral_density import ReconstructFunctionFromSpectralDensity
from lqrker.utils.common import CommonUtils
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

using_deltas = True
# using_deltas = False

eps_ = 1e-3
dim_in = 1
dim_ctx = dim_in + 1
dim_out = 1

# COLOR_MAP = "seismic"
# COLOR_MAP = "gist_heat"
COLOR_MAP = "copper"

def ker_fun(x,xp,alpha):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	out: [Npoints,Npoints]
	"""

	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	x = squash(x)
	xp = squash(xp)

	xp = xp.T
	return 1./(1.-alpha*x*xp)


def squash(x):
	return 2./(1 + np.exp(-x)) - 1.0


# def f_call_all(xnew,f_samples_all,x_inputs_all):
# 	return np.interp(xnew,x_inputs_all,f_samples_all)


def generate_data(plot_stuff=False,block_plot=False):

	
	Nrollouts = 60
	
	Npred = 71
	
	xmin = -7.0
	xmax = +7.0
	xpred = np.reshape(np.linspace(xmin,xmax,Npred),(-1,dim_in))

	alpha = 0.90
	kXX = ker_fun(xpred,xpred,alpha)

	kXX_chol = np.linalg.cholesky(kXX + 1e-8*np.eye(kXX.shape[0])) # [Npred,Npred]

	mvn0_samples = np.random.randn(Nrollouts,Npred)

	f_samples = kXX_chol @ mvn0_samples.T # [Npred,Nrollouts]

	plot_stuff = False
	if plot_stuff:
		hdl_fig_ker, hdl_splots_ker = plt.subplots(3,figsize=(12,8),sharex=True)
		# hdl_fig_pred.suptitle("Predictions ...", fontsize=16)
		for ss in range(Nrollouts):
			hdl_splots_ker[0].plot(xpred[:,0],f_samples[:,ss],lw=2.,color="crimson",alpha=0.2)
		extent_plot_xpred = [xmin,xmax,xmin,xmax] #  scalars (left, right, bottom, top)
		hdl_splots_ker[1].imshow(kXX,extent=extent_plot_xpred,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=kXX.min(),vmax=kXX.max(),interpolation='nearest')
		hdl_splots_ker[1].set_xlim([xmin,xmax])
		hdl_splots_ker[1].set_ylim([xmin,xmax])
		hdl_splots_ker[1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots_ker[1].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
		hdl_splots_ker[1].set_title(r"$k(x_t,x^\prime_t)$ {0:s}".format("Kernel suppressed polys"),fontsize=fontsize_labels)
		hdl_splots_ker[1].set_xticks([xmin,0.0,xmax])
		hdl_splots_ker[1].set_yticks([xmin,0.0,xmax])


		ker_from_samples = f_samples @ f_samples.T / Nrollouts
		hdl_splots_ker[2].imshow(ker_from_samples,extent=extent_plot_xpred,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=ker_from_samples.min(),vmax=ker_from_samples.max(),interpolation='nearest')
		hdl_splots_ker[2].set_xlim([xmin,xmax])
		hdl_splots_ker[2].set_ylim([xmin,xmax])
		hdl_splots_ker[2].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		hdl_splots_ker[2].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
		hdl_splots_ker[2].set_title(r"$k(x_t,x^\prime_t)$ {0:s}".format("Kernel suppressed polys"),fontsize=fontsize_labels)
		hdl_splots_ker[2].set_xticks([xmin,0.0,xmax])
		hdl_splots_ker[2].set_yticks([xmin,0.0,xmax])


		plt.show(block=True)
		plt.show(block=block_plot)

	# # Concatenate all inputs and outputs to create a callable function:
	# f_samples_all = np.reshape(f_samples.T,(-1)) # Stack each rollout below the other
	# x_inputs_all = np.concatenate([xpred]*Nrollouts,axis=0)
	# Xtrain = xmin + (xmax-xmin)*np.random.rand(Nrollouts*Npred,1)
	# Ytrain = np.reshape(f_call_all(Xtrain[:,0],f_samples_all,x_inputs_all[:,0]),(-1,1))


	# Append a contextual variable:

	# theta_cntxt_vals = 8.*np.pi*np.random.rand(Nrollouts,1)
	theta_cntxt_vals = np.reshape(np.linspace(-4.*np.pi,4.*np.pi,Nrollouts),(-1,1))
	theta_cntxt_vec = theta_cntxt_vals @ np.ones((1,Npred))
	theta_cntxt_vec = np.reshape(theta_cntxt_vec,(-1,1))
	xpred_training = np.concatenate([xpred]*Nrollouts,axis=0)
	xpred_training_cntxt = tf.convert_to_tensor(np.concatenate([xpred_training,theta_cntxt_vec],axis=1),dtype=tf.float32)
	ypred_training = np.reshape(f_samples.T,(-1,1))
	Xtrain = xpred_training_cntxt
	Ytrain = Ytrain = tf.convert_to_tensor(ypred_training,dtype=tf.float32)

	return Xtrain, Ytrain, xpred, Nrollouts, kXX


@hydra.main(config_path="./config",config_name="config")
def train_reconstruction(cfg):

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/kernel_fit_reconstruction/learning_data_seed_80.pickle ./kernel_fit_reconstruction/

	my_seed = 85
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	path2folder = "kernel_fit_reconstruction"

	Xtrain, Ytrain, xpred, Nrollouts, kXX = generate_data(plot_stuff=False,block_plot=False)

	Npred = xpred.shape[0]
	xmin = xpred[0,0]
	xmax = xpred[-1,0]

	if using_deltas:
		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_in]
		Ytrain = tf.identity(Ytrain_deltas)		

	delta_statespace = 1.0 / Xtrain.shape[0]

	spectral_density_list = []
	spectral_density_list += [ExponentiallySuppressedPolynomialsFromData(cfg=cfg.spectral_density.expsup,cfg_sampler=cfg.sampler.hmc,dim=dim_ctx,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain)]

	Nepochs = 1000
	Nsamples_omega = 750
	if using_hybridrobotics:
		Nepochs = 60000
		Nsamples_omega = 1000
	
	omega_lim = 10.0
	# Dw_coarse = (2.*omega_lim)**dim_in / Nsamples_omega # We are trainig a tensor [Nomegas,dim_in]
	Dw_coarse = 1.0 / Nsamples_omega # We are trainig a tensor [Nomegas,dim_in]

	fx_optimized_omegas_and_voxels = np.zeros((Xtrain.shape[0],dim_out))
	Sw_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	varphi_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,dim_ctx))
	delta_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	delta_statespace_trainedNN = np.zeros((dim_out,Xtrain.shape[0],1))

	learning_rate = 1e-2
	stop_loss_val = 1./Ytrain.shape[0]
	# stop_loss_val = 0.01
	lengthscale_loss = 0.01
	loss_reconstruction_evolution = np.zeros((dim_out,Nepochs))
	spectral_density_optimized_list = [None]*dim_out
	# pdb.set_trace()
	for jj in range(dim_out):

		logger.info("Reconstruction for channel {0:d} / {1:d} ...".format(jj+1,dim_out))

		inverse_fourier_toolbox_channel = InverseFourierTransformKernelToolbox(spectral_density=spectral_density_list[jj],dim=dim_ctx)

		reconstructor_fx_deltas_and_omegas = ReconstructFunctionFromSpectralDensity(dim_in=dim_ctx,dw_voxel_init=Dw_coarse,dX_voxel_init=delta_statespace,
																					omega_lim=omega_lim,Nomegas=Nsamples_omega,
																					inverse_fourier_toolbox=inverse_fourier_toolbox_channel,
																					Xtest=Xtrain,Ytest=Ytrain[:,jj:jj+1])

		reconstructor_fx_deltas_and_omegas.train(Nepochs=Nepochs,learning_rate=learning_rate,stop_loss_val=stop_loss_val,lengthscale_loss=lengthscale_loss,print_every=10)


		spectral_density_optimized_list[jj] = reconstructor_fx_deltas_and_omegas.update_internal_spectral_density_parameters()
		Sw_omegas_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.inverse_fourier_toolbox.spectral_values
		varphi_omegas_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.inverse_fourier_toolbox.varphi_values


		# Collect trained variables for each channel:
		omegas_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.get_omegas_weights()
		delta_omegas_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.get_delta_omegas() # [Nomegas,]
		delta_statespace_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.get_delta_statespace() # [Nxpoints,1]

		# Keep track of the loss evolution:
		loss_reconstruction_evolution[jj,...] = reconstructor_fx_deltas_and_omegas.get_loss_history()
		
		# Reconstructed f(xt) at training locations:
		fx_optimized_omegas_and_voxels[:,jj:jj+1] = reconstructor_fx_deltas_and_omegas.reconstruct_function_at(xpred=Xtrain)

		if using_deltas:
			fx_optimized_omegas_and_voxels[:,jj:jj+1] += Xtrain[:,jj:jj+1]


	# Save relevant quantities:
	save_data = True
	path2data = "{0:s}/{1:s}/learning_data_seed_{2:d}.pickle".format(path2project,path2folder,my_seed)
	if save_data:

		data2save = dict(	omegas_trainedNN=omegas_trainedNN,
							Sw_omegas_trainedNN=Sw_omegas_trainedNN,
							varphi_omegas_trainedNN=varphi_omegas_trainedNN,
							delta_omegas_trainedNN=delta_omegas_trainedNN,
							delta_statespace_trainedNN=delta_statespace_trainedNN,
							xpred=xpred,
							Nrollouts=Nrollouts,
							spectral_density_list=spectral_density_list,
							dim_ctx=dim_ctx,
							Dw_coarse=Dw_coarse,
							delta_statespace=delta_statespace,
							omega_lim=omega_lim,
							Nsamples_omega=Nsamples_omega,
							Xtrain=Xtrain,
							Ytrain=Ytrain,
							path2data=path2data,
							kXX=kXX)
		

		logger.info("Saving learned omegas, S_w, varphi_w, delta_w, delta_xt at {0:s} ...".format(path2data))
		file = open(path2data, 'wb')
		pickle.dump(data2save,file)
		file.close()
		logger.info("Done!")


	Ndiv_omega_for_analysis = 301
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_ctx) # [Ndiv**dim_in,dim_in]
	Sw_vec, phiw_vec = spectral_density_optimized_list[0].unnormalized_density(omegapred_analysis)


	hdl_fig, hdl_splots_reconstruct = plt.subplots(1,3,figsize=(30,10),sharex=False)
	extent_plot_omegas = [-omega_lim,omega_lim,-omega_lim,omega_lim] #  scalars (left, right, bottom, top)
	fx_true_testing = Ytrain + Xtrain[:,0:dim_in]
	for ii in range(Nrollouts):
		fx_true_testing_loc = fx_true_testing[ii*Npred:(ii+1)*Npred,0]
		fx_optimized_voxels_coarse_loc = fx_optimized_omegas_and_voxels[ii*Npred:(ii+1)*Npred,0] # The reconstructed function is the same for all Nsamples_nominal_dynsys


		hdl_splots_reconstruct[0].plot(xpred,fx_true_testing_loc,lw=1,color="crimson",alpha=0.35,label="True",linestyle="None",marker=".",markersize=5)
		# hdl_splots_reconstruct[0].plot(xpred,fx_reconstructed,lw=2,color="navy",alpha=0.5)
		hdl_splots_reconstruct[0].plot(xpred,fx_optimized_voxels_coarse_loc,lw=1,color="navy",alpha=0.7,label="Reconstructed",linestyle="None",marker=".",markersize=5)

	# hdl_splots_reconstruct[0].plot(Xtrain,fx_optimized_omegas_and_voxels[:,0],lw=1)
	# hdl_splots_reconstruct[0].plot(Xtrain,fx_discrete_grid,lw=1)
	# hdl_splots_reconstruct[0].set_xlim([-5,2])
	hdl_splots_reconstruct[0].set_xlim([xmin,xmax])
	# hdl_splots_reconstruct[0].set_ylim([-45.,2.])
	hdl_splots_reconstruct[0].set_xticks([xmin,0,xmax])
	hdl_splots_reconstruct[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[0].set_ylabel(r"$f(x_t;\theta_i)$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[0].set_title(r"Reconstruction; $M=20$",fontsize=fontsize_labels)
	
	S_vec_plotting = np.reshape(Sw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
	hdl_splots_reconstruct[1].imshow(S_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=S_vec_plotting.min(),vmax=S_vec_plotting.max(),interpolation='nearest')
	hdl_splots_reconstruct[1].set_title(r"${0:s}$".format("S(\omega)"),fontsize=fontsize_labels)
	hdl_splots_reconstruct[1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1].set_ylabel(r"$\omega_{\theta}$",fontsize=fontsize_labels)
	hdl_splots_reconstruct[1].plot(omegas_trainedNN[0,:,0],omegas_trainedNN[0,:,1],marker=".",color="navy",markersize=7,linestyle="None")


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
	hdl_splots_reconstruct[2].plot(omegas_trainedNN[0,:,0],omegas_trainedNN[0,:,1],marker=".",color="navy",markersize=7,linestyle="None")


	hdl_splots_reconstruct[1].set_xlim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[1].set_ylim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[2].set_xlim([-omega_lim,omega_lim])
	hdl_splots_reconstruct[2].set_ylim([-omega_lim,omega_lim])

	hdl_splots_reconstruct[1].set_xticks([-omega_lim,0,omega_lim])
	hdl_splots_reconstruct[1].set_yticks([-omega_lim,0,omega_lim])
	hdl_splots_reconstruct[2].set_xticks([-omega_lim,0,omega_lim])
	hdl_splots_reconstruct[2].set_yticks([-omega_lim,0,omega_lim])

	savefig = True
	if savefig:
		path2save_fig = "{0:s}/{1:s}/reconstruction_plots{2:d}.png".format(path2project,path2folder,my_seed)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.pause(1)
		plt.show(block=False)



def test_resulting_kernel():


	pass



if __name__ == "__main__":

	train_reconstruction()
