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

@hydra.main(config_path="./config",config_name="config")
def reconstruct(cfg):

	"""
	Get training dataset
	"""
	path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"
	assert path2data != "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model.pickle", "I accidentally overwrote this"
	# Create a TF dataset: https://www.tensorflow.org/datasets/add_dataset
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	dim_x = data_dict["dim_x"]
	dim_u = data_dict["dim_u"]
	Nsteps = data_dict["Nsteps"]
	Ntrajs = data_dict["Ntrajs"]
	deltaT = data_dict["deltaT"]

	dim_in = dim_x + dim_u
	dim_out = dim_x

	# integration_method = "integrate_with_regular_grid"
	# integration_method = "integrate_with_irregular_grid"
	# integration_method = "integrate_with_bayesian_quadrature"
	integration_method = "integrate_with_data"

	if integration_method == "integrate_with_regular_grid":
		spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.dubinscar,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,use_nominal_model=True,Xtrain=None,Ytrain=None)

		Xtrain_backup = tf.identity(Xtrain)
		Ytrain_backup = tf.identity(Ytrain)

		Xtrain = tf.identity(spectral_density.xdata)
		Ytrain = tf.identity(spectral_density.fdata)

		# Testing dataset (we use the training dataset)
		xpred_testing = spectral_density.xdata
		fx_true_testing = spectral_density.fdata

	elif integration_method == "integrate_with_data":
		spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.dubinscar,cfg.sampler.hmc,dim=dim_in,integration_method=integration_method,use_nominal_model=True,Xtrain=Xtrain,Ytrain=Ytrain)

		# Testing dataset (we use the training dataset)
		xpred_testing = tf.identity(Xtrain)
		fx_true_testing = tf.identity(Ytrain)


	# # Discrete grid:
	# # L = 200.0; Ndiv = 5 # 5**5=3125 # works
	# # L = 100.0; Ndiv = 3 # 3**5=243 # reasonable for being just Ndiv=3
	# # L = 50.0; Ndiv = 5 # 5**5=3125
	# L = 10.0; Ndiv = 3 # 5**5=3125
	# cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_in
	# assert Ndiv % 2 != 0 and Ndiv > 2, "Ndiv must be an odd positive integer"
	# j_indices = CommonUtils.create_Ndim_grid(xmin=-(Ndiv-1)//2,xmax=(Ndiv-1)//2,Ndiv=Ndiv,dim=dim_in) # [Ndiv**dim_x,dim_x]
	# omegas_weights = tf.cast((math.pi/L) * j_indices,dtype=tf.float32)
	# # _, _, omegas_weights = spectral_density.get_Wpoints_discrete(L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False)
	# Nsamples_omega = omegas_weights.shape[0]

	omegas_weights = None
	# Nsamples_omega = 20
	# Nepochs = 10
	Nsamples_omega = 1000
	omega_lim = 1.0
	Nepochs = 200
	extent_plot_statespace = [xpred_testing[0,0],xpred_testing[-1,0],xpred_testing[0,1],xpred_testing[-1,1]] #  scalars (left, right, bottom, top)
	fx_optimized_omegas_and_voxels = np.zeros((xpred_testing.shape[0],dim_out))
	omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,dim_in))
	delta_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	delta_statespace_trainedNN = np.zeros((dim_out,Xtrain.shape[0],1))

	loss_reconstruction_evolution = np.zeros((dim_out,Nepochs))
	for jj in range(dim_out):

		logger.info("Reconstruction for channel {0:d} / {1:d} ...".format(jj+1,dim_out))

		spectral_density.update_fdata(fdata=fx_true_testing[:,jj:jj+1])
		inverse_fourier_toolbox_channel = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in,dim_out_ind=None)

		reconstructor_fx_deltas_and_omegas = ReconstructFunctionFromSpectralDensity(dim_in=dim_in,omega_lim=omega_lim,Nomegas=Nsamples_omega,
																					inverse_fourier_toolbox=inverse_fourier_toolbox_channel,
																					Xtrain=xpred_testing,Ytrain=fx_true_testing[:,jj:jj+1],
																					omegas_weights=omegas_weights)
		reconstructor_fx_deltas_and_omegas.train(Nepochs=Nepochs,learning_rate=1e-1,stop_loss_val=0.0001,print_every=10)

		# Collect trained variables for each channel:
		omegas_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.get_omegas_weights()
		delta_omegas_trainedNN[jj,:,0] = reconstructor_fx_deltas_and_omegas.get_delta_omegas(reconstructor_fx_deltas_and_omegas.delta_omegas_pre_activation) # [Nomegas,]
		delta_statespace_trainedNN[jj,...] = reconstructor_fx_deltas_and_omegas.get_delta_statespace(reconstructor_fx_deltas_and_omegas.delta_statespace_preactivation) # [Nxpoints,1]

		# Keep track of the loss evolution:
		loss_reconstruction_evolution[jj,...] = reconstructor_fx_deltas_and_omegas.get_loss_history()
		
		# Reconstructed f(xt) at training locations:
		if integration_method == "integrate_with_regular_grid":
			fx_optimized_omegas_and_voxels[:,jj:jj+1] = reconstructor_fx_deltas_and_omegas.reconstruct_function_at(xpred=Xtrain_backup)
		else:
			fx_optimized_omegas_and_voxels[:,jj:jj+1] = reconstructor_fx_deltas_and_omegas.reconstruct_function_at(xpred=xpred_testing)








		raise ValueError("Maybe we need more data???????????????????????????????")

		raise ValueError("Make sure # which_features: "RRPDiscreteCosineFeaturesVariableIntegrationStep"")






































	# Save relevant quantities:
	# save_data = True
	save_data = False
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_learned_spectral_density_parameters_irregular_grid_omegalim1p0_omegas_within_lims.pickle"
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_learned_spectral_density_parameters_irregular_grid_omegalim0p5.pickle"
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_learned_spectral_density_parameters_irregular_grid.pickle"
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_learned_spectral_density_parameters_regular_Xgrid_and_omega_grid.pickle"
	# path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_learned_spectral_density_parameters_regular_Xgrid_irregular_omega_grid.pickle"
	if save_data:

		Sw_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
		varphi_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))

		for jj in range(dim_out):

			# In order to evaluate the spectral density and angle we need the correct data channel and the voxels:
			spectral_density.update_fdata(fdata=fx_true_testing[:,jj:jj+1])
			spectral_density.update_dX_voxels(dX_new=delta_statespace_trainedNN[jj,...])
			Sw_omegas_trainedNN[jj,...], varphi_omegas_trainedNN[jj,...] = spectral_density.unnormalized_density(omegas_trainedNN[jj,...]) # This is not going to work because we need the right delta_statespace_trainedNN dimensions

		data2save = dict(	omegas_trainedNN=omegas_trainedNN,
							Sw_omegas_trainedNN=Sw_omegas_trainedNN,
							varphi_omegas_trainedNN=varphi_omegas_trainedNN,
							delta_omegas_trainedNN=delta_omegas_trainedNN,
							delta_statespace_trainedNN=delta_statespace_trainedNN)
		
		logger.info("Saving learned omegas, S_w, varphi_w, delta_w, delta_xt at {0:s} ...".format(path2save))
		file = open(path2save, 'wb')
		pickle.dump(data2save,file)
		file.close()
		logger.info("Done!")
	


	"""
	Discrete grid of omega in the first two dimensions for plotting purposes. Slicing the rest to zero
	"""

	Ndiv_omega_for_analysis = 71
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=2) # [Ndiv**dim_in,dim_in]

	omegapred_analysis_fist_two_dims = tf.concat([omegapred_analysis,tf.zeros((omegapred_analysis.shape[0],3))],axis=1)

	COLOR_MAP = "summer"
	hdl_fig, hdl_splots_omegas = plt.subplots(dim_out,3,figsize=(14,10),sharex=False)
	hdl_fig.suptitle(r"Spectral density, phase and integrand for each channel; Dubins car kernel",fontsize=fontsize_labels)
	
	extent_plot_omegas = [omegapred_analysis[0,0],omegapred_analysis[-1,0],omegapred_analysis[0,1],omegapred_analysis[-1,1]] #  scalars (left, right, bottom, top)
	for jj in range(dim_out):
		
		spectral_density.update_fdata(fdata=fx_true_testing[:,jj:jj+1])
		spectral_density.update_dX_voxels(dX_new=delta_statespace_trainedNN[jj,...])

		# Spectral density and angle:
		Sw_vec, phiw_vec = spectral_density.unnormalized_density(omegapred_analysis_fist_two_dims)

		# Integrand:
		inverse_fourier_toolbox_channel = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in,dim_out_ind=None)
		inverse_fourier_toolbox_channel.update_spectral_density_and_angle(omegapred=omegapred_analysis_fist_two_dims,Dw=None,dX=None)
		fx_integrand_unit_voxels_jj = inverse_fourier_toolbox_channel.get_fx_integrand_variable_voxels(xpred=Xtrain,Dw_vec=1.0) # We set Dw_vec=1 because we just wanna see the integrand
		fx_integrand_averaged_states_jj = np.mean(fx_integrand_unit_voxels_jj,axis=0)


		# Spectral density:
		# S_vec_plotting = np.reshape(Sw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		S_vec_plotting = np.reshape(Sw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		hdl_splots_omegas[jj,0].imshow(S_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=S_vec_plotting.min(),vmax=S_vec_plotting.max(),interpolation='nearest')
		my_title = "S_{0:d}(\omega)".format(jj+1)
		hdl_splots_omegas[jj,0].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		if jj == dim_out-1: hdl_splots_omegas[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		hdl_splots_omegas[jj,0].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)


		# Varphi:
		if np.any(phiw_vec != 0.0):
			# phi_vec_plotting = np.reshape(phiw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
			phi_vec_plotting = np.reshape(phiw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
			hdl_splots_omegas[jj,1].imshow(phi_vec_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=phi_vec_plotting.min(),vmax=phi_vec_plotting.max(),interpolation='nearest')
			my_title = "\\varphi_{0:d}(\omega)".format(jj+1)
			hdl_splots_omegas[jj,1].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		else:
			for jj in range(dim_out): hdl_splots_omegas[jj,1].set_xticks([],[]); hdl_splots_omegas[jj,1].set_yticks([],[])


		# Plotting the integrand (without the voxels, obviously):
		fx_integrand_averaged_states_jj_plotting = np.reshape(fx_integrand_averaged_states_jj,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		hdl_splots_omegas[jj,2].imshow(fx_integrand_averaged_states_jj_plotting,extent=extent_plot_omegas,origin="lower",cmap=plt.get_cmap(COLOR_MAP),vmin=fx_integrand_averaged_states_jj_plotting.min(),vmax=fx_integrand_averaged_states_jj_plotting.max(),interpolation='nearest')
		my_title = "(1/T)\sum_t g_{0:d}(x_t;\omega)".format(jj+1)
		hdl_splots_omegas[jj,2].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		hdl_splots_omegas[jj,2].set_xlim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,2].set_ylim([-omega_lim,omega_lim])

		# Add the resulting omegas:
		hdl_splots_omegas[jj,2].plot(omegas_trainedNN[jj,:,0],omegas_trainedNN[jj,:,1],marker=".",color="indigo",markersize=3,linestyle="None")

		if jj == dim_out-1: hdl_splots_omegas[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		if jj == dim_out-1: hdl_splots_omegas[jj,1].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		if jj == dim_out-1: hdl_splots_omegas[jj,2].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)



	plot_state_transition_reconstruction = True
	if plot_state_transition_reconstruction and integration_method == "integrate_with_data":
		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,1,figsize=(16,14),sharex=False,sharey=False)
		hdl_fig.suptitle(r"State transition - Reconstructed; $x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state = np.reshape(hdl_splots_next_state,(-1,1))
		xpred_testing_for_transition_plot = xpred_testing.numpy()[0:1000,:]

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(xpred_testing_for_transition_plot[:,jj])
			xt_sorted = xpred_testing_for_transition_plot[ind_xt_sorted,jj]
			
			fx_optimized_omegas_and_voxels_sorted = fx_optimized_omegas_and_voxels[ind_xt_sorted,jj]
			# hdl_splots_next_state[jj,0].plot(xt_sorted,fx_optimized_omegas_and_voxels_sorted,marker=".",markersize=3,linestyle="None",color="navy",alpha=0.3)

			fx_true_testing_sorted = fx_true_testing.numpy()[ind_xt_sorted,jj]
			hdl_splots_next_state[jj,0].plot(xt_sorted,fx_true_testing_sorted,marker=".",markersize=3,linestyle="None",color="navy",alpha=0.3,label="True")
			hdl_splots_next_state[jj,0].plot(xt_sorted,fx_optimized_omegas_and_voxels_sorted,marker=".",markersize=3,linestyle="None",color="crimson",alpha=0.3,label="Reconstructed")

			hdl_splots_next_state[jj,0].set_xlim([xt_sorted.min(),xt_sorted.max()])

		hdl_splots_next_state[0,0].set_ylabel(r"$f_1(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state[1,0].set_ylabel(r"$f_2(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state[2,0].set_ylabel(r"$f_3(x_t)$",fontsize=fontsize_labels)

		hdl_splots_next_state[0,0].set_xlabel(r"$x_{t,1}$",fontsize=fontsize_labels)
		hdl_splots_next_state[1,0].set_xlabel(r"$x_{t,2}$",fontsize=fontsize_labels)
		hdl_splots_next_state[2,0].set_xlabel(r"$x_{t,3}$",fontsize=fontsize_labels)

		# hdl_splots_next_state[0,0].set_title("Reconstructed dynamics",fontsize=fontsize_labels)
		# hdl_splots_next_state[0,1].set_title("True dynamics",fontsize=fontsize_labels)
		# hdl_splots_next_state[-1,0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		# hdl_splots_next_state[-1,1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		# 
		lgnd = hdl_splots_next_state[-1,0].legend(loc="best",fontsize=fontsize_labels)
		lgnd.legendHandles[0]._legmarker.set_markersize(20)
		lgnd.legendHandles[1]._legmarker.set_markersize(20)

		# plt.show(block=True)





	# # Trajectories:
	# hdl_fig, hdl_splots_statespace = plt.subplots(1,2,figsize=(14,10),sharex=True,sharey=True)
	# Ntrajs_cutted = Xtrain.shape[0] // Nsteps
	# for jj in range(min(Ntrajs_cutted,2)):
	# # for jj in range(Ntrajs_cutted):

	# 	fx_optimized_omegas_and_voxels_sliced = fx_optimized_omegas_and_voxels[jj*Nsteps:(jj+1)*Nsteps,...]

	# 	hdl_splots_statespace[0].plot(fx_optimized_omegas_and_voxels_sliced[:,0],fx_optimized_omegas_and_voxels_sliced[:,1],color="gray",linestyle="--",marker="None",linewidth=0.5,alpha=0.6)
	# 	hdl_splots_statespace[0].plot(fx_optimized_omegas_and_voxels_sliced[0,0],fx_optimized_omegas_and_voxels_sliced[0,1],color="olivedrab",marker="o",markersize=3,linestyle="None",label="init")
	# 	hdl_splots_statespace[0].plot(fx_optimized_omegas_and_voxels_sliced[-1,0],fx_optimized_omegas_and_voxels_sliced[-1,1],color="black",marker="x",markersize=3,linestyle="None",label="end")


	# 	fx_true_testing_sliced = fx_true_testing[jj*Nsteps:(jj+1)*Nsteps,...]

	# 	hdl_splots_statespace[1].plot(fx_true_testing_sliced[:,0],fx_true_testing_sliced[:,1],color="gray",linestyle="--",marker="None",linewidth=0.5,alpha=0.6)
	# 	hdl_splots_statespace[1].plot(fx_true_testing_sliced[0,0],fx_true_testing_sliced[0,1],color="olivedrab",marker="o",markersize=3,linestyle="None",label="init")
	# 	hdl_splots_statespace[1].plot(fx_true_testing_sliced[-1,0],fx_true_testing_sliced[-1,1],color="black",marker="x",markersize=3,linestyle="None",label="end")
	# 	


	# Trajectories:
	plot_state_trajectories_reconstruction = True
	if integration_method == "integrate_with_regular_grid": fx_true_testing = Ytrain_backup;
	if plot_state_trajectories_reconstruction:
		hdl_fig, hdl_splots_statespace = plt.subplots(1,1,figsize=(14,10),sharex=True,sharey=True)
		hdl_fig.suptitle(r"Reconstructed trajectories $(x_{t,1},x_{t,2})$",fontsize=fontsize_labels)
		Ntrajs_cutted = Xtrain.shape[0] // Nsteps
		colors = ["navy","crimson","darkgreen"]
		for jj in range(min(Ntrajs_cutted,3)):
		# for jj in range(Ntrajs_cutted):

			fx_optimized_omegas_and_voxels_sliced = fx_optimized_omegas_and_voxels[jj*Nsteps:(jj+1)*Nsteps,...]

			label_reconstructed = None
			if jj == 0: label_reconstructed = "Reconstructed"
			hdl_splots_statespace.plot(fx_optimized_omegas_and_voxels_sliced[:,0],fx_optimized_omegas_and_voxels_sliced[:,1],color=colors[jj],linestyle="-",marker="None",linewidth=2.0,alpha=0.7,label=label_reconstructed)
			hdl_splots_statespace.plot(fx_optimized_omegas_and_voxels_sliced[0,0],fx_optimized_omegas_and_voxels_sliced[0,1],color="olivedrab",marker="o",markersize=6,linestyle="None")
			hdl_splots_statespace.plot(fx_optimized_omegas_and_voxels_sliced[-1,0],fx_optimized_omegas_and_voxels_sliced[-1,1],color="black",marker="x",markersize=6,linestyle="None")

			fx_true_testing_sliced = fx_true_testing[jj*Nsteps:(jj+1)*Nsteps,...]

			label_true = None
			if jj == 0: label_true = "Training data"
			hdl_splots_statespace.plot(fx_true_testing_sliced[:,0],fx_true_testing_sliced[:,1],color=colors[jj],linestyle="-",marker="None",linewidth=2.0,alpha=0.3,label=label_true)
			hdl_splots_statespace.plot(fx_true_testing_sliced[0,0],fx_true_testing_sliced[0,1],color="olivedrab",marker="o",markersize=6,linestyle="None")
			hdl_splots_statespace.plot(fx_true_testing_sliced[-1,0],fx_true_testing_sliced[-1,1],color="black",marker="x",markersize=6,linestyle="None")

		hdl_splots_statespace.set_xlabel(r"$x_{t,1}$",fontsize=fontsize_labels)
		hdl_splots_statespace.set_ylabel(r"$x_{t,2}$",fontsize=fontsize_labels)
		hdl_splots_statespace.legend(loc="best",fontsize=fontsize_labels)

	# plt.show(block=True)


	# Loss:
	plot_reconstruction_loss_evolution = True
	if plot_reconstruction_loss_evolution:

		# Loss evolution:
		hdl_fig, hdl_splots_loss = plt.subplots(dim_out,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle(r"Reconstruction log-loss $\log \mathcal{L}(\omega_j,\Delta \omega_j,\Delta x_t)$ for each channel",fontsize=fontsize_labels)
		for jj in range(dim_out):
			hdl_splots_loss[jj].plot(np.arange(1,loss_reconstruction_evolution.shape[1]+1),np.log(loss_reconstruction_evolution[jj,...]),linestyle="-",marker=".",linewidth=0.5,markersize=3,color="navy")
			hdl_splots_loss[jj].set_ylabel(r"$\log \mathcal{L}(\cdot)$",fontsize=fontsize_labels)
		hdl_splots_loss[-1].set_xlabel(r"Epochs",fontsize=fontsize_labels)


	plt.show(block=True)


if __name__ == "__main__":

	reconstruct()
