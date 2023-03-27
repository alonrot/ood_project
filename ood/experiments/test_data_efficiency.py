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
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity, KinkSharpSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity, QuadrupedSpectralDensity
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


# Counter to save data:
counter = 1001

path2folder = "data_efficiency_test_with_dubinscar"
# path2folder = "data_quadruped_experiments_03_13_2023"

using_deltas = True
# using_deltas = False

def load_data_dubins_car(path2project,ratio):

	path2data = "{0:s}/dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle".format(path2project)
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xdataset = data_dict["Xtrain"]
	Ydataset = data_dict["Ytrain"]
	dim_x = data_dict["dim_x"]
	dim_u = data_dict["dim_u"]
	Nsteps = data_dict["Nsteps"]
	Ntrajs = data_dict["Ntrajs"] # This is wrongly set to the same value as Nsteps

	dim_in = dim_x + dim_u
	dim_out = dim_x

	Xdataset_batch = np.reshape(Xdataset,(-1,Nsteps,Xdataset.shape[1])) # [Ntrajs,Nsteps,dim_x+dim_u]
	Ydataset_batch = np.reshape(Ydataset,(-1,Nsteps,Ydataset.shape[1])) # [Ntrajs,Nsteps,dim_x]

	Ntrajs4test = 10
	Ntrajs4train = Xdataset_batch.shape[0] - Ntrajs4test
	Xtest_batch = Xdataset_batch[-Ntrajs4test::,...] # [Ntrajs4test,Nsteps,dim_x+dim_u]
	Ytest_batch = Ydataset_batch[-Ntrajs4test::,...] # [Ntrajs4test,Nsteps,dim_x]

	Xtrain_batch = Xdataset_batch[0:Ntrajs4train,...] # [Ntrajs4train,Nsteps,dim_x+dim_u]
	Ytrain_batch = Ydataset_batch[0:Ntrajs4train,...] # [Ntrajs4train,Nsteps,dim_x]

	logger.info("Splitting dataset:")
	logger.info(" * Testing with {0:d} trajectories".format(Ntrajs4test))
	logger.info(" * Training with {0:d} trajectories".format(Ntrajs4train))


	# Return the trajectories vectorized:
	Xtrain = np.reshape(Xtrain_batch,(-1,Xtrain_batch.shape[2]))
	Ytrain = np.reshape(Ytrain_batch,(-1,Ytrain_batch.shape[2]))

	# Return the trajectories vectorized:
	Xtest = np.reshape(Xtest_batch,(-1,Xtest_batch.shape[2]))
	Ytest = np.reshape(Ytest_batch,(-1,Ytest_batch.shape[2]))

	if using_deltas:
		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_x]
		Ytrain = tf.identity(Ytrain_deltas)

		Ytest_deltas = Ytest - Xtest[:,0:dim_x]
		Ytest = tf.identity(Ytest_deltas)


	return Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data

@hydra.main(config_path="./config",config_name="config")
def reconstruct(cfg):

	savefig = True

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	# Load data:
	Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data_dubins_car(path2project,ratio) # Dubins car

	spectral_density_list = [None]*dim_out
	for jj in range(dim_out):
		spectral_density_list[jj] = DubinsCarSpectralDensity(cfg=cfg.spectral_density.dubinscar,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",use_nominal_model=True,Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
	


	"""
	0) Load data with a particular ratio
	1) Reconstruct
	2) Load MOrrtp and compute x_{t+1} log-evidence and RMSE
	3) Go to 0) with a higher ratio

	Repeat for GPSSM with standard kernel. We hope to see that our model does better with less data

	makse sure using_deltas = True
	"""

	xpred_training = tf.identity(Xtrain)
	fx_training = tf.identity(Ytrain)


	delta_statespace = 1.0 / Xtrain.shape[0]

	Nepochs = 13
	Nsamples_omega = 30
	if using_hybridrobotics:
		Nepochs = 6200
		Nsamples_omega = 1500
	
	omega_lim = 3.0
	Dw_coarse = (2.*omega_lim)**dim_in / Nsamples_omega # We are trainig a tensor [Nomegas,dim_in]
	# Dw_coarse = 1.0 / Nsamples_omega # We are trainig a tensor [Nomegas,dim_in]

	extent_plot_statespace = [xpred_training[0,0],xpred_training[-1,0],xpred_training[0,1],xpred_training[-1,1]] #  scalars (left, right, bottom, top)
	fx_optimized_omegas_and_voxels = np.zeros((xpred_training.shape[0],dim_out))
	Sw_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	varphi_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,dim_in))
	delta_omegas_trainedNN = np.zeros((dim_out,Nsamples_omega,1))
	delta_statespace_trainedNN = np.zeros((dim_out,Xtrain.shape[0],1))

	learning_rate_list = [1e-3,1e-3,1e-3]
	stop_loss_val = 1./fx_training.shape[0]
	# stop_loss_val = 0.01
	lengthscale_loss = 0.01
	loss_reconstruction_evolution = np.zeros((dim_out,Nepochs))
	spectral_density_optimized_list = [None]*dim_out
	# pdb.set_trace()
	for jj in range(dim_out):

		logger.info("Reconstruction for channel {0:d} / {1:d} ...".format(jj+1,dim_out))

		inverse_fourier_toolbox_channel = InverseFourierTransformKernelToolbox(spectral_density=spectral_density_list[jj],dim=dim_in)

		reconstructor_fx_deltas_and_omegas = ReconstructFunctionFromSpectralDensity(dim_in=dim_in,dw_voxel_init=Dw_coarse,dX_voxel_init=delta_statespace,
																					omega_lim=omega_lim,Nomegas=Nsamples_omega,
																					inverse_fourier_toolbox=inverse_fourier_toolbox_channel,
																					Xtest=xpred_training,Ytest=fx_training[:,jj:jj+1])

		reconstructor_fx_deltas_and_omegas.train(Nepochs=Nepochs,learning_rate=learning_rate_list[jj],stop_loss_val=stop_loss_val,lengthscale_loss=lengthscale_loss,print_every=10)


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
		fx_optimized_omegas_and_voxels[:,jj:jj+1] = reconstructor_fx_deltas_and_omegas.reconstruct_function_at(xpred=xpred_training)

		if using_deltas:
			fx_optimized_omegas_and_voxels[:,jj:jj+1] += xpred_training[:,jj:jj+1]


	# Save relevant quantities:
	save_data = True
	# save_data = False
	name_file_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	path2save = "{0:s}/{1:s}/reconstruction_data_{2:d}.pickle".format(path2project,path2folder,name_file_data)
	if save_data:

		data2save = dict(	omegas_trainedNN=omegas_trainedNN,
							Sw_omegas_trainedNN=Sw_omegas_trainedNN,
							varphi_omegas_trainedNN=varphi_omegas_trainedNN,
							delta_omegas_trainedNN=delta_omegas_trainedNN,
							delta_statespace_trainedNN=delta_statespace_trainedNN,
							spectral_density_list=spectral_density_list,
							Dw_coarse=Dw_coarse,
							delta_statespace=delta_statespace,
							omega_lim=omega_lim,
							Nsamples_omega=Nsamples_omega,
							Xtrain=Xtrain,
							Ytrain=Ytrain,
							path2data=path2data)		
		
		logger.info("Saving data at {0:s} ...".format(path2save))
		file = open(path2save, 'wb')
		pickle.dump(data2save,file)
		file.close()
		logger.info("Done!")






	"""
	Discrete grid of omega in the first two dimensions for plotting purposes. Slicing the rest to zero
	"""

	Ndiv_omega_for_analysis = 71
	if using_hybridrobotics: Ndiv_omega_for_analysis = 121
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=2) # [Ndiv**dim_in,dim_in]

	omegapred_analysis_fist_two_dims = tf.concat([omegapred_analysis,tf.zeros((omegapred_analysis.shape[0],3))],axis=1)

	COLOR_MAP = "summer"
	hdl_fig, hdl_splots_omegas = plt.subplots(dim_out,3,figsize=(14,10),sharex=False)
	hdl_fig.suptitle(r"Spectral density, phase and integrand for each channel; Dubins car kernel",fontsize=fontsize_labels)
	
	extent_plot_omegas = [omegapred_analysis[0,0],omegapred_analysis[-1,0],omegapred_analysis[0,1],omegapred_analysis[-1,1]] #  scalars (left, right, bottom, top)
	for jj in range(dim_out):
		
		# spectral_density.update_fdata(fdata=fx_true_testing[:,jj:jj+1])
		# spectral_density.update_dX_voxels(dX_new=delta_statespace_trainedNN[jj,...])

		# Spectral density and angle:
		Sw_vec, phiw_vec = spectral_density_optimized_list[jj].unnormalized_density(omegapred_analysis_fist_two_dims)

		# Integrand:
		# inverse_fourier_toolbox_channel = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_in,dim_out_ind=None)
		# inverse_fourier_toolbox_channel.update_spectral_density_and_angle(omegapred=omegapred_analysis_fist_two_dims,Dw=None,dX=None)
		# fx_integrand_unit_voxels_jj = inverse_fourier_toolbox_channel.get_fx_integrand_variable_voxels(xpred=Xtrain,Dw_vec=1.0) # We set Dw_vec=1 because we just wanna see the integrand
		# fx_integrand_averaged_states_jj = np.mean(fx_integrand_unit_voxels_jj,axis=0)


		# Spectral density:
		# S_vec_plotting = np.reshape(Sw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		S_vec_plotting = np.reshape(Sw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		hdl_splots_omegas[jj,0].imshow(S_vec_plotting,extent=extent_plot_omegas,origin="upper",cmap=plt.get_cmap(COLOR_MAP),vmin=S_vec_plotting.min(),vmax=S_vec_plotting.max(),interpolation='nearest')
		my_title = "S_{0:d}(\omega)".format(jj+1)
		hdl_splots_omegas[jj,0].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		if jj == dim_out-1: hdl_splots_omegas[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		hdl_splots_omegas[jj,0].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)
		hdl_splots_omegas[jj,0].set_xlim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,0].set_ylim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,0].set_xticks([-omega_lim,0,omega_lim])
		hdl_splots_omegas[jj,0].set_yticks([-omega_lim,0,omega_lim])


		# Varphi:
		if np.any(phiw_vec != 0.0):
			# phi_vec_plotting = np.reshape(phiw_vec[:,jj:jj+1],(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
			phi_vec_plotting = np.reshape(phiw_vec,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
			hdl_splots_omegas[jj,1].imshow(phi_vec_plotting,extent=extent_plot_omegas,origin="upper",cmap=plt.get_cmap(COLOR_MAP),vmin=phi_vec_plotting.min(),vmax=phi_vec_plotting.max(),interpolation='nearest')
			my_title = "\\varphi_{0:d}(\omega)".format(jj+1)
			hdl_splots_omegas[jj,1].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		else:
			for jj in range(dim_out): hdl_splots_omegas[jj,1].set_xticks([],[]); hdl_splots_omegas[jj,1].set_yticks([],[])
		hdl_splots_omegas[jj,1].set_xlim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,1].set_ylim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,1].set_xticks([-omega_lim,0,omega_lim])
		hdl_splots_omegas[jj,1].set_yticks([-omega_lim,0,omega_lim])



		# Plotting the integrand (without the voxels, obviously):
		# fx_integrand_averaged_states_jj_plotting = np.reshape(fx_integrand_averaged_states_jj,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
		# hdl_splots_omegas[jj,2].imshow(fx_integrand_averaged_states_jj_plotting,extent=extent_plot_omegas,origin="upper",cmap=plt.get_cmap(COLOR_MAP),vmin=fx_integrand_averaged_states_jj_plotting.min(),vmax=fx_integrand_averaged_states_jj_plotting.max(),interpolation='nearest')
		my_title = "(1/T)\sum_t g_{0:d}(x_t;\omega)".format(jj+1)
		hdl_splots_omegas[jj,2].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
		hdl_splots_omegas[jj,2].set_xlim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,2].set_ylim([-omega_lim,omega_lim])
		hdl_splots_omegas[jj,2].set_xticks([-omega_lim,0,omega_lim])
		hdl_splots_omegas[jj,2].set_yticks([-omega_lim,0,omega_lim])

		# Add the resulting omegas:
		hdl_splots_omegas[jj,0].plot(omegas_trainedNN[jj,:,0],omegas_trainedNN[jj,:,1],marker=".",color="indigo",markersize=2,linestyle="None")
		hdl_splots_omegas[jj,1].plot(omegas_trainedNN[jj,:,0],omegas_trainedNN[jj,:,1],marker=".",color="indigo",markersize=2,linestyle="None")

		if jj == dim_out-1: hdl_splots_omegas[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		if jj == dim_out-1: hdl_splots_omegas[jj,1].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
		if jj == dim_out-1: hdl_splots_omegas[jj,2].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
	

	if savefig:
		# path2save_fig = "{0:s}/{1:s}/spectral_density_Nepochs{2:d}.png".format(path2project,path2folder,Nepochs)
		path2save_fig = "{0:s}/{1:s}/spectral_density_counter_{2:d}.png".format(path2project,path2folder,Nepochs)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.pause(1)
		plt.show(block=False)


	plot_state_transition_reconstruction = True
	if plot_state_transition_reconstruction:
		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,1,figsize=(16,14),sharex=False,sharey=False)
		hdl_fig.suptitle(r"State transition - Reconstructed; $x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state = np.reshape(hdl_splots_next_state,(-1,1))
		xpred_testing_for_transition_plot = xpred_testing.numpy()[0:1000,:]

		if using_deltas:
			fx_true_testing_plotting = fx_true_testing + xpred_testing[:,0:dim_out]
		else:
			fx_true_testing_plotting = fx_true_testing

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(xpred_testing_for_transition_plot[:,jj])
			xt_sorted = xpred_testing_for_transition_plot[ind_xt_sorted,jj]
			
			fx_optimized_omegas_and_voxels_sorted = fx_optimized_omegas_and_voxels[ind_xt_sorted,jj]
			# hdl_splots_next_state[jj,0].plot(xt_sorted,fx_optimized_omegas_and_voxels_sorted,marker=".",markersize=3,linestyle="None",color="navy",alpha=0.3)

			# fx_true_testing_sorted = fx_true_testing.numpy()[ind_xt_sorted,jj]
			fx_true_testing_sorted = fx_true_testing_plotting.numpy()[ind_xt_sorted,jj]
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

		if savefig:
			# path2save_fig = "{0:s}/{1:s}/state_transition_Nepochs{2:d}.png".format(path2project,path2folder,Nepochs)
			path2save_fig = "{0:s}/{1:s}/state_transition_counter_{2:d}.png".format(path2project,path2folder,Nepochs)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1)
			plt.show(block=False)


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

			if using_deltas:
				fx_true_testing_sliced += xpred_testing[jj*Nsteps:(jj+1)*Nsteps,0:dim_out]

			label_true = None
			if jj == 0: label_true = "Training data"
			hdl_splots_statespace.plot(fx_true_testing_sliced[:,0],fx_true_testing_sliced[:,1],color=colors[jj],linestyle="-",marker="None",linewidth=2.0,alpha=0.3,label=label_true)
			hdl_splots_statespace.plot(fx_true_testing_sliced[0,0],fx_true_testing_sliced[0,1],color="olivedrab",marker="o",markersize=6,linestyle="None")
			hdl_splots_statespace.plot(fx_true_testing_sliced[-1,0],fx_true_testing_sliced[-1,1],color="black",marker="x",markersize=6,linestyle="None")

		hdl_splots_statespace.set_xlabel(r"$x_{t,1}$",fontsize=fontsize_labels)
		hdl_splots_statespace.set_ylabel(r"$x_{t,2}$",fontsize=fontsize_labels)
		hdl_splots_statespace.legend(loc="best",fontsize=fontsize_labels)


		if savefig:
			# path2save_fig = "{0:s}/{1:s}/state_trajectories_Nepochs{2:d}.png".format(path2project,path2folder,Nepochs)
			path2save_fig = "{0:s}/{1:s}/state_trajectories_counter_{2:d}.png".format(path2project,path2folder,Nepochs)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1)
			plt.show(block=False)


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


		if savefig:
			# path2save_fig = "{0:s}/{1:s}/reconstruction_loss_Nepochs{2:d}.png".format(path2project,path2folder,Nepochs)
			path2save_fig = "{0:s}/{1:s}/reconstruction_loss_counter_{2:d}.png".format(path2project,path2folder,Nepochs)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1)
			plt.show(block=False)


	plt.show(block=True)


if __name__ == "__main__":

	reconstruct()




	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/"*6200.png" ./data_quadruped_experiments_03_13_2023/
	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/"*6200.pickle" ./data_quadruped_experiments_03_13_2023/






