import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from datetime import datetime
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
from lqrker.models import MultiObjectiveReducedRankProcess
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


path2folder = "data_efficiency_test_with_dubinscar"

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


	assert ratio > 0.0 and ratio <= 1.0
	Ntest_max = int(Xtest.shape[0] * ratio)
	Xtest = Xtest[0:Ntest_max,:]
	Ytest = Ytest[0:Ntest_max,:]

	if using_deltas:
		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_x]
		Ytrain = tf.identity(Ytrain_deltas)

		Ytest_deltas = Ytest - Xtest[:,0:dim_x]
		Ytest = tf.identity(Ytest_deltas)

	return Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data


def reconstruct_for_ratio(cfg,ratio=1.0):

	savefig = True

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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
		Nepochs = 5000
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
	path2save = "{0:s}/{1:s}/reconstruction_data_{2:s}.pickle".format(path2project,path2folder,name_file_date)
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
							Xtest=Xtest,
							Ytest=Ytest,
							ratio=ratio,
							path2data=path2data)		
		
		logger.info("Saving data at {0:s} ...".format(path2save))
		file = open(path2save, 'wb')
		pickle.dump(data2save,file)
		file.close()
		logger.info("Done!")


	return data2save


def compute_model_error(cfg):


	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 


	file_name = "reconstruction_data_2023_03_26_22_25_20.pickle"
	path2load_full = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
	file = open(path2load_full, 'rb')
	data_dict = pickle.load(file)
	file.close()

	omegas_trainedNN = tf.convert_to_tensor(data_dict["omegas_trainedNN"],dtype=tf.float32)
	Sw_omegas_trainedNN = tf.convert_to_tensor(data_dict["Sw_omegas_trainedNN"],dtype=tf.float32)
	varphi_omegas_trainedNN = tf.convert_to_tensor(data_dict["varphi_omegas_trainedNN"],dtype=tf.float32)
	delta_omegas_trainedNN = tf.convert_to_tensor(data_dict["delta_omegas_trainedNN"],dtype=tf.float32)
	delta_statespace_trainedNN = tf.convert_to_tensor(data_dict["delta_statespace_trainedNN"],dtype=tf.float32)

	spectral_density_list = data_dict["spectral_density_list"]
	omega_lim = data_dict["omega_lim"]
	Nsamples_omega = data_dict["Nsamples_omega"]
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	Xtest = data_dict["Xtest"]
	Ytest = data_dict["Ytest"]

	logger.info("\n\n")
	logger.info(" * omega_lim: {0:f}".format(omega_lim))
	logger.info(" * Nsamples_omega: {0:d}".format(Nsamples_omega))

	dim_x = Ytest.shape[1]
	dim_u = Xtest.shape[1] - Ytrain.shape[1]

	# Initialize GP model:
	dim_in = dim_x + dim_u
	dim_out = dim_x
	logger.info(" * Initializing GP model ...")
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_in,cfg,spectral_density_list,Xtrain,Ytrain,using_deltas=using_deltas)

	MO_mean_test, MO_std_test = rrtp_MO.predict_at_locations(Xtest)


	plot_state_transition_reconstruction = True
	savefig = False
	if plot_state_transition_reconstruction:
		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,1,figsize=(16,14),sharex=False,sharey=False)
		hdl_fig.suptitle(r"State transition - Reconstructed; $\Delta x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state = np.reshape(hdl_splots_next_state,(-1,1))

		assert using_deltas == True

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(Ytest[:,jj])
			delta_fx_next_sorted = Ytest.numpy()[ind_xt_sorted,jj]
			delta_MO_mean_test_sorted = MO_mean_test.numpy()[ind_xt_sorted,jj]

			hdl_splots_next_state[jj,0].plot(delta_fx_next_sorted,linestyle="-",color="crimson",alpha=0.3,lw=3.0,label="Training data")
			hdl_splots_next_state[jj,0].plot(delta_MO_mean_test_sorted,linestyle="-",color="navy",alpha=0.7,lw=1.0,label="Reconstructed")

		hdl_splots_next_state[0,0].set_ylabel(r"$\Delta f_1(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state[1,0].set_ylabel(r"$\Delta f_2(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state[2,0].set_ylabel(r"$\Delta f_3(x_t)$",fontsize=fontsize_labels)

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
			path2save_fig = "{0:s}/{1:s}/state_transition_{2:s}.png".format(path2project,path2folder,name_file_date)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1)
			plt.show(block=True)


@hydra.main(config_path="./config",config_name="config")
def main(cfg):

	reconstruct_for_ratio(cfg,ratio=1.0)

	# compute_model_error(cfg)




if __name__ == "__main__":

	main()


	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/"*6200.png" ./data_quadruped_experiments_03_13_2023/
	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/"*6200.pickle" ./data_quadruped_experiments_03_13_2023/






