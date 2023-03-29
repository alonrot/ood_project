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



# GP flow:
import gpflow as gpf
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("github")
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# plt.rc('legend',fontsize=fontsize_labels+2)
plt.rc('legend',fontsize=fontsize_labels//2)


# path2folder = "data_efficiency_test_with_dubinscar"
path2folder = "data_efficiency_test_with_quadruped_data_03_25_2023"

using_deltas = True
# using_deltas = False


def load_data(path2project,path2file,ratio):

	path2data = "{0:s}/{1:s}".format(path2project,path2file)
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xdataset = data_dict["Xtrain"]
	Ydataset = data_dict["Ytrain"]
	# Ntrajs = data_dict["Ntrajs"] # This is wrongly set to the same value as Nsteps
	Ntrajs = None

	assert ratio > 0.0 and ratio <= 1.0

	if "dubins" in path2file:

		raise NotImplementedError("Code this up by bascially sampling random state-action-state tuples according to the ratios")

		dim_x = data_dict["dim_x"]
		dim_u = data_dict["dim_u"]
		Nsteps = data_dict["Nsteps"]

		dim_in = dim_x + dim_u
		dim_out = dim_x

		Xdataset_batch = np.reshape(Xdataset,(-1,Nsteps,Xdataset.shape[1])) # [Ntrajs,Nsteps,dim_x+dim_u]
		Ydataset_batch = np.reshape(Ydataset,(-1,Nsteps,Ydataset.shape[1])) # [Ntrajs,Nsteps,dim_x]

		# Split dataset: LAst 10 trajectories will be for testing; the rest are traning data
		Ntrajs4test = 10
		Ntrajs4train = Xdataset_batch.shape[0] - Ntrajs4test
		Xtest_batch = Xdataset_batch[-Ntrajs4test::,...] # [Ntrajs4test,Nsteps,dim_x+dim_u]
		Ytest_batch = Ydataset_batch[-Ntrajs4test::,...] # [Ntrajs4test,Nsteps,dim_x]

		Xtrain_batch = Xdataset_batch[0:Ntrajs4train,...] # [Ntrajs4train,Nsteps,dim_x+dim_u]
		Ytrain_batch = Ydataset_batch[0:Ntrajs4train,...] # [Ntrajs4train,Nsteps,dim_x]

		logger.info("Splitting dataset:")
		logger.info(" * Testing with {0:d} / {1:d} trajectories".format(Ntrajs4test,Xdataset_batch.shape[0]))
		logger.info(" * Training with {0:d} / {1:d} trajectories".format(Ntrajs4train,Xdataset_batch.shape[0]))

		# Return the trajectories vectorized:
		Xtrain = np.reshape(Xtrain_batch,(-1,Xtrain_batch.shape[2]))
		Ytrain = np.reshape(Ytrain_batch,(-1,Ytrain_batch.shape[2]))

		# Return the trajectories vectorized:
		Xtest = np.reshape(Xtest_batch,(-1,Xtest_batch.shape[2]))
		Ytest = np.reshape(Ytest_batch,(-1,Ytest_batch.shape[2]))

		# Slice according to requested ratio:
		Ntrain_max = int(Xtrain.shape[0] * ratio)
		logger.info(" * Requested ratio: {0:2.2f} | Training with {1:d} / {2:d} datapoints".format(ratio,Ntrain_max,Xtrain.shape[0]))
		Xtrain = Xtrain[0:Ntrain_max,:]
		Ytrain = Ytrain[0:Ntrain_max,:]


	if "quadruped" in path2file:

		# Test using 10% of the data. Traing using the rest a percentage of the rest, indicated by "ratio"
		Ndataset = Xdataset.shape[0]
		ind_samples = np.arange(0,Xdataset.shape[0])
		Ntest = int(Ndataset*0.1)
		ind_samples_test = np.random.choice(ind_samples,size=(Ntest),replace=False)

		# Take the rest by creating a mask:
		mask = np.ones(Ndataset) == 1
		for ii in range(Ndataset):
			if np.any(ii == ind_samples_test):
				mask[ii] = False

		# mask = np.prod(np.reshape(ind_samples,(-1,1)) - ind_samples_test.T,axis=1) != 0
		ind_samples_train = ind_samples[mask]

		Ntrain_with_perc = int(ratio*(Ndataset - Ntest))
		ind_samples_train_perc = np.random.choice(ind_samples_train,size=(Ntrain_with_perc,),replace=False)

		Xtest = Xdataset[ind_samples_test,:]
		Ytest = Ydataset[ind_samples_test,:]

		Xtrain = Xdataset[ind_samples_train_perc,:]
		Ytrain = Ydataset[ind_samples_train_perc,:]

		dim_in = Xtrain.shape[1]
		dim_out = dim_x = Ytrain.shape[1]


		Xtest = tf.convert_to_tensor(Xtest,dtype=tf.float32)
		Ytest = tf.convert_to_tensor(Ytest,dtype=tf.float32)
		Xtrain = tf.convert_to_tensor(Xtrain,dtype=tf.float32)
		Ytrain = tf.convert_to_tensor(Ytrain,dtype=tf.float32)

		logger.info(" * Requested ratio: {0:2.2f} | Training with {1:d} / {2:d} datapoints".format(ratio,Ntrain_with_perc,len(ind_samples_train_perc)))
		logger.info(" *                             Testing with {0:d} datapoints".format(len(ind_samples_test)))

		Nsteps = None

		# pdb.set_trace()


	if using_deltas:
		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_x]
		Ytrain = tf.identity(Ytrain_deltas)

		Ytest_deltas = Ytest - Xtest[:,0:dim_x]
		Ytest = tf.identity(Ytest_deltas)



	return Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data


def train_MOrrtp_by_reconstructing(cfg,ratio):

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	assert path2folder in ["dubins_car_reconstruction","data_efficiency_test_with_quadruped_data_03_25_2023"]


	if path2folder == "dubins_car_reconstruction":
		path2file = "dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"
		Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data(path2project,path2file,ratio)
		spectral_density_list = [None]*dim_out
		for jj in range(dim_out):
			spectral_density_list[jj] = DubinsCarSpectralDensity(cfg=cfg.spectral_density.dubinscar,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",use_nominal_model=True,Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
	

	if path2folder == "data_efficiency_test_with_quadruped_data_03_25_2023":
		path2file = "data_quadruped_experiments_03_25_2023/joined_go1trajs_trimmed_2023_03_25.pickle"
		Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data(path2project,path2file,ratio)
		spectral_density_list = [None]*dim_out
		for jj in range(dim_out):
			spectral_density_list[jj] = QuadrupedSpectralDensity(cfg=cfg.spectral_density.quadruped,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])



	xpred_training = tf.identity(Xtrain)
	fx_training = tf.identity(Ytrain)

	delta_statespace = 1.0 / Xtrain.shape[0]

	Nepochs = 13
	Nsamples_omega = 30
	if using_hybridrobotics:
		Nepochs = 3000
		Nsamples_omega = 1000

		# Nepochs = 13
		# Nsamples_omega = 30

	
	omega_lim = 5.0
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
	
	path2save = "{0:s}/{1:s}/reconstruction_data_{2:s}.pickle".format(path2project,path2folder,name_file_date)
	logger.info("Saving data at {0:s} ...".format(path2save))
	file = open(path2save, 'wb')
	pickle.dump(data2save,file)
	file.close()
	logger.info("Done!")


	path2log_file = "{0:s}/{1:s}/MOrrtp_trained_log_file_{2:s}.txt".format(path2project,path2folder,name_file_date)
	logger.info("Writing ratio to log file at {0:s} ...".format(path2log_file))
	file = open(path2log_file, 'w')
	file.write("ratio: {0:2.2f}".format(ratio))
	file.close()
	logger.info("Done!")

	plt.pause(2.0)


	return name_file_date


def train_gpssm(cfg,ratio):

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	assert path2folder in ["dubins_car_reconstruction","data_efficiency_test_with_quadruped_data_03_25_2023"]


	if path2folder == "dubins_car_reconstruction":
		path2file = "dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"
		Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data(path2project,path2file,ratio)
		spectral_density_list = [None]*dim_out
		for jj in range(dim_out):
			spectral_density_list[jj] = DubinsCarSpectralDensity(cfg=cfg.spectral_density.dubinscar,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",use_nominal_model=True,Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
	

	if path2folder == "data_efficiency_test_with_quadruped_data_03_25_2023":
		path2file = "data_quadruped_experiments_03_25_2023/joined_go1trajs_trimmed_2023_03_25.pickle"
		Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data(path2project,path2file,ratio)
		spectral_density_list = [None]*dim_out
		for jj in range(dim_out):
			spectral_density_list[jj] = QuadrupedSpectralDensity(cfg=cfg.spectral_density.quadruped,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])

	# name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	# using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	# logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	# path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	# if using_hybridrobotics:
	# 	path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	# # Load data:
	# Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data(path2project,ratio) # Dubins car

	# Based on: https://gpflow.github.io/GPflow/develop/notebooks/advanced/multioutput.html#
	MAXITER = reduce_in_tests(2000)
	# MAXITER = 10

	N = Xtrain.shape[0]  # number of points
	D = Xtrain.shape[1]  # number of input dimensions
	L = Ytrain.shape[1]  # number of latent GPs
	P = Ytrain.shape[1]  # number of observations = output dimensions
	# M = 20  # number of inducing points
	M_per_dim = 3

	X = tf.cast(Xtrain,dtype=tf.float64)
	Y = tf.cast(Ytrain,dtype=tf.float64)
	data = X, Y


	Zinit = CommonUtils.create_Ndim_grid(xmin=-5.0,xmax=+5.0,Ndiv=M_per_dim,dim=D) # [Ndiv**dim_in,dim_in]
	Zinit = tf.cast(Zinit,dtype=tf.float64)
	Zinit = Zinit.numpy()
	M = Zinit.shape[0]

	def optimize_model_with_scipy(model):
		optimizer = gpf.optimizers.Scipy()
		optimizer.minimize(
			model.training_loss_closure(data),
			variables=model.trainable_variables,
			method="l-bfgs-b",
			options={"disp": 50, "maxiter": MAXITER, "gtol": 1e-16, "ftol": 1e-16},
		)

	which_kernel = "matern"
	# which_kernel = "se"

	assert which_kernel in ["matern","se"]

	# Create list of kernels for each output
	if which_kernel == "se": kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) + gpf.kernels.Linear(variance=1.0) for _ in range(P)] # Adding a linear kernel
	# kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) for _ in range(P)]
	if which_kernel == "matern": kern_list = [gpf.kernels.Matern52(variance=1.0,lengthscales=0.1*np.ones(D)) + gpf.kernels.Linear(variance=1.0) for _ in range(P)] # Adding a linear kernel

	
	# Create multi-output kernel from kernel list:
	use_coregionalization = True
	if use_coregionalization:
		kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(P, L))  # Notice that we initialise the mixing matrix W
	else:
		kernel = gpf.kernels.SeparateIndependent(kern_list)

	
	# initialization of inducing input locations, one set of locations per output
	Zs = [Zinit.copy() for _ in range(P)]
	
	# initialize as list inducing inducing variables
	iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
	
	# create multi-output inducing variables from iv_list
	iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)

	# create SVGP model as usual and optimize
	model_gpflow = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(variance=0.5), inducing_variable=iv, num_latent_gps=P)

	# print_summary(model_gpflow)

	# pdb.set_trace()
	
	# if not using_hybridrobotics: MAXITER = 1
	optimize_model_with_scipy(model_gpflow)


	# # MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(zu_vec)
	# zu_vec_np = tf.cast(zu_vec,dtype=tf.float64).numpy()
	# mu, var = model_gpflow.predict_f(Xnew=zu_vec_np, full_cov=False, full_output_cov=True)
	# # pdb.set_trace()

	# print_summary(model_gpflow)

	# plot_model(model_gpflow)
	# print_summary(model_gpflow.kernel)
	# print("m.kernel.kernel.kernels[0].lengthscales: ",str(m.kernel.kernel.kernels[0].lengthscales))

	# pdb.set_trace()
	# html_string = data.show_batch().data
	# soup = BeautifulSoup(html_string)

	# plot_model(model_gpflow)

	# Save function to predict:
	model_gpflow.compiled_predict_f = tf.function(
		lambda Xnew: model_gpflow.predict_f(Xnew, full_cov=False, full_output_cov=True),
		# lambda xnew: model_gpflow.predict_f(xnew, full_cov=False, full_output_cov=True),
		input_signature=[tf.TensorSpec(shape=[None, D], dtype=tf.float64)],
	)


	# Save inducing points:
	model_gpflow.get_induced_pointsZ_list = tf.function(
		lambda: tf.concat([ind_var.Z for ind_var in model_gpflow.inducing_variable.inducing_variable_list],axis=0),
		input_signature=[],
	)
	# loaded_model.inducing_variable.inducing_variable_list[0].Z.numpy()


	# Save model:
	path2save = "{0:s}/{1:s}/gpssm_trained_model_gpflow_{2:s}".format(path2project,path2folder,name_file_date)
	logger.info("Saving gpflow model at {0:s} ...".format(path2save))
	tf.saved_model.save(model_gpflow, path2save)
	logger.info("Done!")
	
	path2save_others = "{0:s}/{1:s}/gpssm_trained_model_gpflow_{2:s}_relevant_data.pickle".format(path2project,path2folder,name_file_date)
	data2save = dict(Xtrain=Xtrain,
					Ytrain=Ytrain,
					Xtest=Xtest,
					Ytest=Ytest,
					dim_in=dim_in,
					dim_out=dim_out,
					Nsteps=Nsteps,
					path2data=path2data)

	file = open(path2save_others, 'wb')
	logger.info("Saving model data at {0:s} ...".format(path2save_others))
	pickle.dump(data2save,file)
	file.close()
	logger.info("Done!")


	path2log_file = "{0:s}/{1:s}/gpssm_trained_model_gpflow_log_file_{2:s}.txt".format(path2project,path2folder,name_file_date)
	logger.info("Writing ratio to log file at {0:s} ...".format(path2log_file))
	file = open(path2log_file, 'w')
	file.write("ratio: {0:2.2f}\n".format(ratio))
	file.write("MAXITER: {0:d}\n".format(MAXITER))
	file.write("which_kernel: {0:s}\n".format(which_kernel))
	file.close()
	logger.info("Done!")

	return name_file_date


def load_MOrrtp_model(cfg,path2project,file_name):

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

	# Delta predictions (on the full dataset):
	MO_mean_test, MO_std_test = rrtp_MO.predict_at_locations(Xtest)


	return MO_mean_test, MO_std_test, Ytest.numpy(), Xtest, Ytrain, Xtrain


def load_gpssm(path2project,file_name):

	# Load gpflow model:
	path2load_full = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
	logger.info("Loading gpflow model from {0:s} ...".format(path2load_full))
	loaded_model = tf.saved_model.load(path2load_full)
	logger.info("Done!")

	# Load other quantities:
	path2load_full = "{0:s}/{1:s}/{2:s}_relevant_data.pickle".format(path2project,path2folder,file_name)
	logger.info("Loading data from {0:s} ...".format(path2load_full))
	file = open(path2load_full, 'rb')
	data_dict = pickle.load(file)
	file.close()
	logger.info("Done!")

	Xtest = data_dict["Xtest"]
	Ytest = data_dict["Ytest"]
	Ytrain = data_dict["Ytrain"]
	Xtrain = data_dict["Xtrain"]

	print_summary(loaded_model)

	Zinduced = loaded_model.get_induced_pointsZ_list()
	print(Zinduced)

	MO_mean_test, MO_cov_full_test = loaded_model.compiled_predict_f(tf.cast(Xtest,dtype=tf.float64))
	# MO_std_test: [Npoints,dim_out,dim_out]

	# Extract diagonal:
	MO_std_test = np.zeros((MO_cov_full_test.shape[0],MO_cov_full_test.shape[1]))
	for ii in range(MO_std_test.shape[0]):
		MO_std_test[ii,:] = np.sqrt(np.diag(MO_cov_full_test[ii,...]))

	return MO_mean_test, MO_std_test, Ytest.numpy(), Xtest, Ytrain, Xtrain

def compute_model_error_for_selected_model(cfg,dict_all,which_model,which_ratio,plot_data_analysis=True):

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 
	
	file_name = dict_all[which_model][which_ratio]

	if which_model == "MOrrtp":
		MO_mean_test, MO_std_test, Ytest, Xtest, Ytrain, Xtrain = load_MOrrtp_model(cfg,path2project,file_name)
	elif which_model == "gpssm_se" or which_model == "gpssm_matern":
		MO_mean_test, MO_std_test, Ytest, Xtest, Ytrain, Xtrain = load_gpssm(path2project,file_name)

	dim_out = Ytest.shape[1]
	dim_out = Ytest.shape[1]

	# Compute RMSE and log-evidence:
	log_evidence_mat = np.zeros((Ytest.shape[0],dim_out))
	mse_mat = np.zeros((Ytest.shape[0],dim_out))
	for dd in range(dim_out):
		log_evidence_mat[:,dd] = scipy.stats.norm.logpdf(x=Ytest[:,dd],loc=MO_mean_test[:,dd],scale=MO_std_test[:,dd])
		mse_mat[:,dd] = (Ytest[:,dd] - MO_mean_test[:,dd])**2

	if plot_data_analysis:

		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,3,figsize=(16,14),sharex=False,sharey=False)
		hdl_fig.suptitle(r"State transition - Reconstructed; $\Delta x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)

		assert using_deltas == True

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(Ytest[:,jj])
			delta_fx_next_sorted = Ytest[ind_xt_sorted,jj]
			delta_MO_mean_test_sorted = MO_mean_test.numpy()[ind_xt_sorted,jj]

			hdl_splots_next_state[jj,0].plot(delta_fx_next_sorted,linestyle="-",color="crimson",alpha=0.3,lw=3.0,label="Test data")
			hdl_splots_next_state[jj,0].plot(delta_MO_mean_test_sorted,linestyle="-",color="navy",alpha=0.7,lw=1.0,label="Reconstructed")

		hdl_splots_next_state[0,0].set_ylabel(r"$\Delta f_1(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state[1,0].set_ylabel(r"$\Delta f_2(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state[2,0].set_ylabel(r"$\Delta f_3(x_t)$",fontsize=fontsize_labels)

		# hdl_splots_next_state[0,0].set_xlabel(r"$x_{t,1}$",fontsize=fontsize_labels)
		# hdl_splots_next_state[1,0].set_xlabel(r"$x_{t,2}$",fontsize=fontsize_labels)
		# hdl_splots_next_state[2,0].set_xlabel(r"$x_{t,3}$",fontsize=fontsize_labels)

		# hdl_splots_next_state[0,0].set_title("Reconstructed dynamics",fontsize=fontsize_labels)
		# hdl_splots_next_state[0,1].set_title("True dynamics",fontsize=fontsize_labels)
		# hdl_splots_next_state[-1,0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		# hdl_splots_next_state[-1,1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)

		lgnd = hdl_splots_next_state[-1,0].legend(loc="best",fontsize=fontsize_labels)
		lgnd.legendHandles[0]._legmarker.set_markersize(20)
		lgnd.legendHandles[1]._legmarker.set_markersize(20)


		# negative log-evidence:
		hdl_splots_next_state[0,1].set_title("Negative log-evidence",fontsize=fontsize_labels)
		for dd in range(dim_out):
			hdl_splots_next_state[dd,1].plot(-log_evidence_mat[:,dd])


		# mean squared error:
		hdl_splots_next_state[0,2].set_title("MSE",fontsize=fontsize_labels)
		for dd in range(dim_out):
			hdl_splots_next_state[dd,2].plot(mse_mat[:,dd])


		hdl_fig_data, hdl_splots_data = plt.subplots(5,1,figsize=(12,8),sharex=True)
		hdl_fig_data.suptitle("Test trajectory for predictions",fontsize=fontsize_labels)
		hdl_splots_data[0].set_title("Inputs",fontsize=fontsize_labels)
		hdl_splots_data[0].plot(Xtest[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[1].plot(Xtest[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[2].plot(Xtest[:,2],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[3].plot(Xtest[:,3],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[4].plot(Xtest[:,4],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)


		hdl_fig_data, hdl_splots_data = plt.subplots(5,2,figsize=(12,8),sharex=True)
		hdl_fig_data.suptitle("Training data",fontsize=fontsize_labels)
		hdl_splots_data[0,0].set_title("Inputs")
		hdl_splots_data[0,0].plot(Xtrain[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[1,0].plot(Xtrain[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[2,0].plot(Xtrain[:,2],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[3,0].plot(Xtrain[:,3],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[4,0].plot(Xtrain[:,4],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)


		hdl_splots_data[0,1].set_title("Outputs",fontsize=fontsize_labels)
		for jj in range(dim_out):
			hdl_splots_data[jj,1].plot(Ytest[:,jj],lw=1,alpha=0.5,color="crimson")
			hdl_splots_data[jj,1].plot(MO_mean_test[:,jj],lw=1,alpha=0.5,color="navy")
			hdl_splots_data[jj,1].plot(MO_mean_test[:,jj] - 2.*MO_std_test[:,jj],lw=1,color="navy",alpha=0.5)
			hdl_splots_data[jj,1].plot(MO_mean_test[:,jj] + 2.*MO_std_test[:,jj],lw=1,color="navy",alpha=0.5)

		plt.show(block=False)


	log_evidence_tot = np.mean(-log_evidence_mat)
	mse_tot = np.mean(mse_mat)

	return log_evidence_tot, mse_tot


def get_dictionary_log_dubins_car():

	# """
	# All experiments log
	# """
	# # << Our model >>
	# file_name = "reconstruction_data_2023_03_26_22_48_31.pickle" # Ratio: 0.25 | Nepochs: 5000
	# file_name = "reconstruction_data_2023_03_26_22_41_28.pickle" # Ratio: 1.0 | Nepochs: 5000
	# file_name = "reconstruction_data_2023_03_27_14_51_36.pickle" # Ratio: 0.25 | Nepochs: 5000 | dbg
	# file_name = "reconstruction_data_2023_03_27_14_56_21.pickle" # Ratio: 0.25 | Nepochs: 5000 | omega_lim = 5.0 | on hybridrob

	# # << GPSSM >>
	# file_name = "gpssm_trained_model_gpflow_2023_03_27_14_03_22" # Ratio 1.0 | dbg
	# file_name = "gpssm_trained_model_gpflow_2023_03_27_15_07_51" # Ratio 0.25 | on hybridrob
	# dict_gpssm_standard = dict(	p25="gpssm_trained_model_gpflow_2023_03_27_15_29_39", # on hybridrob, 2000 iters
	# 							p50="gpssm_trained_model_gpflow_2023_03_27_15_31_25", # on hybridrob, 2000 iters
	# 							p75="gpssm_trained_model_gpflow_2023_03_27_15_34_09", # on hybridrob, 2000 iters
	# 							p100="gpssm_trained_model_gpflow_2023_03_27_15_37_49") # on hybridrob, 2000 iters


	# Selected dictionary:
	# dict_MOrrtp = dict(p25="reconstruction_data_2023_03_27_14_56_21.pickle",p100="reconstruction_data_2023_03_26_22_48_31.pickle")
	dict_MOrrtp = dict(	p25="reconstruction_data_2023_03_27_16_01_20.pickle", # trained for 10000 epochs, works best with noise: value_init: 0.1
						p50="reconstruction_data_2023_03_27_16_02_54.pickle", # trained for 10000 epochs, works best with noise: value_init: 0.1
						p75="reconstruction_data_2023_03_27_16_07_04.pickle", # trained for 10000 epochs, works best with noise: value_init: 0.1
						p100="reconstruction_data_2023_03_27_16_12_23.pickle") # trained for 10000 epochs, works best with noise: value_init: 0.1



	dict_gpssm_standard_SE = dict(	p25="gpssm_trained_model_gpflow_2023_03_27_15_29_39", # SE, 2000 iters
								p50="gpssm_trained_model_gpflow_2023_03_27_15_31_25", # SE, 2000 iters
								p75="gpssm_trained_model_gpflow_2023_03_27_15_34_09", # SE, 2000 iters
								p100="gpssm_trained_model_gpflow_2023_03_27_15_37_49") # SE, 2000 iters


	# dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_27_17_26_55", # Matern52, 2000 iters
	# 							p50="gpssm_trained_model_gpflow_2023_03_27_17_28_49", # Matern52, 2000 iters
	# 							p75="gpssm_trained_model_gpflow_2023_03_27_17_31_45", # Matern52, 2000 iters
	# 							p100="gpssm_trained_model_gpflow_2023_03_27_17_35_40") # Matern52, 2000 iters

	dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_27_18_19_33", # Matern52, 500 iters
								p50="gpssm_trained_model_gpflow_2023_03_27_18_20_08", # Matern52, 500 iters
								p75="gpssm_trained_model_gpflow_2023_03_27_18_20_56", # Matern52, 500 iters
								p100="gpssm_trained_model_gpflow_2023_03_27_18_21_57") # Matern52, 500 iters

	# dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_27_19_50_51", # Matern52, 1000 iters
	# 							p50="gpssm_trained_model_gpflow_2023_03_27_19_51_53", # Matern52, 1000 iters
	# 							p75="gpssm_trained_model_gpflow_2023_03_27_19_53_24", # Matern52, 1000 iters
	# 							p100="gpssm_trained_model_gpflow_2023_03_27_19_55_23") # Matern52, 1000 iters


	dict_all = dict(MOrrtp=dict_MOrrtp,gpssm_se=dict_gpssm_standard_SE,gpssm_matern=dict_gpssm_standard_matern)

	return dict_all


def get_dictionary_log_quadruped():

	dict_all_list = []


	# Batch 1:
	dict_MOrrtp = dict(	p25="reconstruction_data_2023_03_29_03_47_07.pickle",
								p50="reconstruction_data_2023_03_29_03_48_49.pickle",
								p75="reconstruction_data_2023_03_29_03_50_30.pickle",
								p100="reconstruction_data_2023_03_29_03_52_10.pickle")

	dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_29_14_00_47",
	p50="gpssm_trained_model_gpflow_2023_03_29_14_01_48",
	p75="gpssm_trained_model_gpflow_2023_03_29_14_03_17",
	p100="gpssm_trained_model_gpflow_2023_03_29_14_05_20")


	dict_gpssm_standard_SE = None
	dict_all_list += [dict(MOrrtp=dict_MOrrtp,gpssm_se=dict_gpssm_standard_SE,gpssm_matern=dict_gpssm_standard_matern)]


	# Batch 2:
	dict_MOrrtp = dict(	p25="reconstruction_data_2023_03_29_03_53_53.pickle",
								p50="reconstruction_data_2023_03_29_03_55_33.pickle",
								p75="reconstruction_data_2023_03_29_03_57_13.pickle",
								p100="reconstruction_data_2023_03_29_03_58_54.pickle")
	
	dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_29_14_07_53",
	p50="gpssm_trained_model_gpflow_2023_03_29_14_08_49",
	p75="gpssm_trained_model_gpflow_2023_03_29_14_10_18",
	p100="gpssm_trained_model_gpflow_2023_03_29_14_12_15")


	dict_gpssm_standard_SE = None
	dict_all_list += [dict(MOrrtp=dict_MOrrtp,gpssm_se=dict_gpssm_standard_SE,gpssm_matern=dict_gpssm_standard_matern)]


	# Batch 3:
	dict_MOrrtp = dict(	p25="reconstruction_data_2023_03_29_04_00_37.pickle",
								p50="reconstruction_data_2023_03_29_04_02_16.pickle",
								p75="reconstruction_data_2023_03_29_04_03_57.pickle",
								p100="reconstruction_data_2023_03_29_04_05_37.pickle")

	dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_29_14_14_43",
	p50="gpssm_trained_model_gpflow_2023_03_29_14_15_42",
	p75="gpssm_trained_model_gpflow_2023_03_29_14_17_11",
	p100="gpssm_trained_model_gpflow_2023_03_29_14_19_12")


	dict_gpssm_standard_SE = None
	dict_all_list += [dict(MOrrtp=dict_MOrrtp,gpssm_se=dict_gpssm_standard_SE,gpssm_matern=dict_gpssm_standard_matern)]


	# Batch 4:
	dict_MOrrtp = dict(	p25="reconstruction_data_2023_03_29_04_07_20.pickle",
								p50="reconstruction_data_2023_03_29_04_08_59.pickle",
								p75="reconstruction_data_2023_03_29_04_10_40.pickle",
								p100="reconstruction_data_2023_03_29_04_12_20.pickle")

	dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_29_14_21_42",
	p50="gpssm_trained_model_gpflow_2023_03_29_14_22_39",
	p75="gpssm_trained_model_gpflow_2023_03_29_14_24_07",
	p100="gpssm_trained_model_gpflow_2023_03_29_14_26_09")


	dict_gpssm_standard_SE = None
	dict_all_list += [dict(MOrrtp=dict_MOrrtp,gpssm_se=dict_gpssm_standard_SE,gpssm_matern=dict_gpssm_standard_matern)]


	# Batch 5:
	dict_MOrrtp = dict(	p25="reconstruction_data_2023_03_29_04_14_03.pickle",
								p50="reconstruction_data_2023_03_29_04_15_43.pickle",
								p75="reconstruction_data_2023_03_29_04_17_22.pickle",
								p100="reconstruction_data_2023_03_29_04_19_02.pickle")
	dict_gpssm_standard_matern = dict(	p25="gpssm_trained_model_gpflow_2023_03_29_14_28_43",
	p50="gpssm_trained_model_gpflow_2023_03_29_14_29_40",
	p75="gpssm_trained_model_gpflow_2023_03_29_14_31_08",
	p100="gpssm_trained_model_gpflow_2023_03_29_14_33_08")


	dict_gpssm_standard_SE = None
	dict_all_list += [dict(MOrrtp=dict_MOrrtp,gpssm_se=dict_gpssm_standard_SE,gpssm_matern=dict_gpssm_standard_matern)]


	return dict_all_list


def get_log_evidence_evolution(cfg,which_model,ratio_list,ratio_names_list,plotting=True):

	# dict_all = get_dictionary_log_dubins_car()
	dict_all_list = get_dictionary_log_quadruped()
	Nbatches = len(dict_all_list)
	log_evidence_tot_vec = np.zeros((Nbatches,len(ratio_list)))
	mse_tot_vec = np.zeros((Nbatches,len(ratio_list)))

	for bb in range(Nbatches):

		for tt in range(len(ratio_list)):
			log_evidence_tot, mse_tot = compute_model_error_for_selected_model(cfg,dict_all_list[bb],which_model=which_model,which_ratio=ratio_names_list[tt],plot_data_analysis=plotting)
			logger.info("log_evidence_tot: {0:f}".format(log_evidence_tot))
			logger.info("mse_tot: {0:f}".format(mse_tot))

			log_evidence_tot_vec[bb,tt] = log_evidence_tot
			mse_tot_vec[bb,tt] = mse_tot

		logger.info("log_evidence_tot_vec: {0:s}".format(str(log_evidence_tot_vec[bb,:])))
		logger.info("mse_tot_vec: {0:s}".format(str(mse_tot_vec[bb,:])))
		
		if plotting: plt.show(block=True)



	return log_evidence_tot_vec, mse_tot_vec

@hydra.main(config_path="./config",config_name="config")
def training_for_multiple_ratios(cfg):

	ratio_list = [0.25,0.5,0.75,1.0]
	ratio_names_list = ["p25","p50","p75","p100"]

	# which_model = "MOrrtp"
	which_model = "gpssm"

	# Training models:
	name_file_date = []
	for ratio in ratio_list:
		if which_model == "gpssm": name_file_date += [train_gpssm(cfg,ratio=ratio)]
		if which_model == "MOrrtp": name_file_date += [train_MOrrtp_by_reconstructing(cfg,ratio=ratio)]

	logger.info("name_file_date: {0:s}".format(str(name_file_date)))
	logger.info("ratio_list: {0:s}".format(str(ratio_list)))



@hydra.main(config_path="./config",config_name="config")
def statistical_comparison(cfg):

	which_model_list = ["gpssm_se","gpssm_matern","MOrrtp"]
	which_model_list_legend = ["GPSSM - SE kernel", "GPSSM - Matern kernel", "rrGPSSM (ours)"]

	ratio_list = [0.25,0.5,0.75,1.0]
	ratio_names_list = ["p25","p50","p75","p100"]

	# DBG:
	which_model = "gpssm_matern"
	log_evidence_tot_vec, mse_tot_vec = get_log_evidence_evolution(cfg,which_model,ratio_list,ratio_names_list,plotting=False)

	print(log_evidence_tot_vec)
	print(mse_tot_vec)

	pdb.set_trace()

	log_evidence_per_model_list = []
	mse_per_model_list = []
	marker_list = ["s","v","*"]
	for model in which_model_list:

		log_evidence, mse = get_log_evidence_evolution(cfg,which_model=model,ratio_list=ratio_list,ratio_names_list=ratio_names_list,plotting=False)

		# log_evidence = np.random.rand(4)
		# mse = np.random.rand(4)

		log_evidence_per_model_list += [log_evidence]
		mse_per_model_list += [mse]


	hdl_fig_data, hdl_splots_data = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_fig_data.suptitle("Data efficiency assessment",fontsize=fontsize_labels)
	ratio_list_plot = (np.array(ratio_list)*100).astype(dtype=int)
	for mm in range(len(which_model_list)):
		hdl_splots_data.plot(ratio_list_plot,log_evidence_per_model_list[mm],lw=1,alpha=0.7,color="darkgreen",marker=marker_list[mm],markersize=5,label=which_model_list_legend[mm])
		hdl_splots_data.set_xticks([])
		hdl_splots_data.set_ylabel(r"$-\log p(\Delta x_{t+1})$",fontsize=fontsize_labels)

		# hdl_splots_data[1].plot(ratio_list_plot,mse_per_model_list[mm],lw=1,alpha=0.7,color="darkgreen",marker=marker_list[mm],markersize=5)
		# hdl_splots_data[1].set_xticks([])
		# hdl_splots_data[1].set_ylabel(r"RMSE",fontsize=fontsize_labels)


	hdl_splots_data.set_xticks(ratio_list_plot)
	hdl_splots_data.set_xlabel(r"\% of training data",fontsize=fontsize_labels)


	print(log_evidence_per_model_list)
	# 312.37,  17.15,  69.26 , -1.57
	# 150.92,  941.16, 2529.22, 624.69
	# 0.80, -1.71, -1.94, -2.01

	plt.show(block=True)

	# hdl_splots_data[-1].set_xticks([])

if __name__ == "__main__":

	my_seed = 1
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)


	Nrepeats = 5
	name_file_date_list = []
	for _ in range(Nrepeats):
		training_for_multiple_ratios()

	# statistical_comparison()

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_efficiency_test_with_dubinscar/"*2023_03_27_19_55_23*" ./data_efficiency_test_with_dubinscar/

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_efficiency_test_with_quadruped_data_03_25_2023/"*" ./data_efficiency_test_with_quadruped_data_03_25_2023/

	# python test_data_efficiency.py gpmodel.using_hybridrobotics=False


