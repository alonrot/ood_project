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


def train_MOrrtp_by_reconstructing(cfg,ratio=1.0):

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


def train_gpssm(cfg,ratio=1.0):

	savefig = True

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	# Load data:
	Xtrain, Ytrain, Xtest, Ytest, dim_in, dim_out, Nsteps, path2data = load_data_dubins_car(path2project,ratio) # Dubins car

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


	# Create list of kernels for each output
	# kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) + gpf.kernels.Linear(variance=1.0) for _ in range(P)] # Adding a linear kernel
	kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) for _ in range(P)]
	
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

def load_MOrrtp_model(path2project,file_name):

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



def compute_model_error_for_selected_model(cfg,dict_all,which_model="MOrrtp",which_ratio="p25"):

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 
	
	file_name = dict_all[which_model][which_ratio]

	if which_model == "MOrrtp":
		MO_mean_test, MO_std_test, Ytest, Xtest, Ytrain, Xtrain = load_MOrrtp_model(path2project,file_name)
	elif which_model == "gpssm":
		MO_mean_test, MO_std_test, Ytest, Xtest, Ytrain, Xtrain = load_gpssm(path2project,file_name)

	dim_out = Ytest.shape[1]
	dim_out = Ytest.shape[1]

	# Compute RMSE and log-evidence:
	log_evidence_mat = np.zeros((Ytest.shape[0],dim_out))
	mse_mat = np.zeros((Ytest.shape[0],dim_out))
	for dd in range(dim_out):
		log_evidence_mat[:,dd] = scipy.stats.norm.logpdf(x=Ytest[:,dd],loc=MO_mean_test[:,dd],scale=MO_std_test[:,dd])
		mse_mat[:,dd] = (Ytest[:,dd] - MO_mean_test[:,dd])**2

	analyze_data = True
	if analyze_data:

		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,3,figsize=(16,14),sharex=False,sharey=False)
		hdl_fig.suptitle(r"State transition - Reconstructed; $\Delta x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)

		assert using_deltas == True

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(Ytest[:,jj])
			delta_fx_next_sorted = Ytest[ind_xt_sorted,jj]
			delta_MO_mean_test_sorted = MO_mean_test.numpy()[ind_xt_sorted,jj]

			hdl_splots_next_state[jj,0].plot(delta_fx_next_sorted,linestyle="-",color="crimson",alpha=0.3,lw=3.0,label="Training data")
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

		plt.show(block=True)


	log_evidence_tot = np.mean(-log_evidence_mat)
	mse_tot = np.mean(mse_mat)

	return log_evidence_tot, mse_tot


def get_dictionary_log():

	# """
	# All experiments log
	# """
	# # << Our model >>
	# # file_name = "reconstruction_data_2023_03_26_22_48_31.pickle" # Ratio: 0.25 | Nepochs: 5000
	# file_name = "reconstruction_data_2023_03_26_22_41_28.pickle" # Ratio: 1.0 | Nepochs: 5000

	# # << GPSSM >>
	file_name = "gpssm_trained_model_gpflow_2023_03_27_14_03_22" # Ratio 1.0 | dbg


	# Selected dictionary:
	dict_MOrrtp = dict(p25="reconstruction_data_2023_03_26_22_48_31.pickle",p100="reconstruction_data_2023_03_26_22_48_31.pickle")
	dict_gpssm_standard = dict(p25="gpssm_trained_model_gpflow_2023_03_27_14_20_01",p100="gpssm_trained_model_gpflow_2023_03_27_14_03_22")
	dict_all = dict(MOrrtp=dict_MOrrtp,gpssm=dict_gpssm_standard)

	return dict_all


@hydra.main(config_path="./config",config_name="config")
def main(cfg):

	# Training models:
	# train_MOrrtp_by_reconstructing(cfg,ratio=0.25)
	train_gpssm(cfg,ratio=0.25)


	# # Assessing model performance:
	# dict_all = get_dictionary_log()
	# log_evidence_tot, mse_tot = compute_model_error_for_selected_model(cfg,dict_all,which_model="gpssm",which_ratio="p25")
	# logger.info("log_evidence_tot: {0:f}".format(log_evidence_tot))
	# logger.info("mse_tot: {0:f}".format(mse_tot))



if __name__ == "__main__":

	my_seed = 1
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)


	main()


	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_efficiency_test_with_dubinscar/reconstruction_data_2023_03_26_22_48_31.pickle ./data_efficiency_test_with_dubinscar/


	# python test_data_efficiency.py gpmodel.using_hybridrobotics=False




