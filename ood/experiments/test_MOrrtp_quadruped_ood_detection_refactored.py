import tensorflow as tf
import gpflow
import pdb
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
from matplotlib.collections import LineCollection
import numpy as np
import scipy
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, QuadrupedSpectralDensity
from lqrker.models import MultiObjectiveReducedRankProcess
from scipy import stats
import hydra
from omegaconf import OmegaConf
import pickle
import control
from lqrker.utils.common import CommonUtils
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)
from min_jerk_gen import min_jerk

from test_dubin_car import get_sequence_of_feedback_gains_finite_horizon_LQR, rollout_with_finitie_horizon_LQR, generate_trajectories, generate_reference_trajectory

from predictions_interface import Predictions


# GP flow:
import gpflow as gpf
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("github")
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 30
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

using_deltas = True
assert using_deltas == True

# path2folder = "data_quadruped_experiments_03_25_2023"
path2folder = "data_quadruped_experiments_03_29_2023"


"""
DEPRECATED
"""

# def fix_pickle_datafile(cfg,path2project,path2folder):

# 	"""
# 	NOTE: This piece of code needed to be called only once, to fix the pickle file.
# 	It genrated the file /Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/from_hybridrob/reconstruction_data_2023_03_27_01_12_35.pickle
# 	"""
# 	raise NotImplementedError("Only needed to be called once to fix a pickle file")

# 	file_name = "reconstruction_data_2023_03_26_21_55_08.pickle" # Trained model on hybridrob for 50000 iters; data subsampled at 10 Hz
# 	path2load_full = "{0:s}/{1:s}/from_hybridrob/{2:s}".format(path2project,path2folder,file_name)
# 	file = open(path2load_full, 'rb')
# 	data_dict = pickle.load(file)
# 	file.close()

# 	# omegas_trainedNN = tf.convert_to_tensor(data_dict["omegas_trainedNN"],dtype=tf.float32)
# 	# Sw_omegas_trainedNN = tf.convert_to_tensor(data_dict["Sw_omegas_trainedNN"],dtype=tf.float32)
# 	# varphi_omegas_trainedNN = tf.convert_to_tensor(data_dict["varphi_omegas_trainedNN"],dtype=tf.float32)
# 	# delta_omegas_trainedNN = tf.convert_to_tensor(data_dict["delta_omegas_trainedNN"],dtype=tf.float32)
# 	# delta_statespace_trainedNN = tf.convert_to_tensor(data_dict["delta_statespace_trainedNN"],dtype=tf.float32)
	
# 	path2data = "{0:s}/data_quadruped_experiments_03_25_2023/joined_go1trajs_trimmed_2023_03_25.pickle".format(path2project)
# 	logger.info("Loading {0:s} ...".format(path2data))
# 	file = open(path2data, 'rb')
# 	data_dict4spectral = pickle.load(file)
# 	file.close()

# 	Xtrain = data_dict4spectral["Xtrain"]
# 	Ytrain = data_dict4spectral["Ytrain"]
# 	state_and_control_full_list = data_dict4spectral["state_and_control_full_list"]
# 	state_next_full_list = data_dict4spectral["state_next_full_list"]

# 	dim_x = Ytrain.shape[1]
# 	dim_u = Xtrain.shape[1] - dim_x
# 	Nsteps = Xtrain.shape[0]
# 	Ntrajs = None

# 	dim_in = dim_x + dim_u
# 	dim_out = dim_x

# 	if using_deltas:
# 		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_x]
# 		Ytrain = tf.identity(Ytrain_deltas)

# 	Xtrain = tf.cast(Xtrain,dtype=tf.float32)
# 	Ytrain = tf.cast(Ytrain,dtype=tf.float32)

# 	# Spectral density:
# 	spectral_density_list = [None]*dim_out
# 	for jj in range(dim_out):
# 		spectral_density_list[jj] = QuadrupedSpectralDensity(cfg=cfg.spectral_density.quadruped,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
# 		spectral_density_list[jj].update_Wsamples_from_file(path2data=path2load_full,ind_out=jj)


# 	data_dict.update(spectral_density_list=spectral_density_list,
# 					omega_lim=5.0,
# 					Nsamples_omega=1500,
# 					Xtrain=Xtrain,
# 					Ytrain=Ytrain,
# 					state_and_control_full_list=state_and_control_full_list,
# 					state_next_full_list=state_next_full_list,
# 					path2data=path2data)


# 	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# 	file_name = "reconstruction_data_{0:s}.pickle".format(name_file_date) # Trained model on hybridrob for 50000 iters; data subsampled at 10 Hz
# 	path2load_full = "{0:s}/{1:s}/from_hybridrob/{2:s}".format(path2project,path2folder,file_name)
# 	file = open(path2load_full, 'wb')
# 	logger.info("Saving data at {0:s} ...".format(path2load_full))
# 	pickle.dump(data_dict,file)
# 	file.close()
# 	logger.info("Done!")

# 	pdb.set_trace()

	
def load_quadruped_experiments_03_29_2023(path2project):

	path2data = "{0:s}/data_quadruped_experiments_03_29_2023/joined_go1trajs_trimmed_2023_03_29_circle_walking.pickle".format(path2project)
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()

	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	state_and_control_full_list = data_dict["state_and_control_full_list"]
	state_next_full_list = data_dict["state_next_full_list"]
	dim_x = Ytrain.shape[1]
	dim_u = Xtrain.shape[1] - dim_x
	Nsteps = Xtrain.shape[0]
	Ntrajs = None

	dim_in = dim_x + dim_u
	dim_out = dim_x

	if using_deltas:
		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_x]
		Ytrain = tf.identity(Ytrain_deltas)

	Xtrain = tf.cast(Xtrain,dtype=tf.float32)
	Ytrain = tf.cast(Ytrain,dtype=tf.float32)

	return Xtrain, Ytrain, dim_in, dim_out, Nsteps, Ntrajs, path2data, state_and_control_full_list, state_next_full_list


def train_gpssm(cfg):

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	Xtrain, Ytrain, dim_in, dim_out, Nsteps, Ntrajs, path2data, state_and_control_full_list, state_next_full_list = load_quadruped_experiments_03_29_2023(path2project)

	# Based on: https://gpflow.github.io/GPflow/develop/notebooks/advanced/multioutput.html#
	MAXITER = reduce_in_tests(500)
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
	# which_kernel = "matern_nolin"
	# which_kernel = "se"

	assert which_kernel in ["matern","se","matern_nolin"]

	# Create list of kernels for each output
	if which_kernel == "se": kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) + gpf.kernels.Linear(variance=1.0) for _ in range(P)] # Adding a linear kernel
	if which_kernel == "matern_nolin": kern_list = [gpf.kernels.Matern52(variance=1.0,lengthscales=0.1*np.ones(D)) for _ in range(P)]
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
	
	# if not using_hybridrobotics: MAXITER = 1
	optimize_model_with_scipy(model_gpflow)

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
	path2save = "{0:s}/{1:s}/gpssm_gpflow_trained_on_quadruped_walking_circles_{2:s}".format(path2project,path2folder,name_file_date)
	logger.info("Saving gpflow model at {0:s} ...".format(path2save))
	tf.saved_model.save(model_gpflow, path2save)
	logger.info("Done!")
	
	path2save_others = "{0:s}/{1:s}/gpssm_gpflow_trained_on_quadruped_walking_circles_{2:s}_relevant_data.pickle".format(path2project,path2folder,name_file_date)
	data2save = dict(Xtrain=Xtrain,
					Ytrain=Ytrain,
					dim_in=dim_in,
					dim_out=dim_out,
					state_and_control_full_list=state_and_control_full_list,
					state_next_full_list=state_next_full_list)

	file = open(path2save_others, 'wb')
	logger.info("Saving model data at {0:s} ...".format(path2save_others))
	pickle.dump(data2save,file)
	file.close()
	logger.info("Done!")



def select_trajectory_from_path(path2project,path2folder,file_name,ind_which_traj):

	path2load_full = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
	file = open(path2load_full, 'rb')
	data_dict = pickle.load(file)
	file.close()

	Xtest = data_dict["Xtrain"] # The data is saved in the dictionary as "Xtrain" by default, but here we actually use it as test data
	Ytest = data_dict["Ytrain"] # The data is saved in the dictionary as "Ytrain" by default, but here we actually use it as test data

	dim_x = Ytest.shape[1]
	dim_u = Xtest.shape[1] - Ytest.shape[1]

	if using_deltas:
		Ytest_deltas = Ytest - Xtest[:,0:dim_x]
		Ytest = tf.identity(Ytest_deltas)

	Xtest = tf.cast(Xtest,dtype=tf.float32)
	Ytest = tf.cast(Ytest,dtype=tf.float32)

	state_and_control_full_list = data_dict["state_and_control_full_list"]
	state_next_full_list = data_dict["state_next_full_list"]

	z_vec_real = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,0:dim_x],dtype=tf.float32)
	u_vec_tf = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,dim_x::],dtype=tf.float32)
	zu_vec = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj],dtype=tf.float32)


	return z_vec_real, u_vec_tf, zu_vec, Xtest, Ytest



def load_model_ours(cfg,path2project):

	"""
	# Quadruped moving around the room to random waypoints:
	# file_name = "reconstruction_data_2023_03_26_21_55_08.pickle" # Trained model on hybridrob for 50000 iters per dim; data subsampled at 10 Hz
	file_name = "reconstruction_data_2023_03_27_01_23_40.pickle" # Trained model on hybridrob for 50000 iters per dim; data subsampled at 10 Hz || Completed the missing fields using the above function fix_pickle_datafile()
	# file_name = "reconstruction_data_2023_03_27_01_23_40.pickle" # [same precision, more omegas, not worth it] Trained model on hybridrob with 2000 omegas. 10000 iters per dim; data subsampled at 10 Hz || Completed the missing fields using the above function fix_pickle_datafile()
	"""

	# Quadruped following a circle:
	# file_name = "reconstruction_data_2023_03_29_23_11_35.pickle" # Trained model on hybridrob for 10000 iters per dim; data subsampled at 10 Hz
	file_name = "reconstruction_data_2023_03_30_10_44_21.pickle" # Trained model on hybridrob for 100000 iters per dim; data subsampled at 10 Hz

	path2load_full = "{0:s}/{1:s}/from_hybridrob/{2:s}".format(path2project,path2folder,file_name)
	file = open(path2load_full, 'rb')
	data_dict = pickle.load(file)
	file.close()

	spectral_density_list = data_dict["spectral_density_list"]
	omega_lim = data_dict["omega_lim"]
	Nsamples_omega = data_dict["Nsamples_omega"]
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"] # This is already deltas when the file is file_name = "reconstruction_....pickle"
	state_and_control_full_list = data_dict["state_and_control_full_list"]
	state_next_full_list = data_dict["state_next_full_list"]

	logger.info("\n\n")
	logger.info(" * omega_lim: {0:f}".format(omega_lim))
	logger.info(" * Nsamples_omega: {0:d}".format(Nsamples_omega))

	dim_x = Ytrain.shape[1]
	dim_u = Xtrain.shape[1] - Ytrain.shape[1]

	# Initialize GP model:
	dim_in = dim_x + dim_u
	dim_out = dim_x
	logger.info(" * Initializing GP model ...")
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_in,cfg,spectral_density_list,Xtrain,Ytrain,using_deltas=using_deltas)

	return rrtp_MO, Xtrain, Ytrain, state_and_control_full_list, state_next_full_list


def load_model_gpssm(path2project):

	file_name = "gpssm_gpflow_trained_on_quadruped_walking_circles_2023_03_30_16_00_54"

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

	Ytrain = data_dict["Ytrain"]
	Xtrain = data_dict["Xtrain"]
	state_and_control_full_list = data_dict["state_and_control_full_list"]
	state_next_full_list = data_dict["state_next_full_list"]

	print_summary(loaded_model)

	Zinduced = loaded_model.get_induced_pointsZ_list()
	print(Zinduced)

	return loaded_model, Xtrain, Ytrain, state_and_control_full_list, state_next_full_list


def compute_predictions_gpflow(cfg):

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	my_seed = 78
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)	

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	loaded_model, Xtrain, Ytrain, state_and_control_full_list, state_next_full_list = load_model_gpssm(path2project)

	dim_out = Ytrain.shape[1]
	dim_x = Ytrain.shape[1]

	ind_which_traj = 0
	# use_same_data_the_model_was_trained_on = True
	use_same_data_the_model_was_trained_on = False
	if use_same_data_the_model_was_trained_on:

		# pdb.set_trace()

		z_vec_real = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,0:dim_x],dtype=tf.float32)
		u_vec_tf = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,dim_x::],dtype=tf.float32)
		zu_vec = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj],dtype=tf.float32)
		Xtest = Xtrain
		Ytest = Ytrain # already deltas when the file is file_name = "reconstruction_....pickle"

	else:
		path2folder_data_diff_env = "data_quadruped_experiments_03_29_2023"

		# Scenario 1: just walking
		# file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_walking.pickle"

		# Scenario 2: rope pulling
		# file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_rope.pickle"

		# Scenario 3: rocky terrain
		file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_rocky.pickle"

		# Scenario 4: poking
		# file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_poking.pickle"


		z_vec_real, u_vec_tf, zu_vec, Xtest, Ytest = select_trajectory_from_path(path2project=path2project,path2folder=path2folder_data_diff_env,file_name=file_name_data_diff_env,ind_which_traj=ind_which_traj)


	# analyze_data = True
	analyze_data = False
	if analyze_data:

		# # Delta predictions:
		# MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(zu_vec)
		# deltas_real = state_next_full_list[ind_which_traj] - state_and_control_full_list[ind_which_traj][:,0:dim_x]

		# Delta predictions (on the full dataset):
		MO_mean_pred, MO_cov_full_test = loaded_model.compiled_predict_f(tf.cast(Xtest,dtype=tf.float64))
		# MO_std_test: [Npoints,dim_out,dim_out]

		# Extract diagonal:
		MO_std_pred = np.zeros((MO_cov_full_test.shape[0],MO_cov_full_test.shape[1]))
		for ii in range(MO_std_pred.shape[0]):
			MO_std_pred[ii,:] = np.sqrt(np.diag(MO_cov_full_test[ii,...]))


		if tf.is_tensor(Ytest):
			deltas_real = Ytest.numpy()
		else:
			deltas_real = Ytest


		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,1,figsize=(12,8),sharex=False,sharey=False)
		hdl_fig.suptitle(r"Predicted state transition on the test data; $\Delta x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state = np.reshape(hdl_splots_next_state,(-1,1))

		assert using_deltas == True

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(deltas_real[:,jj])
			delta_fx_next_sorted = deltas_real[ind_xt_sorted,jj]
			delta_MO_mean_test_sorted = MO_mean_pred.numpy()[ind_xt_sorted,jj]

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

		
		hdl_fig_data, hdl_splots_data = plt.subplots(5,1,figsize=(12,8),sharex=True)
		hdl_fig_data.suptitle("Test trajectory for predictions",fontsize=fontsize_labels)
		hdl_splots_data[0].set_title("Inputs",fontsize=fontsize_labels)
		hdl_splots_data[0].plot(z_vec_real[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[1].plot(z_vec_real[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[2].plot(z_vec_real[:,2],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[3].plot(u_vec_tf[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[4].plot(u_vec_tf[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)


		hdl_fig_data, hdl_splots_data = plt.subplots(5,2,figsize=(12,8),sharex=True)
		hdl_fig_data.suptitle("Data",fontsize=fontsize_labels)
		hdl_splots_data[0,0].set_title("Training data")
		hdl_splots_data[0,0].plot(Xtrain[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[1,0].plot(Xtrain[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[2,0].plot(Xtrain[:,2],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[3,0].plot(Xtrain[:,3],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[4,0].plot(Xtrain[:,4],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)


		hdl_splots_data[0,1].set_title("Test data + predictions",fontsize=fontsize_labels)
		for jj in range(dim_out):
			hdl_splots_data[jj,1].plot(Ytrain[:,jj],lw=1,alpha=0.5,color="crimson")
			hdl_splots_data[jj,1].plot(MO_mean_pred[:,jj],lw=1,alpha=0.5,color="navy")
			hdl_splots_data[jj,1].plot(MO_mean_pred[:,jj] - 2.*MO_std_pred[:,jj],lw=1,color="navy",alpha=0.5)
			hdl_splots_data[jj,1].plot(MO_mean_pred[:,jj] + 2.*MO_std_pred[:,jj],lw=1,color="navy",alpha=0.5)
			# hdl_splots_data[jj,1].fill_between(MO_mean_pred[:,jj] - 2.*MO_std_pred[:,jj],MO_mean_pred[:,jj] + 2.*MO_std_pred[:,jj],color="navy",alpha=0.5)

		# hdl_splots_data[1,1].plot(Ytrain[:,1],lw=1,alpha=0.5,color="crimson")
		# hdl_splots_data[2,1].plot(Ytrain[:,2],lw=1,alpha=0.5,color="crimson")

		# hdl_splots_data[1,1].plot(MO_mean_pred[:,1],lw=1,alpha=0.3,color="navy")
		# hdl_splots_data[1,1].fill_between(MO_mean_pred[:,0] - 2.*MO_std_pred,MO_mean_pred[:,0] + 2.*MO_std_pred,color="navy",alpha=0.2)
		
		# hdl_splots_data[2,1].plot(MO_mean_pred[:,2],lw=1,alpha=0.3,color="navy")
		# hdl_splots_data[2,1].fill_between(MO_mean_pred[:,0] - 2.*MO_std_pred,MO_mean_pred[:,0] + 2.*MO_std_pred,color="navy",alpha=0.2)


		plt.show(block=True)



	if using_hybridrobotics:
		# Nhorizon_rec = 40
		Nhorizon_rec = 10
		# Nsteps_tot = z_vec_real.shape[0]-Nhorizon_rec
		# Nsteps_tot = z_vec_real.shape[0] // 2
		Nsteps_tot = z_vec_real.shape[0]
		Nsteps_tot = 600
		Nepochs = 200
		Nrollouts = 5
		Nchunks = 4
	else:

		Nhorizon_rec = 15
		# Nsteps_tot = 40
		# Nsteps_tot = z_vec_real.shape[0]
		Nsteps_tot = z_vec_real.shape[0] // 4
		Nsteps_tot = 20
		Nepochs = 200
		Nrollouts = 10
		Nchunks = 4


	assert Nsteps_tot > Nhorizon_rec



	# Receding horizon predictions:
	Nsteps_tot = min(z_vec_real.shape[0]-Nhorizon_rec,Nsteps_tot)
	loss_val_per_step = np.zeros(Nsteps_tot)
	x_traj_pred_all_vec = np.zeros((Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x))
	noise_mat = np.random.randn(Nrollouts,dim_x)
	loss_elbo_evidence_avg_vec = np.zeros(Nsteps_tot)
	loss_elbo_entropy_vec = np.zeros(Nsteps_tot)
	loss_prior_regularizer_vec = np.zeros(Nsteps_tot)
	for tt in range(Nsteps_tot):

		time_init = time.time()

		x_traj_real_applied = z_vec_real[tt:tt+Nhorizon_rec,:]
		x_traj_real_applied_tf = tf.reshape(x_traj_real_applied,(1,Nhorizon_rec,dim_x))
		u_applied_tf = u_vec_tf[tt:tt+Nhorizon_rec,:]
		logger.info("Prediction with horizon = {0:d}; tt: {1:d} / {2:d} | ".format(Nhorizon_rec,tt+1,Nsteps_tot))

		x0_tf = tf.cast(x_traj_real_applied_tf[0,0:1,:],dtype=tf.float64) # [Npoints,self.dim_in], with Npoints=1
		u_applied_tf = tf.cast(u_applied_tf,dtype=tf.float64) # [Npoints,self.dim_in], with Npoints=1

		for rr in range(Nrollouts):

			x_traj_pred_all_vec[tt,rr,0:1,:] = x0_tf.numpy()
			for ppp in range(Nhorizon_rec-1):

				zu_vec_tf = tf.convert_to_tensor(np.concatenate((x_traj_pred_all_vec[tt,rr,ppp:ppp+1,:],u_applied_tf.numpy()[ppp:ppp+1,:]),axis=1),dtype=tf.float64)
				MO_mean_pred, cov_full = loaded_model.compiled_predict_f(zu_vec_tf)

				cov_full_chol = tf.linalg.cholesky(cov_full)[0,...]

				ft_vec = MO_mean_pred + noise_mat[rr:rr+1,:]@tf.transpose(cov_full_chol) # [1,dim_x]
				if using_deltas:
					x_traj_pred_all_vec[tt,rr,ppp+1:ppp+2,:] = ft_vec + x_traj_pred_all_vec[tt,rr,ppp:ppp+1,:]
				else:
					x_traj_pred_all_vec[tt,rr,ppp+1:ppp+2,:] = ft_vec


				loss_elbo_entropy_vec[tt] += -0.5*np.sum(np.log(np.diag(cov_full[0,...])))

		# Compute losses:
		loss_elbo_entropy_vec[tt] = loss_elbo_entropy_vec[tt] / ((Nhorizon_rec-1)*Nrollouts*dim_out)
		loss_elbo_evidence_avg_vec[tt] = tf.reduce_mean((x_traj_pred_all_vec[tt,...] - tf.expand_dims(x_traj_real_applied,axis=0))**2) / 0.001

		time_elapsed = time.time() - time_init

		logger.info("time_elapsed: {0:2.2f}".format(time_elapsed))

	savedata = True
	if savedata:
		data2save = dict(x_traj_pred_all_vec=x_traj_pred_all_vec,u_vec_tf=u_vec_tf,z_vec_real=z_vec_real,loss_val_per_step=loss_val_per_step,Xtrain=Xtrain,Ytrain=Ytrain)
		data2save.update(	loss_elbo_evidence_avg_vec=loss_elbo_evidence_avg_vec,
							loss_elbo_entropy_vec=loss_elbo_entropy_vec,
							loss_prior_regularizer_vec=loss_prior_regularizer_vec)

		file_name = "predicted_trajs_{0:s}.pickle".format(name_file_date)
		path2save_receding_horizon = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
		logger.info("Saving at {0:s} ...".format(path2save_receding_horizon))
		file = open(path2save_receding_horizon, 'wb')
		pickle.dump(data2save,file)
		file.close()



def compute_predictions_our_model(cfg):

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	my_seed = 78
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)


	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	rrtp_MO, Xtrain, Ytrain, state_and_control_full_list, state_next_full_list = load_model_ours(cfg,path2project)

	dim_out = Ytrain.shape[1]

	ind_which_traj = 0
	# use_same_data_the_model_was_trained_on = True
	use_same_data_the_model_was_trained_on = False
	if use_same_data_the_model_was_trained_on:

		# pdb.set_trace()

		z_vec_real = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,0:dim_x],dtype=tf.float32)
		u_vec_tf = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,dim_x::],dtype=tf.float32)
		zu_vec = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj],dtype=tf.float32)
		Xtest = Xtrain
		Ytest = Ytrain # already deltas when the file is file_name = "reconstruction_....pickle"

	else:
		path2folder_data_diff_env = "data_quadruped_experiments_03_29_2023"

		# Scenario 1: just walking
		file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_walking.pickle"

		# Scenario 2: rope pulling
		# file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_rope.pickle"

		# Scenario 3: rocky terrain
		# file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_rocky.pickle"

		# Scenario 4: poking
		# file_name_data_diff_env = "joined_go1trajs_trimmed_2023_03_29_circle_poking.pickle"


		z_vec_real, u_vec_tf, zu_vec, Xtest, Ytest = select_trajectory_from_path(path2project=path2project,path2folder=path2folder_data_diff_env,file_name=file_name_data_diff_env,ind_which_traj=ind_which_traj)


	# analyze_data = True
	analyze_data = False
	if analyze_data:

		# # Delta predictions:
		# MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(zu_vec)
		# deltas_real = state_next_full_list[ind_which_traj] - state_and_control_full_list[ind_which_traj][:,0:dim_x]


		# Delta predictions (on the full dataset):
		MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(Xtest)
		if tf.is_tensor(Ytest):
			deltas_real = Ytest.numpy()
		else:
			deltas_real = Ytest


		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,1,figsize=(12,8),sharex=False,sharey=False)
		hdl_fig.suptitle(r"Predicted state transition on the test data; $\Delta x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)
		hdl_splots_next_state = np.reshape(hdl_splots_next_state,(-1,1))

		assert using_deltas == True

		for jj in range(dim_out):
			ind_xt_sorted = np.argsort(deltas_real[:,jj])
			delta_fx_next_sorted = deltas_real[ind_xt_sorted,jj]
			delta_MO_mean_test_sorted = MO_mean_pred.numpy()[ind_xt_sorted,jj]

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

		
		hdl_fig_data, hdl_splots_data = plt.subplots(5,1,figsize=(12,8),sharex=True)
		hdl_fig_data.suptitle("Test trajectory for predictions",fontsize=fontsize_labels)
		hdl_splots_data[0].set_title("Inputs",fontsize=fontsize_labels)
		hdl_splots_data[0].plot(z_vec_real[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[1].plot(z_vec_real[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[2].plot(z_vec_real[:,2],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[3].plot(u_vec_tf[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[4].plot(u_vec_tf[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)


		hdl_fig_data, hdl_splots_data = plt.subplots(5,2,figsize=(12,8),sharex=True)
		hdl_fig_data.suptitle("Data",fontsize=fontsize_labels)
		hdl_splots_data[0,0].set_title("Training data")
		hdl_splots_data[0,0].plot(Xtrain[:,0],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[1,0].plot(Xtrain[:,1],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[2,0].plot(Xtrain[:,2],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[3,0].plot(Xtrain[:,3],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)
		hdl_splots_data[4,0].plot(Xtrain[:,4],lw=1,alpha=0.3,color="navy",marker=".",markersize=2)


		hdl_splots_data[0,1].set_title("Test data + predictions",fontsize=fontsize_labels)
		for jj in range(dim_out):
			hdl_splots_data[jj,1].plot(Ytrain[:,jj],lw=1,alpha=0.5,color="crimson")
			hdl_splots_data[jj,1].plot(MO_mean_pred[:,jj],lw=1,alpha=0.5,color="navy")
			hdl_splots_data[jj,1].plot(MO_mean_pred[:,jj] - 2.*MO_std_pred[:,jj],lw=1,color="navy",alpha=0.5)
			hdl_splots_data[jj,1].plot(MO_mean_pred[:,jj] + 2.*MO_std_pred[:,jj],lw=1,color="navy",alpha=0.5)
			# hdl_splots_data[jj,1].fill_between(MO_mean_pred[:,jj] - 2.*MO_std_pred[:,jj],MO_mean_pred[:,jj] + 2.*MO_std_pred[:,jj],color="navy",alpha=0.5)

		# hdl_splots_data[1,1].plot(Ytrain[:,1],lw=1,alpha=0.5,color="crimson")
		# hdl_splots_data[2,1].plot(Ytrain[:,2],lw=1,alpha=0.5,color="crimson")

		# hdl_splots_data[1,1].plot(MO_mean_pred[:,1],lw=1,alpha=0.3,color="navy")
		# hdl_splots_data[1,1].fill_between(MO_mean_pred[:,0] - 2.*MO_std_pred,MO_mean_pred[:,0] + 2.*MO_std_pred,color="navy",alpha=0.2)
		
		# hdl_splots_data[2,1].plot(MO_mean_pred[:,2],lw=1,alpha=0.3,color="navy")
		# hdl_splots_data[2,1].fill_between(MO_mean_pred[:,0] - 2.*MO_std_pred,MO_mean_pred[:,0] + 2.*MO_std_pred,color="navy",alpha=0.2)


		plt.show(block=True)


	if using_hybridrobotics:
		# Nhorizon_rec = 40
		Nhorizon_rec = 30
		# Nsteps_tot = z_vec_real.shape[0]-Nhorizon_rec
		# Nsteps_tot = z_vec_real.shape[0] // 2
		Nsteps_tot = z_vec_real.shape[0]
		# Nsteps_tot = 40
		Nepochs = 200
		Nrollouts = 20
		Nchunks = 4
	else:

		# # Nsteps_tot = z_vec_real.shape[0]
		# Nsteps_tot = 50
		# Nchunks = 4

		# Nhorizon_rec = 10 # Will be overwritten if Nchunks is passed to get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon() and it's not None
		# Nrollouts = 5

		# # Nsteps_tot = 50
		# # Nhorizon_rec = 10
		# # Nrollouts = 5

		# Nepochs = 50


		Nhorizon_rec = 15
		# Nsteps_tot = 40
		# Nsteps_tot = z_vec_real.shape[0]
		Nsteps_tot = z_vec_real.shape[0] // 4
		Nepochs = 200
		Nrollouts = 10
		Nchunks = 4


	assert Nsteps_tot > Nhorizon_rec

	# Prepare the training and its loss; the latter compares the true trajectory with the predicted one, in chunks.
	learning_rate = 1e-1
	
	stop_loss_val = -1000.
	scale_loss_entropy = 0.1
	scale_prior_regularizer = 0.1

	rrtp_MO.update_dataset_predictive_loss(	z_vec_real=z_vec_real,u_traj_real=u_vec_tf,
											learning_rate=learning_rate,Nepochs=Nepochs,stop_loss_val=stop_loss_val,
											scale_loss_entropy=scale_loss_entropy,scale_prior_regularizer=scale_prior_regularizer,
											Nrollouts=Nrollouts)


	# file_name = "tensors4prediction_module_cpp_{0:s}.pickle".format(name_file_date)
	path2save_tensors = "{0:s}/{1:s}".format(path2project,path2folder)
	tensors4predictions = rrtp_MO.export_tensors_needed_for_sampling_predictions_using_sampled_model_instances(path2save_tensors)

	dim_in = tensors4predictions["dim_in"]
	dim_out = tensors4predictions["dim_out"]
	phi_samples_all_dim = tensors4predictions["phi_samples_all_dim"]
	W_samples_all_dim = tensors4predictions["W_samples_all_dim"]
	mean_beta_pred_all_dim = tensors4predictions["mean_beta_pred_all_dim"]
	cov_beta_pred_chol_all_dim = tensors4predictions["cov_beta_pred_chol_all_dim"]
	noise_mat = rrtp_MO.sample_mv0[...,0] # We slice matrix [Nrollouts,Nomegas,1]

	# pdb.set_trace()
	predictions_module = Predictions(dim_in,dim_out,phi_samples_all_dim,W_samples_all_dim,mean_beta_pred_all_dim,cov_beta_pred_chol_all_dim,noise_mat,Nrollouts,Nhorizon_rec)
	# predictions_module = None
	
	# Receding horizon predictions:
	savedata = True
	
	loss_avg, x_traj_pred_all_vec, loss_val_per_step = rrtp_MO.get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon(Nsteps_tot,Nhorizon_rec,when2sample="once_per_class_instantiation",predictions_module=predictions_module)

	if savedata:
		data2save = dict(x_traj_pred_all_vec=x_traj_pred_all_vec,u_vec_tf=u_vec_tf,z_vec_real=z_vec_real,loss_val_per_step=loss_val_per_step,Xtrain=Xtrain,Ytrain=Ytrain)

		if rrtp_MO.loss_elbo_evidence_avg_vec is not None:
			data2save.update(	loss_elbo_evidence_avg_vec=rrtp_MO.loss_elbo_evidence_avg_vec,
								loss_elbo_entropy_vec=rrtp_MO.loss_elbo_entropy_vec,
								loss_prior_regularizer_vec=rrtp_MO.loss_prior_regularizer_vec)

		file_name = "predicted_trajs_{0:s}.pickle".format(name_file_date)
		path2save_receding_horizon = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
		logger.info("Saving at {0:s} ...".format(path2save_receding_horizon))
		file = open(path2save_receding_horizon, 'wb')
		pickle.dump(data2save,file)
		file.close()





	
def plot_predictions(cfg,file_name):

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments"

	path2load = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
	file = open(path2load, 'rb')
	data_dict = pickle.load(file)
	file.close()

	x_traj_pred_all_vec = data_dict["x_traj_pred_all_vec"] # [Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x]
	z_vec_real = data_dict["z_vec_real"]
	loss_val_per_step_in = data_dict["loss_val_per_step"]
	# pdb.set_trace()

	if "loss_elbo_evidence_avg_vec" in data_dict.keys():
		loss_elbo_evidence_avg_vec = data_dict["loss_elbo_evidence_avg_vec"]
		loss_elbo_entropy_vec = data_dict["loss_elbo_entropy_vec"] # Good indicator; it's jsut the log-predictive std!!
		loss_prior_regularizer_vec = data_dict["loss_prior_regularizer_vec"]

	loss_lik_lim = np.inf
	loss4colors = -loss_elbo_entropy_vec**2
	my_title = "OoD detection"

	# Debugging:
	if file_name == "predicted_trajs_2023_03_30_11_25_22.pickle": # rope
		loss_lik_lim = 25000.
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = -loss_elbo_entropy_vec**2
	if file_name == "predicted_trajs_2023_03_30_13_12_01.pickle": # rope
		loss_lik_lim = 1e7
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = -np.log(loss_elbo_entropy_vec)
	if file_name == "predicted_trajs_2023_03_30_11_47_32.pickle": # walking
		loss_lik_lim = 5000.
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = -np.log(loss_elbo_entropy_vec)
	if file_name == "predicted_trajs_2023_03_30_11_54_19.pickle": # rocky
		loss_lik_lim = 10000.
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = loss_elbo_entropy_vec
	if file_name == "predicted_trajs_2023_03_30_12_07_26.pickle": # poking
		loss_lik_lim = 25000.
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = loss_elbo_entropy_vec
	


	# Used for paper:
	if file_name == "predicted_trajs_2023_03_30_14_11_55.pickle": # walking [this]
		loss_lik_lim = np.inf
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = loss_val_per_step_in_mod
		my_title = r"OoD detection - No change"
	if file_name == "predicted_trajs_2023_03_30_14_03_40.pickle": # rope [this]
		loss_lik_lim = 5e7
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = -np.log(loss_elbo_entropy_vec)
		my_title = r"OoD detection - Rope pulling"
	if file_name == "predicted_trajs_2023_03_30_13_34_28.pickle": # rocky [this]
		loss_lik_lim = 1e7
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = -loss_elbo_entropy_vec**2
		my_title = r"OoD detection - Rocky terrain"
	if file_name == "predicted_trajs_2023_03_30_13_58_31.pickle": # poking [this]
		loss_lik_lim = 1e7
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss4colors = -np.log(loss_elbo_entropy_vec)
		my_title = r"OoD detection - Poking"






	# Rescale loss for plotting:
	loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
	loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
	# loss_val_per_step_in_mod[loss_val_per_step_in_mod > 25000.0] = 25000.0 # rope
	# loss_val_per_step_in_mod = loss_val_per_step_in / 1000.0

	# add_val = 0.0
	# if np.amin(loss_val_per_step_in_mod) < 0.0:
	# 	add_val = abs(np.amin(loss_val_per_step_in_mod))
	# loss_val_per_step = np.log(loss_val_per_step_in_mod + add_val + 1e-10)

	loss_val_per_step = loss_val_per_step_in_mod





	plotting_analysis_loss = True
	if plotting_analysis_loss:

		hdl_fig_loss, hdl_splots_loss = plt.subplots(4,1,figsize=(17,7),sharex=True)
		hdl_splots_loss[0].plot(loss_val_per_step_in)
		hdl_splots_loss[0].plot(loss_val_per_step)
		hdl_splots_loss[0].set_ylim([loss_val_per_step.min(),loss_val_per_step.max()*1.1])
		hdl_splots_loss[1].plot(loss_elbo_evidence_avg_vec)
		hdl_splots_loss[2].plot(loss_elbo_entropy_vec)
		hdl_splots_loss[3].plot(loss_prior_regularizer_vec)
		# plt.show(block=True)
		plt.show(block=False)


	# ELBO:
	# (1) E_q[p(y|z)] -> this can be seen as an entropy and hence, as information. Is it gain in epistemic uncertainty? -> very high values, scale down
	# (2) H(q(z)) -> this is how confident the model is
	# (3) Prior: we choose it as to penalize values too far from each other
	# 
	# Note that (2) can be used for decision making -> when the model is not confident, we act more conservatively. Use it for plots!




	# The problem is the mean, not the stochastic term. Improve the model by having a better reconstruction. Then, add stochasticity with VERY small noise
	# For some reason the learnign isn't working as it should; MAYBE TRY SHUFFLING THE DATA, IT WORKED WELL FOR DATA EFFICIENCY TEST
	# SERIOUESLY, TRY SHUFFLING THE DATA
	# Claire wants to see a comparison against a sadard GPSSM model; do it. With GPSSM we should see all in bad color
	# I guess we're not comparing against other GPSSM from the literature, too bad... look for code, but no time




	# ABOUT THE LATEX MODEL DATA EFFICIENCY
	# USE SMALLER RATIOS [0.05, 0.1, 0.15, 0.2] AND USE ONLY 0.05 TESTING DATA. SEE WHAT HAPPENS. WHEN USING 0.1 FOR TESTING, BUT THEN 0.9*0.2, THAT'S ALREADY 0.18, WHICH IS NOT PRECISELY SCARCE WRT THE TESTING SET



	Nsteps_tot = x_traj_pred_all_vec.shape[0]

	z_vec_real = z_vec_real[0:Nsteps_tot,:] # We might be doing predictions for a littler shorter time than the duration of the trajectory than trajectories

	Nrollouts = x_traj_pred_all_vec.shape[1]
	time_steps = np.arange(1,Nsteps_tot+1)
	list_xticks_loss = list(range(0,Nsteps_tot+1,200)); list_xticks_loss[0] = 1
	thres_OoD = 5.0
	loss_min = np.amin(loss_val_per_step)
	loss_max = np.amax(loss_val_per_step)
	# loss_max = 300.0

	def color_gradient(loss_val):
		rho_loss = (loss_val - loss_min) / (loss_max - loss_min)
		color_rbg_list = [np.array([220,20,60])/255. , np.array([0,0,128])/255. ] # [crimson, navy]
		color_with_loss = color_rbg_list[0]*rho_loss + color_rbg_list[1]*(1.-rho_loss)
		return color_with_loss

	def is_OoD_loss_based(loss_val_current,loss_thres):
		return loss_val_current > loss_thres

	hdl_fig_pred_sampling_rec, hdl_splots_sampling_rec = plt.subplots(1,1,figsize=(10,10),sharex=False)
	hdl_splots_sampling_rec = [hdl_splots_sampling_rec]
	# hdl_fig_pred_sampling_rec.suptitle("Simulated trajectory predictions ...", fontsize=fontsize_labels)
	# hdl_splots_sampling_rec[0].plot(z_vec_real[0:tt+1,0],z_vec_real[0:tt+1,1],linestyle="-",color="navy",lw=2.0,label="Real traj - nominal dynamics",alpha=0.3)

	
	# hdl_splots_sampling_rec[0].plot(z_vec_real[:,0],z_vec_real[:,1],linestyle="-",color="navy",lw=2.0,label="With nominal dynamics",alpha=0.7)

	z_vec_real_4colors = np.reshape(z_vec_real[:,0:2],(-1,1,2))
	segments = np.concatenate([z_vec_real_4colors[:-1], z_vec_real_4colors[1:]], axis=1)
	norm = plt.Normalize(loss4colors.min(), loss4colors.max())
	lc = LineCollection(segments, cmap='coolwarm', norm=norm, alpha=0.7)
	# Set the values used for colormapping
	# pdb.set_trace()
	lc.set_array(loss4colors)
	lc.set_linewidth(3)

	line = hdl_splots_sampling_rec[0].add_collection(lc)

	if "Xtrain" in data_dict.keys():
		Xtrain = data_dict["Xtrain"]
		hdl_splots_sampling_rec[0].plot(Xtrain[:,0],Xtrain[:,1],linestyle="-",color="grey",lw=1.0,alpha=0.4) # Overlay the entire training set
	tt = 0

	hdl_splots_sampling_rec[0].set_aspect('equal', 'box')

	# color_robot = color_gradient(loss_val_per_step[tt])
	color_robot = "darkorange"

	Nhor = x_traj_pred_all_vec.shape[2]
	Nhor = 15
	logger.info(" <<<<<<<<<<<< [WARNING] >>>>>>>>>>>>>>>>>")
	logger.info(" USING REDUCED HORIZON THAN AVAILABE!!!!")
	# pdb.set_trace() # leave it to make sure we don't forget

	hdl_plt_dubins_real, = hdl_splots_sampling_rec[0].plot(z_vec_real[tt,0],z_vec_real[tt,1],marker="s",markersize=15,color=color_robot,label="Tracking experimental data - Quadruped",alpha=0.5)
	# hdl_splots_sampling_rec[0].set_xlim([-6.0,5.0])
	# hdl_splots_sampling_rec[0].set_ylim([-3.5,1.5])
	hdl_splots_sampling_rec[0].set_title(my_title, fontsize=fontsize_labels)
	hdl_splots_sampling_rec[0].set_xlabel(r"$x_1$ [m]", fontsize=fontsize_labels)
	hdl_splots_sampling_rec[0].set_ylabel(r"$x_2$ [m]", fontsize=fontsize_labels)
	hdl_splots_sampling_rec[0].set_xticks([-2.0,0.0,1.0])
	hdl_splots_sampling_rec[0].set_yticks([1.5,3.0,4.5])
	hdl_plt_predictions_list = []
	color_rollouts = "darkorange"
	for ss in range(Nrollouts):
		# Nhor = 15
		# Nhor = x_traj_pred_all_vec.shape[2]
		if ss == 0: lw = 3.0
		if ss > 0: lw = 0.5
		hdl_plt_predictions_list += hdl_splots_sampling_rec[0].plot(x_traj_pred_all_vec[0,ss,0:Nhor,0],x_traj_pred_all_vec[0,ss,0:Nhor,1],linestyle="-",color=color_rollouts,lw=lw,label="Sampled trajs",alpha=0.7)

	# # Loss evolution:
	# hdl_plt_artist_loss_title = hdl_splots_sampling_rec[1].set_title("ELBO prediction loss", fontsize=fontsize_labels)
	# hdl_plt_artist_loss, = hdl_splots_sampling_rec[1].plot(time_steps[0:1],loss_val_per_step[0:1],linestyle="-",color="darkorange",lw=2.0,alpha=0.8)
	# hdl_splots_sampling_rec[1].set_xlim([1,Nsteps_tot+1])
	# hdl_splots_sampling_rec[1].set_xticks(list_xticks_loss)
	# hdl_splots_sampling_rec[1].set_xlabel("Time step", fontsize=fontsize_labels)
	# hdl_splots_sampling_rec[1].set_ylim([loss_min,thres_OoD*3.])
	# hdl_splots_sampling_rec[1].axhline(y=thres_OoD,color="palegoldenrod",lw=2.0,linestyle='-')
	
	plt.show(block=False)
	# plt.pause(0.5)
	plt_pause_sec = 0.0005
	# pdb.set_trace()
	input()
	

	for tt in range(Nsteps_tot):

		is_OoD = is_OoD_loss_based(loss_val_per_step[tt],thres_OoD)
		# hdl_plt_dubins_real.set_markerfacecolor("red" if is_OoD else "green")
		# hdl_plt_dubins_real.set_markeredgecolor("red" if is_OoD else "green")

		# color_robot = color_gradient(loss_val_per_step[tt])
		color_robot = "grey"
		hdl_plt_dubins_real.set_markeredgecolor(color_robot)

		hdl_plt_dubins_real.set_xdata(z_vec_real[tt,0])
		hdl_plt_dubins_real.set_ydata(z_vec_real[tt,1])
		
		for ss in range(Nrollouts):
			hdl_plt_predictions_list[ss].set_xdata(x_traj_pred_all_vec[tt,ss,0:Nhor,0])
			hdl_plt_predictions_list[ss].set_ydata(x_traj_pred_all_vec[tt,ss,0:Nhor,1])
			# hdl_splots_sampling_rec[0].plot(x_traj_pred_all_vec[tt,ss,:,0],x_traj_pred_all_vec[tt,ss,:,1],linestyle="-",color="crimson",lw=0.5,label="Sampled trajs",alpha=0.3)

		# # Right plot with loss evolution
		# hdl_plt_artist_loss.set_xdata(time_steps[0:tt+1])
		# hdl_plt_artist_loss.set_ydata(loss_val_per_step[0:tt+1])
		# # hdl_splots_sampling_rec[1].set_ylim([loss_min,np.amax(loss_val_per_step[0:tt+1])*1.1])
		# # hdl_splots_sampling_rec[1].set_title("Prediction loss; {0:s}".format("OoD = {0:s}".format(str(is_OoD))), fontsize=fontsize_labels)
		# hdl_plt_artist_loss_title.set_text("Prediction loss | OoD = {0:s}".format(str(is_OoD)))
		# hdl_plt_artist_loss_title.set_color(color_robot)
		

		plt.show(block=False)
		plt.pause(plt_pause_sec)

	plt.show(block=True)


def plots4paper_ood(cfg):

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments"


	file_name_list = [	"predicted_trajs_2023_03_30_14_11_55.pickle",
						"predicted_trajs_2023_03_30_14_03_40.pickle",
						"predicted_trajs_2023_03_30_13_34_28.pickle",
						"predicted_trajs_2023_03_30_13_58_31.pickle"]
	hdl_fig_pred_sampling_rec, hdl_splots_sampling_rec = plt.subplots(2,2,figsize=(10,10))
	ind_plot_row = [0,0,1,1]
	ind_plot_col = [0,1,0,1]
	for aa in range(4):

		file_name = file_name_list[aa]

		hdl_splots_sampling_cur = hdl_splots_sampling_rec[ind_plot_row[aa],ind_plot_col[aa]]

		path2load = "{0:s}/{1:s}/{2:s}".format(path2project,path2folder,file_name)
		file = open(path2load, 'rb')
		data_dict = pickle.load(file)
		file.close()

		x_traj_pred_all_vec = data_dict["x_traj_pred_all_vec"] # [Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x]
		z_vec_real = data_dict["z_vec_real"]
		loss_val_per_step_in = data_dict["loss_val_per_step"]

		if "loss_elbo_evidence_avg_vec" in data_dict.keys():
			loss_elbo_evidence_avg_vec = data_dict["loss_elbo_evidence_avg_vec"]
			loss_elbo_entropy_vec = data_dict["loss_elbo_entropy_vec"] # Good indicator; it's jsut the log-predictive std!!
			loss_prior_regularizer_vec = data_dict["loss_prior_regularizer_vec"]

		loss_lik_lim = np.inf
		loss4colors = -loss_elbo_entropy_vec**2
		my_title = "OoD detection"
		Nhor_list = [x_traj_pred_all_vec.shape[2]]*4
		# Used for paper:
		if file_name == "predicted_trajs_2023_03_30_14_11_55.pickle": # walking [this]
			loss_lik_lim = np.inf
			loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
			loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
			loss4colors = loss_val_per_step_in_mod
			my_title = r"OoD detection - No change"
			aux = 0 + 640*np.array([0.035,1./4,1./2,3./4*1.1])
			Nhor_list = [30]*4
		if file_name == "predicted_trajs_2023_03_30_14_03_40.pickle": # rope [this]
			loss_lik_lim = 5e7
			loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
			loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
			loss4colors = -np.log(loss_elbo_entropy_vec)
			my_title = r"OoD detection - Rope pulling"
			aux = 0 + 640*np.array([1./4*0.5,1./2,3./4*1.1,1.065])
			Nhor_list = [20,10,10,20]
		if file_name == "predicted_trajs_2023_03_30_13_34_28.pickle": # rocky [this]
			loss_lik_lim = 1e7
			loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
			loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
			loss4colors = -loss_elbo_entropy_vec**2
			my_title = r"OoD detection - Rocky terrain"
			aux = 0 + 640*np.array([1./4*0.4,1./4,1./2*0.9,3./4*1.15])
			Nhor_list = [20,10,10,20]
		if file_name == "predicted_trajs_2023_03_30_13_58_31.pickle": # poking [this]
			loss_lik_lim = 1e7
			loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
			loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
			loss4colors = -np.log(loss_elbo_entropy_vec)
			my_title = r"OoD detection - Poking"
			aux = 0 + 640*np.array([1./4*0.4,1./4,1./2*1.2,3./4*1.15])
			Nhor_list = [20,5,5,20]

		tt_for_showing_ood = aux.astype(int)


		# Rescale loss for plotting:
		loss_val_per_step_in_mod = np.copy(loss_val_per_step_in)
		loss_val_per_step_in_mod[loss_val_per_step_in_mod > loss_lik_lim] = loss_lik_lim 
		loss_val_per_step = loss_val_per_step_in_mod
		Nsteps_tot = x_traj_pred_all_vec.shape[0]
		z_vec_real = z_vec_real[0:Nsteps_tot,:] # We might be doing predictions for a littler shorter time than the duration of the trajectory than trajectories
		Nrollouts = x_traj_pred_all_vec.shape[1]

		z_vec_real_4colors = np.reshape(z_vec_real[:,0:2],(-1,1,2))
		segments = np.concatenate([z_vec_real_4colors[:-1], z_vec_real_4colors[1:]], axis=1)
		norm = plt.Normalize(loss4colors.min(), loss4colors.max())
		lc = LineCollection(segments, cmap='coolwarm', norm=norm, alpha=0.7)
		lc.set_array(loss4colors)
		lc.set_linewidth(3)
		line = hdl_splots_sampling_cur.add_collection(lc)

		if "Xtrain" in data_dict.keys():
			Xtrain = data_dict["Xtrain"]
			hdl_splots_sampling_cur.plot(Xtrain[:,0],Xtrain[:,1],linestyle="-",color="grey",lw=1.0,alpha=0.4) # Overlay the entire training set
		
		
		
		
		logger.info(" <<<<<<<<<<<< [WARNING] >>>>>>>>>>>>>>>>>")
		logger.info(" USING REDUCED HORIZON THAN AVAILABE!!!!")
		# pdb.set_trace() # leave it to make sure we don't forget

		# color_robot = "darkorange"
		color_robot = "yellowgreen"
		# hdl_splots_sampling_cur.set_title(my_title, fontsize=fontsize_labels)
		hdl_splots_sampling_cur.set_xlim([-2.2,1.2])

		hdl_splots_sampling_cur.set_aspect('equal', 'box')

		hdl_splots_sampling_cur.set_xticks([])
		hdl_splots_sampling_cur.set_yticks([])
		hdl_plt_predictions_list = []
		color_rollouts = color_robot

		for ii in range(len(tt_for_showing_ood)):
			tt = tt_for_showing_ood[ii]
			hdl_splots_sampling_cur.plot(z_vec_real[tt,0],z_vec_real[tt,1],marker="s",markersize=10,color=color_robot,alpha=0.3)
			for ss in range(Nrollouts):
				Nhor = Nhor_list[ii]
				# Nhor = 15
				# Nhor = x_traj_pred_all_vec.shape[2]
				if ss == 0: lw = 3.0
				if ss > 0: lw = 0.1
				hdl_plt_predictions_list += hdl_splots_sampling_cur.plot(x_traj_pred_all_vec[tt,ss,0:Nhor,0],x_traj_pred_all_vec[tt,ss,0:Nhor,1],linestyle="-",color=color_rollouts,lw=lw,label="Sampled trajs",alpha=0.7)
			

	hdl_splots_sampling_rec[1,0].set_xticks([-2.0,0.0,1.0])
	hdl_splots_sampling_rec[1,0].set_xlabel(r"$x_1$ [m]", fontsize=fontsize_labels)
	# hdl_splots_sampling_rec[1,1].set_xticks([-2.0,0.0,1.0])
	hdl_splots_sampling_rec[1,1].set_xlabel(r"$x_1$ [m]", fontsize=fontsize_labels)
	
	# hdl_splots_sampling_rec[0,0].set_yticks([1.5,3.0,4.5])
	hdl_splots_sampling_rec[1,0].set_yticks([1.5,3.0,4.5])
	hdl_splots_sampling_rec[0,0].set_ylabel(r"$x_2$ [m]", fontsize=fontsize_labels)
	hdl_splots_sampling_rec[1,0].set_ylabel(r"$x_2$ [m]", fontsize=fontsize_labels)


	# hdl_fig_pred_sampling_rec.subplots_adjust(right=0.8)
	# cbar_ax = hdl_fig_pred_sampling_rec.add_axes([0.85, 0.15, 0.05, 0.7])
	# hdl_fig_pred_sampling_rec.colorbar(line, cax=cbar_ax)

	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	savefig = True
	if savefig:
		path2save_fig = "{0:s}/plotting/plots4paper/ood_quadruped_rrMOtp{1:s}.png".format(path2project,name_file_date)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig_pred_sampling_rec.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.show(block=True)


	plt.show(block=True)


@hydra.main(config_path="./config",config_name="config")
def main(cfg):

	# train_gpssm(cfg)

	# compute_predictions_our_model(cfg)

	compute_predictions_gpflow(cfg)


	# # ==============================================================
	# # With Quadruped data from data_quadruped_experiments_03_25_2023
	# # ==============================================================
	# # All with recostructed model file_name = "reconstruction_data_2023_03_27_01_23_40.pickle"
	# # file_name = "predicted_trajs_2023_03_27_02_02_52.pickle"
	# # file_name = "predicted_trajs_2023_03_27_02_31_51.pickle"
	# # file_name = "predicted_trajs_2023_03_27_02_37_01.pickle" # Working alright, but could be better
	# # file_name = "predicted_trajs_2023_03_27_12_03_52.pickle" # Sampling rollouts no with Gaussian samples cutoff of 0.8; noise parameter in the model set to 0.005
	# file_name = "predicted_trajs_2023_03_27_12_16_02.pickle" # Sampling rollouts no with Gaussian samples cutoff of 0.8; noise parameter in the model set to 0.003; ovelaying all the trainign data | not too bad
	# plot_predictions(cfg,file_name)


	# ===========================================================================
	# With Quadruped data from data_quadruped_experiments_03_29_2023 || OUR MODEL
	# ===========================================================================
	# All with recostructed model file_name = "reconstruction_data_2023_03_29_23_11_35.pickle" (walking on a circle)
	# file_name = "predicted_trajs_2023_03_29_23_34_13.pickle" # DBG; noise: 0.01
	# file_name = "predicted_trajs_2023_03_30_00_25_21.pickle" # hybridrob, Nhor: 30, Nrollouts: 20
	# file_name = "predicted_trajs_2023_03_30_01_12_50.pickle" # DBG, Nhor: 10, Nrollouts: 10; noise: 0.008
	# file_name = "predicted_trajs_2023_03_30_02_00_56.pickle" # DBG, Nhor: 10, Nrollouts: 10; noise: 0.008; no stochasticy, just the mean
	# file_name = "predicted_trajs_2023_03_30_02_04_15.pickle" # DBG, Nhor: 30, Nrollouts: 10; noise: 0.008; no stochasticy, just the mean
	# file_name = "predicted_trajs_2023_03_30_02_38_09.pickle" # DBG, Nhor: 30, Nrollouts: 1; noise: 0.008; no stochasticy, just the mean (trained on walking circle; tested on 1/5 of the rope data)
	# file_name = "predicted_trajs_2023_03_30_03_01_59.pickle" # DBG, Nhor: 30, Nrollouts: 1; noise: 0.008; no stochasticy, just the mean (trained on walking circle; tested on full if the rope data)
	# file_name = "predicted_trajs_2023_03_30_11_25_22.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.008; first rollout is the mean || trained on walking circle; tested on: rope
	# file_name = "predicted_trajs_2023_03_30_11_47_32.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.008; first rollout is the mean || trained on walking circle; tested on: walking
	# file_name = "predicted_trajs_2023_03_30_11_54_19.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.008; first rollout is the mean || trained on walking circle; tested on: rocky
	# file_name = "predicted_trajs_2023_03_30_12_07_26.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.008; first rollout is the mean || trained on walking circle; tested on: poking
	# file_name = "predicted_trajs_2023_03_30_13_04_32.pickle" # DBG, Nhor: 15, Nrollouts: 10; noise: 0.0001; first rollout is the mean || trained on walking circle; tested on: rope
	# file_name = "predicted_trajs_2023_03_30_13_12_01.pickle" # DBG, Nhor: 15, Nrollouts: 10; noise: 0.0001; first rollout is the mean || trained on walking circle; tested on: rope, longer time
	# plot_predictions(cfg,file_name)


	# =======================================================================
	# With Quadruped data from data_quadruped_experiments_03_29_2023 || GPSSM
	# =======================================================================
	# All trained from data from walking on a circle
	# file_name = "predicted_trajs_2023_03_30_17_11_19.pickle" # DBG
	# file_name = "predicted_trajs_2023_03_30_17_20_27.pickle" # DBG, Nhor: 10, Nrollouts: 5 || trained on walking circle; tested on: walking
	file_name = "predicted_trajs_2023_03_30_17_31_00.pickle" # DBG, Nhor: 10, Nrollouts: 5 || trained on walking circle; tested on: rope
	plot_predictions(cfg,file_name)



	# ===========================================================================
	# With Quadruped data from data_quadruped_experiments_03_29_2023 || FOR PAPER || OUR MODEL
	# ===========================================================================
	# file_name = "predicted_trajs_2023_03_30_14_11_55.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.0001; first rollout is the mean || trained on walking circle; tested on: walking
	file_name = "predicted_trajs_2023_03_30_14_03_40.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.0001; first rollout is the mean || trained on walking circle; tested on: rope
	# file_name = "predicted_trajs_2023_03_30_13_34_28.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.0001; first rollout is the mean || trained on walking circle; tested on: rocky
	# file_name = "predicted_trajs_2023_03_30_13_58_31.pickle" # hybridrob, Nhor: 30, Nrollouts: 20; noise: 0.0001; first rollout is the mean || trained on walking circle; tested on: poking
	plot_predictions(cfg,file_name)

	# plots4paper_ood(cfg)



if __name__ == "__main__":

	main()

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/predicted_trajs_55.pickle ./data_quadruped_experiments_03_25_2023/
	# export PYTHONPATH=$PYTHONPATH:/Users/alonrot/work/code_projects_WIP/ood_project/ood/predictions_module/build
	# export PYTHONPATH=$PYTHONPATH:/home/amarco/code_projects/ood_project/ood/predictions_module/build

	# scp -P 4444 -r ./data_quadruped_experiments_03_25_2023/from_hybridrob/reconstruction_data_2023_03_27_01_23_40.pickle amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/from_hybridrob/

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/predicted_trajs_2023_03_27_02_37_01.pickle ./data_quadruped_experiments_03_25_2023/




	# scp -P 4444 -r ./data_quadruped_experiments_03_29_2023/from_hybridrob/reconstruction_data_2023_03_29_23_11_35.pickle amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_29_2023/from_hybridrob/
	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_29_2023/"*2023_03_30_16_00_54*" ./data_quadruped_experiments_03_29_2023/
	# scp -P 4444 -r ./data_quadruped_experiments_03_29_2023/from_hybridrob/reconstruction_data_2023_03_29_23_11_35.pickle amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_29_2023/from_hybridrob/
	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_29_2023/predicted_trajs_2023_03_30_17_31_00.pickle ./data_quadruped_experiments_03_29_2023/















