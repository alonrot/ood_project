import tensorflow as tf
# import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity
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

from lqrker.spectral_densities import DubinsCarSpectralDensity
from test_dubin_car import get_sequence_of_feedback_gains_finite_horizon_LQR, rollout_with_finitie_horizon_LQR, generate_trajectories, generate_reference_trajectory


dyn_sys_true = DubinsCarSpectralDensity._controlled_dubinscar_dynamics

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

# GP flow:
import gpflow as gpf
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("github")
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary


using_deltas = True
# using_deltas = False

my_seed = 80


def load_data(cfg):

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments"
	else:
		path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"


	path2data = "{0:s}/data_quadruped_experiments_03_13_2023/joined_go1trajs_trimmed.pickle".format(path2project) # mac

	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	state_and_control_full_list = data_dict["state_and_control_full_list"] # Identical to the training data, only that trajectories are sorted out in a list; Will be used as test data
	state_next_full_list = data_dict["state_next_full_list"] # Identical to the training data, only that trajectories are sorted out in a list; Will be used as test data
	dim_x = Ytrain.shape[1]
	dim_u = Xtrain.shape[1] - Ytrain.shape[1]

	# Trying modifications:
	do_modifications_to_test_data = False
	if do_modifications_to_test_data: # Expose the robot to an altered trajectory at test time and hence, during rollouts
		for state_and_control_full_el in state_and_control_full_list:
			# state_and_control_full_el[:,3] = 1.1*state_and_control_full_el[:,3]
			# state_and_control_full_el[:,4] = 1.1*state_and_control_full_el[:,4]
			state_and_control_full_el[:,0] = 1.15*state_and_control_full_el[:,0]
			state_and_control_full_el[:,1] = 1.15*state_and_control_full_el[:,1]

	# Convert to tensor:
	Xtrain = tf.convert_to_tensor(Xtrain,dtype=tf.float32)
	Ytrain = tf.convert_to_tensor(Ytrain,dtype=tf.float32)
	state_and_control_full_list = [tf.convert_to_tensor(state_and_control_full_el,dtype=tf.float32) for state_and_control_full_el in state_and_control_full_list]
	state_next_full_list = [tf.convert_to_tensor(state_next_full_el,dtype=tf.float32) for state_next_full_el in state_next_full_list]


	if using_deltas:
		Ytrain_deltas = Ytrain - Xtrain[:,0:dim_x]
		Ytrain = tf.identity(Ytrain_deltas)

	dim_X = dim_x + dim_u

	return Xtrain, Ytrain, state_and_control_full_list, state_next_full_list, dim_x, dim_u, dim_X, using_hybridrobotics, path2project



@hydra.main(config_path="./config",config_name="config")
def main_train_model(cfg: dict):

	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	Xtrain, Ytrain, state_and_control_full_list, state_next_full_list, dim_x, dim_u, dim_X, using_hybridrobotics, path2project = load_data(cfg)


	# Based on: https://gpflow.github.io/GPflow/develop/notebooks/advanced/multioutput.html#
	MAXITER = reduce_in_tests(2000)

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
	kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) + gpf.kernels.Linear(variance=1.0) for _ in range(P)] # Adding a linear kernel
	# kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) for _ in range(P)]
	
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
	path2save_model = path2project+"/data_quadruped_experiments_03_13_2023_gpflow"
	model_name = "model_{0:d}_coregionalization_{1:s}".format(my_seed,str(use_coregionalization))
	path2save_model_full = "{0:s}/{1:s}".format(path2save_model,model_name)
	logger.info("Saving model at {0:s} ...".format(path2save_model_full))
	tf.saved_model.save(model_gpflow, path2save_model_full)
	logger.info("Done!")


@hydra.main(config_path="./config",config_name="config")
def main_test_model(cfg: dict):

	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	Xtrain, Ytrain, state_and_control_full_list, state_next_full_list, dim_x, dim_u, dim_X, using_hybridrobotics, path2project = load_data(cfg)

	# Load model:
	path2save_model = path2project+"/data_quadruped_experiments_03_13_2023_gpflow"
	# path2save_model = path2project+"/data_quadruped_experiments_03_13_2023_gpflow/from_hybridrobotics"
	# model_name = "model_12" # No coregionalization, adding linear kernel
	model_name = "model_80_coregionalization_True" # not adding linear kernel; optimizaiton finished prematurely
	path2save_model_full = "{0:s}/{1:s}".format(path2save_model,model_name)
	logger.info("Loading model from {0:s} ...".format(path2save_model))
	loaded_model = tf.saved_model.load(path2save_model_full)
	logger.info("Done!")

	print_summary(loaded_model)

	Zinduced = loaded_model.get_induced_pointsZ_list()
	print(Zinduced)

	# Select test trajectory:
	ind_which_traj = 1
	zu_vec = state_and_control_full_list[ind_which_traj]
	z_vec = zu_vec[:,0:dim_x]
	u_vec = zu_vec[:,dim_x::]
	if using_deltas:
		z_next_vec = state_next_full_list[ind_which_traj] - state_and_control_full_list[ind_which_traj][:,0:dim_x]
	else:
		z_next_vec = state_next_full_list[ind_which_traj]


	# Plotting:
	plotting_selected_trajs = False
	if plotting_selected_trajs and not using_hybridrobotics:


		MO_mean_pred, _ = loaded_model.compiled_predict_f(tf.cast(zu_vec,dtype=tf.float64))

		MO_mean_pred = tf.cast(MO_mean_pred,dtype=tf.float32)

		if using_deltas:
			z_next_vec_plotting = z_next_vec + zu_vec[:,0:dim_x]
			MO_mean_pred_plotting = MO_mean_pred + zu_vec[:,0:dim_x]
		else:
			z_next_vec_plotting = z_next_vec
			MO_mean_pred_plotting = MO_mean_pred


		hdl_fig_pred, hdl_splots_pred = plt.subplots(4,figsize=(12,8),sharex=True)
		hdl_fig_pred.suptitle("Predictions ...", fontsize=16)
		for ii in range(len(state_and_control_full_list)):
			zu_el = state_and_control_full_list[ii]
			zu_el_next = state_next_full_list[ii]
			hdl_splots_pred[0].cla()
			hdl_splots_pred[0].plot(zu_el[:,0],zu_el[:,1],linestyle="-",color="grey",lw=3.0,label=r"Real traj - Input",alpha=0.3)
			hdl_splots_pred[0].plot(zu_el_next[:,0],zu_el_next[:,1],linestyle="-",color="navy",lw=1.0,label=r"Real traj - Input",alpha=0.5)

			hdl_splots_pred[1].cla()
			hdl_splots_pred[1].plot(zu_el[:,2])

			hdl_splots_pred[2].cla()
			hdl_splots_pred[2].plot(zu_el[:,3])

			hdl_splots_pred[3].cla()
			hdl_splots_pred[3].plot(zu_el[:,4])

			plt.show(block=False)

			# pdb.set_trace()
			logger.info("ii: {0:d}".format(ii))
			# plt.pause(0.1)
			# if ii == 15:
			# 	plt.show(block=True)

		hdl_fig_pred, hdl_splots_pred = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_fig_pred.suptitle("Predictions ...", fontsize=16)
		hdl_splots_pred.plot(z_vec[:,0],z_vec[:,1],linestyle="--",color="grey",lw=1.0,label=r"Real traj - Input",alpha=0.3)
		hdl_splots_pred.plot(z_next_vec_plotting[:,0],z_next_vec_plotting[:,1],linestyle="-",color="navy",lw=2.0,label=r"Real traj - Next state",alpha=0.3)
		hdl_splots_pred.plot(MO_mean_pred_plotting[:,0],MO_mean_pred_plotting[:,1],linestyle="-",color="navy",lw=2.0,label=r"Predicted traj - Next dynamics",alpha=0.7)


		# plt.show(block=False)
		plt.show(block=True)
		plt.pause(4.)


	z_vec_tf = tf.convert_to_tensor(value=z_vec,dtype=tf.float32)
	u_vec_tf = tf.convert_to_tensor(value=u_vec,dtype=tf.float32)

	z_vec_real = z_vec_tf
	z_vec_changed_dyn_tf = None

	if using_hybridrobotics:
		# Nhorizon_rec = 40
		Nhorizon_rec = 25
		# Nsteps_tot = z_vec_real.shape[0]-Nhorizon_rec
		Nsteps_tot = z_vec_real.shape[0]
		# Nsteps_tot = z_vec_real.shape[0]
		Nepochs = 200
		Nrollouts = 15
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


		Nhorizon_rec = 25
		Nsteps_tot = z_vec_real.shape[0]
		# Nsteps_tot = z_vec_real.shape[0] // 8
		Nepochs = 200
		Nrollouts = 40
		Nchunks = 4

	assert Nsteps_tot > Nhorizon_rec


	# Prepare the training and its loss; the latter compares the true trajectory with the predicted one, in chunks.
	learning_rate = 1e-1
	epochs = 5
	Nhorizon = 10
	Nrollouts = 20
	train = False
	stop_loss_val = -1000.
	scale_loss_entropy = 0.1
	scale_prior_regularizer = 0.1
	z_vec_tf = tf.convert_to_tensor(value=z_vec,dtype=tf.float32)
	# z_vec_changed_dyn_tf = tf.convert_to_tensor(value=z_vec_changed_dyn,dtype=tf.float32)
	z_vec_changed_dyn_tf = None 
	u_vec_tf = tf.convert_to_tensor(value=u_vec,dtype=tf.float32)

	# compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to = "nominal_model"
	# # compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to = "altered_model"
	# assert compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to in ["nominal_model","altered_model"]
	# if compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to == "nominal_model":
	# 	z_vec_real = z_vec_tf # [Nsteps,dim_out]
	# if compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to == "altered_model":
	# 	z_vec_real = z_vec_changed_dyn_tf # [Nsteps,dim_out]


	# Receding horizon predictions:
	plotting_receding_horizon_predictions = True
	savedata = True
	recompute = True
	# recompute = False
	path2save_receding_horizon = path2project+"/data_quadruped_experiments_03_13_2023_gpflow"
	file_name = "trajs_ind_traj_{0:d}.pickle".format(my_seed)
	if plotting_receding_horizon_predictions and recompute:
		Nsteps_tot = min(z_vec_real.shape[0]-Nhorizon_rec,Nsteps_tot)
		loss_val_per_step = np.zeros(Nsteps_tot)
		x_traj_pred_all_vec = np.zeros((Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x))
		for tt in range(Nsteps_tot):
			noise_mat = np.random.randn(Nrollouts,dim_x)

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
					MO_mean_pred, cov_full = loaded_model.compiled_predict_f(zu_vec_tf) # cov_full turns out to be alwys diagonal because tGPflow cannot handle full covariance

					xnext = MO_mean_pred + np.sqrt(np.diag(cov_full[0,...]))*noise_mat[rr:rr+1,:] # [1,dim_x]

					x_traj_pred_all_vec[tt,rr,ppp+1:ppp+2,:] = xnext

		if savedata:
			data2save = dict(x_traj_pred_all_vec=x_traj_pred_all_vec,u_vec_tf=u_vec_tf,z_vec_real=z_vec_real,z_vec_tf=z_vec_tf,z_vec_changed_dyn_tf=z_vec_changed_dyn_tf,loss_val_per_step=loss_val_per_step)
			path2save_full = "{0:s}/{1:s}".format(path2save_receding_horizon,file_name)
			logger.info("Saving at {0:s} ...".format(path2save_full))
			file = open(path2save_full, 'wb')
			pickle.dump(data2save,file)
			file.close()
			logger.info("Done!")
			return


	elif plotting_receding_horizon_predictions:

		# file_name = "trajs_ind_traj_75.pickle" # Using trajectory from nominal model; trained GPflow model: model_12
		file_name = "trajs_ind_traj_48.pickle" # Using trajectory from altered model; trained GPflow model: model_12
		# file_name = "trajs_ind_traj_12.pickle" # Using coregionalization, no linear kernel, and trajectory from altered model; trained GPflow model: model_13_coregionalization_True; -> training ended prematurely; not working
		# file_name = "trajs_ind_traj_72.pickle" # Using coregionalization, with linear kernel, and trajectory from altered model; trained GPflow model: model_13_coregionalization_True; -> no big difference wrt trajs_ind_traj_48

		path2save_full = "{0:s}/{1:s}".format(path2save_receding_horizon,file_name)
		file = open(path2save_full, 'rb')
		data_dict = pickle.load(file)
		file.close()

		x_traj_pred_all_vec = data_dict["x_traj_pred_all_vec"] # [Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x]
		z_vec_tf = data_dict["z_vec_tf"]
		z_vec_changed_dyn_tf = data_dict["z_vec_changed_dyn_tf"]
		z_vec_real = data_dict["z_vec_real"]
		loss_val_per_step = data_dict["loss_val_per_step"]

		Nsteps_tot = x_traj_pred_all_vec.shape[0]
		Nrollouts = x_traj_pred_all_vec.shape[1]
		time_steps = np.arange(1,Nsteps_tot+1)
		list_xticks_loss = list(range(0,Nsteps_tot+1,40)); list_xticks_loss[0] = 1
		thres_OoD = 10.0
		loss_min = np.amin(loss_val_per_step)

		def is_OoD_loss_based(loss_val_current,loss_thres):
			return loss_val_current > loss_thres

		hdl_fig_pred_sampling_rec, hdl_splots_sampling_rec = plt.subplots(1,2,figsize=(17,7),sharex=False)
		# hdl_fig_pred_sampling_rec.suptitle("Simulated trajectory predictions ...", fontsize=fontsize_labels)
		# hdl_splots_sampling_rec[0].plot(z_vec_real[0:tt+1,0],z_vec_real[0:tt+1,1],linestyle="-",color="navy",lw=2.0,label="Real traj - nominal dynamics",alpha=0.3)
		hdl_splots_sampling_rec[0].plot(z_vec_tf[:,0],z_vec_tf[:,1],linestyle="-",color="navy",lw=2.0,label="With nominal dynamics",alpha=0.7)
		if z_vec_changed_dyn_tf is not None: hdl_splots_sampling_rec[0].plot(z_vec_changed_dyn_tf[:,0],z_vec_changed_dyn_tf[:,1],linestyle="-",color="navy",lw=2.0,label="With changed dynamics",alpha=0.15)
		hdl_plt_dubins_real, = hdl_splots_sampling_rec[0].plot(z_vec_real[tt,0],z_vec_real[tt,1],marker="*",markersize=14,color="darkgreen",label="Dubins car")
		# hdl_splots_sampling_rec[0].set_xlim([-6.0,5.0])
		# hdl_splots_sampling_rec[0].set_ylim([-3.5,1.5])
		hdl_splots_sampling_rec[0].set_title("Dubins car", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[0].set_xlabel(r"$x_1$", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[0].set_ylabel(r"$x_2$", fontsize=fontsize_labels)
		hdl_plt_predictions_list = []
		for ss in range(Nrollouts):
			hdl_plt_predictions_list += hdl_splots_sampling_rec[0].plot(x_traj_pred_all_vec[0,ss,:,0],x_traj_pred_all_vec[0,ss,:,1],linestyle="-",color="darkorange",lw=0.5,label="Sampled trajs",alpha=0.5)

		# Loss evolution:
		hdl_plt_artist_loss_title = hdl_splots_sampling_rec[1].set_title("Prediction loss", fontsize=fontsize_labels)
		hdl_plt_artist_loss, = hdl_splots_sampling_rec[1].plot(time_steps[0:1],loss_val_per_step[0:1],linestyle="-",color="darkorange",lw=2.0,alpha=0.8)
		hdl_splots_sampling_rec[1].set_xlim([1,Nsteps_tot+1])
		hdl_splots_sampling_rec[1].set_xticks(list_xticks_loss)
		hdl_splots_sampling_rec[1].set_xlabel("Time step", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[1].set_ylim([loss_min,thres_OoD*3.])
		hdl_splots_sampling_rec[1].axhline(y=thres_OoD,color="palegoldenrod",lw=2.0,linestyle='-')
		
		plt.show(block=False)
		plt.pause(0.5)
		plt_pause_sec = 0.01
		

		for tt in range(Nsteps_tot):

			is_OoD = is_OoD_loss_based(loss_val_per_step[tt],thres_OoD)

			hdl_plt_dubins_real.set_markerfacecolor("red" if is_OoD else "green")
			hdl_plt_dubins_real.set_markeredgecolor("red" if is_OoD else "green")

			hdl_plt_dubins_real.set_xdata(z_vec_real[tt,0])
			hdl_plt_dubins_real.set_ydata(z_vec_real[tt,1])
			
			for ss in range(Nrollouts):
				hdl_plt_predictions_list[ss].set_xdata(x_traj_pred_all_vec[tt,ss,:,0])
				hdl_plt_predictions_list[ss].set_ydata(x_traj_pred_all_vec[tt,ss,:,1])
				# hdl_splots_sampling_rec[0].plot(x_traj_pred_all_vec[tt,ss,:,0],x_traj_pred_all_vec[tt,ss,:,1],linestyle="-",color="crimson",lw=0.5,label="Sampled trajs",alpha=0.3)

			hdl_plt_artist_loss.set_xdata(time_steps[0:tt+1])
			hdl_plt_artist_loss.set_ydata(loss_val_per_step[0:tt+1])
			# hdl_splots_sampling_rec[1].set_ylim([loss_min,np.amax(loss_val_per_step[0:tt+1])*1.1])
			# hdl_splots_sampling_rec[1].set_title("Prediction loss; {0:s}".format("OoD = {0:s}".format(str(is_OoD))), fontsize=fontsize_labels)
			hdl_plt_artist_loss_title.set_text("Prediction loss | OoD = {0:s}".format(str(is_OoD)))
			hdl_plt_artist_loss_title.set_color("red" if is_OoD else "green")
			

			plt.show(block=False)
			plt.pause(plt_pause_sec)

		plt.show(block=True)


if __name__ == "__main__":

	# main_train_model()

	main_test_model()



