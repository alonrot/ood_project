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

from bs4 import BeautifulSoup

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
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary


def alter_dynamics_flag_time_based(tt,Nsteps):

	# Time-based:
	flag_alter = False
	if tt < Nsteps/2:
		flag_alter = True

	return flag_alter

def alter_dynamics_flag_state_based(state_curr):

	flag_alter = False
	if np.any(abs(state_curr[0,0:2]) > 2.0):
		flag_alter = True

	return flag_alter



@hydra.main(config_path="./config",config_name="config")
def main_train_model(cfg: dict):

	my_seed = 12
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	# Get training data:
	path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"

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

	dim_X = dim_x + dim_u

	gpf.config.set_default_float(np.float64)
	gpf.config.set_default_summary_fmt("notebook")
	np.random.seed(0)
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
			options={"disp": 50, "maxiter": MAXITER},
		)


	# Create list of kernels for each output
	kern_list = [gpf.kernels.SquaredExponential(variance=1.0,lengthscales=0.1*np.ones(D)) + gpf.kernels.Linear(variance=1.0) for _ in range(P)]
	
	# Create multi-output kernel from kernel list
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
	
	# MAXITER = 1
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

	model_gpflow.compiled_predict_f = tf.function(
		lambda Xnew: model_gpflow.predict_f(Xnew, full_cov=False, full_output_cov=True),
		# lambda xnew: model_gpflow.predict_f(xnew, full_cov=False, full_output_cov=True),
		input_signature=[tf.TensorSpec(shape=[None, D], dtype=tf.float64)],
	)

	# Save model:
	path2save_model = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_receding_gpflow"
	model_name = "model_{0:d}".format(my_seed)
	path2save_model_full = "{0:s}/{1:s}".format(path2save_model,model_name)
	logger.info("Saving model at {0:s} ...".format(path2save_model_full))
	tf.saved_model.save(model_gpflow, path2save_model_full)
	logger.info("Done!")

@hydra.main(config_path="./config",config_name="config")
def main_test_model(cfg: dict):

	my_seed = 12
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	# Get training data:
	path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"

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

	dim_X = dim_x + dim_u

	# Load model:
	path2save_model = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_receding_gpflow"
	model_name = "model_{0:d}".format(my_seed)
	path2save_model_full = "{0:s}/{1:s}".format(path2save_model,model_name)
	logger.info("Loading model from {0:s} ...".format(path2save_model))
	loaded_model = tf.saved_model.load(path2save_model_full)
	logger.info("Done!")


	# Trajectory selector:
	Ntrajs_for_sel = Xtrain.shape[0]//Nsteps
	Xtrain_for_sel = tf.reshape(Xtrain,(Ntrajs_for_sel,Nsteps,dim_X))
	Ytrain_for_sel = tf.reshape(Ytrain,(Ntrajs_for_sel,Nsteps,dim_x))
	ind_traj_selected = np.random.randint(low=0,high=Ntrajs_for_sel)
	zu_vec = Xtrain_for_sel[ind_traj_selected,...]
	zu_next_vec = Ytrain_for_sel[ind_traj_selected,...]
	z_vec = Xtrain_for_sel[ind_traj_selected,:,0:dim_x].numpy()
	u_vec = Xtrain_for_sel[ind_traj_selected,:,dim_x::].numpy()

	zu_vec_np = tf.cast(zu_vec,dtype=tf.float64)
	MO_mean_pred, var = loaded_model.compiled_predict_f(zu_vec_np)


	hdl_fig_pred, hdl_splots_pred = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_fig_pred.suptitle("Predictions ...", fontsize=16)
	hdl_splots_pred.plot(zu_vec[:,0],zu_vec[:,1],linestyle="-",color="grey",lw=2.0,label=r"Real traj - Input",alpha=0.3)
	hdl_splots_pred.plot(zu_next_vec[:,0],zu_next_vec[:,1],linestyle="-",color="navy",lw=2.0,label=r"Real traj - Next state",alpha=0.3)
	hdl_splots_pred.plot(MO_mean_pred[:,0],MO_mean_pred[:,1],linestyle="-",color="navy",lw=2.0,label=r"Predicted traj - Next dynamics",alpha=0.7)


	plt.show(block=True)
	plt.pause(1.)


	# Generate control sequence (u_vec) using the right nominal model, but then apply it to the changed dynamics.
	z_vec_changed_dyn = np.zeros((Nsteps,dim_x))
	z_vec_changed_dyn[0,:] = z_vec[0,:]
	for tt in range(Nsteps-1):

		use_nominal_model = alter_dynamics_flag_state_based(state_curr=z_vec_changed_dyn[tt:tt+1,:])
		# use_nominal_model = True

		# When using disturbance:
		z_vec_changed_dyn[tt+1:tt+2,:] = dyn_sys_true(state_vec=z_vec_changed_dyn[tt:tt+1,:],control_vec=u_vec[tt:tt+1,:],use_nominal_model=use_nominal_model,control_vec_prev=None)


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
	z_vec_changed_dyn_tf = tf.convert_to_tensor(value=z_vec_changed_dyn,dtype=tf.float32)
	u_vec_tf = tf.convert_to_tensor(value=u_vec,dtype=tf.float32)

	compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to = "nominal_model"
	# compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to = "altered_model"
	assert compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to in ["nominal_model","altered_model"]
	if compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to == "nominal_model":
		z_vec_real = z_vec_tf # [Nsteps,dim_out]
	if compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to == "altered_model":
		z_vec_real = z_vec_changed_dyn_tf # [Nsteps,dim_out]


	# Receding horizon predictions:
	plotting_receding_horizon_predictions = True
	savedata = True
	# recompute = True
	recompute = False
	path2save_receding_horizon = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_receding_gpflow"
	file_name = "trajs_ind_traj_{0:d}.pickle".format(ind_traj_selected)
	if plotting_receding_horizon_predictions and recompute:
		Nhorizon_rec = 50
		Nsteps_tot = z_vec_real.shape[0]-Nhorizon_rec
		loss_val_per_step = np.zeros(Nsteps_tot)
		x_traj_pred_all_vec = np.zeros((Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x))
		for tt in range(Nsteps_tot):

			x_traj_real_applied = z_vec_real[tt:tt+Nhorizon_rec,:]
			x_traj_real_applied_tf = tf.reshape(x_traj_real_applied,(1,Nhorizon_rec,dim_x))
			u_applied_tf = u_vec_tf[tt:tt+Nhorizon_rec,:]
			str_progress_bar = "Prediction with horizon = {0:d}; tt: {1:d} / {2:d} | ".format(Nhorizon_rec,tt+1,Nsteps_tot)
			loss_val_per_step[tt], x_traj_pred, y_traj_pred = rrtp_MO._get_negative_log_evidence_and_predictive_trajectory_chunk(x_traj_real_applied_tf,u_applied_tf,Nsamples=1,
																												Nrollouts=Nrollouts,str_progress_bar=str_progress_bar,from_prior=False,
																												scale_loss_entropy=scale_loss_entropy,
																												scale_prior_regularizer=scale_prior_regularizer,
																												sample_fx_once=True)
			x_traj_pred_all_vec[tt,...] = np.concatenate([x_traj_pred,y_traj_pred[:,-1::,:]],axis=1) # [Nsteps_tot,Nrollouts,Nhorizon_rec,self.dim_out]


		if savedata:
			data2save = dict(x_traj_pred_all_vec=x_traj_pred_all_vec,u_vec_tf=u_vec_tf,z_vec_real=z_vec_real,z_vec_tf=z_vec_tf,z_vec_changed_dyn_tf=z_vec_changed_dyn_tf,loss_val_per_step=loss_val_per_step)
			path2save_full = "{0:s}/{1:s}".format(path2save_receding_horizon,file_name)
			logger.info("Saving at {0:s} ...".format(path2save_full))
			file = open(path2save_full, 'wb')
			pickle.dump(data2save,file)
			file.close()
			return


	elif plotting_receding_horizon_predictions:

		# file_name = "trajs_ind_traj_46.pickle"

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
		hdl_splots_sampling_rec[0].plot(z_vec_changed_dyn_tf[:,0],z_vec_changed_dyn_tf[:,1],linestyle="-",color="navy",lw=2.0,label="With changed dynamics",alpha=0.15)
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



