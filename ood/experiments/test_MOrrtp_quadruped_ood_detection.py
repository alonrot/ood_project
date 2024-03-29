import tensorflow as tf
# import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import gpflow
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
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


# export PYTHONPATH=$PYTHONPATH:/Users/alonrot/work/code_projects_WIP/ood_project/ood/predictions_module/build
from predictions_interface import Predictions


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



def test_dbg_cpp_predictions_module(rrtp_MO,predictions_module,Nsteps_tot,Nhorizon_rec,when2sample):

	tt = 0

	str_progress_bar = "Prediction with horizon = {0:d}; tt: {1:d} / {2:d} | ".format(Nhorizon_rec,tt+1,Nsteps_tot)

	x_traj_real_applied = rrtp_MO.z_vec_real[tt:tt+Nhorizon_rec,:]
	x_traj_real_applied_tf = tf.reshape(x_traj_real_applied,(1,Nhorizon_rec,rrtp_MO.dim_out))
	u_applied_tf = rrtp_MO.u_traj_real[tt:tt+Nhorizon_rec,:]

	loss_val_new, x_traj_pred, y_traj_pred = rrtp_MO._get_negative_log_evidence_and_predictive_trajectory_chunk(x_traj_real_applied_tf,u_applied_tf,Nsamples=1,
																										Nrollouts=rrtp_MO.Nrollouts,str_progress_bar=str_progress_bar,from_prior=False,
																										scale_loss_entropy=rrtp_MO.scale_loss_entropy,
																										scale_prior_regularizer=rrtp_MO.scale_prior_regularizer,
																										when2sample=when2sample,
																										predictions_module=None)



	x0_tf = tf.convert_to_tensor(value=x_traj_real_applied_tf[0,0:1,:],dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
	u_applied_tf = tf.convert_to_tensor(value=u_applied_tf,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1

	xtraj_sampled_all_rollouts_list = predictions_module.run_all_rollouts_from_current_state(x0_tf.numpy().astype(dtype=np.float64),u_applied_tf.numpy().astype(dtype=np.float64))

	xtraj_sampled_all_rollouts = np.stack(xtraj_sampled_all_rollouts_list,axis=0)

	x_traj_pred_cpp = xtraj_sampled_all_rollouts[:,0:-2,:]
	y_traj_pred_cpp = xtraj_sampled_all_rollouts[:,1:-1,:]

	x_traj_pred_cpp = tf.convert_to_tensor(value=x_traj_pred_cpp,dtype=tf.float32)
	y_traj_pred_cpp = tf.convert_to_tensor(value=y_traj_pred_cpp,dtype=tf.float32)


	# pdb.set_trace()



	hdl_fig_pred, hdl_splots_pred = plt.subplots(1,figsize=(12,8),sharex=True)
	hdl_fig_pred.suptitle("Predictions ...", fontsize=16)

	hdl_splots_pred.plot(x_traj_pred[0,:,0],x_traj_pred[0,:,1])
	hdl_splots_pred.plot(x_traj_pred_cpp[0,:,0],x_traj_pred_cpp[0,:,1])

	plt.show(block=True)

	# pdb.set_trace()



def initialize_MOrrp_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,path2project,use_nominal_model_for_spectral_density=True):
	"""
	<<< Initialize GP model >>>
	"""
		# Spectral density:
	dim_in = dim_X
	dim_out = Ytrain.shape[1]
	spectral_density_list = [None]*dim_out
	# path2load = "{0:s}/data_quadruped_experiments_03_13_2023/learning_data_Nepochs4500.pickle".format(path2project) # not using deltas, trained in hybridrobotics
	# path2load = "{0:s}/data_quadruped_experiments_03_13_2023/learning_data_Nepochs300.pickle".format(path2project) # using deltas, trained on mac
	# path2load = "{0:s}/data_quadruped_experiments_03_13_2023/learning_data_Nepochs6000.pickle".format(path2project) # using deltas, trained on hybridrobotics
	# path2load = "{0:s}/data_quadruped_experiments_03_13_2023/learning_data_Nepochs6100.pickle".format(path2project) # using deltas, trained on hybridrobotics, cut data a bit
	path2load = "{0:s}/data_quadruped_experiments_03_13_2023/learning_data_Nepochs6200.pickle".format(path2project) # using deltas, trained on hybridrobotics, cut data a lot
	for jj in range(dim_out):
		spectral_density_list[jj] = QuadrupedSpectralDensity(cfg=cfg.spectral_density.quadruped,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
		spectral_density_list[jj].update_Wsamples_from_file(path2data=path2load,ind_out=jj)

	print("Initializing GP model ...")
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_X,cfg,spectral_density_list,Xtrain,Ytrain,using_deltas=using_deltas)
	return rrtp_MO


@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict):

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/predicted_trajs_55.pickle ./data_quadruped_experiments_03_13_2023/

	# my_seed = 60 # hybridrobotics, with value_init: 0.006
	# my_seed = 61 # mac, with value_init: 0.006
	# my_seed = 62 # mac, with value_init: 0.006, full time, 20 rollouts
	my_seed = 78
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments"
	else:
		path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"


	# path2data = "{0:s}/data_quadruped_experiments_03_13_2023/joined_go1trajs.pickle".format(path2project) # mac
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

	# Initialize GP model:
	dim_X = dim_x + dim_u
	which_kernel = "quadruped"
	# which_kernel = "matern"
	rrtp_MO = initialize_MOrrp_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,path2project,use_nominal_model_for_spectral_density=True)
	
	# Select test trajectory:
	ind_which_traj = 1
	zu_vec = state_and_control_full_list[ind_which_traj]
	z_vec = zu_vec[:,0:dim_x]
	u_vec = zu_vec[:,dim_x::]
	if using_deltas:
		z_next_vec = state_next_full_list[ind_which_traj] - state_and_control_full_list[ind_which_traj][:,0:dim_x]
	else:
		z_next_vec = state_next_full_list[ind_which_traj]

	# Predictions:
	MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(zu_vec)

	# Plotting:
	plotting_selected_trajs = False
	if plotting_selected_trajs and not using_hybridrobotics:
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
		Nsteps_tot = z_vec_real.shape[0] // 2
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


		Nhorizon_rec = 50
		Nsteps_tot = z_vec_real.shape[0]
		# Nsteps_tot = z_vec_real.shape[0] // 8
		Nepochs = 200
		Nrollouts = 40
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


	path2save_tensors = "{0:s}/data_quadruped_experiments_03_13_2023".format(path2project)
	tensors4predictions = rrtp_MO.export_tensors_needed_for_sampling_predictions_using_sampled_model_instances(path2save_tensors)

	dim_in = tensors4predictions["dim_in"]
	dim_out = tensors4predictions["dim_out"]
	phi_samples_all_dim = tensors4predictions["phi_samples_all_dim"]
	W_samples_all_dim = tensors4predictions["W_samples_all_dim"]
	mean_beta_pred_all_dim = tensors4predictions["mean_beta_pred_all_dim"]
	cov_beta_pred_chol_all_dim = tensors4predictions["cov_beta_pred_chol_all_dim"]


	# noise_mat = CommonUtils.sample_standard_multivariate_normal_inside_confidence_set(Nsamples=Nrollouts,Nels=phi_samples_all_dim.shape[1],min_prob_chi2=0.99)
	# noise_mat = np.random.randn(Nrollouts,phi_samples_all_dim.shape[1])
	# pdb.set_trace()

	# pdb.set_trace()
	noise_mat = rrtp_MO.sample_mv0[...,0]
	# pdb.set_trace()
	predictions_module = Predictions(dim_in,dim_out,phi_samples_all_dim,W_samples_all_dim,mean_beta_pred_all_dim,cov_beta_pred_chol_all_dim,noise_mat,Nrollouts,Nhorizon_rec)
	# predictions_module = None
	

	# Receding horizon predictions:
	savedata = True
	recompute = True
	# recompute = False
	path2save_receding_horizon = "{0:s}/data_quadruped_experiments_03_13_2023".format(path2project)
	if recompute:


		# test_dbg_cpp_predictions_module(rrtp_MO,predictions_module,Nsteps_tot,Nhorizon_rec,when2sample="once_per_class_instantiation")
		
		loss_avg, x_traj_pred_all_vec, loss_val_per_step = rrtp_MO.get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon(Nsteps_tot,Nhorizon_rec,when2sample="once_per_class_instantiation",predictions_module=predictions_module)


		if savedata:
			data2save = dict(x_traj_pred_all_vec=x_traj_pred_all_vec,u_vec_tf=u_vec_tf,z_vec_real=z_vec_real,z_vec_tf=z_vec_tf,z_vec_changed_dyn_tf=z_vec_changed_dyn_tf,loss_val_per_step=loss_val_per_step)
			file_name = "predicted_trajs_{0:d}.pickle".format(my_seed)
			path2save_full = "{0:s}/{1:s}".format(path2save_receding_horizon,file_name)
			logger.info("Saving at {0:s} ...".format(path2save_full))
			file = open(path2save_full, 'wb')
			pickle.dump(data2save,file)
			file.close()
			return

	else:

		# ==============================================================
		# With Quadruped data from data_quadruped_experiments_03_13_2023
		# ==============================================================
		# file_name = "predicted_trajs_50.pickle" # using deltas, reconstruction loss trained on mac, predictions done on mac
		# file_name = "predicted_trajs_51.pickle" # using deltas, reconstruction loss trained on mac, predictions done on mac, longer horizon, more noise
		# file_name = "predicted_trajs_52.pickle" # using deltas, reconstruction loss trained on mac, predictions done on mac, longer horizon, more noise
		# file_name = "predicted_trajs_53.pickle" # using deltas, reconstruction loss trained on hybridrobotics, predictions done on hybridrobotics, longer horizon, more noise
		# file_name = "predicted_trajs_55.pickle" # using deltas, reconstruction loss trained on hybridrobotics with different learning rates, predictions done on hybridrobotics, shorter horizon, same noise as above, cut a bit the beginning and the end of the traectories
		# file_name = "predicted_trajs_57.pickle" # using deltas, reconstruction loss trained on hybridrobotics with different learning rates, predictions done on hybridrobotics, shorter horizon, same noise as above, trimmed the data, cutting off the beginning and the end
		# file_name = "predicted_trajs_58.pickle" # dbg
		# file_name = "predicted_trajs_60.pickle" # dbg, hybridrobotics, with value_init: 0.006
		# file_name = "predicted_trajs_61.pickle" # dbg, mac, with value_init: 0.006
		# file_name = "predicted_trajs_62.pickle" # mac, with value_init: 0.0025, short horizon, looks good!!!
		# file_name = "predicted_trajs_63.pickle" # hybridrobotics, with value_init: 0.0025, looks good!!!
		# file_name = "predicted_trajs_64.pickle" # hybridrobotics, with value_init: 0.0025, longer horizon, looks good!!!
		# file_name = "predicted_trajs_66.pickle" # hybridrobotics, with value_init: 0.0025, looks good!!!
		# file_name = "predicted_trajs_68.pickle" # hybridrobotics, with value_init: 0.0025, change in input traj -> doesn't do too bad
		# file_name = "predicted_trajs_70.pickle" # c++ implementation 
		# file_name = "predicted_trajs_71.pickle" # dbg: comparison vs c++ implementation || This is the python version
		# file_name = "predicted_trajs_72.pickle" # dbg: comparison vs c++ implementation || This is the c++ version
		# file_name = "predicted_trajs_73.pickle" # dbg: comparison vs c++ implementation || This is the Python version
		# file_name = "predicted_trajs_73_cpp.pickle" # dbg: comparison vs c++ implementation || This is the cpp version
		# file_name = "predicted_trajs_74.pickle" # dbg: comparison vs c++ implementation || This is the cpp version
		# file_name = "predicted_trajs_75.pickle" # dbg: comparison vs c++ implementation || This is the cpp version
		# file_name = "predicted_trajs_76.pickle" # Working in cpp!
		file_name = "predicted_trajs_78.pickle" # Working in cpp!


		path2save_full = "{0:s}/{1:s}".format(path2save_receding_horizon,file_name)
		file = open(path2save_full, 'rb')
		data_dict = pickle.load(file)
		file.close()

		x_traj_pred_all_vec = data_dict["x_traj_pred_all_vec"] # [Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x]
		z_vec_tf = data_dict["z_vec_tf"]
		z_vec_changed_dyn_tf = data_dict["z_vec_changed_dyn_tf"]
		z_vec_real = data_dict["z_vec_real"]
		loss_val_per_step = data_dict["loss_val_per_step"]

		# pdb.set_trace()

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
		tt = 0
		hdl_plt_dubins_real, = hdl_splots_sampling_rec[0].plot(z_vec_real[tt,0],z_vec_real[tt,1],marker="*",markersize=14,color="darkgreen",label="Tracking experimental data - Quadruped")
		# hdl_splots_sampling_rec[0].set_xlim([-6.0,5.0])
		# hdl_splots_sampling_rec[0].set_ylim([-3.5,1.5])
		hdl_splots_sampling_rec[0].set_title("Tracking experimental data - Quadruped", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[0].set_xlabel(r"$x_1$", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[0].set_ylabel(r"$x_2$", fontsize=fontsize_labels)
		hdl_plt_predictions_list = []
		for ss in range(Nrollouts):
			# Nhor = 3
			Nhor = x_traj_pred_all_vec.shape[2]
			hdl_plt_predictions_list += hdl_splots_sampling_rec[0].plot(x_traj_pred_all_vec[0,ss,0:Nhor,0],x_traj_pred_all_vec[0,ss,0:Nhor,1],linestyle="-",color="darkorange",lw=0.5,label="Sampled trajs",alpha=0.5)

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
		plt_pause_sec = 0.005
		pdb.set_trace()
		

		for tt in range(Nsteps_tot):

			is_OoD = is_OoD_loss_based(loss_val_per_step[tt],thres_OoD)

			hdl_plt_dubins_real.set_markerfacecolor("red" if is_OoD else "green")
			hdl_plt_dubins_real.set_markeredgecolor("red" if is_OoD else "green")

			hdl_plt_dubins_real.set_xdata(z_vec_real[tt,0])
			hdl_plt_dubins_real.set_ydata(z_vec_real[tt,1])
			
			for ss in range(Nrollouts):
				# Nhor = 3
				Nhor = x_traj_pred_all_vec.shape[2]
				hdl_plt_predictions_list[ss].set_xdata(x_traj_pred_all_vec[tt,ss,0:Nhor,0])
				hdl_plt_predictions_list[ss].set_ydata(x_traj_pred_all_vec[tt,ss,0:Nhor,1])
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


	# Before conditioning (prior):
	plotting_dict["title_fig"] = "Predictions || Using prior, no training"
	loss_val, = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=False,plotting_dict=plotting_dict,Nrollouts=Nrollouts,from_prior=True)
	logger.info("loss_total (before conditioning; prior): {0:f}".format(loss_val))

	# plt.show(block=True)

	# # Train:
	# if train:
	# 	rrtp_MO.train_MOrrp_predictive()
	# else:
	# 	# rrtp_MO = assign_weights_v1(rrtp_MO,log_noise_std_per_dim,log_prior_variance_per_dim)
	# 	# rrtp_MO = assign_weights_v1(rrtp_MO,weights_list)
	# 	rrtp_MO = assign_weights_v2(rrtp_MO,weights_list)

	# After training to predict:
	plotting_dict["title_fig"] = "Predictions || Using posterior after training H-step ahead)"
	loss_val, = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=True,plotting_dict=plotting_dict,Nrollouts=Nrollouts)
	logger.info("loss_total (after training): {0:f}".format(loss_val))

	plt.show(block=True)

	# deprecated()


if __name__ == "__main__":

	main()



