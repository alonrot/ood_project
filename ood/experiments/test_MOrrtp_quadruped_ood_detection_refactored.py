import tensorflow as tf
import gpflow
import pdb
import math
from datetime import datetime
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
assert using_deltas == True

path2folder = "data_quadruped_experiments_03_25_2023"


def fix_pickle_datafile(cfg,path2project,path2folder):

	"""
	NOTE: This piece of code needed to be called only once, to fix the pickle file.
	It genrated the file /Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/from_hybridrob/reconstruction_data_2023_03_27_01_12_35.pickle
	"""
	raise NotImplementedError

	file_name = "reconstruction_data_2023_03_26_21_55_08.pickle" # Trained model on hybridrob for 50000 iters; data subsampled at 10 Hz
	path2load_full = "{0:s}/{1:s}/from_hybridrob/{2:s}".format(path2project,path2folder,file_name)
	file = open(path2load_full, 'rb')
	data_dict = pickle.load(file)
	file.close()

	# omegas_trainedNN = tf.convert_to_tensor(data_dict["omegas_trainedNN"],dtype=tf.float32)
	# Sw_omegas_trainedNN = tf.convert_to_tensor(data_dict["Sw_omegas_trainedNN"],dtype=tf.float32)
	# varphi_omegas_trainedNN = tf.convert_to_tensor(data_dict["varphi_omegas_trainedNN"],dtype=tf.float32)
	# delta_omegas_trainedNN = tf.convert_to_tensor(data_dict["delta_omegas_trainedNN"],dtype=tf.float32)
	# delta_statespace_trainedNN = tf.convert_to_tensor(data_dict["delta_statespace_trainedNN"],dtype=tf.float32)
	
	path2data = "{0:s}/data_quadruped_experiments_03_25_2023/joined_go1trajs_trimmed_2023_03_25.pickle".format(path2project)
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict4spectral = pickle.load(file)
	file.close()

	Xtrain = data_dict4spectral["Xtrain"]
	Ytrain = data_dict4spectral["Ytrain"]
	state_and_control_full_list = data_dict4spectral["state_and_control_full_list"]
	state_next_full_list = data_dict4spectral["state_next_full_list"]

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

	# Spectral density:
	spectral_density_list = [None]*dim_out
	for jj in range(dim_out):
		spectral_density_list[jj] = QuadrupedSpectralDensity(cfg=cfg.spectral_density.quadruped,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
		spectral_density_list[jj].update_Wsamples_from_file(path2data=path2load_full,ind_out=jj)


	data_dict.update(spectral_density_list=spectral_density_list,
					omega_lim=5.0,
					Nsamples_omega=1500,
					Xtrain=Xtrain,
					Ytrain=Ytrain,
					state_and_control_full_list=state_and_control_full_list,
					state_next_full_list=state_next_full_list,
					path2data=path2data)


	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	file_name = "reconstruction_data_{0:s}.pickle".format(name_file_date) # Trained model on hybridrob for 50000 iters; data subsampled at 10 Hz
	path2load_full = "{0:s}/{1:s}/from_hybridrob/{2:s}".format(path2project,path2folder,file_name)
	file = open(path2load_full, 'wb')
	logger.info("Saving data at {0:s} ...".format(path2load_full))
	pickle.dump(data_dict,file)
	file.close()
	logger.info("Done!")

	pdb.set_trace()

	

def compute_predictions(cfg):


	name_file_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	my_seed = 78
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)


	using_hybridrobotics = cfg.gpmodel.using_hybridrobotics
	logger.info("using_hybridrobotics: {0:s}".format(str(using_hybridrobotics)))

	path2project = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments"
	if using_hybridrobotics:
		path2project = "/home/amarco/code_projects/ood_project/ood/experiments" 

	# fix_pickle_datafile(cfg,path2project,path2folder)

	# file_name = "reconstruction_data_2023_03_26_21_55_08.pickle" # Trained model on hybridrob for 50000 iters; data subsampled at 10 Hz
	file_name = "reconstruction_data_2023_03_27_01_23_40.pickle" # Trained model on hybridrob for 50000 iters; data subsampled at 10 Hz || Completed the missing fields using the above function fix_pickle_datafile()
	path2load_full = "{0:s}/{1:s}/from_hybridrob/{2:s}".format(path2project,path2folder,file_name)
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


	ind_which_traj = 0
	z_vec_real = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,0:dim_x],dtype=tf.float32)
	u_vec_tf = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj][:,dim_x::],dtype=tf.float32)
	zu_vec = tf.convert_to_tensor(value=state_and_control_full_list[ind_which_traj],dtype=tf.float32)

	# Predictions:
	MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(zu_vec)
	# pdb.set_trace()
	deltas_real = state_next_full_list[ind_which_traj] - state_and_control_full_list[ind_which_traj][:,0:dim_x]

	plot_state_transition_reconstruction = False
	if plot_state_transition_reconstruction:
		hdl_fig, hdl_splots_next_state = plt.subplots(dim_out,1,figsize=(16,14),sharex=False,sharey=False)
		hdl_fig.suptitle(r"State transition - Reconstructed; $\Delta x_{t+1,d} = f_d(x_t)$",fontsize=fontsize_labels)
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
		# 
		lgnd = hdl_splots_next_state[-1,0].legend(loc="best",fontsize=fontsize_labels)
		lgnd.legendHandles[0]._legmarker.set_markersize(20)
		lgnd.legendHandles[1]._legmarker.set_markersize(20)

		plt.show(block=True)


	if using_hybridrobotics:
		# Nhorizon_rec = 40
		Nhorizon_rec = 15
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

		Nhorizon_rec = 30
		Nsteps_tot = 40
		# Nsteps_tot = z_vec_real.shape[0]
		# Nsteps_tot = z_vec_real.shape[0] // 8
		Nepochs = 200
		Nrollouts = 15
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


	noise_mat = rrtp_MO.sample_mv0[...,0]
	predictions_module = Predictions(dim_in,dim_out,phi_samples_all_dim,W_samples_all_dim,mean_beta_pred_all_dim,cov_beta_pred_chol_all_dim,noise_mat,Nrollouts,Nhorizon_rec)
	# predictions_module = None
	
	# Receding horizon predictions:
	savedata = True
	
	loss_avg, x_traj_pred_all_vec, loss_val_per_step = rrtp_MO.get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon(Nsteps_tot,Nhorizon_rec,when2sample="once_per_class_instantiation",predictions_module=predictions_module)

	if savedata:
		data2save = dict(x_traj_pred_all_vec=x_traj_pred_all_vec,u_vec_tf=u_vec_tf,z_vec_real=z_vec_real,loss_val_per_step=loss_val_per_step)
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
	hdl_splots_sampling_rec[0].plot(z_vec_real[:,0],z_vec_real[:,1],linestyle="-",color="navy",lw=2.0,label="With nominal dynamics",alpha=0.7)
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


@hydra.main(config_path="./config",config_name="config")
def main(cfg):

	compute_predictions(cfg)


	# ==============================================================
	# With Quadruped data from data_quadruped_experiments_03_25_2023
	# ==============================================================
	# file_name = "predicted_trajs_2023_03_27_02_02_52.pickle"
	# file_name = "predicted_trajs_2023_03_27_02_31_51.pickle"
	# plot_predictions(cfg,file_name)




if __name__ == "__main__":

	main()

	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/predicted_trajs_55.pickle ./data_quadruped_experiments_03_25_2023/
	# export PYTHONPATH=$PYTHONPATH:/Users/alonrot/work/code_projects_WIP/ood_project/ood/predictions_module/build

	# scp -P 4444 -r ./data_quadruped_experiments_03_25_2023/from_hybridrob/reconstruction_data_2023_03_27_01_23_40.pickle amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/from_hybridrob/


	# scp -P 4444 -r amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_25_2023/predicted_trajs_2023_03_27_02_31_51.pickle ./data_quadruped_experiments_03_25_2023/
