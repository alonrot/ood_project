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
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity, QuadrupedSpectralDensity
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

def initialize_MOrrp_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,use_nominal_model_for_spectral_density=True):
	"""
	<<< Initialize GP model >>>
	"""

	assert which_kernel == "quadruped"
	
	# Spectral density:
	dim_in = dim_X
	dim_out = Ytrain.shape[1]
	spectral_density_list = [None]*dim_out
	# path2load = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/learning_data_Nepochs4500.pickle" # mac
	path2load = "/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/learning_data_Nepochs4500.pickle" # hybridrobotics
	for jj in range(dim_out):
		spectral_density_list[jj] = QuadrupedSpectralDensity(cfg=cfg.spectral_density.quadruped,cfg_sampler=cfg.sampler.hmc,dim=dim_in,integration_method="integrate_with_data",Xtrain=Xtrain,Ytrain=Ytrain[:,jj:jj+1])
		spectral_density_list[jj].update_Wsamples_from_file(path2data=path2load,ind_out=jj)
		# spectral_density_list[jj].update_Wsamples_from_file(path2load)


	# # Randomly sampled uniform grid:
	# omega_min = -0.314
	# omega_max = +0.134
	# Nsamples = 3125
	# spectral_density.update_Wsamples_uniform(omega_min,omega_max,Nsamples)

	

	print("Initializing GP model ...")
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_X,cfg,spectral_density_list,Xtrain,Ytrain)
	# rrtp_MO = MultiObjectiveReducedRankProcess(dim_X,cfg,spectral_density,Xtrain,Ytrain)
	# rrtp_MO.train_model()

	return rrtp_MO



@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict):

	my_seed = 14
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/joined_go1trajs.pickle" # mac
	path2data = "/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023/joined_go1trajs.pickle" # hybridrobotics
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	dim_x = Ytrain.shape[1]
	dim_u = Xtrain.shape[1] - Ytrain.shape[1]

	# Initialize GP model:
	dim_X = dim_x + dim_u
	which_kernel = "quadruped"
	# which_kernel = "matern"
	rrtp_MO = initialize_MOrrp_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,use_nominal_model_for_spectral_density=True)
	
	# Trajectory selector:
	# See dubins car version of this file...

	Nsteps = 510
	zu_vec = Xtrain[-Nsteps::,...]
	zu_next_vec = Ytrain[-Nsteps::,...]
	z_vec = Xtrain[-Nsteps::,0:dim_x]
	u_vec = Xtrain[-Nsteps::,dim_x::]

	MO_mean_pred, MO_std_pred = rrtp_MO.predict_at_locations(zu_vec)

	hdl_fig_pred, hdl_splots_pred = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_fig_pred.suptitle("Predictions ...", fontsize=16)
	hdl_splots_pred.plot(zu_vec[:,0],zu_vec[:,1],linestyle="-",color="grey",lw=2.0,label=r"Real traj - Input",alpha=0.3)
	hdl_splots_pred.plot(zu_next_vec[:,0],zu_next_vec[:,1],linestyle="-",color="navy",lw=2.0,label=r"Real traj - Next state",alpha=0.3)
	hdl_splots_pred.plot(MO_mean_pred[:,0],MO_mean_pred[:,1],linestyle="-",color="navy",lw=2.0,label=r"Predicted traj - Next dynamics",alpha=0.7)

	plt.show(block=False)
	plt.pause(1.)

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
	u_vec_tf = tf.convert_to_tensor(value=u_vec,dtype=tf.float32)
	z_vec_real = z_vec_tf
	z_vec_changed_dyn_tf = None
	rrtp_MO.update_dataset_predictive_loss(	z_vec_real=z_vec_tf,u_traj_real=u_vec_tf,Nhorizon=Nhorizon,
											learning_rate=learning_rate,epochs=epochs,stop_loss_val=stop_loss_val,
											scale_loss_entropy=scale_loss_entropy,scale_prior_regularizer=scale_prior_regularizer,
											Nrollouts=Nrollouts)

	# Visualize samples:
	plotting_visualize_samples = False
	if plotting_visualize_samples:
		plt_pause_sec = 1.0
		plt_samples_ylabels = [r"$x_1$",r"$x_2$",r"$\theta$"]
		hdl_fig_pred_sampling, hdl_splots_sampling = plt.subplots(dim_x,1,figsize=(12,8),sharex=True)
		Nchunks = Nsteps//Nhorizon
		z_vec_real_in_chunks = np.reshape(z_vec_real,(Nchunks,Nhorizon,dim_x)) # z_vec_real: [Nsteps,dim_out] || z_vec_real_in_chunks: [Nchunks,Nhorizon,dim_out]
		hdl_fig_pred_sampling.suptitle("Sampling trajectories ...", fontsize=16)
		time_steps = np.arange(1,z_vec_real_in_chunks.shape[1]+1)
		hdl_splots_sampling[-1].set_xlabel(r"Horizon time steps")
		for ii in range(Nchunks):

			for dd in range(dim_x):
				hdl_splots_sampling[dd].cla()
				hdl_splots_sampling[dd].plot(time_steps,z_vec_real_in_chunks[ii,:,dd],linestyle="-",color="navy",lw=2.0,label=r"Real traj",alpha=0.8)
				hdl_splots_sampling[dd].plot(time_steps,z_vec_real_in_chunks[ii,:,dd],linestyle="None",color="navy",marker=".",alpha=0.8,markersize=8)
				hdl_splots_sampling[dd].set_ylabel(plt_samples_ylabels[dd],fontsize=fontsize_labels)
				for ss in range(Nrollouts):
					hdl_splots_sampling[dd].plot(time_steps,x_traj_pred_chunks[ii,ss,:,dd],linestyle="-",color="navy",lw=1.0,label=r"Sampled trajs",alpha=0.2)

			plt.show(block=False)
			plt.pause(plt_pause_sec)

		plt.show(block=True)


	# Receding horizon predictions:
	plotting_receding_horizon_predictions = True
	savedata = True
	recompute = True
	# recompute = False
	# path2save_receding_horizon = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023" # mac
	path2save_receding_horizon = "/home/amarco/code_projects/ood_project/ood/experiments/data_quadruped_experiments_03_13_2023" # hybridrobotics
	file_name = "trajs_ind_traj_{0:d}.pickle".format(my_seed)
	if plotting_receding_horizon_predictions and recompute:
		Nhorizon_rec = 15
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

		file_name = "trajs_ind_traj_12.pickle" # from hybrid


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
		tt = 0
		hdl_plt_dubins_real, = hdl_splots_sampling_rec[0].plot(z_vec_real[tt,0],z_vec_real[tt,1],marker="*",markersize=14,color="darkgreen",label="Dubins car")
		# hdl_splots_sampling_rec[0].set_xlim([-6.0,5.0])
		# hdl_splots_sampling_rec[0].set_ylim([-3.5,1.5])
		hdl_splots_sampling_rec[0].set_title("Dubins car", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[0].set_xlabel(r"$x_1$", fontsize=fontsize_labels)
		hdl_splots_sampling_rec[0].set_ylabel(r"$x_2$", fontsize=fontsize_labels)
		hdl_plt_predictions_list = []
		for ss in range(Nrollouts):
			Nhor = 3
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
		plt_pause_sec = 0.01
		

		for tt in range(Nsteps_tot):

			is_OoD = is_OoD_loss_based(loss_val_per_step[tt],thres_OoD)

			hdl_plt_dubins_real.set_markerfacecolor("red" if is_OoD else "green")
			hdl_plt_dubins_real.set_markeredgecolor("red" if is_OoD else "green")

			hdl_plt_dubins_real.set_xdata(z_vec_real[tt,0])
			hdl_plt_dubins_real.set_ydata(z_vec_real[tt,1])
			
			for ss in range(Nrollouts):
				Nhor = 3
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

	# Train:
	if train:
		rrtp_MO.train_MOrrp_predictive()
	else:
		# rrtp_MO = assign_weights_v1(rrtp_MO,log_noise_std_per_dim,log_prior_variance_per_dim)
		# rrtp_MO = assign_weights_v1(rrtp_MO,weights_list)
		rrtp_MO = assign_weights_v2(rrtp_MO,weights_list)

	# After training to predict:
	plotting_dict["title_fig"] = "Predictions || Using posterior after training H-step ahead)"
	loss_val, = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=True,plotting_dict=plotting_dict,Nrollouts=Nrollouts)
	logger.info("loss_total (after training): {0:f}".format(loss_val))

	plt.show(block=True)

	# deprecated()


if __name__ == "__main__":

	main()



