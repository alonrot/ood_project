import tensorflow as tf
# import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import pickle
import gpflow
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
import hydra
from omegaconf import OmegaConf
import pickle
import control
from lqrker.utils.parsing import get_logger
from lqrker.utils.common import CommonUtils
logger = get_logger(__name__)
from min_jerk_gen import min_jerk

from lqrker.spectral_densities import DubinsCarSpectralDensity
from test_dubin_car import get_sequence_of_feedback_gains_finite_horizon_LQR, rollout_with_finitie_horizon_LQR, generate_trajectories, generate_reference_trajectory

from test_MOrrtp_vanderpol import initialize_GPmodel_with_existing_data

dyn_sys_true = DubinsCarSpectralDensity._controlled_dubinscar_dynamics

# tf.compat.v1.enable_v2_behavior()


# tf.debugging.enable_check_numerics()

# tf.compat.v1.disable_eager_execution()

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


def get_training_data(save_data_dict=None,use_nominal_model=True):

	# print(OmegaConf.to_yaml(cfg))

	# Generate random points:
	name2save_base = "dubinscar_ood_detection"
	path2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/{}".format(name2save_base)
	ref_pars = dict()
	Nbatches = 10
	Nsimus = 20
	# plotting = True
	# Nbatches = 1
	# Nsimus = 2
	X_tot = [None]*Nbatches
	Y_tot = [None]*Nbatches
	ref_xt_list = [None]*Nbatches
	ref_ut_list = [None]*Nbatches

	for bb in range(Nbatches):

		# For each reference trajectory (randomdly sampled), generate a batch of Nsimus roll-outs. Each roll-out starts from a randomly sampled initial condition
		# Each roll-out is carried with a model-based controller that knows the true dynamics
		ref_pars["rad"] = 1.5 + 1.5*np.random.rand()
		ref_pars["sign_Y"] = 2*np.random.randint(low=0,high=2) - 1
		ref_pars["sign_xT"] = 1.0
		assert ref_pars["sign_xT"] == 1.0, "[DBG]: The controller fails to track the reference when ref_pars['sign_xT']=-1; investigate why"

		X, Y, deltaT, x0, ref_xt, ref_ut, Nsteps = generate_trajectories(ref_pars,Nsimus=Nsimus,plotting=False,include_ut_in_X=True,batch_nr=bb,block=True,use_nominal_model=use_nominal_model)

		assert deltaT == 0.01, "This is the deltaT used inside DubinsCarSpectralDensity(), and it should be kept this way"

		X_tot[bb] = X
		Y_tot[bb] = Y

		ref_xt_list[bb] = ref_xt
		ref_ut_list[bb] = ref_ut

	dim_x = Y.shape[1]
	dim_u = X.shape[1] - dim_x

	# Plot data:
	plotting_data = False
	if plotting_data:
		hdl_fig_data, hdl_splots_data = plt.subplots(1,1,figsize=(12,8),sharex=True)
		for bb in range(Nbatches):
			state_reshaped = tf.reshape(X_tot[bb][:,0:dim_x],[Nsimus,Nsteps-1,dim_x])
			hdl_splots_data.plot(ref_xt_list[bb][:,0],ref_xt_list[bb][:,1],color="firebrick",linestyle="-",marker="None",linewidth=2.,alpha=0.8)
			for ss in range(Nsimus):
				hdl_splots_data.plot(state_reshaped[ss,:,0],state_reshaped[ss,:,1],color="gray",linestyle="--",marker="None",linewidth=0.5,alpha=0.6)
				hdl_splots_data.plot(state_reshaped[ss,0,0],state_reshaped[ss,0,1],color="olivedrab",marker="o",markersize=3,linestyle="None",label="init")
				# hdl_splots_data.plot(state_reshaped[ss,-1,0],state_reshaped[ss,-1,1],color="sienna",marker="x",markersize=3,linestyle="None",label="end")
				hdl_splots_data.plot(state_reshaped[ss,-1,0],state_reshaped[ss,-1,1],color="black",marker="x",markersize=3,linestyle="None",label="end")

				if ss == 0 and bb == 0:
					hdl_splots_data.legend(markerscale=3.)
		
		hdl_splots_data.set_xticks([])
		hdl_splots_data.set_yticks([])
		hdl_fig_data.suptitle(r"Input data to GP model - Dubins car - Finite horizon LQR",fontsize=fontsize_labels)
		plt.show(block=True)

	Xtrain = tf.convert_to_tensor(value=np.concatenate(X_tot,axis=0),dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=np.concatenate(Y_tot,axis=0),dtype=np.float32)

	if save_data_dict is not None:
		if save_data_dict["save"]:
			data2save = dict(Xtrain=Xtrain,Ytrain=Ytrain,dim_x=dim_x,dim_u=dim_u,Nsteps=Nsteps,deltaT=deltaT)
			file = open(save_data_dict["path2data"], 'wb')
			pickle.dump(data2save,file)
			file.close()

	return Xtrain, Ytrain, dim_x, dim_u, Nsteps, deltaT



@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict):

	my_seed = 4
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	# Get training data:
	generate_data = False
	# generate_data = True
	path2data="/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model.pickle"
	# path2data="/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_disturbance_model.pickle"
	# save_data_dict = dict(save=True,path2data=path2data)
	save_data_dict = dict(save=False,path2data=path2data)
	# Create a TF dataset: https://www.tensorflow.org/datasets/add_dataset
	if generate_data:
		Xtrain, Ytrain, dim_x, dim_u, Nsteps, deltaT = get_training_data(save_data_dict,use_nominal_model=True)
	else:
		file = open(path2data, 'rb')
		data_dict = pickle.load(file)
		file.close()
		Xtrain = data_dict["Xtrain"]
		Ytrain = data_dict["Ytrain"]
		dim_x = data_dict["dim_x"]
		dim_u = data_dict["dim_u"]
		Nsteps = data_dict["Nsteps"]
		deltaT = data_dict["deltaT"]

	# tf.debugging.experimental.enable_dump_debug_info(
	# 	"/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/tfdbg2_logdir",
	# 	tensor_debug_mode="FULL_HEALTH",
	# 	circular_buffer_size=-1)

	# Initialize GP model:
	dim_X = dim_x + dim_u
	which_kernel = "dubinscar"
	# which_kernel = "matern"
	print("tf.executing_eagerly(): ",tf.executing_eagerly())
	# tf.compat.v1.disable_eager_execution() # https://www.activestate.com/resources/quick-reads/how-to-debug-tensorflow/
	# spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.dubinscar,cfg.sampler.hmc,dim=dim_X)
	rrtp_MO, Ndiv = initialize_GPmodel_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel)

	# pdb.set_trace()

	# New reference trajectory, for the true system to follow
	ref_pars = dict()
	ref_pars["rad"] = 2.0
	ref_pars["sign_Y"] = +1
	ref_pars["sign_xT"] = +1

	ref_xt_vec, ref_ut_vec = generate_reference_trajectory(ref_pars,Nsteps,deltaT)

	# Roll-out ONE trajectory with a model-based controller that knows the true dynamics, following the above reference
	x0 = ref_xt_vec[0,:]
	x0_noise_std = 1.0
	x0_mod = x0 + np.array([0.2,0.2,0.5])*np.random.randn(3)*x0_noise_std
	T = 10.0 # Not used, get rid of
	z_vec, u_vec, t_vec = rollout_with_finitie_horizon_LQR(x0_mod,deltaT,T,Nsteps,ref_xt_vec,ref_ut_vec,use_nominal_model=True)
	# z_vec: [Nsteps,dim_out]
	# u_vec: [Nsteps,dim_in]

	# Generate control sequence (u_vec) using the right nominal model, but then apply it to the changed dynamics.
	z_vec_changed_dyn = np.zeros((Nsteps,dim_x))
	z_vec_changed_dyn[0,:] = z_vec[0,:]
	for tt in range(Nsteps-1):
		z_vec_changed_dyn[tt+1:tt+2,:] = dyn_sys_true(state_vec=z_vec_changed_dyn[tt:tt+1,:],control_vec=u_vec[tt:tt+1,:],use_nominal_model=False)


	# pdb.set_trace()


	# At every x_t, predict x_{t:t+H} states. 
	# Compare those predictions with the observed y_{t:t+H}. 
	# Compute OOD. 
	# Repeat for time t+H+1
	Nhorizon = 20
	Nrollouts = 5
	# x0 = np.random.rand(1,dim_x)

	train = False
	# log_noise_std_per_dim = tf.constant([-2.4124577,-2.2396216,-2.4339094]) # 50 iters
	# log_prior_variance_per_dim = tf.constant([0.1828889,2.7030554,0.9408228]) # 50 iters

	# log_noise_std_per_dim = tf.constant([-3.9496725, -2.8052473, -4.253263]) # 115 iters
	# log_prior_variance_per_dim = tf.constant([-4.871695, 6.299472, -4.5974174]) # 115 iters

	# # Weights: wrong, using matern kernel ... 88 iterations
	# weights_list = []
	# weights_list += [dict(log_noise_std=-1.055355,log_prior_variance=-1.9315833,log_prior_mean_factor=6.438162)]
	# weights_list += [dict(log_noise_std=-1.589952,log_prior_variance=-1.0690501,log_prior_mean_factor=6.441733)]
	# weights_list += [dict(log_noise_std=+0.367157,log_prior_variance=+5.218601,log_prior_mean_factor=-1.8352405)]

	# Weights: 85 iterations
	weights_list = []
	weights_list += [dict(log_noise_std=-3.2072575,log_prior_variance=-2.3769002,log_prior_mean_factor=-2.1808307e-06)]
	weights_list += [dict(log_noise_std=-3.375641,log_prior_variance=5.378116,log_prior_mean_factor=3.5671384e-08)]
	weights_list += [dict(log_noise_std=-3.5728772,log_prior_variance=-1.7348373,log_prior_mean_factor=-1.6369743e-06)]


	# Prepare the training and its loss; the latter compares the true trajectory with the predicted one, in chunks.
	learning_rate = 5e-2
	epochs = 200
	# epochs = 2
	stop_loss_val = -1000.
	z_vec_tf = tf.convert_to_tensor(value=z_vec,dtype=tf.float32)
	z_vec_changed_dyn_tf = tf.convert_to_tensor(value=z_vec_changed_dyn,dtype=tf.float32)
	u_vec_tf = tf.convert_to_tensor(value=u_vec,dtype=tf.float32)
	rrtp_MO.update_dataset_predictive_loss(	z_vec_real=z_vec_changed_dyn_tf,u_traj_real=u_vec_tf,Nhorizon=Nhorizon,
											learning_rate=learning_rate,epochs=epochs,stop_loss_val=stop_loss_val)

	# plt.show(block=True)

	# Before training to predict:
	plotting_dict = dict(plotting=True,block_plot=False,title_fig="Predictions || Using prior, no training",ref_xt_vec=None,z_vec=None,z_vec_changed_dyn=None)
	plotting_dict["ref_xt_vec"] = ref_xt_vec
	plotting_dict["z_vec"] = z_vec
	plotting_dict["z_vec_changed_dyn"] = z_vec_changed_dyn
	# rrtp_MO.set_dbg_flag(True)
	loss_val = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=False,plotting_dict=plotting_dict,Nrollouts=Nrollouts)
	logger.info("loss_total (before training): {0:f}".format(loss_val))

	plt.show(block=True)

	# Before conditioning (prior):
	plotting_dict["title_fig"] = "Predictions || Using posterior, after training one-step ahead"
	loss_val = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=False,plotting_dict=plotting_dict,Nrollouts=Nrollouts,from_prior=True)
	logger.info("loss_total (before conditioning; prior): {0:f}".format(loss_val))

	# plt.show(block=True)

	# Train:
	if train:
		rrtp_MO.train_MOrrp_predictive()
	else:
		# rrtp_MO = assign_weights(rrtp_MO,log_noise_std_per_dim,log_prior_variance_per_dim)
		rrtp_MO = assign_weights(rrtp_MO,weights_list)

	# After training to predict:
	plotting_dict["title_fig"] = "Predictions || Using posterior after training H-step ahead)"
	loss_val = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=True,plotting_dict=plotting_dict,Nrollouts=Nrollouts)
	logger.info("loss_total (after training): {0:f}".format(loss_val))

	plt.show(block=True)

	# deprecated()

def assign_weights(rrtp_MO,weights_list):

	dd = 0
	for weights in weights_list:
		rrtp_MO.rrgpMO[dd].log_noise_std.assign(value=[weights["log_noise_std"]])
		rrtp_MO.rrgpMO[dd].log_prior_variance.assign(value=[weights["log_prior_variance"]])
		rrtp_MO.rrgpMO[dd].log_prior_mean_factor.assign(value=[weights["log_prior_mean_factor"]])
		dd += 1

	return rrtp_MO

def deprecated():

	
	x_traj_real_list = []
	x_traj_pred_list = []
	hdl_fig_pred, hdl_splots_pred = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots_pred.set_xlabel(r"$x_1$"); hdl_splots_pred.set_ylabel(r"$x_2$")
	loss_val_vec = np.zeros(Nsteps//Nhorizon)
	for ii in range(Nsteps//Nhorizon):

		logger.info("Iteration {0:d}".format(ii+1))

		# Extract chunk of real trajectory, to compare the predictions with:
		x_traj_real = z_vec[ii*Nhorizon:(ii+1)*Nhorizon,:]
		x_traj_real_list += [x_traj_real]

		u_applied = u_vec[ii*Nhorizon:(ii+1)*Nhorizon,:]

		# Negative log evidence (numpy/tensorflow selected inside)
		x_traj_real_applied = np.reshape(x_traj_real,(1,Nhorizon,Ytrain.shape[1]))
		x_traj_real_applied_tf = tf.convert_to_tensor(value=x_traj_real_applied,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
		u_applied_tf = tf.convert_to_tensor(value=u_applied,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
		loss_per_dim, x_traj_pred, y_traj_pred = rrtp_MO.get_negative_log_evidence_predictive(x_traj_real_applied_tf,u_applied_tf,Nsamples=1,Nrollouts=15,update_features=False)
		x_traj_pred_list += [x_traj_pred] # x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		loss_val = tf.math.reduce_sum(loss_per_dim)
		logger.info("loss_val: {0:f}".format(loss_val))
		loss_val_vec[ii] = loss_val
		# print("x_traj_pred:",str(x_traj_pred))

		# # Roll-outs tensorflow:
		# x0_tf = tf.convert_to_tensor(value=x_traj_real[0:1,:],dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
		# u_applied_tf = tf.convert_to_tensor(value=u_applied,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
		# x_traj_pred, _ = rrtp_MO.sample_state_space_from_prior_recursively_tf(x0=x0_tf,Nsamples=1,Nrollouts=15,u_traj=u_applied_tf,traj_length=-1,sort=False,plotting=False)
		# x_traj_pred_list += [x_traj_pred] # x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		# print("x_traj_pred:",str(x_traj_pred))

		# # Roll-outs numpy:
		# x0_in = x_traj_real[0:1,:]
		# x_traj_pred, _ = rrtp_MO.sample_state_space_from_prior_recursively(x0=x0_in,Nsamples=Nsamples,Nrollouts=Nrollouts,u_traj=u_applied,traj_length=-1,sort=False,plotting=False)
		# x_traj_pred_list += [x_traj_pred] # x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		# print("x_traj_pred:",str(x_traj_pred))

		# Plot stuff:
		hdl_splots_pred.plot(x_traj_real[:,0],x_traj_real[:,1],marker=".",linestyle="-",color="r",lw=1)
		# for ss in range(x_traj_pred.shape[2]):
		# 	hdl_splots_pred.plot(x_traj_pred[:,0,ss],x_traj_pred[:,1,ss],marker=".",linestyle="-",color="grey",lw=0.5)
		for ss in range(x_traj_pred.shape[0]):
			hdl_splots_pred.plot(x_traj_pred[ss,:,0],x_traj_pred[ss,:,1],marker=".",linestyle="-",color="grey",lw=0.5)


	logger.info("loss_val_vec: {0:s}".format(str(loss_val_vec)))
	logger.info("loss_total: {0:f}".format(np.sum(loss_val_vec)))

	plt.show(block=True)

if __name__ == "__main__":

	main()



