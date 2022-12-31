import tensorflow as tf
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
from scipy import stats
import hydra
from omegaconf import OmegaConf
import pickle
import control
from lqrker.utils.parsing import get_logger
# from lqrker.models import MultiObjectiveReducedRankProcess
# from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity
from lqrker.utils.common import CommonUtils
logger = get_logger(__name__)
from min_jerk_gen import min_jerk

from test_dubin_car import get_sequence_of_feedback_gains_finite_horizon_LQR, rollout_with_finitie_horizon_LQR, generate_trajectories, generate_reference_trajectory

from test_MOrrtp_vanderpol import initialize_GPmodel_with_existing_data


markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict):

	# print(OmegaConf.to_yaml(cfg))

	my_seed = 4
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

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

		X, Y, deltaT, x0, ref_xt, ref_ut, Nsteps = generate_trajectories(ref_pars,Nsimus=Nsimus,plotting=False,include_ut_in_X=True,batch_nr=bb,block=True)

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

	# Initialize GP model:
	which_kernel = "dubinscar"
	# which_kernel = "matern"
	Xtrain = tf.convert_to_tensor(value=np.concatenate(X_tot,axis=0),dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=np.concatenate(Y_tot,axis=0),dtype=np.float32)
	dim_X = dim_x + dim_u
	rrtp_MO, Ndiv = initialize_GPmodel_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel)

	# New reference trajectory, for the true system to follow
	ref_pars = dict()
	ref_pars["rad"] = 2.0
	ref_pars["sign_Y"] = +1
	ref_pars["sign_xT"] = +1

	ref_xt_vec, ref_ut_vec = generate_reference_trajectory(ref_pars,Nsteps,deltaT)

	# Roll-out with a model-based controller that knows the true dynamics, following the above reference
	x0 = ref_xt_vec[0,:]
	x0_noise_std = 1.0
	x0_mod = x0 + np.array([0.2,0.2,0.5])*np.random.randn(3)*x0_noise_std
	T = 10.0
	z_vec, u_vec, t_vec = rollout_with_finitie_horizon_LQR(x0_mod,deltaT,T,Nsteps,ref_xt_vec,ref_ut_vec)


	# At every x_t, predict x_{t:t+H} states. 
	# Compare those predictions with the observed y_{t:t+H}. 
	# Compute OOD. 
	# Repeat for time t+H+1
	Nhorizon = 20
	# x0 = np.random.rand(1,dim_x)
	
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

		x_traj_real_applied = np.reshape(x_traj_real,(1,Nhorizon,Ytrain.shape[1]))
		loss_val, x_traj_pred, y_traj_pred = rrtp_MO.get_loss_gaussian_predictive(x_traj_real_applied,u_applied,Nsamples=1,Nrollouts=15,update_features=False)
		x_traj_pred_list += [x_traj_pred] # x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		logger.info("loss_val: {0:f}".format(loss_val))
		loss_val_vec[ii] = loss_val

		# x0_in = x_traj_real[0:1,:]
		# x_traj_pred, _ = rrtp_MO.sample_state_space_from_prior_recursively(x0=x0_in,Nsamples=Nsamples,Nrollouts=Nrollouts,u_traj=u_applied,traj_length=-1,sort=False,plotting=False)
		# x_traj_pred_list += [x_traj_pred] # x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]

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



