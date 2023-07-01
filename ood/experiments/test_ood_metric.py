# import tensorflow as tf
# import tensorflow.compat.v2 as tf
# import tensorflow_probability as tfp
import gpflow
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, QuadrupedSpectralDensity, DubinsCarSpectralDensity
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
import sys

from test_dubin_car import get_sequence_of_feedback_gains_finite_horizon_LQR, rollout_with_finitie_horizon_LQR, generate_trajectories, generate_reference_trajectory

sys.path.append('/Users/alonrot/work/code_projects_WIP/catkin_real_robot_ws/src/unitree_ros_to_real_forked/unitree_legged_real/nodes/python')
from utils.generate_vel_profile import get_velocity_profile_given_waypoints, generate_random_set_of_waypoints, generate_waypoints_in_circle

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


def fun_dubins_car_dynamics(state_vec, control_vec, pars_model):

	deltaT = 0.01 # NOTE: move this up to user choices

	# Model parameters:
	c1 = pars_model["c1"]
	c2 = pars_model["c2"]
	c3 = pars_model["c3"]
	c4 = pars_model["c4"]
	c5 = pars_model["c5"]

	# State:
	x = state_vec[:,0:1]
	y = state_vec[:,1:2]
	th = state_vec[:,2:3]

	# Control input
	u1 = control_vec[:,0:1]
	u2 = control_vec[:,1:2]

	# Integrate dynamics:
	x_next = deltaT*u1*c1*np.cos(th) + x + c2
	y_next = deltaT*u1*c1*np.sin(th) + y + c3
	th_next = deltaT*u2*c4 + th + c5

	if np.any(np.isinf(x_next)):
	# if np.any(np.isinf(x_next)):
		print("x_next is inf")
		pdb.set_trace()

	if np.any(np.isinf(y_next)):
	# if np.any(np.isinf(y_next)):
		print("x_next is inf")
		pdb.set_trace()

	if np.any(np.isinf(th_next)):
	# if np.any(np.isinf(th_next)):
		print("x_next is inf")
		pdb.set_trace()

	xyth_next = np.array([np.squeeze(x_next),np.squeeze(y_next),np.squeeze(th_next)])

	if np.all(xyth_next == 0.0):
	# if np.all(xyth_next == 0.0):
		print("xyth_next is all zeroes")
		pdb.set_trace()

	return xyth_next

def rollout_many_for_given_control_sequence(state_curr,vel_seq,Nhor,Nrollouts):

	# Nominal parameters:
	pars_names = ["c1","c2","c3","c4","c5"]
	pars_nom = np.array([[1.,0.,0.,1.,0.]]).T

	# Randomize model parameters:
	pars_rand = pars_nom + abs(0.01*np.random.randn(len(pars_names),Nrollouts))

	pars_rand[0,:] = pars_nom[0,0] # c1
	pars_rand[1,:] = pars_nom[1,0] # c2
	pars_rand[2,:] = pars_nom[2,0] # c3
	pars_rand[3,:] = pars_nom[3,0] # c4
	# pars_rand[4,:] = pars_nom[4,0] # c5

	# Roll out models:
	states_mat_pred = np.zeros((Nrollouts,Nhor+1,state_curr.shape[1]))
	for rr in range(Nrollouts):
		pars_model = dict(zip(pars_names,pars_rand[:,rr]))
		# pdb.set_trace()
		states_mat_pred[rr,...] = rollout_once_model_for_given_control_sequence(state_curr,vel_seq,pars_model)
		# try:
		# except:
		# 	pdb.set_trace()

	return states_mat_pred # [Nrollouts,Nhor+1,D]


def rollout_once_model_for_given_control_sequence(state_curr,vel_seq,pars_model):

	Nhor = vel_seq.shape[0]
	states_mat_pred = np.zeros((Nhor+1,state_curr.shape[1]))
	states_mat_pred[0,:] = state_curr
	for tt in range(Nhor):
		states_mat_pred[tt+1,:] = fun_dubins_car_dynamics(states_mat_pred[tt:tt+1,:], vel_seq[tt:tt+1,:], pars_model)

	if np.any(np.isnan(states_mat_pred)):
		pdb.set_trace()

	return states_mat_pred # [Nhor+1,D]


def main():

	np.random.seed(1)

	Nwaypoints = 4
	xlim = [-2.0,1.2]
	ylim = [0.0,5.5]
	rate_freq_send_commands_for_trajs = 100 # Hz
	time_tot = Nwaypoints*10.0 # sec

	state_tot, vel_tot, pos_waypoints = generate_random_set_of_waypoints(Nwaypoints,xlim,ylim,rate_freq_send_commands_for_trajs,time_tot,block_plot=False,plotting=True)
	# state_tot: 	[Nsteps_tot,3]
	# vel_tot: 		[Nsteps_tot,2]

	# plt.show(block=True)

	Nsteps = vel_tot.shape[0]
	Nrollouts = 10
	Nhor = 5
	states_mat_pred_all_times = np.zeros((Nsteps,Nrollouts,Nhor+1,state_tot.shape[1]))
	for tt in range(Nsteps-Nhor):
		states_mat_pred_all_times[tt,...] = rollout_many_for_given_control_sequence(state_curr=state_tot[tt:tt+1,:],vel_seq=vel_tot[tt:tt+Nhor,:],Nhor=Nhor,Nrollouts=Nrollouts)

	# True system: (for now, just take the first)
	pars_names = ["c1","c2","c3","c4","c5"]
	pars_nom = np.array([[1.,0.,0.,1.,0.]]).T
	pars_model_nom = dict(zip(pars_names,pars_nom))
	
	# Roll out model:
	state0 = state_tot[0:1,:]
	states_true_sys = rollout_once_model_for_given_control_sequence(state_curr=state0,vel_seq=vel_tot,pars_model=pars_model_nom) # states_true_sys: [Nsteps,D]

	hdl_fig_pred_sampling_rec, hdl_splots_sampling_rec = plt.subplots(1,2,figsize=(17,7),sharex=False)
	# hdl_fig_pred_sampling_rec.suptitle("Simulated trajectory predictions ...", fontsize=fontsize_labels)
	# hdl_splots_sampling_rec[0].plot(z_vec_real[0:tt+1,0],z_vec_real[0:tt+1,1],linestyle="-",color="navy",lw=2.0,label="Real traj - nominal dynamics",alpha=0.3)
	hdl_splots_sampling_rec[0].plot(states_true_sys[:,0],states_true_sys[:,1],linestyle="-",color="navy",lw=2.0,label="With nominal dynamics",alpha=0.4)

	hdl_splots_sampling_rec[0].set_title("Tracking experimental data - Quadruped", fontsize=fontsize_labels)
	hdl_splots_sampling_rec[0].set_xlabel(r"$x_1$", fontsize=fontsize_labels)
	hdl_splots_sampling_rec[0].set_ylabel(r"$x_2$", fontsize=fontsize_labels)
	Nhor = states_mat_pred_all_times.shape[1]
	tt_steps = np.arange(0,states_mat_pred_all_times.shape[0],Nhor)
	for tt in tt_steps:
		for rr in range(Nrollouts):
			hdl_splots_sampling_rec[0].plot(states_mat_pred_all_times[tt,rr,:,0],states_mat_pred_all_times[tt,rr,:,1],linestyle="-",color="darkgray",lw=0.5,label="Sampled trajs",alpha=0.5)


	plt.show(block=True)

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




if __name__ == "__main__":

	main()

