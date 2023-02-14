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
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity
from lqrker.models import MultiObjectiveReducedRankProcess
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

# from test_MOrrtp_vanderpol import initialize_GPmodel_with_existing_data

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


def initialize_MOrrp_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,use_nominal_model_for_spectral_density=True):
	"""
	<<< Initialize GP model >>>
	"""
	
	# dim_y = dim_x

	print("Initializing Spectral density ...")
	if which_kernel == "vanderpol":
		spectral_density = VanDerPolSpectralDensity(cfg.spectral_density.vanderpol,cfg.sampler.hmc,dim=dim_X)
	elif which_kernel == "matern":
		spectral_density = MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_X)
	elif which_kernel == "dubinscar":
		spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.dubinscar,cfg.sampler.hmc,dim=dim_X,use_nominal_model=use_nominal_model_for_spectral_density)
	else:
		raise ValueError("Possibilities: [vanderpol,matern,dubinscar]")


	# Regular grid:
	# omega_min = -6.
	# omega_max = +6.
	# Ndiv = 31
	# cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_x
	# spectral_density.update_Wpoints_regular(omega_min,omega_max,Ndiv)


	# Discrete grid:
	# L = 200.0; Ndiv = 5 # 5**5=3125 # works
	L = 50.0; Ndiv = 3 # 3**5=243
	# L = 10.0; Ndiv = 5 # 5**5=3125
	# L = 10.0; Ndiv = 7 # 7**5=16807
	# L = 30.0
	# Ndiv = 61
	cfg.gpmodel.hyperpars.weights_features.Nfeat = Ndiv**dim_X
	spectral_density.update_Wpoints_discrete(L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False)



	# # Randomly sampled uniform grid:
	# omega_min = -0.314
	# omega_max = +0.134
	# Nsamples = 3125
	# spectral_density.update_Wsamples_uniform(omega_min,omega_max,Nsamples)

	

	print("Initializing GP model ...")
	rrtp_MO = MultiObjectiveReducedRankProcess(dim_X,cfg,spectral_density,Xtrain,Ytrain)
	# rrtp_MO.train_model()

	return rrtp_MO



def get_training_data_from_vel_profile(save_data_dict=None,use_nominal_model=True,z0_mean=np.zeros(3)):
	"""

	TODO:

	1) Add mean to z0
	2) Figure out return arguments
	3) Wrap entire thing into a callable and generate data for different z0_mean

	Questions:
		1) If I train S() and varphi() with 90% of the dataset and then test model preditctions in the reaming 10%, how well does it do?
		2) How can address the ELBO training for better predictions?

	"""


	dyn_sys_true = DubinsCarSpectralDensity._controlled_dubinscar_dynamics

	deltaT = 0.01
	assert deltaT == 0.01, "This is the deltaT used inside DubinsCarSpectralDensity(), and it should be kept this way"

	Nsteps = 1001

	# Generate randomly the middle points of vel_lin_way_points, vel_ang_way_points, and also z0
	Ntrajectories = 100
	Nwaypoints = 3 + 2
	vel_lin_way_points = np.zeros((Ntrajectories,Nwaypoints))
	vel_ang_way_points = np.zeros((Ntrajectories,Nwaypoints))

	vel_lin_max = 2.0 # [m/s]
	vel_lin_way_points[:,1:-1] = vel_lin_max*np.random.rand(Ntrajectories,Nwaypoints-2)

	vel_ang_max = 90 * math.pi/180. # [rad/sec]
	vel_ang_way_points[:,1:-1] = -vel_ang_max + 2.*vel_ang_max*np.random.rand(Ntrajectories,Nwaypoints-2)

	vel_points = np.stack((vel_lin_way_points,vel_ang_way_points),axis=2)

	vel_profile_batch = np.zeros((Ntrajectories,Nsteps,2))
	for ii in range(Ntrajectories):
		vel_profile_batch[ii,...],_ = min_jerk(pos=vel_points[ii,...], dur=Nsteps, vel=None, acc=None, psg=None) # [Nsteps, D]


	# Initial conditions:
	z0_x_lim = 0.1; z0_y_lim = 0.1; z0_th_lim = math.pi; 
	z0_lim = np.reshape(np.array([z0_x_lim,z0_y_lim,z0_th_lim]),(1,-1))
	z0_vec = -z0_lim + 2.*z0_lim*np.random.rand(Ntrajectories,3)

	z_vec_true = np.zeros((Ntrajectories,Nsteps,3))
	u_vec = vel_profile_batch
	for ii in range(Ntrajectories):

		z_vec_true[ii,0,:] = z0_vec[ii,:]
		for tt in range(Nsteps-1):
			z_vec_true[ii,tt+1:tt+2,:] = dyn_sys_true(state_vec=z_vec_true[ii,tt:tt+1,:],control_vec=u_vec[ii,tt:tt+1,:],use_nominal_model=use_nominal_model)

	# Velocity profiles:
	hdl_fig_data, hdl_splots_data = plt.subplots(3,1,figsize=(12,8),sharex=True)
	time_vec = np.arange(0,Nsteps)*deltaT
	for ii in range(Ntrajectories):
		hdl_splots_data[0].plot(time_vec,u_vec[ii,:,0],lw=1,alpha=0.3,color="navy")
		hdl_splots_data[1].plot(time_vec,u_vec[ii,:,1],lw=1,alpha=0.3,color="navy")
		hdl_splots_data[2].plot(u_vec[ii,:,0],u_vec[ii,:,1],lw=1,alpha=0.3,color="navy")
	hdl_splots_data[1].set_xlabel(r"$t$ [sec]",fontsize=fontsize_labels)
	hdl_splots_data[0].set_ylabel(r"$v$ [m/s]",fontsize=fontsize_labels)
	hdl_splots_data[1].set_ylabel(r"$\omega$ [rad/s]",fontsize=fontsize_labels)


	# Trajectories:
	hdl_fig_data, hdl_splots_data = plt.subplots(1,1,figsize=(12,8),sharex=False)
	for ii in range(Ntrajectories):
		hdl_splots_data.plot(z_vec_true[ii,:,0],z_vec_true[ii,:,1],alpha=0.3,color="navy")
		hdl_splots_data.plot(z_vec_true[ii,-1,0],z_vec_true[ii,-1,1],marker="x",color="black",markersize=5)
		hdl_splots_data.plot(z_vec_true[ii,0,0],z_vec_true[ii,0,1],marker=".",color="green",markersize=3)
	hdl_splots_data.set_xlabel(r"$x$ [m]",fontsize=fontsize_labels)
	hdl_splots_data.set_ylabel(r"$y$ [m]",fontsize=fontsize_labels)


	plt.show(block=True)



def get_training_data_from_waypoints(save_data_dict=None,use_nominal_model=True):
	"""
	

	:return:
	Xtrain: [(Nsteps-3)*Ntrajs, dim_x+dim_u]
	Ytrain: [(Nsteps-3)*Ntrajs, dim_x]
	"""

	dyn_sys_true = DubinsCarSpectralDensity._controlled_dubinscar_dynamics

	deltaT = 0.01
	assert deltaT == 0.01, "This is the deltaT used inside DubinsCarSpectralDensity(), and it should be kept this way"

	Nsteps = 201+2 # We add 2 because numerical differentiation will suppress 2 points

	# Generate randomly the middle points of vel_lin_way_points, vel_ang_way_points, and also z0
	Ntrajs = 3000
	Nwaypoints = 4
	x_lim = [0.0,5.0]; y_lim = [-5.0,5.0]; 
	# th_lim_low = [-math.pi/2., math.pi/2.]; # Why do we not need the heading? Because it's inferred from [x(t),y(t)] as th = arctan(yd(t) / xd(t)), where xd(t) = d/dt x(t)
	# s_lim = np.reshape(np.array([x_lim;y_lim;th_lim]),(2,-1))
	s_lim = np.vstack((x_lim,y_lim))
	pos_waypoints = s_lim[:,0] + (s_lim[:,1]-s_lim[:,0])*np.random.rand(Ntrajs,Nwaypoints,2)

	# Set the initial position to zero without loss of generality:
	pos_waypoints[:,0,:] = np.zeros(2)

	# Force the final position to be at a random value for x in [4,5]:
	pos_waypoints[:,-1,0] = 4. + (5.-4.)*np.random.rand(Ntrajs)

	# # Initial conditions:
	# z0_x_lim = 0.1; z0_y_lim = 0.1; z0_th_lim = math.pi;
	# z0_lim = np.reshape(np.array([z0_x_lim,z0_y_lim,z0_th_lim]),(1,-1))
	# z0_vec = -z0_lim + 2.*z0_lim*np.random.rand(Ntrajs,3)

	# Sort out the x positions in increasing order to prevent weird turns:
	pos_waypoints[:,:,0] = np.sort(pos_waypoints[:,:,0],axis=1)

	pos_profile_batch = np.zeros((Ntrajs,Nsteps,2))
	for ii in range(Ntrajs):
		if ii % 10 == 0: logger.info(" Generating trajectory {0:d} / {1:d} ...".format(ii+1,Ntrajs))
		pos_profile_batch[ii,...],_ = min_jerk(pos=pos_waypoints[ii,...], dur=Nsteps, vel=None, acc=None, psg=None) # [Nsteps, D]


	# Velocity profiles and heading with numerical differentiation:
	vel_profile_batch = np.zeros((Ntrajs,Nsteps-1,2))
	th_profile_batch = np.zeros((Ntrajs,Nsteps-1,1))
	th_vel_profile_batch = np.zeros((Ntrajs,Nsteps-2,1))
	vel_profile_batch[...,0] = np.diff(pos_profile_batch[...,0],axis=1) / deltaT
	vel_profile_batch[...,1] = np.diff(pos_profile_batch[...,1],axis=1) / deltaT
	vel_mod_profile_batch = np.sqrt(vel_profile_batch[...,0:1]**2 + vel_profile_batch[...,1:2]**2)
	th_profile_batch[...,0] = np.arctan2(vel_profile_batch[...,1], vel_profile_batch[...,0])
	th_vel_profile_batch[...,0] = np.diff(th_profile_batch[...,0],axis=1) / deltaT

	# Subselect those very smooth ones:
	vx_is_positive = np.all(vel_profile_batch[...,0] >= -0.5,axis=1)
	th_is_within_range = np.all(abs(th_profile_batch[...,0]) <= 0.98*math.pi/2.,axis=1)
	ind_smooth = np.arange(0,Ntrajs)[vx_is_positive & th_is_within_range]

	logger.info("Smooth trajectories: {0:d} / {1:d}".format(len(ind_smooth),Ntrajs))

	Nsteps_tot = th_vel_profile_batch.shape[1]
	state_tot = np.concatenate((pos_profile_batch[ind_smooth,0:Nsteps_tot,:],th_profile_batch[ind_smooth,0:Nsteps_tot,:]),axis=2) # [Ntrajs,Nsteps_tot,3], with Nsteps_tot=Nsteps-2 due to the integration issues
	vel_tot = np.concatenate((vel_mod_profile_batch[ind_smooth,0:Nsteps_tot,:],th_vel_profile_batch[ind_smooth,:,:]),axis=2) # [Ntrajs,Nsteps_tot,2], with Nsteps_tot=Nsteps-2 due to the integration issues

	# Get a round number of trajectories:
	# pdb.set_trace()
	# if state_tot.shape[0] > 300 and vel_tot.shape[0] > 300:
	# 	state_tot = state_tot[0:300,...]
	# 	vel_tot = vel_tot[0:300,...]

	state_and_control_tot = np.concatenate((state_tot,vel_tot),axis=2)

	# The data needs to be reshaped in this particular way; it's incorrect to do this: Xtot = np.reshape(Ntrajs*(Nsteps-3),dim_x+dim_u); Xtrain_np = Xtot[0:-1,:]; Ytrain_np = Xtot[1::,:]
	Xtrain_np = np.reshape(state_and_control_tot[:,0:-1,:],(state_and_control_tot.shape[0]*(state_and_control_tot.shape[1]-1),state_and_control_tot.shape[2]),order="C") # order="C" -> last axis index changing fastest
	Ytrain_np = np.reshape(state_tot[:,1::,:],(state_tot.shape[0]*(state_tot.shape[1]-1),state_tot.shape[2]),order="C") # order="C" -> last axis index changing fastest

	Xtrain = tf.convert_to_tensor(value=Xtrain_np,dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=Ytrain_np,dtype=np.float32)

	# Velocity profiles:
	hdl_fig_data, hdl_splots_data = plt.subplots(4,1,figsize=(12,8),sharex=True)
	time_vec = np.arange(0,Nsteps-1)*deltaT
	for ii in ind_smooth:
		hdl_splots_data[0].plot(time_vec,vel_profile_batch[ii,:,0],lw=1,alpha=0.3,color="navy")
		hdl_splots_data[1].plot(time_vec,vel_profile_batch[ii,:,1],lw=1,alpha=0.3,color="navy")
		hdl_splots_data[2].plot(time_vec,vel_mod_profile_batch[ii,:],lw=1,alpha=0.3,color="navy")
		hdl_splots_data[3].plot(time_vec[0:-1],th_vel_profile_batch[ii,:,0],lw=1,alpha=0.3,color="navy")
	hdl_splots_data[2].set_xlabel(r"$t$ [sec]",fontsize=fontsize_labels)
	hdl_splots_data[0].set_ylabel(r"$v_x$ [m/s]",fontsize=fontsize_labels)
	hdl_splots_data[1].set_ylabel(r"$v_y$ [m/s]",fontsize=fontsize_labels)
	hdl_splots_data[2].set_ylabel(r"$v$ [m/s]",fontsize=fontsize_labels)
	hdl_splots_data[3].set_ylabel(r"$\dot{\theta}$ [rad/s]",fontsize=fontsize_labels)


	# Trajectories:
	hdl_fig_data, hdl_splots_data = plt.subplots(1,1,figsize=(12,8),sharex=False)
	for ii in ind_smooth:
		hdl_splots_data.plot(pos_profile_batch[ii,:,0],pos_profile_batch[ii,:,1],alpha=0.3,color="navy")
		hdl_splots_data.plot(pos_profile_batch[ii,-1,0],pos_profile_batch[ii,-1,1],marker="x",color="black",markersize=5)
		hdl_splots_data.plot(pos_profile_batch[ii,0,0],pos_profile_batch[ii,0,1],marker=".",color="green",markersize=3)
	hdl_splots_data.set_xlabel(r"$x$ [m]",fontsize=fontsize_labels)
	hdl_splots_data.set_ylabel(r"$y$ [m]",fontsize=fontsize_labels)


	dim_x = state_tot.shape[2]; dim_u = vel_tot.shape[2]; Nsteps = state_tot.shape[1]-1
	if save_data_dict is not None:
		if save_data_dict["save"]:
			data2save = dict(Xtrain=Xtrain,Ytrain=Ytrain,dim_x=dim_x,dim_u=dim_u,Nsteps=Nsteps,Ntrajs=Ntrajs,deltaT=deltaT)
			file = open(save_data_dict["path2data"], 'wb')
			pickle.dump(data2save,file)
			file.close()
	else:
		plt.show(block=True)

	return Xtrain, Ytrain, dim_x, dim_u, Nsteps, Ntrajs, deltaT


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

def merge_data(path2data_list):

	Xtrain = []; Ytrain = []
	for path in path2data_list:
		file = open(path, 'rb')
		data_dict = pickle.load(file)
		file.close()
		Xtrain += [data_dict["Xtrain"]]
		Ytrain += [data_dict["Ytrain"]]
		dim_x = data_dict["dim_x"]
		dim_u = data_dict["dim_u"]
		Nsteps = data_dict["Nsteps"]
		deltaT = data_dict["deltaT"]

	Xtrain = tf.concat(Xtrain,axis=0)
	Ytrain = tf.concat(Ytrain,axis=0)
	data2save = dict(Xtrain=Xtrain,Ytrain=Ytrain,dim_x=dim_x,dim_u=dim_u,Nsteps=Nsteps,deltaT=deltaT)
	path2data_save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_merged.pickle"
	logger.info("Saving to {0:s} ...".format(path2data_save))
	file = open(path2data_save, 'wb')
	pickle.dump(data2save,file)
	file.close()


@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict):

	my_seed = 4
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	# get_training_data_from_waypoints(save_data_dict=None,use_nominal_model=True)

	merge_existing_data = False
	if merge_existing_data:
		path2data_list = ["/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model.pickle"]
		path2data_list += ["/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_disturbance_model_2.pickle"]
		merge_data(path2data_list)
		return;

	# Get training data:
	generate_data = False
	# generate_data = True
	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints.pickle"
	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter.pickle"
	path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs.pickle"
	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model.pickle" # I accidentally overwrote this
	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_disturbance_model.pickle" # additive in linear velocity; distur = -2.0
	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_disturbance_model_2.pickle" # multiplicative in linear and angular velocities; distur_v = 2.0; distur_w = 3.0
	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_merged.pickle" # merged
	assert path2data != "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model.pickle", "I accidentally overwrote this"
	save_data_dict = dict(save=generate_data,path2data=path2data)
	# Create a TF dataset: https://www.tensorflow.org/datasets/add_dataset
	if generate_data:
		logger.info("Generating new data ... || path2data = {0:s} || save = {1:s}".format(save_data_dict["path2data"],str(save_data_dict["save"])))
		Xtrain, Ytrain, dim_x, dim_u, Nsteps, Ntrajs, deltaT = get_training_data_from_waypoints(save_data_dict,use_nominal_model=True)
		# Xtrain, Ytrain, dim_x, dim_u, Nsteps, deltaT = get_training_data(save_data_dict,use_nominal_model=False)
		# get_training_data_from_vel_profile(save_data_dict=None,use_nominal_model=True)
	else:
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

	# tf.debugging.experimental.enable_dump_debug_info(
	# 	"/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/tfdbg2_logdir",
	# 	tensor_debug_mode="FULL_HEALTH",
	# 	circular_buffer_size=-1)
	# 	

	# Initialize GP model:
	dim_X = dim_x + dim_u
	which_kernel = "dubinscar"
	# which_kernel = "matern"
	# print("tf.executing_eagerly(): ",tf.executing_eagerly())
	# tf.compat.v1.disable_eager_execution() # https://www.activestate.com/resources/quick-reads/how-to-debug-tensorflow/
	# spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.dubinscar,cfg.sampler.hmc,dim=dim_X)
	# rrtp_MO, Ndiv = initialize_GPmodel_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,use_nominal_model_for_spectral_density=True)
	# pdb.set_trace()
	rrtp_MO = initialize_MOrrp_with_existing_data(cfg,dim_X,Xtrain,Ytrain,which_kernel,use_nominal_model_for_spectral_density=True)
	

	# pdb.set_trace()

	# # New reference trajectory, for the true system to follow
	# ref_pars = dict()
	# ref_pars["rad"] = 2.0
	# ref_pars["sign_Y"] = +1
	# ref_pars["sign_xT"] = +1

	# ref_xt_vec, ref_ut_vec = generate_reference_trajectory(ref_pars,Nsteps,deltaT)
	ref_xt_vec = None

	# # Roll-out ONE trajectory with a model-based controller that knows the true dynamics, following the above reference
	# x0 = ref_xt_vec[0,:]
	# x0_noise_std = 1.0
	# x0_mod = x0 + np.array([0.2,0.2,0.5])*np.random.randn(3)*x0_noise_std
	# T = 10.0 # Not used, get rid of
	# z_vec, u_vec, t_vec = rollout_with_finitie_horizon_LQR(x0_mod,deltaT,T,Nsteps,ref_xt_vec,ref_ut_vec,use_nominal_model=True)
	# # z_vec: [Nsteps,dim_out]
	# # u_vec: [Nsteps,dim_in]
	

	# Take the last trajectory as the good one:
	# pdb.set_trace()
	# Nsteps = 1000
	z_vec = Xtrain[-Nsteps::,0:3].numpy()
	u_vec = Xtrain[-Nsteps::,3::].numpy()
	

	# Generate control sequence (u_vec) using the right nominal model, but then apply it to the changed dynamics.
	z_vec_changed_dyn = np.zeros((Nsteps,dim_x))
	z_vec_changed_dyn[0,:] = z_vec[0,:]
	for tt in range(Nsteps-1):

		use_nominal_model = tt > Nsteps/2
		# use_nominal_model = True
		z_vec_changed_dyn[tt+1:tt+2,:] = dyn_sys_true(state_vec=z_vec_changed_dyn[tt:tt+1,:],control_vec=u_vec[tt:tt+1,:],use_nominal_model=use_nominal_model)


		"""
		Here, every chunk of H time steps we compare the elapsed observations with the predicted ones for past steps [t-H,t-H+1,...,t]

		len_rrtp_MO_list = len(rrtp_MO_list)
		log_evidence = np.zeros(len_rrtp_MO_list)
		for mm in range(len_rrtp_MO_list):
			log_evidence[mm] = rrtp_MO_list[mm]._get_negative_log_evidence_and_predictive_trajectory_chunk() -> 

		Select the best log evidence. Switch to the model with the best one.

		
		IMORTANT NOTE: With this approach we can only do model ANALYSIS. We can identify which one is the right model. But we can't do control at all, unless we do stochastic MPC.
			1) At time t, we compare the observations from times [t-H,t-H+1,...,t] with the predictions we did starting from time t-H. For the same control sequence, different models will predict different trajectories. If one of them coincides with the observed one, with a low lsot, then that's our model. Otheriwse, we need to 
			collect data and re-condition. But even when we have the model in our stack, the question is, how the hell do we give to the robot a new control sequence that it can follow? We have one that the robot is using to traverse a room. But in the middle of the room, the friction changes. Then that control
			sequence is no longer valid. Imagine this is a real scenario. Where do we get a new control sequence from? If one of the models is correct, then, I guess, one possiblity is using stochastic MPC, no? I can actually do non-linear MPC by locally linearizing, subject to some disturbance. Remember that we can
			linearize our model at cost O(1) because all it takes is to compute the derivatives of the features. Then, we can decompose matrix A in something deterministic (the mean) and a stochastic component (a random disturbabce) and just use standard stcohastc MPC tools.
		"""


	# pdb.set_trace()


	# At every x_t, predict x_{t:t+H} states. 
	# Compare those predictions with the observed y_{t:t+H}. 
	# Compute OOD. 
	# Repeat for time t+H+1
	Nhorizon = 20
	Nrollouts = 5
	# x0 = np.random.rand(1,dim_x)

	train = True
	# log_noise_std_per_dim = tf.constant([-2.4124577,-2.2396216,-2.4339094]) # 50 iters
	# log_prior_variance_per_dim = tf.constant([0.1828889,2.7030554,0.9408228]) # 50 iters

	# log_noise_std_per_dim = tf.constant([-3.9496725, -2.8052473, -4.253263]) # 115 iters
	# log_prior_variance_per_dim = tf.constant([-4.871695, 6.299472, -4.5974174]) # 115 iters

	# # Weights: wrong, using matern kernel ... 88 iterations
	# weights_list = []
	# weights_list += [dict(log_noise_std=-1.055355,log_prior_variance=-1.9315833,log_prior_mean_factor=6.438162)]
	# weights_list += [dict(log_noise_std=-1.589952,log_prior_variance=-1.0690501,log_prior_mean_factor=6.441733)]
	# weights_list += [dict(log_noise_std=+0.367157,log_prior_variance=+5.218601,log_prior_mean_factor=-1.8352405)]

	# # Weights: 85 iterations (with assign_weights_v1())
	# weights_list = []
	# weights_list += [dict(log_noise_std=-3.2072575,	log_prior_variance=-2.3769002,	log_prior_mean_factor=-2.1808307e-06)]
	# weights_list += [dict(log_noise_std=-3.375641,	log_prior_variance=5.378116,	log_prior_mean_factor=3.5671384e-08)]
	# weights_list += [dict(log_noise_std=-3.5728772,	log_prior_variance=-1.7348373,	log_prior_mean_factor=-1.6369743e-06)]

	# Weights:

	# Prepare the training and its loss; the latter compares the true trajectory with the predicted one, in chunks.
	# learning_rate = 5e-2
	learning_rate = 1e-1
	epochs = 200
	# epochs = 2
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
		z_vec_real = z_vec_tf
	if compute_predictions_over_trajectory_when_nominal_control_sequence_applied_to == "altered_model":
		z_vec_real = z_vec_changed_dyn_tf
	rrtp_MO.update_dataset_predictive_loss(	z_vec_real=z_vec_real,u_traj_real=u_vec_tf,Nhorizon=Nhorizon,
											learning_rate=learning_rate,epochs=epochs,stop_loss_val=stop_loss_val,
											scale_loss_entropy=scale_loss_entropy,scale_prior_regularizer=scale_prior_regularizer,
											Nrollouts=Nrollouts)

	# plt.show(block=True)

	# Before training to predict:
	plotting_dict = dict(plotting=True,block_plot=False,title_fig="Predictions || Dummy",ref_xt_vec=None,z_vec=None,z_vec_changed_dyn=None)
	plotting_dict["title_fig"] = "Predictions || Using posterior, after training one-step ahead"
	plotting_dict["ref_xt_vec"] = ref_xt_vec
	plotting_dict["z_vec"] = z_vec
	plotting_dict["z_vec_changed_dyn"] = z_vec_changed_dyn
	# plotting_dict["z_vec_changed_dyn"] = None
	# rrtp_MO.set_dbg_flag(True)
	loss_val = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=False,plotting_dict=plotting_dict,Nrollouts=Nrollouts)
	logger.info("loss_total (before training): {0:f}".format(loss_val))

	plt.show(block=True)

	# Before conditioning (prior):
	plotting_dict["title_fig"] = "Predictions || Using prior, no training"
	loss_val = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=False,plotting_dict=plotting_dict,Nrollouts=Nrollouts,from_prior=True)
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
	loss_val = rrtp_MO.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=True,plotting_dict=plotting_dict,Nrollouts=Nrollouts)
	logger.info("loss_total (after training): {0:f}".format(loss_val))

	plt.show(block=True)

	# deprecated()

def assign_weights_v1(rrtp_MO,weights_list):

	dd = 0
	for weights in weights_list:
		rrtp_MO.rrgpMO[dd].log_noise_std.assign(value=[weights["log_noise_std"]])
		rrtp_MO.rrgpMO[dd].log_prior_variance.assign(value=[weights["log_prior_variance"]])
		rrtp_MO.rrgpMO[dd].log_prior_mean_factor.assign(value=[weights["log_prior_mean_factor"]])
		dd += 1

	return rrtp_MO

def assign_weights_v2(rrtp_MO,weights_list):
	raise NotImplementedError("assign_weights_v2")

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



