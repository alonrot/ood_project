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




class DynamicsDubinsCar():

	def __init__(self,deltaT,beta_d_vec):

		self.deltaT = deltaT
		assert beta_d_vec.shape[0] == 3
		assert beta_d_vec.ndim == 1
		self.beta_d_vec = beta_d_vec

	def _dyn_fun(self,xt,yt,tht,vt,wt):

		Dxt_next = self.deltaT * tf.math.cos(tht) * vt * self.beta_d_vec[0]
		Dyt_next = self.deltaT * tf.math.sin(tht) * vt * self.beta_d_vec[1]
		Dtht_next = self.deltaT * wt * self.beta_d_vec[2]

		# return np.reshape(np.array([Dxt_next,Dyt_next,Dtht_next]),(-1,1))
		return np.array([Dxt_next,Dyt_next,Dtht_next])

	def __call__(self,x_in,u_in):

		# assert x_in.shape[0] == 1
		# assert x_in.shape[1] == 5
		# assert u_in.shape[0] == 1
		# assert u_in.shape[1] == 2

		# return _dyn_fun(x_in[:,0],x_in[:,1],x_in[:,2],u_in[:,0],u_in[:,1])
		return self._dyn_fun(x_in[0],x_in[1],x_in[2],u_in[0],u_in[1])




@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict):

	deltaT = 0.01

	beta_d_vec_mean = np.reshape(np.array([1.,1.,1.]),(-1,1))
	beta_d_vec_var = np.reshape(np.array([1.,1.,1.]),(-1,1))


	"""
	Construct a range of sampled dynamical systems
	"""


	Nsteps = 101
	T = (Nsteps-1)*deltaT
	t_vec = np.linspace(0,T,Nsteps)
	t_vec = np.reshape(t_vec,(-1,1))
	vt_vec = t_vec * (T - 1./3*t_vec**2) * 10.0
	# wt_vec = t_vec * (T - 1./3*t_vec**2) * 10.0
	wt_vec = tf.math.sin(t_vec / T * 2.*math.pi) * 5.0

	ut_vec = np.concatenate([vt_vec,wt_vec],axis=1)

	"""
	Samples from the model class, directly
	"""

	"""
	Nsamples = 10
	beta_d_vec = beta_d_vec_mean + np.diag(np.sqrt(beta_d_vec_var[:,0])) @ np.random.randn(3,Nsamples) # [3,Nsamples]

	samples_dyn_dubscar = []
	for ss in range(Nsamples):
		samples_dyn_dubscar += [DynamicsDubinsCar(deltaT,beta_d_vec[:,ss])]



	state_rollouts = np.zeros((Nsamples,Nsteps,3))
	for ss in range(Nsamples):
		for tt in range(Nsteps-1):
			Dstate = samples_dyn_dubscar[ss](state_rollouts[ss,tt,:],ut_vec[tt,:])
			state_rollouts[ss,tt+1,:] = state_rollouts[ss,tt,:] + Dstate


	hdl_fig, hdl_splots = plt.subplots(1,2,figsize=(17,7),sharex=False)
	for ss in range(Nsamples):
		hdl_splots[0].plot(state_rollouts[ss,:,0],state_rollouts[ss,:,1],linestyle="-",color="grey")
	hdl_splots[0].plot(state_rollouts[0,0,0],state_rollouts[0,0,1],color="green",marker="o",markersize=5)

	plt.show(block=False)
	"""

	hdl_fig, hdl_splots = plt.subplots(2,4,figsize=(17,7),sharex=False)

	beta_true = beta_d_vec_mean[:,0]*1.1

	# Pick one as true dynamics:
	dyn_dubscar_true = DynamicsDubinsCar(deltaT,beta_true)

	# Collect noisy data:
	state_real_data = np.zeros((Nsteps,3))
	for tt in range(Nsteps-1):
		Dstate = dyn_dubscar_true(state_real_data[tt,:],ut_vec[tt,:])
		state_real_data[tt+1,:] = state_real_data[tt,:] + Dstate
	noise_std = 0.01
	state_real_data += noise_std*np.random.randn(Nsteps,1)
	

	# Train model
	Ncut = Nsteps
	Dstate_data = state_real_data[1::,0:Ncut] - state_real_data[0:-1,0:Ncut]
	Xtrain = np.concatenate([state_real_data[0:-1,0:Ncut],ut_vec[0:-1,0:Ncut]],axis=1)
	Ytrain = Dstate_data

	rrgp = MultiObjectiveReducedRankProcess(dim_in=5,cfg=cfg,spectral_density=None,Xtrain=Xtrain,Ytrain=Ytrain,using_deltas=True)


	# Predict from prior
	Nrollouts = 10
	# pdb.set_trace()
	x0 = tf.zeros((1,3))
	x0_tf = tf.convert_to_tensor(value=x0,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
	u_applied_tf = tf.convert_to_tensor(value=ut_vec,dtype=tf.float32) # [Npoints,self.dim_in]
	# Regardless of self.using_deltas, the function below returns the actual state, not the deltas
	x_traj_pred, y_traj_pred = rrgp._rollout_model_given_control_sequence_tf(x0=x0_tf,Nsamples=1,Nrollouts=Nrollouts,u_traj=u_applied_tf,traj_length=-1,
																			sort=False,plotting=False,str_progress_bar="[hola] ",from_prior=True,
																			when2sample="once_per_class_instantiation") # [Nrollouts,traj_length-1,self.dim_out]


	hdl_splots[0,0].plot(state_real_data[:,0],state_real_data[:,1],linestyle="-",color="navy",alpha=0.3)
	for rr in range(Nrollouts):
		hdl_splots[0,0].plot(x_traj_pred[rr,:,0],x_traj_pred[rr,:,1],linestyle="-",color="grey",linewidth=0.5)
	hdl_splots[0,0].set_title(r"Prior - Samples",fontsize=fontsize_labels)
	hdl_splots[0,0].plot(x0_tf[0,0],x0_tf[0,1],color="green",marker="o",markersize=5)
	hdl_splots[0,0].set_xlim([-0.5,2.5])
	hdl_splots[0,0].set_ylim([-0.5,3.5])
	hdl_splots[0,0].set_ylabel(r"$y_t$",fontsize=fontsize_labels)
	hdl_splots[0,0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)

	# Betas - prior:
	xpred_beta = np.linspace(-2,2,201)
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, beta_d_vec_mean[0], beta_d_vec_var[0])
	hdl_splots[1,0].fill_between(xpred_beta, beta_pdf, color='navy', alpha=0.2)
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, beta_d_vec_mean[1], beta_d_vec_var[1])
	hdl_splots[1,0].fill_between(xpred_beta, beta_pdf, color='navy', alpha=0.2)
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, beta_d_vec_mean[2], beta_d_vec_var[2])
	hdl_splots[1,0].fill_between(xpred_beta, beta_pdf, color='navy', alpha=0.2)
	hdl_splots[1,0].set_xlabel(r"$\gamma$",fontsize=fontsize_labels)



	# Predict from posterior:
	x_traj_pred, y_traj_pred = rrgp._rollout_model_given_control_sequence_tf(x0=x0_tf,Nsamples=1,Nrollouts=Nrollouts,u_traj=u_applied_tf,traj_length=-1,
																			sort=False,plotting=False,str_progress_bar="[hola] ",from_prior=False,
																			when2sample="once_per_class_instantiation") # [Nrollouts,traj_length-1,self.dim_out]

	hdl_splots[0,1].plot(state_real_data[:,0],state_real_data[:,1],linestyle="-",color="red",alpha=0.8,label="Tr. Data")
	for rr in range(Nrollouts):
		hdl_splots[0,1].plot(x_traj_pred[rr,:,0],x_traj_pred[rr,:,1],linestyle="-",color="grey",linewidth=0.5)
	hdl_splots[0,1].set_title(r"Post., Data size = 101",fontsize=fontsize_labels)
	hdl_splots[0,1].set_xlim([-0.5,2.5])
	hdl_splots[0,1].set_ylim([-0.5,3.5])
	hdl_splots[0,1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots[0,1].legend(loc="lower right",fontsize=fontsize_labels*0.8)


	# Betas - posterior:
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[0].mean_beta_predictive, rrgp.rrgpMO[0].chol_cov_beta_predictive)
	hdl_splots[1,1].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)

	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[1].mean_beta_predictive, rrgp.rrgpMO[1].chol_cov_beta_predictive)
	hdl_splots[1,1].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)
	
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[2].mean_beta_predictive, rrgp.rrgpMO[2].chol_cov_beta_predictive)
	hdl_splots[1,1].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)
	hdl_splots[1,1].set_xlabel(r"$\gamma$",fontsize=fontsize_labels)


	# Predict from posterior, less data
	Ncut = 21
	Dstate_data = state_real_data[1::,0:Ncut] - state_real_data[0:-1,0:Ncut]
	Xtrain = np.concatenate([state_real_data[0:-1,0:Ncut],ut_vec[0:-1,0:Ncut]],axis=1)
	Ytrain = Dstate_data
	rrgp.update_model(X=Xtrain,Y=Ytrain)


	x_traj_pred, y_traj_pred = rrgp._rollout_model_given_control_sequence_tf(x0=x0_tf,Nsamples=1,Nrollouts=Nrollouts,u_traj=u_applied_tf,traj_length=-1,
																			sort=False,plotting=False,str_progress_bar="[hola] ",from_prior=False,
																			when2sample="once_per_class_instantiation") # [Nrollouts,traj_length-1,self.dim_out]

	hdl_splots[0,2].plot(state_real_data[0:Ncut,0],state_real_data[0:Ncut,1],linestyle="-",color="red",alpha=0.8,label="Tr. Data")
	for rr in range(Nrollouts):
		hdl_splots[0,2].plot(x_traj_pred[rr,:,0],x_traj_pred[rr,:,1],linestyle="-",color="grey",linewidth=0.5)
	hdl_splots[0,2].set_title(r"Post., Data size = 21",fontsize=fontsize_labels)
	hdl_splots[0,2].set_xlim([-0.5,2.5])
	hdl_splots[0,2].set_ylim([-0.5,3.5])
	hdl_splots[0,2].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots[0,2].legend(loc="lower right",fontsize=fontsize_labels*0.8)


	# Betas - posterior less data:
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[0].mean_beta_predictive, rrgp.rrgpMO[0].chol_cov_beta_predictive)
	hdl_splots[1,2].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)

	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[1].mean_beta_predictive, rrgp.rrgpMO[1].chol_cov_beta_predictive)
	hdl_splots[1,2].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)
	
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[2].mean_beta_predictive, rrgp.rrgpMO[2].chol_cov_beta_predictive)
	hdl_splots[1,2].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)
	hdl_splots[1,2].set_xlabel(r"$\gamma$",fontsize=fontsize_labels)



	# New control input trajetcory - Collect noisy data:
	state_real_data_new_control = np.zeros((Nsteps,3))
	vt_vec = t_vec * (T - 1./3*t_vec**2) * 3.0
	# wt_vec = t_vec * (T - 1./3*t_vec**2) * 10.0
	wt_vec = -tf.math.sin(t_vec / T * math.pi) * 7.5
	ut_vec_new = np.concatenate([vt_vec,wt_vec],axis=1)
	for tt in range(Nsteps-1):
		Dstate = dyn_dubscar_true(state_real_data_new_control[tt,:],ut_vec_new[tt,:])
		state_real_data_new_control[tt+1,:] = state_real_data_new_control[tt,:] + Dstate
	noise_std = 0.01
	state_real_data_new_control += noise_std*np.random.randn(Nsteps,1)

	u_applied_tf_new = tf.convert_to_tensor(value=ut_vec_new,dtype=tf.float32) # [Npoints,self.dim_in]
	x_traj_pred, y_traj_pred = rrgp._rollout_model_given_control_sequence_tf(x0=x0_tf,Nsamples=1,Nrollouts=Nrollouts,u_traj=u_applied_tf_new,traj_length=-1,
																			sort=False,plotting=False,str_progress_bar="[hola] ",from_prior=False,
																			when2sample="once_per_class_instantiation") # [Nrollouts,traj_length-1,self.dim_out]

	hdl_splots[0,3].plot(state_real_data_new_control[:,0],state_real_data_new_control[:,1],linestyle="-",color="navy",alpha=0.3)
	for rr in range(Nrollouts):
		hdl_splots[0,3].plot(x_traj_pred[rr,:,0],x_traj_pred[rr,:,1],linestyle="-",color="grey",linewidth=0.5)
	hdl_splots[0,3].set_title(r"Post., Data size = 21 [new input]",fontsize=fontsize_labels*0.7)
	hdl_splots[0,3].set_xlim([-0.5,0.5])
	hdl_splots[0,3].set_ylim([-0.5,0.5])
	hdl_splots[0,3].set_xlabel(r"$x_t$",fontsize=fontsize_labels)


	# Betas - posterior less data:
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[0].mean_beta_predictive, rrgp.rrgpMO[0].chol_cov_beta_predictive)
	hdl_splots[1,3].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)

	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[1].mean_beta_predictive, rrgp.rrgpMO[1].chol_cov_beta_predictive)
	hdl_splots[1,3].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)
	
	beta_pdf = scipy.stats.norm.pdf(xpred_beta, rrgp.rrgpMO[2].mean_beta_predictive, rrgp.rrgpMO[2].chol_cov_beta_predictive)
	hdl_splots[1,3].fill_between(xpred_beta, beta_pdf[0,:], color='navy', alpha=0.2)
	hdl_splots[1,3].set_xlabel(r"$\gamma$",fontsize=fontsize_labels)


	# Control input trajectory:
	hdl_fig_control, hdl_splots_control = plt.subplots(2,2,figsize=(17,7),sharex=True)
	hdl_splots_control[0,0].plot(t_vec[:,0],ut_vec[:,0],linestyle="-")
	hdl_splots_control[0,0].set_title(r"$u^{(1)}_t$ - Linear velocity [old]",fontsize=fontsize_labels)
	
	hdl_splots_control[0,1].plot(t_vec[:,0],ut_vec[:,1],linestyle="-")
	hdl_splots_control[0,1].set_title(r"$u^{(2)}_t$ - Angular velocity [old]",fontsize=fontsize_labels)

	hdl_splots_control[1,0].plot(t_vec[:,0],ut_vec_new[:,0],linestyle="-")
	hdl_splots_control[1,0].set_title(r"$u^{(1)}_t$ - Linear velocity [new]",fontsize=fontsize_labels)
	hdl_splots_control[1,0].set_xlabel(r"Time [sec]",fontsize=fontsize_labels)
	
	hdl_splots_control[1,1].plot(t_vec[:,0],ut_vec_new[:,1],linestyle="-")
	hdl_splots_control[1,1].set_title(r"$u^{(2)}_t$ - Angular velocity [new]",fontsize=fontsize_labels)
	hdl_splots_control[1,1].set_xlabel(r"Time [sec]",fontsize=fontsize_labels)


	# for dd in range(3):
	# 	print("mean_beta_predictive:",rrgp.rrgpMO[dd].mean_beta_predictive)
	# 	print("chol_cov_beta_predictive:",rrgp.rrgpMO[dd].chol_cov_beta_predictive)
	# 	print("mean_beta_prior:",rrgp.rrgpMO[dd].mean_beta_prior)
	# 	print("chol_cov_beta_prior:",rrgp.rrgpMO[dd].chol_cov_beta_prior)





	plt.show(block=True)

	# Demonstrate that the prior already contains full dynamics trajectorues
	# Think about probabilistic reachability. More useful than just shooting trajectories. It'd be nice to have the probabiity distirbution over a forward/backwards reachable set given some st of control inputs...
	# 1) Sample one beta
	# 2) Roll out the model forward in time
	# 3) Repeat
	# Assume fixed control inputs. This is very similar to probabilistic reachability, no?





if __name__ == "__main__":

	main()