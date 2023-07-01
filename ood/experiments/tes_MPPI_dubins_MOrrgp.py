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


def generate_vel_profile(Nsteps,deltaT=0.01,vt_fac=10.0,wt_fac=5.0):

	T = (Nsteps-1)*deltaT
	t_vec = np.linspace(0,T,Nsteps)
	t_vec = np.reshape(t_vec,(-1,1))
	vt_vec = t_vec * (T - 1./3*t_vec**2) * vt_fac
	# wt_vec = t_vec * (T - 1./3*t_vec**2) * 10.0
	wt_vec = tf.math.sin(t_vec / T * 2.*math.pi) * wt_fac
	ut_vec = np.concatenate([vt_vec,wt_vec],axis=1)

	return ut_vec # [Nsteps,2]


def collect_data(ut_vec,deltaT,Nsteps,beta_true,noise_std=0.01):

	# Pick one as true dynamics:
	dyn_dubscar_true = DynamicsDubinsCar(deltaT,beta_true)

	# Collect noisy data:
	state_real_data = np.zeros((Nsteps,3))
	for tt in range(Nsteps-1):
		Dstate = dyn_dubscar_true(state_real_data[tt,:],ut_vec[tt,:])
		state_real_data[tt+1,:] = state_real_data[tt,:] + Dstate
	state_real_data += noise_std*np.random.randn(Nsteps,1)
	

	# Train model
	Ncut = Nsteps
	Dstate_data = state_real_data[1::,0:Ncut] - state_real_data[0:-1,0:Ncut]
	Xtrain = np.concatenate([state_real_data[0:-1,0:Ncut],ut_vec[0:-1,0:Ncut]],axis=1)
	Ytrain = Dstate_data

	return Xtrain, Ytrain, state_real_data # [Nsteps-1,5], [Nsteps-1,3], [Nsteps,3]


def OoD_metric():
	"""
	Since we're already getting the MPC predictions, we can compute OoD for free
	This will be a delayed metric
	"""
	pass



def main():

	Nsteps = 101
	deltaT = 0.01
	ut_vec = generate_vel_profile(Nsteps,deltaT=deltaT,vt_fac=10.0,wt_fac=5.0) # [Nsteps,2]

	beta_d_vec_mean = np.reshape(np.array([1.,1.,1.]),(-1,1))
	beta_d_vec_var = np.reshape(np.array([1.,1.,1.]),(-1,1))

	beta_true = beta_d_vec_mean[:,0]*1.1
	Xtrain, Ytrain, state_real_data = collect_data(ut_vec,deltaT,Nsteps,beta_true,noise_std=0.01)

	# 1) Create a cost function as th distance from the state to a goal
	# 2) Create some virtual obstacles and assign a high cost to passing through them and to the obstacles
	# 3) Train the model
	# 4) Create an MPC for loop, have a shorter time horizon, H = 10
	# 5) Generate R=10 rollputs for each velocity profiel; sample the V=5 velocity profiles by sampling vt_fac, wt_fac
	# 6) Pick the best control sequence, then pick the best, repeat



















if __name__ == "__main__":

	main()




