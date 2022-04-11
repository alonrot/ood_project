import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPSarkkaFeatures, RRTPRandomFourierFeatures
from lqrker.utils.spectral_densities import MultiDimensionalFourierTransformQuadratureFromData
import numpy as np
import numpy.random as npr
import scipy
from simple_loop_2dpend import simulate_single_pend
import hydra
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


"""
"""


def simulate_inverted_pendulum(Nsteps,visualize=False,x0=None,sort=False):
	
	obs_vec, u_vec, policy_fun = simulate_single_pend(Nsteps,visualize=visualize,plot=False) # [Npoints,dim]


	if sort == True:
		# obs: return np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])
		obs_sorted = np.zeros((Nsteps,4))
		obs_sorted[:,0] = np.arctan2(obs_vec[:,2],obs_vec[:,3]) # signed angle in radians; np.arctan2(cos,sin)
		obs_sorted[:,1] = obs_vec[:,4]
		obs_sorted[:,2] = obs_vec[:,0]
		obs_sorted[:,3] = obs_vec[:,1]

		# obs_sorted: [th, thd, x, xd]

	else:
		obs_sorted = obs_vec


	obs_vec_tf = tf.convert_to_tensor(value=obs_sorted,dtype=np.float32)


	Nskip = 1
	X = obs_vec_tf[0:-1:Nskip,:] # [Npoints,dim_x]
	Y = obs_vec_tf[1::Nskip,:] # [Npoints,dim_x]

	return X,Y,u_vec,policy_fun

@hydra.main(config_path=".",config_name="config.yaml")
def test(cfg: dict) -> None:
	"""

	Train the model to predict one-step ahead

	Then, see how well the model can predict long term trajectories.
	To this end, we evaluate the model at a trining point, then sample a new point from the output distribution, 
	then evaluate the model at that sampled point, and then repeat the process
	"""

	# Dataset from one simulation only:
	Nevals = 60*2+1
	X,Y,u_vec,policy_fun = simulate_inverted_pendulum(Nsteps=Nevals,visualize=False,x0=None,sort=True) # [Nevals,dim] , [Nevals,dim]

	# Debug:
	X = X[:,0:2]
	Y = Y[:,0:2]

	dim_x = X.shape[1]
	dim_y = Y.shape[1]
	
	# Multi-output BLM:
	rrtp_MO = [None]*dim_y
	spectral_density = [None]*dim_y
	MO_mean_pred = [None]*dim_y
	MO_std_pred = [None]*dim_y
	# xpred = tf.reshape([tf.linspace(-5.,+5.,101)]*5,(101,5))
	Xmin = tf.math.reduce_min(X,axis=0)
	Xmax = tf.math.reduce_max(X,axis=0)
	Xmid = (Xmin+Xmax)/2.
	Xmid_ext = Xmid-(Xmid-Xmin)*1.5
	Xmax_ext = Xmid+(Xmax-Xmid)*1.5
	xpred = tf.linspace(Xmid_ext,Xmax_ext,101)

	# Analysis on the spectral density for 1 and 2-dimensional trajectories:
	dbg_plot = False
	if dbg_plot == True:
		if dim_x == 1:

			# Initialize density:
			Yproj = tf.reshape(Y[:,0],(-1,1)) # [Npoints,1]
			spectral_density_dbg = MultiDimensionalFourierTransformQuadratureFromData(cfg.RRTPRandomFourierFeatures.spectral_density_pars, X, Yproj)

			# Sample from the unnormalized density:
			omega_samples = spectral_density_dbg.get_samples(Nsamples=100,state_ind=None)
			omega_samples = tf.squeeze(omega_samples)

			# Because the spectral density is symmetric w.r.t the X=0 axis, we can make all the samples positive:
			omega_samples = tf.math.abs(omega_samples)

			# Plot the density itself:
			Ndiv = 301
			w_lim = tf.reduce_max(omega_samples)
			omega_pred = tf.reshape(tf.linspace(0.0,+w_lim.numpy(),Ndiv),(-1,1))
			Sw_vec = spectral_density_dbg.unnormalized_density(omega_in=omega_pred,log=False)

			hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle("Inverted pendulum simulation x(t)")
			hdl_splots.plot(omega_pred,Sw_vec)
			Sw_height = tf.reduce_min(Sw_vec)
			hdl_splots.plot(omega_samples,[Sw_height.numpy()]*len(omega_samples),linestyle="None",marker="x",markersize=10)
		
		elif dim_x == 2:

			Ndiv = 81
			w_lim = 2.0
			xx = tf.linspace(-w_lim,+w_lim,Ndiv)
			yy = tf.linspace(-w_lim,+w_lim,Ndiv)
			XXpred, YYpred = tf.meshgrid(xx, yy)
			omega_pred = tf.concat( [tf.reshape(XXpred,(-1,1)) , tf.reshape(YYpred,(-1,1))], axis=1 )

			Yproj1 = tf.reshape(Y[:,0],(-1,1)) # [Npoints,1]
			spectral_density_dbg = MultiDimensionalFourierTransformQuadratureFromData(cfg.RRTPRandomFourierFeatures.spectral_density_pars, X, Yproj1)
			S1w_vec = spectral_density_dbg.unnormalized_density(omega_in=omega_pred,log=False)
			S1w = np.reshape(S1w_vec,(Ndiv,Ndiv))

			Yproj2 = tf.reshape(Y[:,1],(-1,1)) # [Npoints,1]
			spectral_density_dbg = MultiDimensionalFourierTransformQuadratureFromData(cfg.RRTPRandomFourierFeatures.spectral_density_pars, X, Yproj2)
			S2w_vec = spectral_density_dbg.unnormalized_density(omega_in=omega_pred,log=False)
			S2w = np.reshape(S2w_vec,(Ndiv,Ndiv))

			hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle("Inverted pendulum simulation x(t)")
			hdl_splots[0].imshow(S1w)
			hdl_splots[1].imshow(S2w)

		plt.show(block=True)


	# Debug:
	xpred = X

	for ii in range(dim_y):

		Yproj = tf.reshape(Y[:,ii],(-1,1)) # [Npoints,1]
		spectral_density[ii] = MultiDimensionalFourierTransformQuadratureFromData(cfg.RRTPRandomFourierFeatures.spectral_density_pars, X, Yproj)

		rrtp_MO[ii] = RRTPRandomFourierFeatures(dim=dim_x,cfg=cfg.RRTPRandomFourierFeatures,spectral_density=spectral_density[ii])
		rrtp_MO[ii].update_spectral_density(None,None)

		rrtp_MO[ii].update_model(X,Y[:,ii]) # Update model indexing the target outputs at the corresponding dimension
		rrtp_MO[ii].train_model()

		# Compute predictive moments:
		MO_mean_pred[ii], cov_pred = rrtp_MO[ii].get_predictive_moments(xpred)
		MO_std_pred[ii] = tf.sqrt(tf.linalg.diag_part(cov_pred))

	# Plot:
	hdl_fig, hdl_splots = plt.subplots(dim_y,1,figsize=(12,8),sharex=True)
	if dim_y == 1:
		hdl_splots = [hdl_splots]
	hdl_fig.suptitle("Inverted pendulum simulation x(t)")
	for ii in range(dim_y):
		hdl_splots[ii].plot(MO_mean_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(MO_mean_pred[ii] + 2.*MO_std_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(MO_mean_pred[ii] - 2.*MO_std_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(Y[:,ii],linestyle="None",color="k",marker=".")

	# # Plot:
	# hdl_fig, hdl_splots = plt.subplots(dim_y+1,1,figsize=(12,8),sharex=True)
	# hdl_fig.suptitle("Inverted pendulum simulation x(t)")
	# for ii in range(dim_y):
	# 	hdl_splots[ii].plot(Y[:,ii],linestyle="None",color="k",marker=".")
	# hdl_splots[dim_y].plot(u_vec,linestyle="None",color="k",marker=".")
	# # plt.show(block=True)
	
	# # Spectral density args:
	# phi0 = tf.reshape(xpred[0,0],(1,-1))
	# u0 = u_vec[0]
	# args_density = dict(phi0=phi0,u0=u0)




	# plt.show(block=True)
	
	# # Predictions:
	# H = 10
	# assert (Nevals-1) % H == 0

	# MO_mean_pred = np.zeros((H,dim_y),dtype=np.float32)
	# MO_std_pred = np.zeros((H,dim_y),dtype=np.float32)
	# x_samples = np.zeros((xpred.shape),dtype=np.float32)
	# for ii in range(0,xpred.shape[0],H):
		
	# 	# Predict H steops ahead, starting at location x_pred_location:
	# 	x_pred_location = tf.reshape(xpred[ii,:],(1,-1))

	# 	# Spectral density args:
	# 	phi0 = x_pred_location[0,0]
	# 	u0 = u_vec[ii]
	# 	args_density = dict(phi0=phi0,u0=u0)

	# 	# Retrain the entire model at each iteration with the new linearization point (x0_lin,u0_lin) 
	# 	# and new S(w;x0_lin,u0_lin); the posterior distribution will change accordingly
	# 	# raise NotImplementedError
	# 	for jj in range(dim_y):

	# 		rrtp_MO[jj].update_spectral_density(args_density,state_ind=jj)

	# 		# Update model:
	# 		rrtp_MO[jj].update_model(X,Y[:,jj])
	# 		logger.warning("[WARNING]: In the line above, do we need to offset with the linearization point??? X-x_pred_location ???")

	# 		# Try to mitigate the training time:
	# 		rrtp_MO[jj].train_model()

	# 	# Predict for horizon H:
	# 	for h in range(H):

	# 		# Compute predictions for each channel:
	# 		for jj in range(dim_y):
	# 			MO_mean_pred[h,jj], cov_pred = rrtp_MO[jj].get_predictive_moments(x_pred_location)
	# 			MO_std_pred[h,jj] = tf.sqrt(tf.linalg.diag_part(cov_pred))

	# 			# Sample from the predictive moments: (assume Gaussian for simplicity):
	# 			x_samples[ii+h,jj] = tf.random.normal(shape=(1,1), mean=MO_mean_pred[h,jj], stddev=MO_std_pred[h,jj])
			
	# 		# Predictive location:
	# 		x_pred_location = tf.reshape(tf.convert_to_tensor(x_samples[ii+h,:]),(1,-1))

	# 	# Plot predictions on top
	# 	for jj in range(dim_y):
	# 		hdl_splots[jj].plot(np.arange(ii,ii+H),x_samples[ii:ii+H,jj],linestyle="--",color="r")

	plt.show(block=True)


if __name__ == "__main__":

	test()


