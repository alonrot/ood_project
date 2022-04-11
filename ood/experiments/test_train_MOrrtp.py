import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPSarkkaFeatures, RRTPRandomFourierFeatures
import numpy as np
import numpy.random as npr
import scipy
from simple_loop_2dpend import simulate_single_pend
import hydra


"""
### TODO
1) Code up the dynamics model: Give flexibility (inheritance) for features to be NNs or Sarkka features. Think about re-using this for the koopman project
2) Record trajectories from the swin-up policy and compute the model posterior
2.5) Sarkka features work funny. To debug this, take the Matlab implementation of the Sarkka paper (was this T. Schon) and try to replicate their results
2.6) If Sarkka doesn't work, try with coding up random Fourier features
3) Code up a OOD function detetion using the likelihood (make a virtual class for the different possible implementations of the OOD function)
2) Create a bunch of new environments with different forces/stops, etc.
2) Achieve the trigger of OOD
"""

"""
1) Try training with X[:,0] as input only
1.5) Compare with a multioutput model in GPflow...
2) We might need different lengthscales for different inputs dimensionalities...
2) Eventually, add a timestamp...?
3) Refactor rrtp to account for NN training
"""


def simulate_inverted_pendulum(Nsteps,x0=None):
	
	obs_vec = simulate_single_pend(Nsteps) # [Npoints,dim]
	obs_vec_tf = tf.convert_to_tensor(value=obs_vec,dtype=np.float32)

	Nskip = 1
	X = obs_vec_tf[0:-1:Nskip,:] # [Npoints,dim_x]
	Y = obs_vec_tf[1::Nskip,:] # [Npoints,dim_x]

	return X,Y

@hydra.main(config_path=".",config_name="config.yaml")
def test(cfg: dict) -> None:
	"""

	Train the model to predict one-step ahead

	Then, see how well the model can predict long term trajectories.
	To this end, we evaluate the model at a trining point, then sample a new point from the output distribution, 
	then evaluate the model at that sampled point, and then repeat the process
	"""

	# Dataset:
	Nevals = 60*2+1
	X,Y = simulate_inverted_pendulum(Nsteps=Nevals) # [Nevals,dim] , [Nevals,dim]

	# # Debug:
	# X = X[:,0:2]
	# Y = Y[:,0:2]

	dim_x = X.shape[1]
	dim_y = Y.shape[1]
	
	# Multi-output BLM:
	rrtp_MO = [None]*dim_y
	MO_mean_pred = [None]*dim_y
	MO_std_pred = [None]*dim_y
	# xpred = tf.reshape([tf.linspace(-5.,+5.,101)]*5,(101,5))
	Xmin = tf.math.reduce_min(X,axis=0)
	Xmax = tf.math.reduce_max(X,axis=0)
	Xmid = (Xmin+Xmax)/2.
	Xmid_ext = Xmid-(Xmid-Xmin)*1.5
	Xmax_ext = Xmid+(Xmax-Xmid)*1.5
	xpred = tf.linspace(Xmid_ext,Xmax_ext,101)

	# Debug:
	xpred = X
	
	for ii in range(dim_y):

		# rrtp_MO[ii] = RRTPSarkkaFeatures(dim=dim_x,cfg=cfg.RRTPSarkkaFeatures)
		rrtp_MO[ii] = RRTPRandomFourierFeatures(dim=dim_x,cfg=cfg.RRTPRandomFourierFeatures)
		rrtp_MO[ii].update_model(X,Y[:,ii])
		rrtp_MO[ii].train_model()

		# Compute predictive moments:
		MO_mean_pred[ii], cov_pred = rrtp_MO[ii].get_predictive_moments(xpred)
		MO_std_pred[ii] = tf.sqrt(tf.linalg.diag_part(cov_pred))


	# Plot:
	hdl_fig, hdl_splots = plt.subplots(dim_y,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Inverted pendulum simulation x(t)")
	for ii in range(dim_y):
		hdl_splots[ii].plot(MO_mean_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(MO_mean_pred[ii] + 2.*MO_std_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(MO_mean_pred[ii] - 2.*MO_std_pred[ii],linestyle="-",color="b")
		hdl_splots[ii].plot(Y[:,ii],linestyle="None",color="k",marker=".")


	plt.show(block=True)
	
	# Predictions:
	H = 10
	assert (Nevals-1) % H == 0

	MO_mean_pred = np.zeros((H,dim_y),dtype=np.float32)
	MO_std_pred = np.zeros((H,dim_y),dtype=np.float32)
	x_samples = np.zeros((xpred.shape),dtype=np.float32)
	for ii in range(0,xpred.shape[0],H):
		
		# Predict H steops ahead, starting at location x_pred_location:
		x_pred_location = tf.reshape(xpred[ii,:],(1,-1))

		# Predict for horizon H:
		for h in range(H):

			# Compute predictions for each channel:
			for jj in range(dim_y):
				MO_mean_pred[h,jj], cov_pred = rrtp_MO[jj].get_predictive_moments(x_pred_location)
				MO_std_pred[h,jj] = tf.sqrt(tf.linalg.diag_part(cov_pred))

				# Sample from the predictive moments: (assume Gaussian for simplicity):
				x_samples[ii+h,jj] = tf.random.normal(shape=(1,1), mean=MO_mean_pred[h,jj], stddev=MO_std_pred[h,jj])
			
			# Predictive location:å
			x_pred_location = tf.reshape(tf.convert_to_tensor(x_samples[ii+h,:]),(1,-1))

		# Plot predictions on top
		for jj in range(dim_y):
			hdl_splots[jj].plot(np.arange(ii,ii+H),x_samples[ii:ii+H,jj],linestyle="--",color="r")

	# plt.show(block=True)


if __name__ == "__main__":

	test()


