import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrblr import ReducedRankBayesianLinearRegression
# from lqrker.models.rrblr import RRTPQuadraticFeatures
import numpy as np
import numpy.random as npr
import scipy

from simple_loop_2dpend import simulate_single_pend



fontsize = 17

"""
### TODO
1) Code up the dynamics model: Give flexibility (inheritance) for features to be NNs or Sarkka features. Think about re-using this for the koopman project
2) Record trajectories from the swin-up policy and compute the model posterior
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

def test():
	
	# Dataset:
	Nevals = 60*2
	X,Y = simulate_inverted_pendulum(Nsteps=Nevals) # [Nevals,dim] , [Nevals,dim]

	# # Debug:
	# X = X[:,2:4]
	# Y = Y[:,2:4]

	# pdb.set_trace()

	dim_x = X.shape[1]
	dim_y = Y.shape[1]

	# Domain limits (needed for Sarkka features)
	L = 10.0*max(tf.math.abs(tf.math.reduce_max(X)),tf.math.abs(tf.math.reduce_min(X)))
	L = L.numpy()

	# Multi-output BLM:
	Nfeat = 200 # Number of features
	sigma_n = 1.0 # Process noise of the underlying dynamics x_{k+1} = f(x_k)
	rrblr_MO = [None]*dim_y
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
		
		rrblr_MO[ii] = ReducedRankBayesianLinearRegression(dim=dim_x,Nfeat=Nfeat,L=L,sigma_n=sigma_n)
		rrblr_MO[ii].update_dataset(X,Y[:,ii])

		# Compute predictive moments:
		MO_mean_pred[ii], cov_pred = rrblr_MO[ii].get_predictive_moments(xpred)
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


if __name__ == "__main__":

	test()


