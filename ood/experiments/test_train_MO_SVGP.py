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

import gpflow as gpf


from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
MAXITER = ci_niter(2000)

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
np.random.seed(0)


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

	dim_x = X.shape[1]
	dim_y = Y.shape[1]

	# Multi-output BLM:
	sigma_n = 1.0 # Process noise of the underlying dynamics x_{k+1} = f(x_k)

	# Inducing points:
	M = 15
	Xmin = tf.math.reduce_min(X,axis=0)
	Xmax = tf.math.reduce_max(X,axis=0)
	Xmid = (Xmin+Xmax)/2.
	Xmid_ext = Xmid-(Xmid-Xmin)*1.5
	Xmax_ext = Xmid+(Xmax-Xmid)*1.5
	Zinit = np.linspace(Xmid_ext,Xmax_ext,M)
	# initialization of inducing input locations (M random points from the training inputs)
	Z = Zinit.copy()
	iv = gpf.inducing_variables.SharedIndependentInducingVariables( gpf.inducing_variables.InducingPoints(Z) )

	# Kernel:
	kernel = gpf.kernels.SharedIndependent( gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=dim_y )
	
	# Create SVGP model:
	m = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=dim_y)
	print_summary(m)

	# Cast data and inducing points:
	Z = np.ndarray.astype(Z,dtype=np.float64)
	X = tf.cast(X,dtype=tf.float64)
	Y = tf.cast(Y,dtype=tf.float64)

	# Optimize:
	def optimize_model_with_scipy(model):
		optimizer = gpf.optimizers.Scipy()
		optimizer.minimize(
			model.training_loss_closure((X,Y)),
			variables=model.trainable_variables,
			method="l-bfgs-b",
			options={"disp": True, "maxiter": MAXITER},
		)

	optimize_model_with_scipy(m)
	# print_summary(m)

	# Prediction points:
	# xpred = tf.linspace(Xmid_ext,Xmax_ext,101)
	# xpred = tf.cast(xpred,dtype=tf.float64)
	xpred = X
	MO_mean_pred, MO_var_pred = m.predict_y(xpred)

	# Plot:
	hdl_fig, hdl_splots = plt.subplots(dim_y,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Inverted pendulum simulation x(t)")
	for ii in range(dim_y):
		hdl_splots[ii].plot(MO_mean_pred[:,ii],linestyle="-",color="b")
		hdl_splots[ii].plot(MO_mean_pred[:,ii] + 2.*MO_var_pred[:,ii]**0.5,linestyle="-",color="b")
		hdl_splots[ii].plot(MO_mean_pred[:,ii] - 2.*MO_var_pred[:,ii]**0.5,linestyle="-",color="b")
		hdl_splots[ii].plot(Y[:,ii],linestyle="None",color="k",marker=".")

	plt.show(block=True)


if __name__ == "__main__":

	test()


