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
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


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


def get_unnormalized_spectral_density(omega_in,X,Y,ind):
	"""

	Approximate the quadrature of a multivariate Fourier transform using data

	X: [Npoints,dim]
	Y: [Npoints,dim]
	omega_in: [dim,1] TODO: Extend for [dim, Npoints_omega]
	
	"""

	Y_proj = tf.reshape(Y[:,ind],(-1,1)) # [Npoints,1]	
	omega_times_X = X @ omega_in  # [Npoints,1]
	real_part = tf.reduce_sum(tf.math.cos(omega_times_X) * Y_proj)
	img_part = tf.reduce_sum(tf.math.sin(omega_times_X) * Y_proj)

	return tf.math.sqrt(real_part**2 + img_part**2)


@hydra.main(config_path=".",config_name="config.yaml")
def test(cfg: dict) -> None:


	# Dataset:
	Nevals = 60*2+1
	X,Y,u_vec,policy_fun = simulate_inverted_pendulum(Nsteps=Nevals,visualize=False,x0=None,sort=True) # [Nevals,dim] , [Nevals,dim]

	# Debug:
	X = X[:,0:2]
	Y = Y[:,0:2]

	dim_x = X.shape[1]
	dim_y = Y.shape[1]

	Ndiv = 81
	xx = tf.linspace(-10.0,+10.0,Ndiv)
	yy = tf.linspace(-10.0,+10.0,Ndiv)
	XXpred, YYpred = tf.meshgrid(xx, yy)

	omega_pred = tf.concat( [tf.reshape(XXpred,(-1,1)) , tf.reshape(YYpred,(-1,1))], axis=1 )

	# omega_vec = tf.reshape(tf.linspace(-10.0,+10.0,201),(1,-1))
	S1w_vec = np.zeros((omega_pred.shape[0],1))
	S2w_vec = np.zeros((omega_pred.shape[0],1))
	for k in range(omega_pred.shape[0]):
		omega_in = tf.reshape(omega_pred[k:k+1,:],(-1,1))
		# pdb.set_trace()
		S1w_vec[k,0] = get_unnormalized_spectral_density(omega_in,X,Y,ind=0)
		S2w_vec[k,0] = get_unnormalized_spectral_density(omega_in,X,Y,ind=1)

	S1w = np.reshape(S1w_vec,(Ndiv,Ndiv))
	S2w = np.reshape(S2w_vec,(Ndiv,Ndiv))


	# Plot:
	if dim_x == 2:
		hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
		hdl_fig.suptitle("Inverted pendulum simulation x(t)")
		hdl_splots[0].imshow(S1w)
		hdl_splots[1].imshow(S2w)


	# S1(w) (exact density for state 1 of cart-pole system)
	omega_pred = np.linspace(-10.0,+10.0,501)
	L = 5.0
	Sw_vec = np.abs( 2.*np.sin(L*omega_pred) -2.*L*omega_pred ) / omega_pred**2
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots.plot(omega_pred,Sw_vec)
	plt.show(block=True)


if __name__ == "__main__":

	test()


