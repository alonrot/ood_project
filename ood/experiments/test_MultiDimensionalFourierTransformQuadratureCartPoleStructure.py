import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPSarkkaFeatures, RRTPRandomFourierFeatures
from lqrker.utils.spectral_densities import MultiDimensionalFourierTransformQuadratureFromData, MultiDimensionalFourierTransformQuadratureCartPoleStructure
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

	u_vec = tf.reshape(tf.cast(tf.constant(u_vec[0:-1]),dtype=tf.float32),(-1,1))

	dim_x = X.shape[1]
	dim_y = Y.shape[1]
	
	plot_dim = 2
	Ndiv = 31
	w_lim = 2.0
	xx1,xx2,xx3,xx4 = tf.meshgrid(*[tf.linspace(-w_lim,+w_lim,Ndiv)]*4)
	omega_pred = tf.concat( [tf.reshape(xx1,(-1,1)), tf.reshape(xx2,(-1,1)), tf.reshape(xx3,(-1,1)), tf.reshape(xx4,(-1,1))], axis=1 )
	
	hdl_fig, hdl_splots = plt.subplots(2,2,figsize=(12,8),sharex=False)
	hdl_fig.suptitle("S(w|x0), showing a slice S(w1,w2,-w_lim,-w_lim)")

	for ii in [0,1]:
		for jj in [0,1]:

			ind = ii*2 + jj

			spectral_density = MultiDimensionalFourierTransformQuadratureCartPoleStructure(cfg.RRTPRandomFourierFeatures.spectral_density_pars, X, u_vec, ind=ind)
			Sw_vec = spectral_density.unnormalized_density(omega_pred,log=False)
			Sw = np.reshape(Sw_vec,(Ndiv,Ndiv,Ndiv,Ndiv))

			if plot_dim == 1:
				pdb.set_trace()
				# hdl_splots[ii,jj].plot(omega_pred[:,0,0,0],Sw[:,0,0,0])
			elif plot_dim == 2:
				hdl_splots[ii,jj].imshow(Sw[:,:,0,0])
			else:
				raise ValueError

	plt.show(block=True)


if __name__ == "__main__":

	test()


