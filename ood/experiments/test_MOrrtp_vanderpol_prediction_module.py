import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from lqrker.models import MultiObjectiveReducedRankProcess
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity
from lqrker.utils.common import CommonUtils
import numpy as np
import scipy
import hydra
from omegaconf import OmegaConf
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

from test_MOrrtp_vanderpol import generate_training_data, initialize_GPmodel_with_existing_data, simulate_nonlinsystem


@hydra.main(config_path="./config",config_name="config")
def main(cfg: dict, block_plot=True, which_kernel="vanderpol") -> None:
	"""
	
	
	
	"""

	# which_kernel = "matern"

	print(OmegaConf.to_yaml(cfg))

	my_seed = 4
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	Nsteps = 500
	dim_x = 2
	Nx0_training = 5
	nonlinear_system_fun_vanderpol = VanDerPolSpectralDensity._controlled_vanderpol_dynamics
	Xlatent_list, Ylatent_list = generate_training_data(Nsteps,dim_x,Nx0_training,nonlinear_system_fun_vanderpol)
	Xtrain = tf.convert_to_tensor(value=np.concatenate(Xlatent_list,axis=0),dtype=np.float32)
	Ytrain = tf.convert_to_tensor(value=np.concatenate(Ylatent_list,axis=0),dtype=np.float32)

	rrtp_MO, Ndiv = initialize_GPmodel_with_existing_data(cfg,dim_x,Xtrain,Ytrain,which_kernel)

	# At every x_t, predict x_{t:t+H} states. Compare those predictions with the observed y_{t:t+H}. Compute OOD. Repeat for time t+H+1
	Nhorizon = 20
	x0 = np.random.rand(1,dim_x)
	
	x_traj_real_list = []
	x_traj_pred_list = []
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots.set_xlabel(r"$x_1$"); hdl_splots.set_ylabel(r"$x_2$")
	for ii in range(Nsteps//Nhorizon):

		print("Iteration {0:d}".format(ii+1))

		# use_nominal_model = ii <= 0.5*(Nsteps//Nhorizon)
		use_nominal_model = True

		x_traj_real, _, _, _ = simulate_nonlinsystem(Nhorizon+1,x0,nonlinear_system_fun_vanderpol,visualize=False,use_nominal_model=use_nominal_model) # [Nsteps-1,dim]
		x_traj_real_list += [x_traj_real]

		# TODO:
		# Update the GP with the acquired trajectory data? -> Maybe not; let's separate here training from testing. This is the testing phase.
		# We trained above with similar trajectories

		x1 = np.reshape(x_traj_real[1,:],(1,dim_x))
		x_traj_pred, _ = rrtp_MO.sample_state_space_from_prior_recursively(x0=x0,x1=x1,traj_length=Nhorizon+1,Nsamples=10,plotting=False,retrain_withx0x1=False) # [Nhorizon-1,dim_y,Nsamples]
		x_traj_pred_list += [x_traj_pred]

		x0 = np.reshape(x_traj_real[-1,:],(1,dim_x))


		# Plot stuff:
		hdl_splots.plot(x_traj_real[:,0],x_traj_real[:,1],marker=".",linestyle="-",color="r",lw=1)
		for ii in range(x_traj_pred.shape[2]):
			hdl_splots.plot(x_traj_pred[:,0,ii],x_traj_pred[:,1,ii],marker=".",linestyle="-",color="grey",lw=0.5)

	plt.show(block=True)




if __name__ == "__main__":

	main()












