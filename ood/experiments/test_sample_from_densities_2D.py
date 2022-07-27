import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from lqrker.spectral_densities import VanDerPolSpectralDensity, MaternSpectralDensity
from lqrker.utils.parsing import dotdict
import hydra

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

def get_samples_and_density(spectral_density):

	W_samples_vec = None
	# W_samples_vec, S_samples_vec, phi_samples_vec = spectral_density.get_samples() # [Nsamples,1,dim], [Nsamples,], [Nsamples,]

	omega_min = -5.
	omega_max = +5.
	Ndiv = 51
	Sw_vec, phiw_vec, omegapred = spectral_density.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,reshape_for_plotting=True)

	return W_samples_vec, Sw_vec, phiw_vec, omegapred


@hydra.main(config_path="./config",config_name="config")
def test(cfg):
	np.random.seed(seed=0)
	dim_x = 2

	spectral_densities = []
	spectral_densities += [VanDerPolSpectralDensity(cfg=cfg.spectral_density.vanderpol,cfg_sampler=cfg.sampler.hmc,dim=dim_x)]
	spectral_densities += [MaternSpectralDensity(cfg=cfg.spectral_density.matern,cfg_sampler=cfg.sampler.hmc,dim=dim_x)]
	labels = ["VanDerPol","Matern"]

	for kk in range(len(spectral_densities)):
		
		hdl_fig, hdl_splots = plt.subplots(dim_x,2,figsize=(12,8),sharex=True)
		W_samples_vec, S_vec_plotting, phi_vec_plotting, omegapred = get_samples_and_density(spectral_densities[kk])
		for jj in range(dim_x):

			hdl_splots[jj,0].imshow(S_vec_plotting[jj,...])
			if W_samples_vec is not None:
				hdl_splots[jj,0].plot(W_samples_vec,0.1*np.ones(W_samples_vec.shape[0]),marker="x",color="green",linestyle="None")
			hdl_splots[jj,0].set_title("Spectral density for {0:s} kernel".format(labels[kk]),fontsize=fontsize_labels)
			# hdl_splots[jj,0].set_xlim([omegapred[0,0],omegapred[-1,0]])
			hdl_splots[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
			hdl_splots[jj,0].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)

		if np.any(phi_vec_plotting != 0.0):
			for jj in range(dim_x):


				hdl_splots[jj,1].imshow(phi_vec_plotting[jj,...])
				hdl_splots[jj,1].set_title("phi(w) for {0:s} kernel".format(labels[kk]),fontsize=fontsize_labels)
				# hdl_splots[jj,1].set_xlim([omegapred[0,0],omegapred[-1,0]])
				hdl_splots[jj,1].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
				hdl_splots[jj,1].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)

	plt.show(block=True)
	plt.pause(1)


if __name__ == "__main__":

	test()


