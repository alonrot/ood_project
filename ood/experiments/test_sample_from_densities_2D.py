import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from lqrker.spectral_densities import VanDerPolSpectralDensity, MaternSpectralDensity, SquaredExponentialSpectralDensity
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

	# omega_min = -5.
	# omega_max = +5.
	# Ndiv = 51
	# Sw_vec, phiw_vec, omegapred = spectral_density.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,reshape_for_plotting=True)


	Ndiv = 81
	L = 100.0
	Sw_vec, phiw_vec, omegapred = spectral_density.get_Wpoints_discrete(L=L,Ndiv=Ndiv,normalize_density_numerically=False,reshape_for_plotting=True)

	if "part_real_dbg" in dir(spectral_density):

		# DBG/debug:
		part_real_dbg = [ np.reshape(spectral_density.part_real_dbg[:,ii],(Ndiv,Ndiv)) for ii in range(spectral_density.dim) ]
		part_real_dbg = np.stack(part_real_dbg) # [dim, Ndiv, Ndiv]
		part_imag_dbg = [ np.reshape(spectral_density.part_imag_dbg[:,ii],(Ndiv,Ndiv)) for ii in range(spectral_density.dim) ]
		part_imag_dbg = np.stack(part_imag_dbg) # [dim, Ndiv, Ndiv]

	else:
		part_real_dbg = Sw_vec
		part_imag_dbg = Sw_vec

	return W_samples_vec, Sw_vec, phiw_vec, omegapred, part_real_dbg, part_imag_dbg


@hydra.main(config_path="./config",config_name="config")
def test(cfg):
	np.random.seed(seed=0)
	dim_x = 2

	spectral_densities = []; labels = []
	spectral_densities += [VanDerPolSpectralDensity(cfg=cfg.spectral_density.vanderpol,cfg_sampler=cfg.sampler.hmc,dim=dim_x)]; labels += ["VanDerPol"]
	spectral_densities += [MaternSpectralDensity(cfg=cfg.spectral_density.matern,cfg_sampler=cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]

	for kk in range(len(spectral_densities)):
		
		hdl_fig, hdl_splots = plt.subplots(dim_x,2,figsize=(14,10),sharex=False)
		hdl_fig.suptitle(r"Spectral density $S(\omega) = [S_1(\omega),S_2(\omega)]$ and spectral phase $\varphi(\omega) = [\varphi_1(\omega), \varphi_2(\omega)]$ for {0:s} kernel".format(labels[kk]),fontsize=fontsize_labels)
		W_samples_vec, S_vec_plotting, phi_vec_plotting, omegapred, part_real_dbg, part_imag_dbg = get_samples_and_density(spectral_densities[kk])

		# # DBG/debug:
		# S_vec_plotting = part_real_dbg
		# phi_vec_plotting = part_imag_dbg
		
		extent_plot = [omegapred[0,0],omegapred[-1,0],omegapred[0,1],omegapred[-1,1]] #  scalars (left, right, bottom, top)
		for jj in range(dim_x):

			hdl_splots[jj,0].imshow(S_vec_plotting[jj,...],extent=extent_plot)
			if W_samples_vec is not None:
				# hdl_splots[jj,0].plot(W_samples_vec,0.1*np.ones(W_samples_vec.shape[0]),marker="x",color="green",linestyle="None")
				raise NotImplementedError
			my_title = "S_{0:d}(\omega)".format(jj+1)
			hdl_splots[jj,0].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
			# hdl_splots[jj,0].set_xlim([omegapred[0,0],omegapred[-1,0]])
			if jj == dim_x-1: hdl_splots[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
			hdl_splots[jj,0].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)

		if np.any(phi_vec_plotting != 0.0):
			for jj in range(dim_x):

				hdl_splots[jj,1].imshow(phi_vec_plotting[jj,...],extent=extent_plot)
				my_title = "\\varphi_{0:d}(\omega)".format(jj+1)
				hdl_splots[jj,1].set_title(r"${0:s}$".format(my_title),fontsize=fontsize_labels)
				# hdl_splots[jj,1].set_xlim([omegapred[0,0],omegapred[-1,0]])
				if jj == dim_x-1: hdl_splots[jj,0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
				hdl_splots[jj,1].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)
		else:
			for jj in range(dim_x): hdl_splots[jj,1].set_xticks([],[]); hdl_splots[jj,1].set_yticks([],[])

	plt.show(block=True)
	plt.pause(1)


if __name__ == "__main__":

	test()


