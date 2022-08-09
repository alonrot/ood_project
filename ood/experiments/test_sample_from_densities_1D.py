import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity, NoNameSpectralDensity, KinkSharpSpectralDensity
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

	# omega_min = -7.
	# omega_max = +7.
	Nsamples = 101
	# L = 500.0
	# Ndiv = 4001

	L = 500.
	Ndiv = 2001
	# S_vec_plotting, _, omegapred = spectral_density.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,normalize_density_numerically=False)
	S_vec_plotting, _, omegapred = spectral_density.get_Wpoints_discrete(L=L,Ndiv=Ndiv,normalize_density_numerically=False)
	_, _, W_samples_vec = spectral_density.get_Wsamples_from_Sw(Nsamples)

	return W_samples_vec, S_vec_plotting, omegapred


@hydra.main(config_path="./config",config_name="config")
def test(cfg):

	np.random.seed(seed=0)
	dim_x = 1
	
	spectral_densities = []; labels = []
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x)]; labels += ["Kink"]
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]
	spectral_densities += [ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x)]; labels += ["Parabola"]
	spectral_densities += [NoNameSpectralDensity(cfg.spectral_density.noname,cfg.sampler.hmc,dim=dim_x)]; labels += ["NoName"]
	spectral_densities += [KinkSharpSpectralDensity(cfg.spectral_density.kinksharp,cfg.sampler.hmc,dim=dim_x)]; labels += ["KinkSharp"]
	Ndensities = len(spectral_densities)


	hdl_fig, hdl_splots = plt.subplots(Ndensities,1,figsize=(12,8),sharex=True)
	for jj in range(Ndensities):

		W_samples_vec, S_vec_plotting, omegapred = get_samples_and_density(spectral_densities[jj])

		hdl_splots[jj].plot(omegapred,S_vec_plotting,lw=2)
		hdl_splots[jj].plot(W_samples_vec,0.1*np.ones(W_samples_vec.shape[0]),marker="x",color="green",linestyle="None")
		hdl_splots[jj].set_title("Spectral density for {0:s} kernel".format(labels[jj]),fontsize=fontsize_labels)
		hdl_splots[jj].set_xlim([omegapred[0,0],omegapred[-1,0]])
		hdl_splots[jj].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	hdl_splots[-1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)

	plt.show(block=True)
	plt.pause(1)


if __name__ == "__main__":

	test()


