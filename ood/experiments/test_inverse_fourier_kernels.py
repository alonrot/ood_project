import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from scipy import integrate
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParabolaSpectralDensity
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
import hydra

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


def get_plotting_quantities(xpred,inverse_fourier_toolbox):


	fx 			= inverse_fourier_toolbox.get_fx(xpred)
	ker_diag 	= inverse_fourier_toolbox.get_kernel_diagonal(xpred)
	cov_diag 	= inverse_fourier_toolbox.get_covariance_diagonal(xpred)

	return fx, ker_diag, cov_diag


@hydra.main(config_path="./config",config_name="config")
def test_ifou(cfg):

	dim_x = 1
	Npoints = 201
	xmin = -5.0
	xmax = +2.0
	xpred = np.linspace(xmin,xmax,Npoints)
	xpred = np.reshape(xpred,(-1,dim_x)) # [Nsteps,dim]

	labels = ["Kink","Matern","SE","Parabola"]

	spectral_densities = []
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x)]
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]
	spectral_densities += [ParabolaSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x)]

	inverse_fourier_toolboxes = []
	for ii in range(len(spectral_densities)):
		inverse_fourier_toolboxes += [InverseFourierTransformKernelToolbox(spectral_densities[ii],dim_x)]

		fx, ker_diag, cov_diag = get_plotting_quantities(xpred,inverse_fourier_toolboxes[ii])


		hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(12,8),sharex=False)
		hdl_fig.suptitle("Using Spectral density {0:s}".format(labels[ii]),fontsize=fontsize_labels)

		if labels[ii] == "Kink":

			# Overlay true function:
			fx_true = spectral_densities[ii]._kink_fun(xpred[:,0])
			hdl_splots[0].plot(xpred[:,0],fx_true,label="kink",color="grey",lw=1)

		if labels[ii] == "Parabola":

			# Overlay true function:
			fx_true = spectral_densities[ii]._parabola_fun(xpred[:,0])
			hdl_splots[0].plot(xpred[:,0],fx_true,label="parabola",color="grey",lw=1)

		hdl_splots[0].plot(xpred[:,0],fx,label=labels[ii],color="red",lw=1,linestyle="--")
		hdl_splots[0].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
		hdl_splots[0].set_xlim([xmin,xmax])

		hdl_splots[1].plot(xpred[:,0],ker_diag,label=labels[ii])
		hdl_splots[1].set_xlim([xmin,xmax])
		hdl_splots[1].set_ylabel(r"$k(x_t,x_t)$",fontsize=fontsize_labels)
		# hdl_splots[1].legend(loc="right")

		hdl_splots[2].plot(xpred[:,0],cov_diag,label=labels[ii])
		hdl_splots[2].set_ylabel(r"$cov(x_t,x_t)$",fontsize=fontsize_labels)
		hdl_splots[2].set_xlim([xmin,xmax])
		
		hdl_splots[2].set_xlabel(r"$x_t$",fontsize=fontsize_labels)

	plt.show(block=True)





if __name__ == "__main__":

	test_ifou()
