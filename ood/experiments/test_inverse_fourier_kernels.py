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
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from ood.utils.common import CommonUtils
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
	xmin = -5.0
	xmax = +2.0
	Ndiv = 201
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]

	spectral_densities = []; labels = []
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x)]; labels += ["Kink"]
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]
	spectral_densities += [ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x)]; labels += ["Parabola"]

	inverse_fourier_toolboxes = []
	for ii in range(len(spectral_densities)):
		inverse_fourier_toolboxes += [InverseFourierTransformKernelToolbox(spectral_densities[ii],dim_x)]

		fx, ker_diag, cov_diag = get_plotting_quantities(xpred,inverse_fourier_toolboxes[ii])

		hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(12,8),sharex=False)
		hdl_fig.suptitle("Using Spectral density {0:s}".format(labels[ii]),fontsize=fontsize_labels)

		fx_true = spectral_densities[ii]._nonlinear_system_fun(xpred)
		hdl_splots[0].plot(xpred[:,0],fx_true,label=labels[ii],color="grey",lw=1)

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
