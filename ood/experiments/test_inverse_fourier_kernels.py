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
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity, KinkSharpSpectralDensity, VanDerPolSpectralDensity
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
	# cov_diag 	= inverse_fourier_toolbox.get_covariance_diagonal(xpred)
	cov_diag = np.zeros(xpred.shape[0])

	return fx, ker_diag, cov_diag


@hydra.main(config_path="./config",config_name="config")
def test_ifou_1D(cfg):
	
	dim_x = 1

	spectral_densities = []; labels = []
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x)]; labels += ["Kink"]
	spectral_densities += [KinkSharpSpectralDensity(cfg.spectral_density.kinksharp,cfg.sampler.hmc,dim=dim_x)]; labels += ["KinkSharp"]
	spectral_densities += [ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x)]; labels += ["Parabola"]
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]

	# Create grid for plotting:
	xmin = -5.0
	xmax = +2.0
	Ndiv = 201
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]

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

	plt.show(block=False)


@hydra.main(config_path="./config",config_name="config")
def test_ifou_2D(cfg):
	
	dim_x = 2

	spectral_densities = []; labels = []
	spectral_densities += [VanDerPolSpectralDensity(cfg=cfg.spectral_density.vanderpol,cfg_sampler=cfg.sampler.hmc,dim=dim_x)]; labels += ["VanDerPol"]
	spectral_densities += [MaternSpectralDensity(cfg=cfg.spectral_density.matern,cfg_sampler=cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]

	# Create grid for plotting:
	xmin = -5.0
	xmax = +5.0
	Ndiv = 51
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]

	inverse_fourier_toolboxes_0 = []
	inverse_fourier_toolboxes_1 = []
	for ii in range(len(spectral_densities)):

		hdl_fig, hdl_splots = plt.subplots(3,2,figsize=(13,9),sharex=False)
		hdl_fig.suptitle("Using Spectral density {0:s}".format(labels[ii]),fontsize=fontsize_labels)

		inverse_fourier_toolboxes_0 += [InverseFourierTransformKernelToolbox(spectral_densities[ii],dim_x,dim_out_ind=0)]
		inverse_fourier_toolboxes_1 += [InverseFourierTransformKernelToolbox(spectral_densities[ii],dim_x,dim_out_ind=1)]

		# Prediction:
		fx_0, ker_diag_0, _ = get_plotting_quantities(xpred,inverse_fourier_toolboxes_0[ii])
		fx_1, ker_diag_1, _ = get_plotting_quantities(xpred,inverse_fourier_toolboxes_1[ii])
		fx_0_plotting = np.reshape(fx_0,(Ndiv,Ndiv))
		fx_1_plotting = np.reshape(fx_1,(Ndiv,Ndiv))
		
		# True function:
		fx_true = spectral_densities[ii]._nonlinear_system_fun(xpred)
		fx_true_0_plotting = np.reshape(fx_true[:,0],(Ndiv,Ndiv))
		fx_true_1_plotting = np.reshape(fx_true[:,1],(Ndiv,Ndiv))

		# Variance k(x,x):
		ker_diag_0_plotting = np.reshape(ker_diag_0,(Ndiv,Ndiv))
		ker_diag_1_plotting = np.reshape(ker_diag_1,(Ndiv,Ndiv))

		hdl_splots[0,0].imshow(fx_true_0_plotting,extent=(xmin,xmax,xmin,xmax))
		hdl_splots[0,0].set_title("True f(x) - Channel 0")
		hdl_splots[1,0].imshow(fx_0_plotting,extent=(xmin,xmax,xmin,xmax))
		hdl_splots[1,0].set_title("Approximate f(x) - Channel 0")
		# hdl_splots[2,0].imshow(np.abs(fx_true_0_plotting-fx_0_plotting),extent=(xmin,xmax,xmin,xmax))
		# hdl_splots[2,0].set_title("Error - Channel 0")
		hdl_splots[2,0].imshow(ker_diag_0_plotting,extent=(xmin,xmax,xmin,xmax))
		hdl_splots[2,0].set_title("Variance k(x,x) - Channel 0")


		hdl_splots[0,1].imshow(fx_true_1_plotting,extent=(xmin,xmax,xmin,xmax))
		hdl_splots[0,1].set_title("True f(x) - Channel 1")
		hdl_splots[1,1].imshow(fx_1_plotting,extent=(xmin,xmax,xmin,xmax))
		hdl_splots[1,1].set_title("Approximate f(x) - Channel 1")
		# hdl_splots[2,1].imshow(np.abs(fx_true_0_plotting-fx_0_plotting),extent=(xmin,xmax,xmin,xmax))
		# hdl_splots[2,1].set_title("Error - Channel 1")
		hdl_splots[2,1].imshow(ker_diag_1_plotting,extent=(xmin,xmax,xmin,xmax))
		hdl_splots[2,1].set_title("Variance k(x,x) - Channel 1")


	plt.show(block=True)



if __name__ == "__main__":

	test_ifou_1D()
	test_ifou_2D()
