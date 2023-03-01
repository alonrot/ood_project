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
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity, KinkSharpSpectralDensity, VanDerPolSpectralDensity, DubinsCarSpectralDensity
from lqrker.spectral_densities.base import SpectralDensityBase
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from lqrker.utils.common import CommonUtils
import hydra
import pickle
from ood.spectral_density_approximation.elliptical_slice_sampler import EllipticalSliceSampler
import tensorflow as tf
# import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


def plotting():

	dim_in = 2
	Nsamples_omega = 6**2
	omega_lim = 3.0
	samples_omega = -omega_lim + 2.*omega_lim*tf.math.sobol_sample(dim=dim_in,num_results=Nsamples_omega,skip=2000 + np.random.randint(0,100)*100)

	omega_lim_plot = omega_lim*4.

	hdl_fig, hdl_splots = plt.subplots(1,3,figsize=(17,6),sharex=False)
	hdl_splots[0].plot(samples_omega[:,0],samples_omega[:,1],linestyle="None",marker="o",markerfacecolor="navy",alpha=0.5,markersize=12)
	hdl_splots[0].set_xlim([-omega_lim_plot,omega_lim_plot])
	hdl_splots[0].set_ylim([-omega_lim_plot,omega_lim_plot])
	hdl_splots[0].set_xticks([])
	hdl_splots[0].set_yticks([])
	hdl_splots[0].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
	hdl_splots[0].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)
	hdl_splots[0].set_title("Irregular grid - Uniform",fontsize=fontsize_labels)

	samples_omega = -omega_lim_plot + 2.*omega_lim_plot*tf.math.sobol_sample(dim=dim_in,num_results=Nsamples_omega,skip=2000 + np.random.randint(0,100)*100)

	hdl_splots[1].plot(samples_omega[:,0],samples_omega[:,1],linestyle="None",marker="o",markerfacecolor="navy",alpha=0.5,markersize=12)
	hdl_splots[1].set_xlim([-omega_lim_plot,omega_lim_plot])
	hdl_splots[1].set_ylim([-omega_lim_plot,omega_lim_plot])
	# hdl_splots[1].set_xticks([])
	hdl_splots[1].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
	# hdl_splots[1].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)
	hdl_splots[1].set_title("Irregular grid - Uniform",fontsize=fontsize_labels)
	hdl_splots[1].set_xticks([])
	hdl_splots[1].set_yticks([])

	samples_omega = CommonUtils.create_Ndim_grid(xmin=-omega_lim_plot,xmax=omega_lim_plot,Ndiv=int(np.sqrt(Nsamples_omega)),dim=dim_in) # [Ndiv**dim_in,dim_in]

	hdl_splots[2].plot(samples_omega[:,0],samples_omega[:,1],linestyle="None",marker="o",markerfacecolor="navy",alpha=0.5,markersize=12)
	hdl_splots[2].set_xlim([-omega_lim_plot,omega_lim_plot])
	hdl_splots[2].set_ylim([-omega_lim_plot,omega_lim_plot])
	# hdl_splots[2].set_xticks([])
	hdl_splots[2].set_xlabel(r"$\omega_1$",fontsize=fontsize_labels)
	# hdl_splots[2].set_ylabel(r"$\omega_2$",fontsize=fontsize_labels)
	hdl_splots[2].set_title("Regular grid",fontsize=fontsize_labels)
	hdl_splots[2].set_xticks([])
	hdl_splots[2].set_yticks([])




	plt.show(block=True)






if __name__ == "__main__":

	plotting()

