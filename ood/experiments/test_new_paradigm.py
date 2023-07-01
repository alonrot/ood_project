import tensorflow as tf
import gpflow
import pdb
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
from matplotlib.collections import LineCollection
import numpy as np
import scipy
from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity, QuadrupedSpectralDensity
from lqrker.models import MultiObjectiveReducedRankProcess
from scipy import stats
import hydra
from omegaconf import OmegaConf
import pickle
import control
from lqrker.utils.common import CommonUtils
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)
from min_jerk_gen import min_jerk
import seaborn as sns

# GP flow:
import gpflow as gpf
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("github")
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary

from presentation_LQRlike_kernel_parabola import compute_error_rates

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 30
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# plt.rc('legend',fontsize=fontsize_labels+2)
plt.rc('legend',fontsize=fontsize_labels-2)

# using_deltas = True
# assert using_deltas == True

# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)


def ftrue_call(x_in):
	"""
	x_in: [Npoints,dim_x]
	"""
	if x_in.ndim == 1:
		x_in = np.reshape(x_in,(-1,1))

	y_true = 1.1*x_in[:,0]**2
	return y_true


def ftrue_degenerated(x_in,dist_simu_real):
	"""
	x_in: [Npoints,dim_x]
	"""
	if x_in.ndim == 1:
		x_in = np.reshape(x_in,(-1,1))

	assert dist_simu_real >= 0

	# pdb.set_trace()
	y_true = ftrue_call(x_in[:,0])

	# Move parameters in a line:
	dim_pars = 3
	pars = np.ones(dim_pars) * dist_simu_real / np.sqrt(dim_pars)

	# y_true_deg = y_true - pars[0]*x_in[:,0] + pars[1]*np.exp(-x_in[:,0]**2) + pars[2]*np.cos(x_in[:,0]*10.0)
	y_true_deg = y_true - pars[0]*x_in[:,0] + pars[1]*np.exp(-x_in[:,0]**2) + 0.2*pars[2]*np.cos(x_in[:,0]*10.0)
	# y_true_deg = y_true - pars[0]*x_in[:,0]**2

	return y_true_deg


def convergence_rate_vs_reality_gap(ind_Nobs=5,plot_block=True):
	"""

	Given the learned feature mapping, which is able to capture the "correct" kernel for the simulator,
	the question is now how does this prior help to learn the true function if the true function shares the
	same structure, but has unmodeled part. How do we capture those?

	For example:
		Simulation: f_simu(x) = a*x^2, with a ~N() -> we learn its kernel, k_prior
		Reality: f_true(x) = a_true*x^2 + b_true*x + c_true

	Assumption: Reality is costly, so we can't really afford a lot of evaluations there.

	1) First, show that k_prior helps to learn the true function w.r.t standard choices
	2) Study "how restrictive is our learned prior for the true model" Did we bias the model too much? And if we did, Is the extra uncertainty enough to capture it?
	"""

	my_seed = 1
	tf.random.set_seed(my_seed)
	np.random.seed(my_seed)


	"""
	Steps to visualize the learnging speed vs. distance (simulation-reality)
	1) 	For each distance value, compute RMSE vs. number of evaluations; average RMSE across many different sampled evaluations (see the code for presentation). 
		Each point in the RMSE curve is computed by solving a GP regression problem, where we use the parabola kernel obtained from f_simu
	2) 	Repeat the above for a standard kernel without hyperparameter optimization. Then, repeat with hyperparameter optimization.


	CONCLUSIONS
	1) As the reality gap increases, standard kernels do as good, but thse tailored models have it really hard because their hypothesis space is too limited


	NEXT STEPS
	1) Visualize the kind of GP samples I get with a parabola kernel VS a SE kernel, and compare them with the true function
	2) 

	"""



	Ndiv = 30
	# dd_vec = np.linspace(0.0,0.1,Ndiv)
	dd_vec = np.logspace(-5,1,num=Ndiv)
	mse_error_mean_list = []
	mse_error_std_list = []
	for ii in range(dd_vec.shape[0]):
		dist = dd_vec[ii]
		ftrue_degenerated_fixed_pars = lambda x_in: ftrue_degenerated(x_in,dist_simu_real=dist)
		mse_error_mean, mse_error_std, _, _ = compute_error_rates(ftrue_call=ftrue_degenerated_fixed_pars,block_plot=False,plotting=False,ind_fig=100,savefig=False)

		mse_error_mean_list += [mse_error_mean]
		mse_error_std_list += [mse_error_std]


	color_list = sns.color_palette("deep",3)
	legend_list = ["SE","ESP","PAR"]
	mse_error_mean_list_cut = np.zeros((Ndiv,3))
	for ee in range(Ndiv):
		mse_error_mean_list_cut[ee,:] = mse_error_mean_list[ee][ind_Nobs,:]


	hdl_fig_error_rates, hdl_splots_error_rates = plt.subplots(1,1,figsize=(12,8),sharex=True)
	for kk in range(mse_error_mean_list_cut.shape[1]):
		hdl_splots_error_rates.plot(dd_vec,mse_error_mean_list_cut[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2",label=legend_list[kk])
		# hdl_splots_error_rates.fill_between(mse_error_mean[:,kk],mse_error_mean[:,kk] - 2.*mse_error_std[:,kk],mse_error_mean[:,kk] + 2.*mse_error_std[:,kk],color="brown",alpha=0.9)
		hdl_splots_error_rates.set_yscale("log")
		hdl_splots_error_rates.set_xscale("log")
		hdl_splots_error_rates.set_ylabel(r"$|f_{true}(x_*) - \mu(x_*)|$",fontsize=fontsize_labels)

		# hdl_splots_error_rates.plot(loglikneg_error_mean[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2")
		# # hdl_splots_error_rates.fill_between(loglikneg_error_mean[:,kk],loglikneg_error_mean[:,kk] - 2.*loglikneg_error_std[:,kk],loglikneg_error_mean[:,kk] + 2.*loglikneg_error_std[:,kk],color="brown",alpha=0.9)
		# hdl_splots_error_rates.set_yscale("log")
		# hdl_splots_error_rates.set_ylabel(r"$-\log p(f_{true}(x_*) | f(x_*))$",fontsize=fontsize_labels)


	hdl_splots_error_rates.set_title(r"RMSE after {0:d} evaluations".format(ind_Nobs),fontsize=fontsize_labels)
	hdl_splots_error_rates.grid(visible=True,which="major",axis="both")
	hdl_splots_error_rates.set_xlabel(r"Distance between real and simulation (reality gap)",fontsize=fontsize_labels)
	hdl_splots_error_rates.set_xlim([dd_vec[0],dd_vec[-1]])
	hdl_splots_error_rates.legend()

	hdl_splots_error_rates.set_ylim([1e-12,1e2])

	plt.show(block=plot_block)




def convergence_rate_vs_Nobs(plot_block=True):

	# Plot stuff:
	hdl_fig_error_rates, hdl_splots_error_rates = plt.subplots(2,1,figsize=(12,8),sharex=True)
	xmin = -1.0
	xmax = +1.0
	epsi = 1e-3
	Npred = 120
	dim_in = 1
	xpred = np.reshape(np.linspace(xmin+epsi,xmax-epsi,Npred),(-1,dim_in))
	ftrue_plotting = ftrue_call(xpred)
	ftrue_degenerated_plotting = ftrue_degenerated(xpred,dist_simu_real=1.0)
	hdl_splots_error_rates[0].plot(xpred[:,0],ftrue_plotting)
	hdl_splots_error_rates[1].plot(xpred[:,0],ftrue_degenerated_plotting)
	hdl_splots_error_rates[1].set_xlim([-1,1])
	plt.show(block=False)


	Ndiv = 6
	# dd_vec = np.linspace(0.0,0.1,Ndiv)
	dd_vec = np.logspace(-5,1,num=Ndiv)
	mse_error_mean_list = []
	mse_error_std_list = []
	savefig = True
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	for ii in range(dd_vec.shape[0]):
		dist = dd_vec[ii]
		print(dist)
		ftrue_degenerated_fixed_pars = lambda x_in: ftrue_degenerated(x_in,dist_simu_real=dist)
		mse_error_mean, mse_error_std, _, _ = compute_error_rates(ftrue_call=ftrue_degenerated_fixed_pars,ker_sum_tradeoff=(ii+1)/dd_vec.shape[0],block_plot=False,plotting=True,ind_fig=ii,savefig=True)

		# print(mse_error_std)

		# mse_error_mean_list += [mse_error_mean]
		# mse_error_std_list += [mse_error_std]


		# if savefig:
		# 	path2save_fig = "{0:s}/RMSE_{1:d}.png".format(path2folder,ii)
		# 	logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		# 	hdl_fig_error_rates.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		# 	logger.info("Done saving fig!")
		# else:
		# 	plt.pause(0.1)

	plt.show(block=True)

@hydra.main(config_path="./config",config_name="config")
def convergence_rate_vs_reality_gap_list(cfg):

	ind_Nobs_list = [2,8]
	for ind in ind_Nobs_list:
		convergence_rate_vs_reality_gap(ind,plot_block=False)

	plt.show(block=True)



if __name__ == "__main__":

	# convergence_rate_vs_reality_gap_list()


	convergence_rate_vs_Nobs()





