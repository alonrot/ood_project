import tensorflow as tf
# import tensorflow_probability as tfp
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
import hydra
from omegaconf import OmegaConf
import pickle
from lqrker.spectral_densities import ExponentiallySuppressedPolynomialsFromData
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from ood.spectral_density_approximation.reconstruct_function_from_spectral_density import ReconstructFunctionFromSpectralDensity
from lqrker.utils.common import CommonUtils
from lqrker.utils.parsing import get_logger
import seaborn as sns
logger = get_logger(__name__)

from test_new_model_deep_karhunen_loeve import LossMomentMatching, SaveBestModel, DeepKarhunenLoeve, FourierLayerNonStationary, test_model

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 28
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

# using_deltas = True
using_deltas = False

eps_ = 1e-3
dim_in = 1
dim_ctx = dim_in + 1
dim_out = 1

# COLOR_MAP = "seismic"
# COLOR_MAP = "gist_heat"
COLOR_MAP = "copper"

saving_counter = 106

figsize_many = (19,4)

def generate_moving_samples(mean,cov,Nsamples):
	"""

	Code parsed from Matlab to Python using Philipp's animations:

	@TechReport{hennig-tr-is-8,
	author = {Philipp Hennig},
	title = {Animating Samples from Gaussian Distributions},
	institution = {Max Planck Institute for Intelligent Systems},
	year = 2013,
	type = {Technical Report},
	number = 8,
	address = {Spemannstra{\ss}e, 72076 T{\"u}bingen, Germany},
	month = {September}
	}

	http://mlss.tuebingen.mpg.de/2013/2013/Hennig_2013_Animating_Samples_from_Gaussian_Distributions.pdf
	Matlab source code: 'matlab_philipp.m'


	"""

	Ndiv = cov.shape[1]

	# pdb.set_trace()
	L = tf.linalg.cholesky(cov + 1e-6*tf.eye(cov.shape[1],dtype=tf.float64)) # cov = L.L^T

	# Sample and normalize 2 noise vectors from N(0,1):
	noise_vec0 = tf.random.normal(shape=(Ndiv,1),mean=0,stddev=1,dtype=np.float64)
	noise_vec0_norm = tf.norm(noise_vec0,ord="euclidean")
	noise_vec0 = noise_vec0 / noise_vec0_norm

	noise_vec1 = tf.random.normal(shape=(Ndiv,1),mean=0,stddev=1,dtype=np.float64) # [Ndiv,1]
	# pdb.set_trace()
	noise_vec1 = noise_vec1 - (tf.transpose(noise_vec1)@noise_vec0)*noise_vec0 # Orthogonalise by Gram-Schmidt (not strictly needed)
	noise_vec1 = noise_vec1 / tf.norm(noise_vec1,ord="euclidean")

	# Divide the phase interval [0,2pi] Ndiv_circle times:
	Ndiv_circle = Nsamples + 2 # We'll need to later remove the last point and the column of M
	phase_samples_aux = np.linspace(0.0,2.*np.pi,Ndiv_circle,dtype=np.float64).reshape(1,-1) # [1,Ndiv_circle]
	phase_samples = phase_samples_aux[:,0:-1] # Exclude the last point because cos(0) = cos(2pi)

	# Span the phases on each element of noise_vec1
	noise_vec = noise_vec1 @ phase_samples # [Ndiv,1] x [1,Ndiv_circle] -> [Ndiv,Ndiv_circle]

	# Project onto sphere:
	theta_vec = tf.reshape(tf.math.sqrt(tf.math.reduce_sum(noise_vec**2,axis=0)),(1,-1)) # [1,Ndiv_circle]
	M = noise_vec0 @ tf.math.cos(theta_vec) + noise_vec*tf.tile(tf.math.sin(theta_vec)/theta_vec,multiples=[Ndiv,1])
	Mnew = M[:,1::] # Exclude the first point, as it is all NaNs because the first element of theta_vec is zero
	Mnew = noise_vec0_norm*Mnew # Rescale with noise_vec0_norm
	
	samples = mean + L[0,:] @ Mnew
	# samples = tf.transpose(L[0,:]) @ Mnew
	# samples = tf.transpose(Mnew) @ tf.transpose(L[0,:])
	# samples = tf.transpose(samples)

	return samples # [Ndiv x Nsamples]

def ker_fun_SE(x,xp,ls=0.2,var_prior=2.):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	out: [Npoints,Npoints]
	"""

	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	# x = squash(x)
	# xp = squash(xp)

	xp = xp.T


	dist = (x - xp)**2
	dist_scaled = dist / ls**2

	return var_prior*np.exp(-dist_scaled)


	# return 1./(1.-alpha*x*xp)
	# return 1./(1.-alpha)*(x**2)*(xp**2)

def ker_fun_exp_sup_polys(x,xp,alpha=0.9):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	out: [Npoints,Npoints]
	"""

	raise ValueError("We shouldn't be calling this anymore, but ker_fun_true_plus_sta() instead")

	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	# x = squash(x)
	# xp = squash(xp)

	xp = xp.T
	return 1./(1.-alpha*x*xp)

def ker_fun_true_plus_sta(x,xp,alpha=0.9):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	ker_sum_tradeoff_global in [0,1]
	ker_sum_tradeoff_global = 0 : only parabola kernel
	ker_sum_tradeoff_global = 1 : only stationary kernel


	out: [Npoints,Npoints]
	"""

	global ker_sum_tradeoff_global

	assert ker_sum_tradeoff_global <= 1.0 and ker_sum_tradeoff_global >= 0.0
	# print("ker_sum_tradeoff_global:",ker_sum_tradeoff_global)

	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	# x = squash(x)
	# xp = squash(xp)

	kXX_par = ker_fun_parabola(x,xp,alpha=0.9)
	kXX_SE = ker_fun_SE(x,xp,ls=0.1,var_prior=0.5)

	kXX = (1.-ker_sum_tradeoff_global)*kXX_par + ker_sum_tradeoff_global*kXX_SE

	return kXX

def ker_fun_parabola(x,xp,alpha=0.9):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	out: [Npoints,Npoints]
	"""

	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	# x = squash(x)
	# xp = squash(xp)

	xp = xp.T
	return 1./(1.-alpha)*(x**2)*(xp**2)


def plotting_posterior(plot_stuff=False,block_plot=False):

	ker_fun_list = [ker_fun_SE,ker_fun_exp_sup_polys,ker_fun_parabola]
	Nsamples = 3
	Npred = 120
	mvn0_samples = np.random.randn(Nsamples,Npred)

	hdl_fig_ker, hdl_splots_ker_fit = plt.subplots(3,1,figsize=(12,8),sharex=True)

	ind_Xevals_sel_tot = [60,90,10,75,25,110,55]
	savefig = False
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	for ee in range(len(ind_Xevals_sel_tot)+1):

		if ee == 0:
			ind_Xevals_sel = []
		else:
			ind_Xevals_sel = ind_Xevals_sel_tot[0:ee]

		for kk in range(len(ker_fun_list)):
			compute_posterior(ker_fun_list[kk],mvn0_samples,hdl_splots_ker_fit[kk],Nsamples,ind_Xevals_sel,Npred,block_plot=False,plotting=True)

			hdl_splots_ker_fit[kk].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
			hdl_splots_ker_fit[kk].set_xticks([])
			hdl_splots_ker_fit[kk].set_yticks([])
			hdl_splots_ker_fit[kk].set_xticks([-1,0,1])
			hdl_splots_ker_fit[kk].set_yticks([])

		hdl_splots_ker_fit[-1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
		# hdl_splots_ker_fit.set_title(r"New samples",fontsize=fontsize_labels)
		# hdl_splots_ker_fit.set_title(titles_list[cc],fontsize=fontsize_labels)


		if savefig:
			path2save_fig = "{0:s}/gpfit{1:d}.png".format(path2folder,ee)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1.)

	plt.show(block=True)


def compute_error_rates(ftrue_call=None,ker_sum_tradeoff=0.5,block_plot=True,plotting=True,ind_fig=100,savefig=False):

	global ker_sum_tradeoff_global
	ker_sum_tradeoff_global = ker_sum_tradeoff

	# ker_fun_list = [ker_fun_SE,ker_fun_exp_sup_polys,ker_fun_parabola]; legend_list = ["SE","ESP","PAR"]
	# ker_fun_list = [ker_fun_parabola,ker_fun_SE,ker_fun_true_plus_sta]; legend_list = ["PAR","SE","SUM"]
	ker_fun_list = [ker_fun_parabola,ker_fun_true_plus_sta]; legend_list = ["PAR","SUM"]
	Nsamples = 1
	Npred = 120
	Nkers = len(ker_fun_list)
	mvn0_samples = np.random.randn(Nsamples,Npred)

	Nevals = 8
	Nrep = 40
	ind_Xevals_sel_tot = np.random.randint(low=0, high=Npred, size=(Nrep,Nevals), dtype=int)
	# pdb.set_trace()
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	mse_error = np.zeros((Nrep,Nevals+1,Nkers))
	loglikneg_error = np.zeros((Nrep,Nevals+1,Nkers))
	for rr in range(Nrep):

		for ee in range(ind_Xevals_sel_tot.shape[1]+1):

			if ee == 0:
				ind_Xevals_sel = []
			else:
				ind_Xevals_sel = ind_Xevals_sel_tot[rr,0:ee]

			for kk in range(Nkers):

				meanpred, stdpred, ftrue = compute_posterior(ker_fun_list[kk],mvn0_samples,None,Nsamples,ind_Xevals_sel,Npred,ftrue_call=ftrue_call,block_plot=False,plotting=False)

				mse_error[rr,ee,kk] = np.sqrt(np.mean((ftrue-meanpred[:,0])**2))
				loglikneg_error[rr,ee,kk] = np.mean(((ftrue-meanpred[:,0])/stdpred)**2)


	mse_error = np.log10(mse_error)


	mse_error_mean = np.mean(mse_error,axis=0) # [Nevals+1,Nkers]
	mse_error_std = np.std(mse_error,axis=0) # [Nevals+1,Nkers]
	loglikneg_error_mean = np.mean(loglikneg_error,axis=0) # [Nevals+1,Nkers]
	loglikneg_error_std = np.std(loglikneg_error,axis=0) # [Nevals+1,Nkers]
	color_list = sns.color_palette("deep",3)
	
	
	if plotting:
		hdl_fig_error_rates, hdl_splots_error_rates = plt.subplots(2,1,figsize=(12,8),sharex=True)

		Nobs_list = list(range(0,Nevals+1))

		for kk in range(Nkers):
			hdl_splots_error_rates[0].plot(Nobs_list,10**mse_error_mean[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2",label=legend_list[kk])
			hdl_splots_error_rates[0].fill_between(Nobs_list,10**(mse_error_mean[:,kk] - 1.*mse_error_std[:,kk]),10**(mse_error_mean[:,kk] + 1.*mse_error_std[:,kk]),color=color_list[kk],alpha=0.4)
			# plt.show(block=False)
			# plt.pause(0.2)
			# pdb.set_trace()
			hdl_splots_error_rates[0].set_yscale("log")
			hdl_splots_error_rates[0].set_ylabel(r"$|f_{true}(x_*) - \mu(x_*)|$",fontsize=fontsize_labels)
			# hdl_splots_error_rates[0].set_yticks([])

			hdl_splots_error_rates[1].plot(Nobs_list,loglikneg_error_mean[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2")
			hdl_splots_error_rates[1].fill_between(Nobs_list,loglikneg_error_mean[:,kk] - 1.*loglikneg_error_std[:,kk],loglikneg_error_mean[:,kk] + 1.*loglikneg_error_std[:,kk],color=color_list[kk],alpha=0.7)
			hdl_splots_error_rates[1].set_yscale("log")
			hdl_splots_error_rates[1].set_ylabel(r"$-\log p(f_{true}(x_*) | f(x_*))$",fontsize=fontsize_labels)

		hdl_splots_error_rates[0].set_title("RMSE",fontsize=fontsize_labels)
		hdl_splots_error_rates[0].set_xticks(Nobs_list)
		hdl_splots_error_rates[0].grid(visible=True,which="major",axis="y")
		hdl_splots_error_rates[0].legend()
		
		hdl_splots_error_rates[1].set_title("log-evidence",fontsize=fontsize_labels)
		hdl_splots_error_rates[1].set_xticks(Nobs_list)
		hdl_splots_error_rates[1].grid(visible=True,which="major",axis="y")
		hdl_splots_error_rates[1].set_xlabel(r"Nr. observations",fontsize=fontsize_labels)

	if savefig:
		path2save_fig = "{0:s}/RMSE_{1:d}.png".format(path2folder,ind_fig)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig_error_rates.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.pause(1.)

	plt.show(block=block_plot)

	return mse_error_mean, mse_error_std, loglikneg_error_mean, loglikneg_error_std


def compute_posterior(ker_fun,mvn0_samples,hdl_splots_ker_fit,Nsamples,ind_Xevals_sel,Npred,ftrue_call=None,block_plot=False,plotting=False):
	
	# mvn0_samples = np.random.randn(Nsamples,Npred)
	xmin = -1.0
	xmax = +1.0
	epsi = 1e-3
	xpred = np.reshape(np.linspace(xmin+epsi,xmax-epsi,Npred),(-1,dim_in))


	if ftrue_call is None:
		ftrue_call = lambda xx: 1.1*xx**2
	ftrue = ftrue_call(xpred[:,0])

	ind_Xevals = np.ones(xpred.shape[0]) != 1.0
	for ind in ind_Xevals_sel:
		ind_Xevals[ind] = True
	
	Xevals = xpred[ind_Xevals,0:1]
	Yevals = np.reshape(np.interp(Xevals[:,0],xpred[:,0],ftrue),(-1,1))
	# kXX_orig_cc0 = ker_fun(xpred,xpred)
	kXX = ker_fun(xpred,xpred)
	meanpred, stdpred, f_samples = little_gp_regression(kXX,ind_Xevals,Yevals,mvn0_samples)

	if plotting:
		hdl_splots_ker_fit.cla()
		# hdl_splots_ker_fit.plot(xpred[:,0],ftrue,linestyle="--",color="black",lw=2,alpha=0.9)
		hdl_splots_ker_fit.plot(xpred[:,0],meanpred[:,0],linestyle="-",color="blue",lw=2,alpha=0.8)
		hdl_splots_ker_fit.fill_between(xpred[:,0],meanpred[:,0] - 2.*stdpred,meanpred[:,0] + 2.*stdpred,color="navy",alpha=0.3,linestyle="None")
		# for ss in range(f_samples.shape[1]):
		# 	hdl_splots_ker_fit.plot(xpred[:,0],f_samples[:,ss],linestyle="--",color="navy",lw=2,alpha=0.5)
		# hdl_splots_ker_fit.plot(Xevals[:,0],Yevals[:,0],color="darkred",marker="o",markersize=7,linestyle="None")
		hdl_splots_ker_fit.set_ylim([-4.0,4.0])
		hdl_splots_ker_fit.set_xlim([xmin,xmax])
		hdl_splots_ker_fit.set_xlabel(r"$x$",fontsize=fontsize_labels)
		# hdl_splots_ker_fit.set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
		# hdl_splots_ker_fit.set_xticks([])
		# hdl_splots_ker_fit.set_yticks([])
		# hdl_splots_ker_fit.set_title(r"New samples",fontsize=fontsize_labels)
		# hdl_splots_ker_fit.set_title(titles_list[cc],fontsize=fontsize_labels)

		plt.show(block=block_plot)
		# plt.pause(0.5)
		# pdb.set_trace()
		# plt.show(block=block_plot)

	return meanpred, stdpred, ftrue, f_samples, xpred, Xevals, Yevals

def little_gp_regression(kxx,ind_Xevals,Yevals,mvn0_samples):

	# kXX_thetas_chol = np.linalg.cholesky(kXX + 1e-5*np.eye(kXX.shape[0])) # [Npred,Npred]

	# kxX = ker_call(xpred,Xevals)
	# kxx = ker_call(xpred,xpred)

	mvn0_samples_norma = mvn0_samples / np.linalg.norm(mvn0_samples)

	# mvn0_samples_norma = 1.5*mvn0_samples_norma # par
	mvn0_samples_norma = 2.5*mvn0_samples_norma # SE

	var_noise = 1e-4

	# Prior:
	if np.all(ind_Xevals == False):
		kxx_plus_noise = kxx + var_noise*np.eye(kxx.shape[0]) # [Npred,Npred]
		stdpred = np.sqrt(np.diag(kxx_plus_noise))
		kxx_chol = np.linalg.cholesky(kxx_plus_noise)
		meanpred = np.zeros((kxx.shape[0],1))
		f_samples = meanpred + (mvn0_samples_norma @ kxx_chol).T # [Npred,Nsamples]

		# pdb.set_trace()
		# raise ValueError("The stdpred here should just be np.diag(kxx_chol), or alternatively get the diagonal of kxx and get its sqrt()")
		# stdpred = np.sqrt(np.diag(kxx_chol))
		# # f_samples = meanpred + kxx_chol @ mvn0_samples.T # [Npred,Nsamples]
		# # pdb.set_trace()
		# # pdb.set_trace()
		# # f_samples = generate_moving_samples(meanpred,kxx,mvn0_samples.shape[0])


	kxX = kxx[:,ind_Xevals]

	kXX = kxX[ind_Xevals,:]

	kXX_inv = np.linalg.inv(kXX + var_noise*np.eye(kXX.shape[0]))
	meanpred = kxX @ kXX_inv @ Yevals
	covpred_mat = kxx - kxX @ kXX_inv @ kxX.T
	# if np.any(np.diag(covpred_mat) < 0.0):
	# 	pdb.set_trace()

	varpred = np.diag(covpred_mat)
	varpred_thres = 1e-16
	if np.any(varpred < varpred_thres):
		# print("@little_gp_regression: Fixing negative variances in varpred\n")
		varpred.flags.writeable = True
		varpred[varpred < varpred_thres] = varpred_thres
	stdpred = np.sqrt(varpred)

	# f_samples = meanpred + covpred_mat @ mvn0_samples.T # [Npred,Nsamples]
	f_samples = meanpred + (mvn0_samples_norma @ covpred_mat).T # [Npred,Nsamples]

	return meanpred, stdpred, f_samples


def illustrate_fit_white_prior(plot_stuff=False,block_plot=False):

	ker_fun_list = ker_fun_SE
	Nsamples = 3
	Npred = 120
	mvn0_samples = np.random.randn(Nsamples,Npred)

	hdl_fig_ker, hdl_splots_illus = plt.subplots(1,2,figsize=figsize_many,sharex=False)

	ind_Xevals_sel_tot = [60,90,10,75,25,110,55]
	Nevals = len(ind_Xevals_sel_tot)
	savefig = False
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	mse_error = np.zeros((Nevals+1))
	loglikneg_error = np.zeros((Nevals+1))
	color_list = sns.color_palette("deep",3)
	for ee in range(Nevals+1):

		if ee == 0:
			ind_Xevals_sel = []
		else:
			ind_Xevals_sel = ind_Xevals_sel_tot[0:ee]

		meanpred, stdpred, ftrue = compute_posterior(ker_fun_list,mvn0_samples,hdl_splots_illus[0],Nsamples,ind_Xevals_sel,Npred,block_plot=False,plotting=True)

		mse_error[ee] = np.mean((ftrue-meanpred[:,0])**2)
		loglikneg_error[ee] = np.mean(((ftrue-meanpred[:,0])/stdpred)**2)

		# hdl_splots_illus[0].set_ylim()
		hdl_splots_illus[0].set_title(r"GP with SE kernel",fontsize=fontsize_labels)
		hdl_splots_illus[0].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
		hdl_splots_illus[0].set_xticks([-1,0,1])
		# hdl_splots_illus[0].set_yticks([])
		hdl_splots_illus[0].set_xlabel(r"$x$",fontsize=fontsize_labels)
		# hdl_splots_illus[0].set_title(r"New samples",fontsize=fontsize_labels)
		# hdl_splots_illus[0].set_title(titles_list[cc],fontsize=fontsize_labels)

		# hdl_splots_illus[0].plot(mse_error_mean[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2",label=legend_list[kk])
		# # hdl_splots_illus[0].fill_between(mse_error_mean[:,kk],mse_error_mean[:,kk] - 2.*mse_error_std[:,kk],mse_error_mean[:,kk] + 2.*mse_error_std[:,kk],color="brown",alpha=0.9)
		# hdl_splots_illus[0].set_yscale("log")
		# hdl_splots_illus[0].set_ylabel(r"$|f_{true}(x_*) - \mu(x_*)|$",fontsize=fontsize_labels)

		hdl_splots_illus[1].plot(list(range(1,ee+1)),mse_error[0:ee],linestyle="-",color=color_list[0],alpha=0.9,marker="s",markersize=5,linewidth="2")
		# hdl_splots_illus[1].fill_between(loglikneg_error[ee],loglikneg_error[ee] - 2.*loglikneg_error_std[:,kk],loglikneg_error[ee] + 2.*loglikneg_error_std[:,kk],color="brown",alpha=0.9)
		hdl_splots_illus[1].set_yscale("log")
		# hdl_splots_illus[1].set_ylabel(r"$-\log p(f_{true}(x_*) | f(x_*))$",fontsize=fontsize_labels)


		# hdl_splots_illus[1].set_title("RMSE",fontsize=fontsize_labels)
		# hdl_splots_illus[1].set_xticks([])
		# hdl_splots_illus[1].grid(visible=True,which="major",axis="y")
		# hdl_splots_illus[1].legend()
		
		# hdl_splots_illus[1].set_title("log-evidence",fontsize=fontsize_labels)
		hdl_splots_illus[1].set_title(r"RMSE",fontsize=fontsize_labels)
		hdl_splots_illus[1].set_xlim([0,Nevals+1])
		hdl_splots_illus[1].set_xticks(list(range(0,ee+1)))
		hdl_splots_illus[1].grid(visible=True,which="major",axis="y")
		hdl_splots_illus[1].set_yticks([0.01,0.1,1.])
		hdl_splots_illus[1].set_ylim([0.001,1.])
		hdl_splots_illus[1].set_xlabel(r"Nr. observations",fontsize=fontsize_labels)

		if savefig:
			path2save_fig = "{0:s}/illustrate_fit_white_prior{1:d}.png".format(path2folder,ee)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1.)

	plt.show(block=True)


def illustrate_fit_three_steps_poster(plot_stuff=False,block_plot=False):

	# which_kernel = "SE"
	# which_kernel = "par"
	# which_kernel = "par_SE"
	which_kernel = "par_rec"
	# which_kernel = "par_rec_SE"

	# model = tf.keras.models.load_model("/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/saved_models_deep_karhunen_loeve/dkl_2023_06_19_18_49_37",\
	# 	custom_objects={"LossMomentMatching": LossMomentMatching, "SaveBestModel": SaveBestModel, "DeepKarhunenLoeve": DeepKarhunenLoeve, "FourierLayerNonStationary": FourierLayerNonStationary})
	# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=LossMomentMatching(),metrics=['accuracy','mean_squared_error'])

	model = DeepKarhunenLoeve(fourier_layer_stationary=False,Nfeatures=20) # Nfeatures = 20 works really well;
	test_model(model)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=LossMomentMatching(),metrics=['accuracy','mean_squared_error'])

	# path2weights = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/saved_models_deep_karhunen_loeve/dkl_weights_2023_06_19_19_40_04" # M = 10
	path2weights = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/saved_models_deep_karhunen_loeve/dkl_weights_2023_06_19_19_47_01" # M = 20	
	model.load_weights(path2weights, skip_mismatch=False, by_name=False, options=None)

	# xmin_plot = -1.0
	# xmax_plot = +1.0
	# xmin = -1.0
	# xmax = +1.0
	# epsi = 1e-3
	# dim_in = 1
	# Npred = 201
	# xpred_plot = np.reshape(np.linspace(xmin_plot+epsi,xmax_plot-epsi,Npred),(-1,dim_in))

	# # pdb.set_trace()
	# kxx = model.kernel(xpred_plot,xpred_plot)

	# hdl_fig_ker, hdl_splots_ker = plt.subplots(1,3,figsize=(24,8),sharex=False)
	# extent_plot_xpred = [xmin,xmax,xmin,xmax] #  scalars (left, right, bottom, top)
	# COLOR_MAP = "copper"

	# cov_pred = kxx.numpy()
	# hdl_splots_ker[0].imshow(cov_pred,extent=extent_plot_xpred,origin="upper",cmap=plt.get_cmap(COLOR_MAP),vmin=cov_pred.min(),vmax=cov_pred.max(),interpolation='nearest')
	# hdl_splots_ker[0].plot([xmin_plot,xmax_plot],[xmax_plot,xmin_plot],linestyle="-",color="green",lw=2,alpha=0.8)
	# hdl_splots_ker[0].set_xlim([xmin_plot,xmax_plot])
	# hdl_splots_ker[0].set_ylim([xmin_plot,xmax_plot])
	# hdl_splots_ker[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	# hdl_splots_ker[0].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
	# hdl_splots_ker[0].set_title(r"$k(x_t,x^\prime_t)$ [true]",fontsize=fontsize_labels)
	# hdl_splots_ker[0].set_xticks([])
	# hdl_splots_ker[0].set_yticks([])

	# plt.show(block=True)


	def ker_par_recons(x1,x2):
		"""
		x1: [Npoints1,dim]
		x2: [Npoints2,dim]

		out: [Npoints1,Npoints2]
		"""
		return model.kernel(x1,x2).numpy()

	def ker_par_recons_plus_SE(x1,x2):
		"""
		x1: [Npoints1,dim]
		x2: [Npoints2,dim]

		out: [Npoints1,Npoints2]
		"""
		kXX_par_recons = model.kernel(x1,x2).numpy()

		global ker_sum_tradeoff_global

		assert ker_sum_tradeoff_global <= 1.0 and ker_sum_tradeoff_global >= 0.0
		# print("ker_sum_tradeoff_global:",ker_sum_tradeoff_global)

		kXX_SE = ker_fun_SE(x1,x2,ls=0.1,var_prior=0.7)

		kXX = (1.-ker_sum_tradeoff_global)*kXX_par_recons + ker_sum_tradeoff_global*kXX_SE

		return kXX



	Nevals = 2
	ftrue_call = None

	# which_fun = "real_in_simu"
	which_fun = "real_outside_simu"

	if which_fun == "real_in_simu":
		ftrue_call = lambda xx: 1.1*xx**2
	if which_fun == "real_outside_simu":
		ftrue_call = lambda xx: 1.1*xx**2 + 0.4


	if which_kernel == "SE":
		ker_fun_list = ker_fun_SE
	if which_kernel == "par":
		ker_fun_list = ker_fun_parabola
	if which_kernel == "par_SE":
		ker_fun_list = ker_fun_true_plus_sta
		global ker_sum_tradeoff_global
		
		# # Step 1
		if which_fun == "real_in_simu":
			ker_sum_tradeoff_global = 0.05 # [0: par, 1: SE]
		# ftrue_call = lambda xx: 1.1*xx**2

		# Step 2
		if which_fun == "real_outside_simu":
			ker_sum_tradeoff_global = 0.45 # [0: par, 1: SE]
		# ftrue_call = lambda xx: 1.1*xx**2 + 0.4
	
	if which_kernel == "par_rec":
		ker_fun_list = ker_par_recons

	if which_kernel == "par_rec_SE":
		ker_fun_list = ker_par_recons_plus_SE
		# global ker_sum_tradeoff_global

		# # Step 1
		if which_fun == "real_in_simu":
			ker_sum_tradeoff_global = 0.01 # [0: par, 1: SE]
		# ftrue_call = lambda xx: 1.1*xx**2

		# Step 2
		if which_fun == "real_outside_simu":
			ker_sum_tradeoff_global = 0.2 # [0: par, 1: SE]
		# ftrue_call = lambda xx: 1.1*xx**2 + 0.4



	if which_kernel == "par":
		my_seed = 6 # with mvn0_samples_norma = 1.5*mvn0_samples_norma
	elif which_kernel == "SE":
		my_seed = 3 # with mvn0_samples_norma = 2.5*mvn0_samples_norma
	else:
		my_seed = 1






	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	Nsamples = 3
	Npred = 120
	mvn0_samples = np.random.randn(Nsamples,Npred)

	hdl_fig_ker, hdl_splots_illus = plt.subplots(1,1,figsize=(6,4),sharex=False)
	hdl_splots_illus = [hdl_splots_illus]

	ind_Xevals_sel_tot = [60,90,10,75,25,110,55] # From [0,120-1]
	
	savefig = False
	# savefig = True
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	color_list = sns.color_palette("deep",3)
	ind_Xevals_sel = ind_Xevals_sel_tot[0:Nevals]
	
	meanpred, stdpred, ftrue, f_samples, xpred, Xevals, Yevals = compute_posterior(ker_fun_list,mvn0_samples,hdl_splots_illus[0],Nsamples,ind_Xevals_sel,Npred,ftrue_call,block_plot=False,plotting=True)

	for ss in range(f_samples.shape[1]):
		# hdl_splots_illus[0].plot(xpred[:,0],f_samples[:,ss],linestyle="--",color="navy",lw=2,alpha=0.5)
		hdl_splots_illus[0].plot(xpred[:,0],f_samples[:,ss],linestyle="-",color="navy",lw=0.5,alpha=0.3)

	if Nevals > 0:
		hdl_splots_illus[0].plot(xpred[:,0],ftrue,linestyle="-",color="crimson",lw=2,alpha=0.9)
		hdl_splots_illus[0].plot(Xevals[:,0],Yevals[:,0],color="darkred",marker="o",markersize=7,linestyle="None")

	# hdl_splots_illus[0].set_ylim()
	# hdl_splots_illus[0].set_title(r"GP with SE kernel",fontsize=fontsize_labels)
	hdl_splots_illus[0].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	# hdl_splots_illus[0].set_xticks([-1,0,1])
	hdl_splots_illus[0].set_xticks([])
	hdl_splots_illus[0].set_yticks([])
	hdl_splots_illus[0].set_xlabel(r"$x$",fontsize=fontsize_labels)
	# hdl_splots_illus[0].set_title(r"New samples",fontsize=fontsize_labels)
	# hdl_splots_illus[0].set_title(titles_list[cc],fontsize=fontsize_labels)

	# hdl_splots_illus[0].plot(mse_error_mean[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2",label=legend_list[kk])
	# # hdl_splots_illus[0].fill_between(mse_error_mean[:,kk],mse_error_mean[:,kk] - 2.*mse_error_std[:,kk],mse_error_mean[:,kk] + 2.*mse_error_std[:,kk],color="brown",alpha=0.9)
	# hdl_splots_illus[0].set_yscale("log")
	# hdl_splots_illus[0].set_ylabel(r"$|f_{true}(x_*) - \mu(x_*)|$",fontsize=fontsize_labels)

	if savefig:
		path2save_fig = "{0:s}/three_steps/fit_kernel_{1:s}_Nevals_{2:d}_fun_{3:s}.png".format(path2folder,which_kernel,Nevals,which_fun)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.show(block=True)

	

def ker_exp_sup(x,xp,b,M=None):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	out: [Npoints,Npoints]
	"""

	dim_in = 1
	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	xp = xp.T

	den = 1. - b*x*xp
	if M is not None:
		assert M > 0
		num = 1. - (b*x*xp)**(M+1)
	else: num = 1.

	ker_mat = num / den
	return ker_mat

def illustrate_KL_construction():


	my_seed = 2
	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	M = None
	# M = 128
	ker_fun = lambda x,xp: ker_exp_sup(x,xp,b=0.999,M=M)

	Nsamples = 2
	Npred = 120
	mvn0_samples = np.random.randn(Nsamples,Npred)

	hdl_fig_ker, hdl_splots_illus = plt.subplots(1,1,figsize=(10,4),sharex=False)
	hdl_splots_illus = [hdl_splots_illus]
	
	# savefig = False
	savefig = True
	path2folder = "/Users/alonrot/work/meetings/presentation_2023_june/pics"
	# path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	color_list = sns.color_palette("deep",3)
	ind_Xevals_sel = [45,115]
	# ind_Xevals_sel = []
	Nevals = len(ind_Xevals_sel)

	def f_call(x):
		xx = np.reshape(x,(-1,1))
		powers = np.array(list(range(0,10)))
		powers = np.reshape(powers,(1,-1))
		return 0.2 * np.sum(xx**powers,axis=1)
	
	meanpred, stdpred, ftrue, f_samples, xpred, Xevals, Yevals = compute_posterior(ker_fun,mvn0_samples,hdl_splots_illus[0],Nsamples,ind_Xevals_sel,Npred,ftrue_call=f_call,block_plot=False,plotting=True)

	hdl_splots_illus[0].cla()

	for ss in range(f_samples.shape[1]):
		# hdl_splots_illus[0].plot(xpred[:,0],f_samples[:,ss],linestyle="--",color="navy",lw=2,alpha=0.5)
		hdl_splots_illus[0].plot(xpred[:,0],f_samples[:,ss],linestyle="--",color="navy",lw=1.25,alpha=0.9)

	if Nevals > 0:
		hdl_splots_illus[0].plot(xpred[:,0],ftrue,linestyle="-",color="crimson",lw=1,alpha=0.9)
		hdl_splots_illus[0].plot(Xevals[:,0],Yevals[:,0],color="darkred",marker="o",markersize=7,linestyle="None")

	hdl_splots_illus[0].plot(xpred[:,0],meanpred[:,0],linestyle="-",color="blue",lw=0.5,alpha=0.8)
	hdl_splots_illus[0].fill_between(xpred[:,0],meanpred[:,0] - 2.*stdpred,meanpred[:,0] + 2.*stdpred,color="navy",alpha=0.3,linestyle="None")
	# hdl_splots_illus[0].set_ylim()
	# hdl_splots_illus[0].set_title(r"GP with SE kernel",fontsize=fontsize_labels)
	hdl_splots_illus[0].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	# hdl_splots_illus[0].set_xticks([-1,0,1])
	hdl_splots_illus[0].set_xticks([])
	hdl_splots_illus[0].set_yticks([])
	hdl_splots_illus[0].set_xlabel(r"$x$",fontsize=fontsize_labels)
	hdl_splots_illus[0].set_ylim([-4.0,4.0])
	hdl_splots_illus[0].set_xlim([-1,1])
	# hdl_splots_illus[0].set_title(r"New samples",fontsize=fontsize_labels)
	# hdl_splots_illus[0].set_title(titles_list[cc],fontsize=fontsize_labels)

	# hdl_splots_illus[0].plot(mse_error_mean[:,kk],linestyle="-",color=color_list[kk],alpha=0.9,marker="s",markersize=5,linewidth="2",label=legend_list[kk])
	# # hdl_splots_illus[0].fill_between(mse_error_mean[:,kk],mse_error_mean[:,kk] - 2.*mse_error_std[:,kk],mse_error_mean[:,kk] + 2.*mse_error_std[:,kk],color="brown",alpha=0.9)
	# hdl_splots_illus[0].set_yscale("log")
	# hdl_splots_illus[0].set_ylabel(r"$|f_{true}(x_*) - \mu(x_*)|$",fontsize=fontsize_labels)

	if savefig:
		if M is None: M_name = 0;
		else: M_name = M;
		path2save_fig = "{0:s}/KL_illustration/prior_kernel_M_{1:d}_Nevals_{2:d}.png".format(path2folder,M_name,Nevals)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.show(block=True)






def illustrate_true_function():

	ker_fun_list = ker_fun_SE
	Nsamples = 3
	Npred = 120
	mvn0_samples = np.random.randn(Nsamples,Npred)

	hdl_fig_ker, hdl_splots_illus = plt.subplots(1,2,figsize=figsize_many,sharex=False)

	ind_Xevals_sel_tot = [60,90,10,75,25,110,55]
	Nevals = len(ind_Xevals_sel_tot)
	savefig = False
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	mse_error = np.zeros((Nevals+1))
	loglikneg_error = np.zeros((Nevals+1))
	color_list = sns.color_palette("deep",3)
	xmin = -1.0
	xmax = +1.0
	epsi = 1e-3
	xpred = np.reshape(np.linspace(xmin+epsi,xmax-epsi,Npred),(-1,dim_in))
	ftrue_call = lambda xx: 1.1*xx**2
	ftrue = ftrue_call(xpred[:,0])
	Xevals = xpred[ind_Xevals_sel_tot,0:1]
	# pdb.set_trace()
	Yevals = np.reshape(np.interp(Xevals[:,0],xpred[:,0],ftrue),(-1,1))
	for ee in range(2):

		hdl_splots_illus[0].plot(xpred[:,0],ftrue,linestyle="--",color="black",lw=2,alpha=0.9)
		if ee == 1: hdl_splots_illus[0].plot(Xevals[:,0],Yevals[:,0],color="darkred",marker="o",markersize=7,linestyle="None")
		hdl_splots_illus[0].set_ylim([-4.0,4.0])
		hdl_splots_illus[0].set_xlim([xmin,xmax])
		hdl_splots_illus[0].set_xlabel(r"$x$",fontsize=fontsize_labels)
		# hdl_splots_illus[0].set_ylabel(r"$f(x_t)$",fontsize=fontsize_labels)
		# hdl_splots_illus[0].set_xticks([])
		# hdl_splots_illus[0].set_yticks([])
		# hdl_splots_illus[0].set_title(r"New samples",fontsize=fontsize_labels)
		# hdl_splots_illus[0].set_title(titles_list[cc],fontsize=fontsize_labels)


		# hdl_splots_illus[0].set_ylim()
		# hdl_splots_illus[0].set_title(r"GP with SE kernel",fontsize=fontsize_labels)
		hdl_splots_illus[0].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
		hdl_splots_illus[0].set_xticks([-1,0,1])
		# hdl_splots_illus[0].set_yticks([])
		# hdl_splots_illus[0].set_title(r"New samples",fontsize=fontsize_labels)
		# hdl_splots_illus[0].set_title(titles_list[cc],fontsize=fontsize_labels)

		if savefig:
			path2save_fig = "{0:s}/illustrate_true_fun{1:d}.png".format(path2folder,ee)
			logger.info("Saving fig at {0:s} ...".format(path2save_fig))
			hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			plt.pause(1.)

	plt.show(block=True)

def illustrate_mutual_info():

	ker_fun_list = [ker_fun_SE,ker_fun_exp_sup_polys,ker_fun_parabola]
	Nsamples = 3
	Npred = 120
	mvn0_samples = np.random.randn(Nsamples,Npred)

	# mvn0_samples = np.random.randn(Nsamples,Npred)
	xmin = -1.0
	xmax = +1.0
	epsi = 1e-3
	xpred = np.reshape(np.linspace(xmin+epsi,xmax-epsi,Npred),(-1,dim_in))

	hdl_fig_ker, hdl_splots_MI = plt.subplots(2,3,figsize=(12,8),sharex=True)

	ind_Xevals_sel_tot = [95]
	savefig = False
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	for ee in range(len(ind_Xevals_sel_tot)+1):

		if ee == 0:

			ind_Xevals_sel = []
			stdpred_prior_list = []
			for kk in range(len(ker_fun_list)):
				_, stdpred_prior, _ = compute_posterior(ker_fun_list[kk],mvn0_samples,hdl_splots_MI[0,kk],Nsamples,ind_Xevals_sel,Npred,block_plot=False,plotting=True)
				stdpred_prior_list += [stdpred_prior]

		else:

			ind_Xevals_sel = ind_Xevals_sel_tot[0:ee]

			for kk in range(len(ker_fun_list)):
				_, stdpred_post, _ = compute_posterior(ker_fun_list[kk],mvn0_samples,hdl_splots_MI[0,kk],Nsamples,ind_Xevals_sel,Npred,block_plot=False,plotting=True)

				# if kk == 2: pdb.set_trace()
				MI_kk = np.log(stdpred_prior_list[kk]) - np.log(stdpred_post)

				# if kk == 2: MI_kk[:] = np.mean(MI_kk)
				# hdl_splots_MI[1,kk].plot(xpred,MI_kk,lw=2.0,alpha=0.5,color="navy")
				hdl_splots_MI[1,kk].fill_between(xpred[:,0],np.zeros((Npred)),MI_kk,color="brown",alpha=0.4,linestyle="None")
				

	hdl_splots_MI[0,0].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	hdl_splots_MI[1,0].set_ylabel(r"MI$(x;X)$",fontsize=fontsize_labels)
	for kk in range(3):
		hdl_splots_MI[0,kk].set_xlabel("")
		hdl_splots_MI[0,kk].set_xticks([])
		hdl_splots_MI[1,kk].set_xlabel(r"$x$",fontsize=fontsize_labels)
		hdl_splots_MI[1,kk].set_xticks([-1,0,1])
		hdl_splots_MI[1,kk].set_ylim([0,8])
	hdl_splots_MI[0,1].set_yticks([])
	hdl_splots_MI[0,2].set_yticks([])
	hdl_splots_MI[1,1].set_yticks([])
	hdl_splots_MI[1,2].set_yticks([])
	# hdl_splots_MI.set_title(r"New samples",fontsize=fontsize_labels)
	# hdl_splots_MI.set_title(titles_list[cc],fontsize=fontsize_labels)




	if savefig:
		path2save_fig = "{0:s}/MI.png".format(path2folder)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.pause(1.)

	plt.show(block=True)







if __name__ == "__main__":


	my_seed = 100 # keep it at 100

	np.random.seed(seed=my_seed)
	tf.random.set_seed(seed=my_seed)

	# plotting_posterior()

	# illustrate_fit_white_prior()

	# illustrate_true_function()

	illustrate_fit_three_steps_poster()

	# illustrate_KL_construction()

	# illustrate_mutual_info()

	# compute_error_rates()





	
