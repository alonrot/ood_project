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
import tensorflow_probability as tfp
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

from bayesquad.examples.example_1d_wrapped import wrapper_1D
from bayesquad.examples.example_2d_wrapped import wrapper_2D


def get_plotting_quantities(xpred,inverse_fourier_toolbox):

	fx 			= inverse_fourier_toolbox.get_fx(xpred)
	ker_diag 	= inverse_fourier_toolbox.get_kernel_diagonal(xpred)
	# cov_diag 	= inverse_fourier_toolbox.get_covariance_diagonal(xpred)
	cov_diag = np.zeros(xpred.shape[0])

	return fx, ker_diag, cov_diag


@hydra.main(config_path="./config",config_name="config")
def test_ifou_1D(cfg):
	
	dim_x = 1

	integration_method = "integrate_with_regular_grid"
	# integration_method = "integrate_with_irregular_grid"
	# integration_method = "integrate_with_bayesian_quadrature"
	# integration_method = "integrate_with_data"

	spectral_densities = []; labels = []
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim_x,integration_method)]; labels += ["Kink"]
	spectral_densities += [KinkSharpSpectralDensity(cfg.spectral_density.kinksharp,cfg.sampler.hmc,dim_x,integration_method)]; labels += ["KinkSharp"]
	spectral_densities += [ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim_x,integration_method)]; labels += ["Parabola"]
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]

	# Create grid for plotting:
	xmin = -5.0
	xmax = +2.0
	Ndiv = 201
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]

	# Integration against spectral density: grid of frequencies
	# L = 500.
	L = 100.
	Ndiv = 2001
	assert Ndiv % 2 != 0 and Ndiv > 2, "Ndiv must be an odd positive integer"
	j_indices = CommonUtils.create_Ndim_grid(xmin=-(Ndiv-1)//2,xmax=(Ndiv-1)//2,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]
	omegapred = tf.cast((math.pi/L) * j_indices,dtype=tf.float32)

	inverse_fourier_toolboxes = []
	for ii in range(len(spectral_densities)):
		inverse_fourier_toolboxes += [InverseFourierTransformKernelToolbox(spectral_densities[ii],dim_x)]
		inverse_fourier_toolboxes[-1].update_spectral_density_and_angle(omegapred)

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


@hydra.main(config_path="./config",config_name="config")
def test_reconstruct_dubinscar(cfg):

	np.random.seed(seed=0)
	dim_x = 3

	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs.pickle"
	path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	dim_x = data_dict["dim_x"]
	dim_u = data_dict["dim_u"]
	Nsteps = data_dict["Nsteps"]
	Ntrajs = data_dict["Ntrajs"]
	deltaT = data_dict["deltaT"]

	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	dim_X = dim_x + dim_u
	spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_X,use_nominal_model=True,data=Xtrain)
	assert spectral_density.dim == 5

	# res_opti = search_best_omega_lim(spectral_density,Xtrain,Ytrain)
	loss_vec, omega_lim_vec, baseline = search_omegas_brute_force(spectral_density,Xtrain,Ytrain)
	# Nevals = 11; Nsamples_per_eval = 3
	# omega_lim_vec = np.linspace(0.5,4.0,Nevals)
	# omega_lim_vec = omega_lim_vec[0:Nevals]
	# loss_vec = np.random.rand(Nevals,Nsamples_per_eval)
	# baseline = 0.1

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,8),sharex=True)
	# hdl_splots.plot(omega_lim_vec,loss_vec,linestyle="None",marker=".")
	hdl_splots.boxplot(loss_vec.T,positions=np.around(omega_lim_vec,decimals=2))
	hdl_splots.axhline(y=baseline, color="gray", linestyle='--', lw=1.0)
	hdl_splots.set_xlabel(r"$\omega_{lim}$",fontsize=fontsize_labels)
	hdl_splots.set_ylabel(r"Reconstruction loss",fontsize=fontsize_labels)

	plt.show(block=True)
	plt.pause(1)

def search_best_omega_lim(spectral_density,Xtrain,Ytrain):

	Nsamples_per_eval = 2
	Nomegas = 1000
	dim_out_ind = 0

	def fun(omega_lim):

		loss_vec = np.zeros(Nsamples_per_eval)
		for jj in range(Nsamples_per_eval):

			logger.info(" * sample: {0:d}".format(jj+1))

			omegapred = -omega_lim + 2.*omega_lim*tf.math.sobol_sample(dim=spectral_density.dim,num_results=Nomegas,skip=2000 + jj*1000)

			# Each time we initialize, we resample:
			inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=spectral_density.dim,dim_out_ind=dim_out_ind,omegapred=omegapred)

			fx_targets = inverse_fourier_toolbox.get_fx(Xtrain)

			loss_vec[jj] = 0.5*tf.reduce_mean((Ytrain[:,dim_out_ind] - fx_targets)**2)

			del inverse_fourier_toolbox

		loss_val = tf.reduce_mean(loss_vec)

		return loss_val.numpy()

	omega_lim0 = np.array([5.0])
	result = scipy.optimize.minimize(fun,omega_lim0,method="L-BFGS-B",bounds=[(0.1,5.0)],options=dict(maxiter=20))

	return result


def search_omegas_brute_force(spectral_density,Xtrain,Ytrain):

	Nevals = 9
	omega_lim_vec = np.linspace(0.5,4.0,Nevals)
	# omega_lim_vec = np.linspace(0.5,0.9,Nevals)
	# omega_lim_vec = omega_lim_vec[0:Nevals]
	Nsamples_per_eval = 4
	# loss_vec = np.zeros(Nevals)
	loss_vec = np.zeros((Nevals,Nsamples_per_eval))
	# omega_lim = 1.0
	Nomegas = 5**5
	dim_out_ind = 0
	for ii in range(Nevals):

		omega_lim = omega_lim_vec[ii]

		logger.info("eval: {0:d}".format(ii+1))
		for jj in range(Nsamples_per_eval):

			logger.info(" * sample: {0:d}".format(jj+1))

			omegapred = -omega_lim + 2.*omega_lim*tf.math.sobol_sample(dim=spectral_density.dim,num_results=Nomegas,skip=2000 + (ii*Nsamples_per_eval + jj)*1000)

			# Each time we initialize, we resample:
			inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=spectral_density.dim,dim_out_ind=dim_out_ind,omegapred=omegapred)

			fx_targets = inverse_fourier_toolbox.get_fx(Xtrain)

			loss_vec[ii,jj] = 0.5*tf.reduce_mean((Ytrain[:,dim_out_ind] - fx_targets)**2)

			del inverse_fourier_toolbox


	Ndiv = 5; L = 10.0
	_, _, omegapred = spectral_density.get_Wpoints_discrete(L=L,Ndiv=Ndiv,normalize_density_numerically=False)
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=spectral_density.dim,dim_out_ind=dim_out_ind,omegapred=omegapred)
	fx_targets = inverse_fourier_toolbox.get_fx(Xtrain)
	baseline = 0.5*tf.reduce_mean((Ytrain[:,dim_out_ind] - fx_targets)**2)
	del inverse_fourier_toolbox

	# loss_vec = np.mean(loss_vec,axis=1)
	

	return loss_vec, omega_lim_vec, baseline



@hydra.main(config_path="./config",config_name="config")
def search_omegas_tf(cfg):

	np.random.seed(seed=0)
	# dim_x = 3

	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs.pickle"
	path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs_for_searching_wlim.pickle"
	logger.info("Loading {0:s} ...".format(path2data))
	file = open(path2data, 'rb')
	data_dict = pickle.load(file)
	file.close()
	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	dim_x = data_dict["dim_x"]
	dim_u = data_dict["dim_u"]
	Nsteps = data_dict["Nsteps"]
	Ntrajs = data_dict["Ntrajs"]
	deltaT = data_dict["deltaT"]

	Xtrain = data_dict["Xtrain"]
	Ytrain = data_dict["Ytrain"]
	dim_X = dim_x + dim_u
	spectral_density = DubinsCarSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_X,use_nominal_model=True,data=Xtrain)
	assert spectral_density.dim == 5


	# dim_in = dim_x + dim_u
	# rec_dyn_sys = ReconstructDynamicalSystem(dim_in,cfg,spectral_density,Xtrain,Ytrain,dim_out_ind=0)
	# rec_dyn_sys.train_model()

	res_opti = search_best_omega_lim(spectral_density,Xtrain,Ytrain)
	# # loss_vec, omega_lim_vec, baseline = search_omegas_brute_force(spectral_density,Xtrain,Ytrain)
	# # Nevals = 11; Nsamples_per_eval = 3
	# # omega_lim_vec = np.linspace(0.5,4.0,Nevals)
	# # omega_lim_vec = omega_lim_vec[0:Nevals]
	# # loss_vec = np.random.rand(Nevals,Nsamples_per_eval)
	# # baseline = 0.1
	# 
	pdb.set_trace()

	# hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,8),sharex=True)
	# # hdl_splots.plot(omega_lim_vec,loss_vec,linestyle="None",marker=".")
	# hdl_splots.boxplot(loss_vec.T,positions=np.around(omega_lim_vec,decimals=2))
	# hdl_splots.axhline(y=baseline, color="gray", linestyle='--', lw=1.0)
	# hdl_splots.set_xlabel(r"$\omega_{lim}$",fontsize=fontsize_labels)
	# hdl_splots.set_ylabel(r"Reconstruction loss",fontsize=fontsize_labels)

	# plt.show(block=True)
	# plt.pause(1)



class ReconstructDynamicalSystem(tf.keras.layers.Layer):

	def __init__(self, dim_in: int, cfg: dict, spectral_density: SpectralDensityBase, Xtrain, Ytrain, dim_out_ind, **kwargs):

		super().__init__(**kwargs)

		self.dim_in = dim_in

		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		self.dim_out_ind = dim_out_ind

		self.learning_rate = 1e-5

		# self.Nomegas = 5**5
		# self.Nomegas = 2000
		self.Nomegas = 1000

		self.stop_loss_val = -500.

		self.epochs = 100

		self.Nsamples_for_averaging_loss = 2

		self.log_omega_lim = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(5.0)), trainable=True,name="log_omega_lim_dim{0:d}".format(self.dim_out_ind))

		omegapred = self.re_sample_omega_grid()

		self.inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=self.dim_in,dim_out_ind=self.dim_out_ind,omegapred=omegapred)


	def get_omega_lim(self):
		return tf.exp(self.log_omega_lim)

	def re_sample_omega_grid(self):
		logger.info("Re sampling omega grid...")
		omega_lim = self.get_omega_lim()
		jj = tf.random.uniform(shape=[1],minval=0,maxval=100,dtype=tf.int32)
		omegapred = -omega_lim + 2.*omega_lim*tf.math.sobol_sample(dim=self.dim_in,num_results=self.Nomegas,skip=2000 + jj*1000)
		logger.info("Done!")
		return omegapred

	def reconstruction_loss(self):
		
		loss_val = 0.0
		for jj in range(self.Nsamples_for_averaging_loss):

			logger.info("Computing loss for sample {0:d} / {1:d} ...".format(jj+1,self.Nsamples_for_averaging_loss))

			omegapred = self.re_sample_omega_grid()
			self.inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred)
			fx_targets = self.inverse_fourier_toolbox.get_fx(self.Xtrain)
			logger.info("Computing loss... 2")
			loss_val += 0.5*tf.reduce_mean((self.Ytrain[:,self.dim_out_ind] - fx_targets)**2)
			# pdb.set_trace()
		
		logger.info("Done!")
		return loss_val / self.Nsamples_for_averaging_loss

	def train_model(self,verbosity=False):
		"""

		"""

		str_banner = " << Training to get omega_lim >> "

		# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
		# optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		# optimizer_list = [tf.keras.optimizers.Adam(learning_rate=self.learning_rate)]*self.dim_out
		# trainable_weights_best_list = [None]*self.dim_out
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		# Use a learning rate scheduler: https://arxiv.org/pdf/1608.03983.pdf
		# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay

		epoch = 0
		done = False
		loss_value_best = float("Inf")
		trainable_weights_best = self.get_weights()
		# for dd in range(self.dim_out):
		# 	trainable_weights_best_list[dd] = self.rrgpMO[dd].get_weights()
		print_every = 1
		while epoch < self.epochs and not done:

			if (epoch+1) % print_every == 0:
				logger.info("="*len(str_banner))
				logger.info(str_banner)
				logger.info(" << Epoch {0:d} / {1:d} >> ".format(epoch+1, self.epochs))
				logger.info("="*len(str_banner))

			# with tf.GradientTape(persistent=True) as tape:
			with tf.GradientTape() as tape:
				# loss_val_per_dim = self.get_negative_log_evidence_predictive_full_trajs_in_batch(self.z_vec_real,self.u_traj_real,self.Nhorizon,update_features=True)
			
				loss_value = self.reconstruction_loss()
				# loss_value = self.get_loss_debug()
				# loss_value = self.get_loss_debug_2(self.z_vec_real,self.u_traj_real,self.Nhorizon)

			# pdb.set_trace()
			# for dd in range(self.dim_out):
			# 	# grads = tape.gradient(loss_value_per_dim[dd], self.rrgpMO[dd].trainable_weights)
			# 	grads = tape.gradient(loss_value, self.rrgpMO[dd].trainable_weights)
			# 	optimizer_list[dd].apply_gradients(zip(grads, self.rrgpMO[dd].trainable_weights))

			# for dd in range(self.dim_out):
			# 	# grads = tape.gradient(loss_value_per_dim[dd], self.rrgpMO[dd].trainable_weights)
			# 	grads = tape.gradient(loss_value, self.trainable_weights)
			# 	print(grads)

			# pdb.set_trace()
			logger.info("Prop gardient")
			grads = tape.gradient(loss_value, self.trainable_weights)
			logger.info("Done!")

			# loss_value = tf.math.reduce_sum(loss_val_per_dim)

			if (epoch+1) % print_every == 0:
				logger.info("    * Predictive loss (current): {0:.4f}".format(float(loss_value)))
				logger.info("    * Predictive loss (best): {0:.4f}".format(float(loss_value_best)))
				logger.info("    * Weights (current): {0:s}".format(self._weights2str(self.trainable_weights)))
				logger.info("    * Weights (best): {0:s}".format(self._weights2str(trainable_weights_best)))
				logger.info("    * Gradients (current): {0:s}".format(self._weights2str(grads)))

			if loss_value <= self.stop_loss_val:
				done = True
			
			if loss_value < loss_value_best:
				loss_value_best = loss_value
				trainable_weights_best = self.get_weights()
				# for dd in range(self.dim_out):
				# 	trainable_weights_best_list[dd] = self.rrgpMO[dd].get_weights()
			
			epoch += 1

		if done == True:
			logger.info(" * Training finished because loss_value = {0:f} (<= {1:f})".format(float(loss_value),float(self.stop_loss_val)))

		self.set_weights(weights=trainable_weights_best)
		# for dd in range(self.dim_out):
		# 	self.rrgpMO[dd].set_weights(weights=trainable_weights_best)

		# if verbosity:
		# 	self.rrgpMO[dd].print_weights_info()

		logger.info("Training finished!")


		pdb.set_trace()

	def _weights2str(self,trainable_weights):
		
		assert len(trainable_weights) > 0
		if tf.is_tensor(trainable_weights[0]):
			which_type = "tfvar"
		elif isinstance(trainable_weights[0],np.ndarray):
			which_type = "nparr"
		elif trainable_weights[0] is None:
			which_type = "none"
		else:
			raise ValueError("trainable_weights has an unspecificed type")

		str_weights = "[ "
		for ii in range(len(trainable_weights)-1):
			# if which_type == "tfvar": str_weights += str(trainable_weights[ii].numpy())
			# elif which_type == "nparr": str_weights += str(trainable_weights[ii])
			# elif which_type == "none": str_weights += str(None)
			try: str_weights += str(trainable_weights[ii].numpy());
			except: str_weights += str(trainable_weights[ii]);
			str_weights += " , "

		try: str_weights += str(trainable_weights[len(trainable_weights)-1].numpy());
		except: str_weights += str(trainable_weights[len(trainable_weights)-1]);
		# if which_type == "tfvar": str_weights += str(trainable_weights[len(trainable_weights)-1].numpy())
		# elif which_type == "nparr": str_weights += str(trainable_weights[len(trainable_weights)-1])
		# elif which_type == "none": str_weights += str(None)

		str_weights += " ]"
		return str_weights





@hydra.main(config_path="./config",config_name="config")
def reconstruction_loss_analysis(cfg):
	
	dim_x = 1

	# Create grid for plotting:
	xmin = -5.0
	xmax = +2.0
	Ndiv = 201
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]

	

	spectral_densities = []; labels = []; fx_true_list = []
	spectral_densities += [ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x,data=xpred)]; labels += ["Parabola"]; fx_true_list += [spectral_densities[-1]._nonlinear_system_fun(xpred)]
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim=dim_x,data=xpred,use_nominal_model=True)]; labels += ["Kink"]; fx_true_list += [spectral_densities[-1]._nonlinear_system_fun(xpred)]
	spectral_densities += [KinkSharpSpectralDensity(cfg.spectral_density.kinksharp,cfg.sampler.hmc,dim=dim_x,data=xpred)]; labels += ["KinkSharp"]; fx_true_list += [spectral_densities[-1]._nonlinear_system_fun(xpred)]
	# pdb.set_trace()
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]; fx_true_list += [spectral_densities[0]._nonlinear_system_fun(xpred)] # matern spectral density for fitting a parabola function
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim=dim_x)]; labels += ["Matern"]; fx_true_list += [spectral_densities[1]._nonlinear_system_fun(xpred)] # matern spectral density for fitting a kink function
	# spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim=dim_x)]; labels += ["SquaredExp"]
	# 

	
	inverse_fourier_toolboxes = []
	Ndiv_omega_for_analysis = 201
	omega_lim = 3.0
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_x) # [Ndiv**dim_x,dim_x]
	hdl_fig, hdl_splots = plt.subplots(len(spectral_densities),2,figsize=(12,8),sharex=False)
	omegapred_analysis_el = omegapred_analysis[0:1,...]
	for ii in range(len(spectral_densities)):

		loss_vec = np.zeros(Ndiv_omega_for_analysis)
		inverse_fourier_toolboxes += [InverseFourierTransformKernelToolbox(spectral_density=spectral_densities[ii],dim=dim_x,dim_out_ind=0,omegapred=omegapred_analysis_el)]
		for jj in range(Ndiv_omega_for_analysis):

			omegapred_analysis_el = omegapred_analysis[jj:jj+1,...]
			
			# pdb.set_trace()
			inverse_fourier_toolboxes[ii].update_spectral_density_and_angle(omegapred_analysis_el)
			fx = inverse_fourier_toolboxes[ii].get_fx(xpred)

			loss_vec[jj] = np.mean(0.5*(fx_true_list[ii] - fx)**2)

			# pdb.set_trace()



		# if ii == 0:
		# 	Dw = 1.0
		# 	omega_in = omegapred_analysis
		# 	inverse_fourier_toolboxes[0].update_spectral_density_and_angle(omega_in)
		# 	fx_inte = inverse_fourier_toolboxes[0].get_fx_integrand(xpred,Dw) # [Npoints, Nomegas]
		# 	loss_val = np.mean(0.5*(np.reshape(fx_true_list[ii],(-1,1)) - fx_inte)**2,axis=0,keepdims=True) # [1, Nomegas]
		# 	loss_val_tp = loss_val.T # [Nomegas,1]
		# 	loss_val_exp = np.exp(-loss_val_tp/40.0)

		# 	pdb.set_trace()




		exp_loss = np.exp(-loss_vec) / np.sum(np.exp(-loss_vec))

		# if ii == len(spectral_densities)-1:
		# 	pdb.set_trace()

		
		# hdl_fig.suptitle("Using Spectral density {0:s}".format(labels[ii]),fontsize=fontsize_labels)
		hdl_splots[ii,0].plot(omegapred_analysis[:,0],loss_vec)
		hdl_splots[ii,1].plot(omegapred_analysis[:,0],exp_loss)
		

	plt.show(block=True)



@hydra.main(config_path="./config",config_name="config")
def reconstruction_loss_with_bayesian_quadrature_1D(cfg):
	
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


	dim_x = 1

	# Create grid for plotting:
	xmin = -5.0
	xmax = +2.0
	Ndiv = 201
	omega_lim = 3.0
	Nomegas = 100
	Nbatches = 10 # Number of iterations
	# Dw = (2.*omega_lim)**dim_x / Nomegas
	Dw = 1.0
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]
	
	spectral_density = KinkSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x,data=xpred);
	# spectral_density = ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x,data=xpred);
	fx_true = spectral_density._nonlinear_system_fun(xpred)
	fx_true = tf.reshape(fx_true,(-1,dim_x))
	omegapred = np.array([[0.0]])
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_x,dim_out_ind=0,omegapred=omegapred)


	# pdb.set_trace()

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=False)
	Ndiv_omega_for_analysis = 201
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_x) # [Ndiv**dim_x,dim_x]

	def integrand(omega_in):
		omega_in = tf.convert_to_tensor(omega_in,dtype=tf.float32)
		inverse_fourier_toolbox.update_spectral_density_and_angle(omega_in)
		fx = inverse_fourier_toolbox.get_fx_integrand(xpred,Dw) # [Npoints, Nomegas]
		loss_val = np.mean(0.5*(fx_true - fx)**2,axis=0,keepdims=True) # [1, Nomegas]
		loss_val_tp = loss_val.T # [Nomegas,1]
		loss_val_exp = np.exp(-loss_val_tp)
		# pdb.set_trace()
		return loss_val_exp

	integrand_loss_vals_exp = integrand(omegapred_analysis)
	int_sum = np.sum(integrand_loss_vals_exp)


	def integrand_normalized(omega_in):
		return 10.*integrand(omega_in) / int_sum

	integrand_loss_vals_exp_nor = integrand_normalized(omegapred_analysis)


	# hdl_splots.plot(omegapred_analysis[:,0],integrand_loss_vals_exp_nor)
	# plt.show(block=True)
	

	omega_locations = wrapper_1D(integrand_normalized,omega_lim,Nomegas,Nbatches)

	# pdb.set_trace()




@hydra.main(config_path="./config",config_name="config")
def reconstruction_loss_with_bayesian_quadrature_2D(cfg):
	
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

	dim_x = 2

	# Create grid for plotting:
	xmin = -5.0
	xmax = +2.0
	Ndiv = 41

	omega_lim = 3.0
	Nomegas = 100
	Nbatches = 50 # Number of iterations
	# Nomegas = 20
	# Nbatches = 2 # Number of iterations
	# Dw = (2.*omega_lim)**dim_x / Nomegas
	Dw = 1.0
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]
	
	spectral_density = VanDerPolSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x,data=xpred);
	fx_true = spectral_density._nonlinear_system_fun(xpred)
	dim_out_ind = 0
	fx_true = fx_true[:,dim_out_ind:dim_out_ind+1]
	omegapred = np.array([[0.0,0.0]])
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_x,dim_out_ind=0,omegapred=omegapred)


	# pdb.set_trace()

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=False)
	Ndiv_omega_for_analysis = 31
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_x) # [Ndiv**dim_x,dim_x]


	# Try NOT transforming the integrand to tbe positive. 
	# The method should handle this.
	# The method is supposed to balance high variance with likelihood
	# Help the method by (a) offsetting the integrand with w->infty values and 
	# (b) scaling up the squared error so that high values are
	# chosen more often
	# 
	# 
	
	loss_squared_offset = -np.mean(0.5*(fx_true)**2)


	def integrand(omega_in):
		omega_in = tf.convert_to_tensor(omega_in,dtype=tf.float32)
		inverse_fourier_toolbox.update_spectral_density_and_angle(omega_in)
		fx = inverse_fourier_toolbox.get_fx_integrand(xpred,Dw) # [Npoints, Nomegas]

		loss_integrand_abs = np.mean(abs(fx),axis=0,keepdims=True) # [1, Nomegas]

		loss_val_squared_neg = -np.mean(0.5*(fx_true - fx)**2,axis=0,keepdims=True) # [1, Nomegas]
		# loss_val_squared_neg_tp = loss_val_squared_neg.T # [Nomegas,1]
		loss_val_squared_neg_exp = np.exp(loss_val_squared_neg) # [1, Nomegas]
		loss_averaged_integrand = np.mean(fx,axis=0,keepdims=True)
		# pdb.set_trace()
		# 
		# 
		# 
		
		loss_val_squared_neg_norm = loss_val_squared_neg - loss_squared_offset
		
		loss_val_squared_neg_norm_exp = np.exp(loss_val_squared_neg_norm)

		loss_val_out = loss_integrand_abs
		# loss_val_out = loss_val_squared_neg_norm_exp
		# loss_val_out = loss_val_squared_neg_norm
		# loss_val_out = loss_val_squared_neg_exp
		# loss_val_out = loss_averaged_integrand

		return loss_val_out

	integrand_loss_vals_exp = integrand(omegapred_analysis)
	int_sum = np.sum(integrand_loss_vals_exp)


	def integrand_normalized(omega_in):
		# return 100.*integrand(omega_in) / int_sum
		return integrand(omega_in)

	# integrand_normalized = integrand

	integrand_loss_vals_exp_nor = integrand_normalized(omegapred_analysis)

	integrand_loss_vals_exp_nor_reshaped = np.reshape(integrand_loss_vals_exp_nor,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis))

	# pdb.set_trace()


	hdl_splots.imshow(integrand_loss_vals_exp_nor_reshaped)
	# hdl_splots.plot(omegapred_analysis[:,0],integrand_loss_vals_exp_nor)
	plt.show(block=False)
	plt.pause(0.1)
	

	omega_locations = wrapper_2D(integrand_normalized,omega_lim,Nomegas,Nbatches)

	# pdb.set_trace()




class SampleReconstructionLoss():

	def __init__(self,dim_in):

		self.num_burnin_steps = 200
		self.Nsamples_per_state0 = 400
		# self.initial_states_sampling = np.array([[0.0,0.0]],dtype=np.float32)
		# self.initial_states_sampling = np.array([[0.0],[0.0]],dtype=np.float32)
		# self.initial_states_sampling = tf.constant([[0.0],[0.0]],dtype=np.float32)
		self.initial_states_sampling = tf.constant([[1.0,1.0]],dtype=np.float32)
		self.step_size_hmc = tf.cast(0.01, tf.float32)
		self.num_leapfrog_steps_hmc = 8
		# self.num_leapfrog_steps_hmc = 20
		self.dim_in = dim_in
		assert self.Nsamples_per_state0 % 2 == 0, "Need an even number, for now"
		self.adaptive_hmc = None


	# @tf.function
	def initialize_HMCsampler(self,log_likelihood_fn):
		"""
		"""

		logger.info("Initializing tfp.mcmc.HamiltonianMonteCarlo()...")
		adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(	
			tfp.mcmc.HamiltonianMonteCarlo(
					target_log_prob_fn=log_likelihood_fn,
					step_size=self.step_size_hmc, # Step size of the leapfrog integration method. If too large, the trajectories can diverge. If we want a different step size for each dimension, a tensor can be entered as well.
					num_leapfrog_steps=self.num_leapfrog_steps_hmc, # How many steps forward the Hamiltonian dynamics are simulated
					state_gradients_are_stopped=False,
					step_size_update_fn=None,
					store_parameters_in_results=False,
					experimental_shard_axis_names=None,
					name=None),
				num_adaptation_steps=int(self.num_burnin_steps * 0.8))

		return adaptive_hmc

	# @tf.function
	def initialize_NoUTurnSamplersampler(self,log_likelihood_fn):
		"""
		"""

		logger.info("Initializing tfp.mcmc.NoUTurnSampler()...")
		# tfb = tfp.bijectors
		# sampler = tfp.mcmc.TransformedTransitionKernel(
		# tfp.mcmc.NoUTurnSampler(
		# 		target_log_prob_fn=log_likelihood_fn,
		# 		step_size=self.step_size_hmc, # Step size of the leapfrog integration method. If too large, the trajectories can diverge. If we want a different step size for each dimension, a tensor can be entered as well.
		# 	    max_tree_depth=10,
		# 	    max_energy_diff=1000.0,
		# 	    unrolled_leapfrog_steps=1,
		# 	    parallel_iterations=10,
		# 	    experimental_shard_axis_names=None,
		# 		name=None),
		# bijector=[tfb.Shift(np.finfo(np.float32).tiny)(tfb.Exp())]
		# )

		# adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
		# 	inner_kernel=sampler,
		# 	num_adaptation_steps=int(0.8 * self.num_burnin_steps),
		# 	target_accept_prob=tf.cast(0.75, tf.float32))


		sampler = tfp.mcmc.SliceSampler(
			target_log_prob_fn=log_likelihood_fn,
			step_size=self.step_size_hmc,
			max_doublings=4,
			experimental_shard_axis_names=None,
			name=None
		)


		return sampler



	# @tf.function
	def get_samples_HMC(self,log_likelihood_fn,Nsamples=None):
		"""

		self.Nsamples_per_state0: int
		self.initial_states_sampling: [Nstates0,dim]
		self.num_burnin_steps: int
		"""

		if self.adaptive_hmc is None:
			# self.adaptive_hmc = self.initialize_HMCsampler(log_likelihood_fn)
			self.adaptive_hmc = self.initialize_NoUTurnSamplersampler(log_likelihood_fn)

		if Nsamples is not None:
			self.Nsamples_per_state0 = Nsamples

		logger.info("Getting MCMC chains for {0} states, with {1} samples each; total: {2} samples".format(self.initial_states_sampling.shape[0],self.Nsamples_per_state0,self.initial_states_sampling.shape[0]*self.Nsamples_per_state0))
		# samples, is_accepted = tfp.mcmc.sample_chain(
		# 	num_results=self.Nsamples_per_state0,
		# 	num_burnin_steps=self.num_burnin_steps,
		# 	current_state=self.initial_states_sampling,
		# 	kernel=self.adaptive_hmc,
		# 	trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

		samples, kernel_results = tfp.mcmc.sample_chain(
			num_results=self.Nsamples_per_state0,
			num_burnin_steps=self.num_burnin_steps,
			current_state=self.initial_states_sampling,
			kernel=self.adaptive_hmc,
			trace_fn=lambda current_state, kernel_results: kernel_results)

		samples = tf.reshape(samples,(-1,self.dim_in))
		# samples = tf.concat([samples,-samples],axis=0)

		return samples


@hydra.main(config_path="./config",config_name="config")
def reconstruction_loss_with_sampler(cfg):




	"""
	Three methods:
	1) Gather samples that maximize the posterior (MAP) with ESS
	2) Do variational inference figuing out the limits of wlim, with sobol sequence
	3) Do batch payesian optimization with local penalization; but we need to use an improvement-based criterion


	"""


	dim_x = 2

	# Create grid for plotting:
	xmin = -5.0
	xmax = +2.0
	Ndiv = 41

	omega_lim = 3.0
	Nomegas = 100
	Nbatches = 10 # Number of iterations
	# Nomegas = 20
	# Nbatches = 2 # Number of iterations
	# Dw = (2.*omega_lim)**dim_x / Nomegas
	Dw = 1.0
	xpred = CommonUtils.create_Ndim_grid(xmin=xmin,xmax=xmax,Ndiv=Ndiv,dim=dim_x) # [Ndiv**dim_x,dim_x]
	
	spectral_density = VanDerPolSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim=dim_x,data=xpred);
	fx_true = spectral_density._nonlinear_system_fun(xpred)
	dim_out_ind = 0
	fx_true = fx_true[:,dim_out_ind:dim_out_ind+1] # [Ndiv,1]
	omegapred = np.array([[0.0,0.0]])
	inverse_fourier_toolbox = InverseFourierTransformKernelToolbox(spectral_density=spectral_density,dim=dim_x,dim_out_ind=0,omegapred=omegapred)

	Ndiv_omega_for_analysis = 51
	omegapred_analysis = CommonUtils.create_Ndim_grid(xmin=-omega_lim,xmax=omega_lim,Ndiv=Ndiv_omega_for_analysis,dim=dim_x) # [Ndiv**dim_x,dim_x]

	def integrand(omega_in):

		omega_in = tf.convert_to_tensor(omega_in,dtype=tf.float32)

		if omega_in.shape[1] != 2 and omega_in.shape[0] == 2:
			omega_in = tf.transpose(omega_in)

		inverse_fourier_toolbox.update_spectral_density_and_angle(omega_in)
		fx = inverse_fourier_toolbox.get_fx_integrand(xpred,Dw) # [Npoints, Nomegas]
		loss_val = tf.reduce_mean(0.5*(fx_true - fx)**2,axis=0,keepdims=True) # [1, Nomegas]
		# loss_val_tp = loss_val.T # [Nomegas,1]
		loss_val_exp = tf.math.exp(-loss_val) # [1, Nomegas]
		# pdb.set_trace()
		return loss_val_exp

	integrand_loss_vals_exp = integrand(omegapred_analysis)
	int_sum = tf.reduce_sum(integrand_loss_vals_exp)


	def integrand_normalized(omega_in):
		return 10.*integrand(omega_in) / int_sum

	# def log_integrand_normalized(omega_in):
	# 	# pdb.set_trace()
	# 	return tf.math.log(integrand_normalized(omega_in)[0,:])
	# 	
	
	# Loss minimum value (will be reached when the integrand goes to zero)
	loss_min = -tf.reduce_mean(0.5*fx_true**2,axis=0,keepdims=True) # [1, Nomegas]

	sigma_noise_stddev = 0.5

	def log_integrand_normalized(omega_in):
		omega_in = tf.convert_to_tensor(omega_in,dtype=tf.float32)

		if omega_in.ndim == 1:
			omega_in = tf.expand_dims(omega_in,axis=0)

		if omega_in.shape[1] != 2 and omega_in.shape[0] == 2:
			omega_in = tf.transpose(omega_in)

		inverse_fourier_toolbox.update_spectral_density_and_angle(omega_in)
		fx = inverse_fourier_toolbox.get_fx_integrand(xpred,Dw) # [Npoints, Nomegas]
		loss_val = -tf.reduce_mean(0.5*((fx_true - fx)/sigma_noise_stddev)**2,axis=0,keepdims=True) # [1, Nomegas]

		# Offset:
		loss_out = loss_val - loss_min # [1, Nomegas]

		loss_out = loss_out[0,:]
		return loss_out
	
	def log_integrand_normalized_to_numpy(omega_in):
		log_int = log_integrand_normalized(omega_in)
		return log_int.numpy()


	integrand_loss_vals_exp = log_integrand_normalized_to_numpy(omegapred_analysis)

	# log_integrand_normalized(omegapred_analysis)


	# pdb.set_trace()

	# sampler = SampleReconstructionLoss(dim_in=dim_x)
	# samples_vec = sampler.get_samples_HMC(log_integrand_normalized)
	# 
	# 
	# 

	sampler = EllipticalSliceSampler(dim_in=dim_x,target_log_lik=log_integrand_normalized_to_numpy,
									Nsamples=100,Nburning=100,
									Nrestarts=4,omega_lim_random_restarts=2.0)
	samples_vec, omega0_restarts = sampler.run_ess()

	COLOUR_MAP = 'summer'
	LOWER_LIMIT = -omega_lim
	UPPER_LIMIT = +omega_lim
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=False)
	integrand_loss_vals_exp_reshaped = np.reshape(integrand_loss_vals_exp,(Ndiv_omega_for_analysis,Ndiv_omega_for_analysis),order="F")
	hdl_splots.imshow(integrand_loss_vals_exp_reshaped,cmap=plt.get_cmap(COLOUR_MAP), vmin=integrand_loss_vals_exp_reshaped.min(), vmax=integrand_loss_vals_exp_reshaped.max(),
                                extent=[LOWER_LIMIT, UPPER_LIMIT, LOWER_LIMIT, UPPER_LIMIT],
                                interpolation='nearest', origin='lower')
	hdl_splots.plot(samples_vec[:,0],samples_vec[:,1],marker=".",color="crimson",markersize=5,linestyle="None")
	hdl_splots.plot(omega0_restarts[:,0],omega0_restarts[:,1],marker="*",color="indigo",markersize=5,linestyle="None")

	plt.show(block=True)
	





if __name__ == "__main__":


	# reconstruction_loss_analysis()
	# reconstruction_loss_with_bayesian_quadrature_1D()
	# 
	# 
	# reconstruction_loss_with_bayesian_quadrature_2D()
	# 
	# reconstruction_loss_with_sampler()

	test_ifou_1D()
	# test_ifou_2D()
	# 
	# test_reconstruct_dubinscar()
	# 
	# 
	# search_omegas_tf()
