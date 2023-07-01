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

from test_dubin_car import get_sequence_of_feedback_gains_finite_horizon_LQR, rollout_with_finitie_horizon_LQR, generate_trajectories, generate_reference_trajectory


# GP flow:
import gpflow as gpf
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("github")
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 30
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

# using_deltas = True
# assert using_deltas == True

# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

class FourierLayerStationary(tf.keras.layers.Layer):
	
	# https://www.tensorflow.org/tutorials/customization/custom_layers
	
	def __init__(self, Nfeatures):
		assert Nfeatures % 2 == 0, "Requiring this for now"
		super(FourierLayerStationary, self).__init__()
		self.Nfeatures = Nfeatures

	def build(self, input_shape):
		"""
		NOTE: input_shape will be figured out at building time:
			_ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.
		"""

		# regularizer_corr_noise_mat = tf.keras.regularizers.L1(l1=0.5) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
		# regularizer=regularizer_corr_noise_mat
		initializer_freqs = tf.keras.initializers.RandomUniform(minval=-5.0, maxval=+5.0)
		self.W = self.add_weight(shape=(self.Nfeatures//2,int(input_shape[-1])),initializer=initializer_freqs,trainable=True,name="freqs_mat")

		# Amplitudes:
		bandwidth = 10.0
		ampli_j = (1.0 / (self.Nfeatures//2))*bandwidth
		self.a = self.add_weight(shape=(self.Nfeatures//2,1),initializer=tf.keras.initializers.Constant(ampli_j),trainable=False,name="amplis")


	def call(self, inputs):
		"""
		inputs: [Npoints,dim_embedding]
		"""
		WX = tf.matmul(self.W,tf.transpose(inputs)) # [Nfeatures/2,Npoints]
		feat = tf.concat(values=(self.a * tf.math.cos(WX) , self.a * tf.math.sin(WX)),axis=0) # [Nfeatures,Npoints]
		return tf.transpose(feat) # [Npoints,Nfeatures]

	def mercer_kernel_stationary(self, z1, z2):
		"""
		z1: [Npoints1,dim_embedding]
		z2: [Npoints2,dim_embedding]

		This is also a stationary kernel on z1, z2
		"""

		# Prior variance:
		var_prior = 2.0
		nu_j = var_prior * (1. / self.W.shape[0]) # self.Nfeatures//2 = self.W.shape[0]

		# Distance between vectors:
		z1_aux = z1[:,tf.newaxis,:]
		dist_z1z2 = z1_aux - z2

		# # Testing:
		# tf.reduce_all(z1[0,0:5] - z2[0,0:5] == dist_z1z2[0,0,0:5])
		# tf.reduce_all(z1[0,0:5] - z2[1,0:5] == dist_z1z2[0,1,0:5])
		# tf.reduce_all(z1[1,0:5] - z2[0,0:5] == dist_z1z2[1,0,0:5])
		# tf.reduce_all(z1[1,0:5] - z2[1,0:5] == dist_z1z2[1,1,0:5])

		Wz1z2 = tf.matmul(dist_z1z2,tf.transpose(self.W)) # [Npoints1,Npoints2,Nfeatures/2]
		ker_mat = tf.reduce_sum(nu_j * tf.math.cos(Wz1z2),axis=2) # [Npoints1,Npoints2]

		return ker_mat

	def mercer_kernel(self, z1, z2, use_standard_mercer=False):
		"""
		z1: [Npoints1,dim_embedding]
		z2: [Npoints2,dim_embedding]

		This function is the standard mercer kernel implementation.
		However, in this case, the kernel is stationary because of the
		way the features are formulated. Hence, we use an alternative 
		simpler formulation that makes use of trigonometric identities
		"""

		if not use_standard_mercer:
			return self.mercer_kernel_stationary(z1,z2)

		phi1 = self.call(z1) # [Npoints1,Nfeatures]
		phi2 = self.call(z2) # [Npoints2,Nfeatures]

		# For each pair of points, we need to compute the dot product:
		ker_mat = phi1 @ tf.transpose(phi2) # [Npoints1,Npoints2]

		return ker_mat

	def mercer_variance(self, z1):
		"""
		z1: [Npoints1,dim_embedding]
		"""
		raise NotImplementedError

class FourierLayerNonStationary(tf.keras.layers.Layer):
	
	# https://www.tensorflow.org/tutorials/customization/custom_layers
	
	def __init__(self, Nfeatures):
		assert Nfeatures % 2 == 0, "Requiring this for now"
		super(FourierLayerNonStationary, self).__init__()
		self.Nfeatures = Nfeatures
		self.is_built = False

	def build(self, input_shape):
		
		# Weights as if sampled from a Gaussian spectral density; amplitues are S(w), phases sampled from U(-pi,pi)
		w_mean = 0.0;
		w_std = 3.0;
		initializer_freqs = tf.keras.initializers.RandomNormal(mean=w_mean, stddev=w_std)
		self.W = self.add_weight(shape=(self.Nfeatures,int(input_shape[-1])),initializer=initializer_freqs,trainable=False,name="freqs_mat")

		var_prior_init = 20.0 / self.Nfeatures
		self.log_var_prior = self.add_weight(shape=(self.Nfeatures,1),initializer=tf.keras.initializers.Constant(tf.math.log(var_prior_init)),trainable=False,name="log_var_prior")

		nu_samples = scipy.stats.norm.pdf(self.W.numpy(), loc=w_mean, scale=w_std) # [Nfeatures,dim_in]
		nu_samples_init = tf.cast(tf.math.reduce_prod(nu_samples**(1/input_shape[-1]),axis=1,keepdims=True),dtype=tf.float32)
		self.nu_j_aux = self.add_weight(shape=(self.Nfeatures,1),initializer=tf.keras.initializers.Constant(nu_samples_init),trainable=False,name="nu_j_aux")

		initializer_psi_j = tf.keras.initializers.RandomUniform(minval=-math.pi, maxval=math.pi)
		self.psi_j = self.add_weight(shape=(self.Nfeatures,1),initializer=initializer_psi_j,trainable=False,name="psi_j")
		
		self.is_built = True

		
		"""
		# Weights as if sampled from a Gaussian spectral density; amplitues are S(w), phases sampled from U(-pi,pi)
		w_mean = 0.0;
		w_std_a = 5.0;
		w_std_b = 1.0;
		initializer_freqs = tf.keras.initializers.RandomNormal(mean=w_mean, stddev=(w_std_a+w_std_b)/2)
		self.W = self.add_weight(shape=(self.Nfeatures//2,int(input_shape[-1])),initializer=initializer_freqs,trainable=False,name="freqs_mat")

		a_samples = scipy.stats.norm.pdf(self.W.numpy(), loc=w_mean, scale=w_std_a) # [Nfeatures/2,dim_in]
		a_samples = np.prod(a_samples**(1/input_shape[-1]),axis=1,keepdims=True)
		
		b_samples = scipy.stats.norm.pdf(self.W.numpy(), loc=w_mean, scale=w_std_b) # [Nfeatures/2,dim_in]
		b_samples = np.prod(b_samples**(1/input_shape[-1]),axis=1,keepdims=True)
		
		bandwidth = 20.0
		ampli_j = (1.0 / (self.Nfeatures//2))*bandwidth

		a_samples_tf = tf.constant(ampli_j*a_samples,dtype=tf.float32) # [Nfeatures/2,1]
		self.a = self.add_weight(shape=(self.Nfeatures//2,1),initializer=tf.keras.initializers.Constant(a_samples_tf),trainable=False,name="amplis_a")

		b_samples_tf = tf.constant(ampli_j*b_samples,dtype=tf.float32) # [Nfeatures/2,1]
		self.b = self.add_weight(shape=(self.Nfeatures//2,1),initializer=tf.keras.initializers.Constant(b_samples_tf),trainable=False,name="amplis_b")
		"""
		
		# TODO:
		# Make bandwidth and w_std part of the optimization

	def get_var_prior(self):
		return tf.math.exp(self.log_var_prior)

	def get_nu_j(self):
		return self.nu_j_aux * self.get_var_prior() # [Nfeatures,1]

	def call(self, inputs):
		"""
		inputs: [Npoints,dim_embedding]
		"""
		WX = tf.matmul(self.W,tf.transpose(inputs)) # [Nfeatures/2,Npoints]
		# feat = self.a * tf.math.cos(WX + self.psi_j) # [Nfeatures,Npoints]
		feat = tf.math.cos(WX + self.psi_j) # [Nfeatures,Npoints]
		return tf.transpose(feat) # [Npoints,Nfeatures]


	# def call(self, inputs):
	# 	"""
	# 	inputs: [Npoints,dim_embedding]
	# 	"""
	# 	WX = tf.matmul(self.W,tf.transpose(inputs)) # [Nfeatures/2,Npoints]
	# 	feat = tf.concat(values=(self.a * tf.math.cos(WX) , self.b * tf.math.sin(WX)),axis=0) # [Nfeatures,Npoints]
	# 	return tf.transpose(feat) # [Npoints,Nfeatures]

	def mercer_kernel(self, z1, z2):
		"""
		z1: [Npoints1,dim_embedding]
		z2: [Npoints2,dim_embedding]
		"""

		# raise

		phi1 = self.call(z1) # [Npoints1,Nfeatures]
		phi2 = self.call(z2) # [Npoints2,Nfeatures]

		phi_times_var = phi1 * tf.transpose(self.get_nu_j()) # [Npoints1,Nfeatures]

		# For each pair of points, we need to compute the dot product:
		ker_mat = phi_times_var @ tf.transpose(phi2) # [Npoints1,Npoints2]

		return ker_mat

	def mercer_variance(self, z1):
		"""
		z1: [Npoints1,dim_embedding]
		"""

		phi1 = self.call(z1) # [Npoints1,Nfeatures]

		phi1_times_var = phi1 * tf.transpose(self.get_nu_j()) # [Npoints1,Nfeatures] = [Npoints1,Nfeatures] * [1,Nfeatures]

		ker_var = tf.math.reduce_sum(phi1*phi1_times_var,axis=1) # [Npoints1,Nfeatures] -> [Npoints1,]

		# tf.print(self.get_nu_j())
		# tf.print(ker_var)

		return ker_var

	def mean(self,z1):
		"""
		z1: [Npoints1,dim_embedding]
		"""

		phi1 = self.call(z1) # [Npoints1,Nfeatures]
		phi1_mean = (1./self.Nfeatures) * tf.math.reduce_sum(phi1,axis=1) # [Npoints,]
		return phi1_mean # [Npoints1,]



class DeepKarhunenLoeve(tf.keras.Model):

	# https://www.tensorflow.org/tutorials/quickstart/advanced

	def __init__(self,fourier_layer_stationary,Nfeatures):
		super(DeepKarhunenLoeve, self).__init__()
		self.embedding1 = tf.keras.layers.Dense(units=128, activation='relu') # units is the output dimension
		self.embedding2 = tf.keras.layers.LSTM(units=64, activation='tanh') # units is the output dimension

		if fourier_layer_stationary:
			self.fourier_layer = FourierLayerStationary(Nfeatures=Nfeatures) # Nfeatures is the output dimension
		else:
			self.fourier_layer = FourierLayerNonStationary(Nfeatures=Nfeatures) # Nfeatures is the output dimension
		
		# self.get_mean = tf.keras.layers.Add() # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add

	def call(self, x):
		"""
		x: [Npoints,dim]

		x_out: [Npoints,2]
		"""
		z = self.pre_embedding(x) # Output: [Npoints,n_out=64]

		if not self.fourier_layer.is_built: self.fourier_layer(z) # Dummy call

		phi_mean = self.fourier_layer.mean(z) # [Npoints,]
		phi_var = self.fourier_layer.mercer_variance(z) # [Npoints,]

		x_out = tf.concat([tf.reshape(phi_mean,(-1,1)),tf.reshape(phi_var,(-1,1))],axis=1) # [Npoints,2]


		"""
		x3 = self.fourier_layer(x2) # Input: [Npoints,n_out=64] || Output: [Npoints,Nfeatures]
		x4_mean = tf.math.reduce_sum(x3,axis=1) # [Npoints,]
		x4_var = self.fourier_layer.mercer_variance(x3) # [Npoints,]
		phi1 = self.call(z1) # [Npoints1,Nfeatures]
		ker_var = tf.math.reduce_sum(phi1*phi1,axes=1) # [Npoints1,]
		"""

		return x_out

	def pre_embedding(self, x):
		"""
		x: [Npoints,dim]
		"""
		# pdb.set_trace()
		x1 = self.embedding1(x) # [Npoints,n_out=128]
		if len(x1.shape) == 2:
			x1 = x1[:,tf.newaxis,:]
		
		x2 = self.embedding2(x1) 	# Input: [Npoints,time_steps=1,dim_in] -> in Tensorflow terminology: [batch,timesteps,feature], https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM#args
									# Output: [Npoints,n_out=64]
		return x2

	def kernel(self,x1,x2=None):
		"""
		x1: [Npoints1,dim]
		x2: [Npoints2,dim]

		out: [Npoints1,Npoints2]
		"""

		z1 = self.pre_embedding(x1) # Output: [Npoints1,n_out=64]

		if x2 is None:
			return self.fourier_layer.mercer_variance(z1) # [Npoints1,]

		z2 = self.pre_embedding(x2) # Output: [Npoints2,n_out=64]

		# # Hack my own code:
		# z1 = tf.concatenate([x1[:,0:1]**2]*z1.shape[1],axis=1)
		# z2 = tf.concatenate([x2[:,0:1]**2]*z2.shape[1],axis=1)

		# pdb.set_trace()

		ker_mat = self.fourier_layer.mercer_kernel(z1,z2) # [Npoints1,Npoints2]

		return ker_mat

def test_model(model):

	# pdb.set_trace()

	x_in1 = np.random.rand(10,1)
	y_in1 = model(x_in1)
	print("x_in1:",x_in1)
	print("y_in1:",y_in1)

	x_in2 = np.random.rand(15,1)
	y_in2 = model(x_in2)
	print("x_in2:",x_in2)
	print("y_in2:",y_in2)

	ker12 = model.kernel(x_in1,x_in2)

	print("ker12:",ker12)





	kxx = model.kernel(x_in1,x_in1)
	cov_pred = kxx.numpy()
	var_pred = np.diag(cov_pred)
	print(var_pred)

	var_pred = model.kernel(x_in1)
	var_pred = var_pred.numpy()
	print(var_pred)
	# stdpred = np.sqrt(var_pred)
	# pdb.set_trace()

	# pdb.set_trace()

def ker_fun_parabola(x,xp,a_min=-1.,a_max=1.):
	"""
	x: [Npoints,1]
	xp: [Npoints,1]

	f(x) = a * x**2, a in U(a_min,a_max)

	E[a] = 0.5*(a_min+a_max), Var[a] = (a_max-a_min)**2 / 12

	out: [Npoints,Npoints]
	"""

	dim_in = 1
	assert x.shape[1] == dim_in
	assert xp.shape[1] == dim_in

	# x = squash(x)
	# xp = squash(xp)

	xp = xp.T

	nu = (a_max - a_min)**2 / 12.

	return nu*(x**2)*(xp**2)

@hydra.main(config_path="./config",config_name="config")
def test_moment_matching(cfg):
	"""

	The equations of the dynamical system are known, but not its parameters

	f_simu(x) = a * x**2 + c, a ~ U(-1,1), c ~ U(-1,1)

	Construct features as phi_j(x) = a_j * x**2 + c_j

	Then, construct linear model as

	f(x) = sum_j beta_j * phi_j(x), beta_j ~ N(m_j,v_j), where m_j = 1/M

	Connection with CLT: E[f(x)] represents the distribution of the empirical mean, which true mean
	is the mean of f_simu and which variance decreases lineary with M.
	The CLT only tells us that as M increases both means will coincide, and that's desirable
	in the sense that we want f(x) to reflect f_simu in mean. 

	Now, we pump variance into the model via beta_j
	"""


	# Simulated function:
	# f_simu = lambda x,a,c: a * x**2 + c
	f_simu = lambda x,a,c: np.exp(-x**2)*np.cos(x*a) + c

	# Empirical moments:
	Nac_samples = 300
	a_samples_simu = np.random.uniform(low=-1.0,high=1.0,size=(Nac_samples,1))
	c_samples_simu = np.random.uniform(low=-0.1,high=0.1,size=(Nac_samples,1))
	f_simu_mean = lambda x: np.mean(f_simu(x,a_samples_simu,c_samples_simu),axis=0) # [Nac_samples]
	f_simu_std = lambda x: np.std(f_simu(x,a_samples_simu,c_samples_simu),axis=0) # [Nac_samples]

	xmin = -2; xmax = 2.; Npred = 201
	xpred = np.linspace(xmin,xmax,Npred)

	meanpred_fsimu = np.zeros(Npred)
	stdpred_fsimu = np.zeros(Npred)

	for kk in range(Npred):
		meanpred_fsimu[kk] = f_simu_mean(xpred[kk])
		stdpred_fsimu[kk] = f_simu_std(xpred[kk])

	M = 100
	a_samples = np.random.uniform(low=-1.0,high=1.0,size=(M,1))
	c_samples = np.random.uniform(low=-0.1,high=0.1,size=(M,1))

	# Features:
	# phi_j = lambda x: a_samples * x**2 + c_samples
	phi_j = lambda x: f_simu(x,a_samples,c_samples)

	# Linear model:
	m_j = 1./M
	nu_j = 1./M

	# Moments BLM:
	meanpred_blm = np.zeros(Npred)
	stdpred_blm = np.zeros(Npred)

	for kk in range(Npred):
		meanpred_blm[kk] = m_j * np.sum(phi_j(xpred[kk]))
		stdpred_blm[kk] = np.sqrt(nu_j * np.sum(phi_j(xpred[kk])**2))

	hdl_fig_ker, hdl_splots_ker = plt.subplots(2,1,figsize=(20,8),sharex=False)
	
	hdl_splots_ker[0].plot(xpred,meanpred_fsimu,linestyle="-",color="navy",lw=2,alpha=0.8)
	hdl_splots_ker[0].fill_between(xpred,meanpred_fsimu-stdpred_fsimu,meanpred_fsimu+stdpred_fsimu,color="navy",alpha=0.4)
	hdl_splots_ker[0].set_xlim([xmin,xmax])
	# hdl_splots_ker[0].set_ylim([0.0,np.amax(stdpred)*1.1])
	hdl_splots_ker[0].set_xlabel(r"$x$",fontsize=fontsize_labels)
	hdl_splots_ker[0].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	hdl_splots_ker[0].set_title(r"Empirical distribution of fsimu",fontsize=fontsize_labels)
	hdl_splots_ker[0].set_xticks([xmin,0,xmax])
	# hdl_splots_ker[0].set_yticks([])

	hdl_splots_ker[1].plot(xpred,meanpred_blm,linestyle="-",color="navy",lw=2,alpha=0.8)
	hdl_splots_ker[1].fill_between(xpred,meanpred_blm-stdpred_blm,meanpred_blm+stdpred_blm,color="navy",alpha=0.4)
	hdl_splots_ker[1].set_xlim([xmin,xmax])
	# hdl_splots_ker[1].set_ylim([0.0,np.amax(stdpred)*1.1])
	hdl_splots_ker[1].set_xlabel(r"$x$",fontsize=fontsize_labels)
	hdl_splots_ker[1].set_ylabel(r"$f(x)$",fontsize=fontsize_labels)
	hdl_splots_ker[1].set_title(r"Gaussian distribution of BLM",fontsize=fontsize_labels)
	hdl_splots_ker[1].set_xticks([xmin,0,xmax])
	# hdl_splots_ker[1].set_yticks([])

	plt.show(block=True)


class SaveBestModel(tf.keras.callbacks.Callback):
	def __init__(self):
		self.best = -float('inf')

	def on_epoch_end(self, epoch, logs=None):
		metric_value = logs["accuracy"]
		if metric_value >= self.best:
			print("\nNew best accuracy: {0:f} at epoch {1:d}".format(metric_value,epoch))
			self.best = metric_value
			self.best_weights = self.model.get_weights()
		# print(self.model.get_weights())

class LossMomentMatching(tf.keras.losses.Loss):
	def __init__(self, name="LossMomentMatching", **kwargs):
		super().__init__(name=name, **kwargs)

	def call(self, y_true, y_pred):
		rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)); # + 100*tf.reduce_sum(tf.cast(y_pred[:,1] < 0.0, tf.float32))
		return rmse


@hydra.main(config_path="./config",config_name="config")
def test_kernels(cfg):

	my_seed = 1
	tf.random.set_seed(my_seed)
	np.random.seed(my_seed)

	# Train an MLP that maps from x to z
	# Fit a simple system, like a parabola

	xmin = -1.0
	xmax = +1.0
	epsi = 1e-3
	dim_in = 1
	Npred = 201
	xpred = np.reshape(np.linspace(xmin+epsi,xmax-epsi,Npred),(-1,dim_in))

	# Simulated function:
	# f_simu = lambda x,a,c: a * x**2 + c
	# f_simu = lambda x,a,c: np.exp(-x**2)*np.cos(x*a) + c
	f_simu = lambda x,a: a*x**2

	# Empirical moments:
	Nac_samples = 300
	a_samples_simu = np.random.uniform(low=-1.0,high=1.0,size=(Nac_samples,1))
	f_simu_mean = lambda x: np.mean(f_simu(x,a_samples_simu),axis=0) # [Nac_samples]
	f_simu_var = lambda x: np.var(f_simu(x,a_samples_simu),axis=0) # [Nac_samples]

	f_simu_mean_vec = np.zeros(Npred)
	f_simu_var_vec = np.zeros(Npred)
	for ii in range(Npred):
		f_simu_mean_vec[ii] = f_simu_mean(xpred[ii,0])
		f_simu_var_vec[ii] = f_simu_var(xpred[ii,0])

	Nevals = 20
	Xevals = xmin + (xmax - xmin)*tf.math.sobol_sample(dim=dim_in,num_results=(Nevals),skip=10000)
	assert Xevals.shape == (Nevals,dim_in)
	Yevals_mean = tf.reshape(np.interp(Xevals[:,0],xpred[:,0],f_simu_mean_vec),(-1,1))
	Yevals_var = tf.reshape(np.interp(Xevals[:,0],xpred[:,0],f_simu_var_vec),(-1,1))

	Yevals = tf.concat([Yevals_mean,Yevals_var],axis=1)

	p_cut = 0.9
	Ntrain = int(p_cut*Nevals)
	x_train = Xevals[0:Ntrain,:]
	x_test = Xevals[Ntrain::,:]

	y_train = Yevals[0:Ntrain,:]
	y_test = Yevals[Ntrain::,:]


	"""
	Follow this tutorial:
	https://www.tensorflow.org/tutorials/quickstart/advanced
	"""
	# mnist = tf.keras.datasets.mnist
	# (x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[..., tf.newaxis]
	x_test = x_test[..., tf.newaxis]

	train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10).batch(4)
	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(4)

	# loss_object = tf.keras.losses.MeanSquaredError()
	# loss_object = tf.keras.losses.MeanAbsoluteError()
	# loss_object = LossMomentMatching()

	# loss_object(y_train[0,:],y_train[1,:])

	model = DeepKarhunenLoeve(fourier_layer_stationary=False,Nfeatures=20) # Nfeatures = 20 works really well; 

	test_model(model)


	# model.compile(): https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=LossMomentMatching(),metrics=['accuracy','mean_squared_error'])
	# Metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics#classes_2
	# accuracy: Calculates how often predictions equal labels.
	# mean_squared_error: Computes the mean squared error between y_true and y_pred.

	save_best_model = SaveBestModel()

	Nepochs = 10000
	# Nepochs = 100
	# model.fit(x_train, y_train, epochs=Nepochs,callbacks=[save_best_model])
	model.fit(x_train, y_train, epochs=Nepochs,callbacks=[save_best_model])
	model.set_weights(save_best_model.best_weights)

	xmin_plot = -1.0
	xmax_plot = +1.0
	xpred_plot = np.reshape(np.linspace(xmin_plot+epsi,xmax_plot-epsi,Npred),(-1,dim_in))

	kxx = model.kernel(xpred_plot,xpred_plot)
	cov_pred = kxx.numpy()
	# stdpred = np.sqrt(np.diag(cov_pred))

	var_pred = model.kernel(xpred_plot)
	var_pred = var_pred.numpy()
	var_pred[var_pred<0.0] = 0.0
	stdpred = np.sqrt(var_pred)
	# pdb.set_trace()

	hdl_fig_ker, hdl_splots_ker = plt.subplots(1,3,figsize=(24,8),sharex=False)
	extent_plot_xpred = [xmin,xmax,xmin,xmax] #  scalars (left, right, bottom, top)
	COLOR_MAP = "copper"

	cov_pred_emp = ker_fun_parabola(xpred_plot,xpred_plot)
	hdl_splots_ker[0].imshow(cov_pred_emp,extent=extent_plot_xpred,origin="upper",cmap=plt.get_cmap(COLOR_MAP),vmin=cov_pred_emp.min(),vmax=cov_pred_emp.max(),interpolation='nearest')
	hdl_splots_ker[0].plot([xmin_plot,xmax_plot],[xmax_plot,xmin_plot],linestyle="-",color="navy",lw=2,alpha=0.8)
	hdl_splots_ker[0].set_xlim([xmin_plot,xmax_plot])
	hdl_splots_ker[0].set_ylim([xmin_plot,xmax_plot])
	hdl_splots_ker[0].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots_ker[0].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
	hdl_splots_ker[0].set_title(r"$k(x_t,x^\prime_t)$ [true]",fontsize=fontsize_labels)
	hdl_splots_ker[0].set_xticks([])
	hdl_splots_ker[0].set_yticks([])

	hdl_splots_ker[1].imshow(cov_pred,extent=extent_plot_xpred,origin="upper",cmap=plt.get_cmap(COLOR_MAP),vmin=cov_pred.min(),vmax=cov_pred.max(),interpolation='nearest')
	hdl_splots_ker[1].plot([xmin_plot,xmax_plot],[xmax_plot,xmin_plot],linestyle="-",color="green",lw=2,alpha=0.8)
	hdl_splots_ker[1].set_xlim([xmin_plot,xmax_plot])
	hdl_splots_ker[1].set_ylim([xmin_plot,xmax_plot])
	hdl_splots_ker[1].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	hdl_splots_ker[1].set_ylabel(r"$x_t^\prime$",fontsize=fontsize_labels)
	hdl_splots_ker[1].set_title(r"$k(x_t,x^\prime_t)$ [reconstructed]",fontsize=fontsize_labels)
	hdl_splots_ker[1].set_xticks([])
	hdl_splots_ker[1].set_yticks([])

	hdl_splots_ker[2].plot(xpred_plot[:,0],f_simu_mean_vec,linestyle="-",color="navy",lw=1.0,alpha=0.9)
	hdl_splots_ker[2].fill_between(xpred_plot[:,0],f_simu_mean_vec-np.sqrt(f_simu_var_vec),f_simu_mean_vec+np.sqrt(f_simu_var_vec),color="navy",alpha=0.4)

	fpred = model(xpred)
	fpred_mean = fpred[:,0]
	fpred_var = fpred[:,1]
	hdl_splots_ker[2].plot(xpred_plot[:,0],fpred_mean,linestyle="-",color="green",lw=1.0,alpha=0.9)
	hdl_splots_ker[2].fill_between(xpred_plot[:,0],fpred_mean-np.sqrt(fpred_var),fpred_mean+np.sqrt(fpred_var),color="green",alpha=0.4)
	hdl_splots_ker[2].set_xlim([xmin_plot,xmax_plot])
	# hdl_splots_ker[2].set_ylim([0.0,np.amax(stdpred)*1.1])
	hdl_splots_ker[2].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	# hdl_splots_ker[2].set_xticks([xmin_plot,0,xmax_plot])
	hdl_splots_ker[2].set_xticks([])
	hdl_splots_ker[2].set_yticks([])


	# hdl_splots_ker[2].plot(xpred_plot[:,0],stdpred,linestyle="-",color="navy",lw=2,alpha=0.8)
	# hdl_splots_ker[2].fill_between(xpred_plot[:,0],np.zeros(Npred),stdpred,color="navy",alpha=0.4)
	# hdl_splots_ker[2].set_xlim([xmin_plot,xmax_plot])
	# hdl_splots_ker[2].set_ylim([0.0,np.amax(stdpred)*1.1])
	# hdl_splots_ker[2].set_xlabel(r"$x_t$",fontsize=fontsize_labels)
	# hdl_splots_ker[2].set_ylabel(r"$\textbf{V}ar[f(x_t)]$",fontsize=fontsize_labels)
	# hdl_splots_ker[2].set_title(r"Variance".format("parabola"),fontsize=fontsize_labels)
	# hdl_splots_ker[2].set_xticks([xmin_plot,0,xmax_plot])
	# hdl_splots_ker[2].set_yticks([])

	# plt.show(block=True)


	savefig = False
	path2folder = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/presentation/parabola_LQRker_like"
	if savefig:
		path2save_fig = "{0:s}/kernel_learning_new_model.png".format(path2folde)
		logger.info("Saving fig at {0:s} ...".format(path2save_fig))
		hdl_fig_error_rates.savefig(path2save_fig,bbox_inches='tight',dpi=600,transparent=True)
		logger.info("Done saving fig!")
	else:
		plt.show(block=False)


	save_model = True
	if save_model:
		where2save = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/saved_models_deep_karhunen_loeve/dkl_weights_{0:s}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
		logger.info("Saving model at {0:s} ...".format(where2save))
		# model.save(where2save)
		model.save_weights(where2save, overwrite=True, save_format=None, options=None)
	else:
		logger.info("Not saving model ...")


	plt.show(block=True)


	# print("Do custom colormaps, black->green, black->blue using matplotlib.colors.Colormap()") # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.colors.ListedColormap.html#matplotlib.colors.ListedColormap
	# also, remove diagonal lines if doing colormaps



	# hdl_fig_parabola_fitting, hdl_splots_parabola_fitting = plt.subplots(1,1,figsize=(12,8),sharex=False)
	# hdl_splots_parabola_fitting = [hdl_splots_parabola_fitting]


	
	# hdl_splots_parabola_fitting[0].plot(Xevals[:,0],Yevals,linestyle="None",marker="o",markersize=7,color="darkgreen")
	# hdl_splots_parabola_fitting[0].plot(xpred_plot[:,0],f_fit,linestyle="-",color="navy",lw=2.0,alpha=0.8)
	# hdl_splots_parabola_fitting[0].set_xlim([xmin,xmax])


	# fft_signal = tf.signal.rfft(f_fit)
	# hdl_splots_parabola_fitting[1].plot(fft_signal,linestyle="-",color="gray",lw=1.0,alpha=0.8)


	# Fourier transform of Parabola f(x) = x^2 for x in [-1,1] and 0 elsewhere:
	a_w = lambda w: (4*w*np.cos(w) + 2.*(-2. + w**2)*np.sin(w))/(w**3) # cos() transform
	b_w = lambda w: np.zeros(w.shape[0]) # sin() transform
	a0 = 2/3.
	# S_w = lambda w: np.sqrt((a_w(w))**2 + (b_w(w))**2)
	w_lim = 20.
	wpred = np.linspace(-w_lim,w_lim,Npred)
	# S_wpred = S_w(wpred)
	a_wpred = a_w(wpred)
	b_wpred = b_w(wpred)


	# S_wpred = S_w(wpred)
	# hdl_splots_parabola_fitting[1].plot(wpred,S_wpred,linestyle="-",color="gray",lw=2.0,alpha=0.8)
	# hdl_splots_parabola_fitting[1].set_xlim([-w_lim,w_lim])

	
	# hdl_splots_parabola_fitting[2].plot(wpred,a_wpred,linestyle="-",color="gray",lw=2.0,alpha=0.8)
	# hdl_splots_parabola_fitting[2].set_xlim([-w_lim,w_lim])

	# hdl_splots_parabola_fitting[3].plot(wpred,b_wpred,linestyle="-",color="gray",lw=2.0,alpha=0.8)
	# hdl_splots_parabola_fitting[3].set_xlim([-w_lim,w_lim])





	# NOTES:
	# 1) The kernel looks stationary, this shouldn't be the case
	# 2) Explore how well the model fits parabola data; does the learned feature z = phi(x) behave like a parabola outside the training data, i.e., z = x**2 ??
	# 3) Compare this kernel with the "brute force approach" I used for the presentation, which is basically multiplying parabola features
	# 4) So, the question is, given z = phi(x), does k(z1,z2) = k(z1-z2) imply k(phi(x1),phi(x2)) = k(x1-x2) ? I don't think so


	plt.show(block=True)

	"""





	# More granularity:
	optimizer = tf.keras.optimizers.Adam()

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	test_loss = tf.keras.metrics.Mean(name='test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


	# Create an instance of the model
	# https://www.tensorflow.org/tutorials/keras/classification#train_the_model
	
	


	# model.fit(train_images, train_labels, epochs=10)
	# model.compile(optimizer='adam',
	#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	#               metrics=['accuracy'])



	"""

if __name__ == "__main__":

	test_kernels()


	# test_moment_matching()





