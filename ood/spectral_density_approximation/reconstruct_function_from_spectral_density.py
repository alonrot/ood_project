import pdb
import math
import numpy as np
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from lqrker.utils.common import CommonUtils
import tensorflow as tf
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class ReconstructFunctionFromSpectralDensity(tf.keras.layers.Layer):

	def __init__(self, dim_in: int, omega_lim: float, Nomegas: int, inverse_fourier_toolbox: InverseFourierTransformKernelToolbox, Xtrain: tf.Tensor, Ytrain: tf.Tensor, omegas_weights=None, **kwargs):

		super().__init__(**kwargs)

		self.dim_in = dim_in
		self.Dw_voxel_val = (2.*omega_lim) / Nomegas
		self.omega_lim = omega_lim
		logger.info("voxel value Dw for reconstructing f(xt): {0:f}".format(float(self.Dw_voxel_val)))
		self.inverse_fourier_toolbox = inverse_fourier_toolbox

		# initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
		# self.delta_omegas = self.add_weight(shape=(Nomegas,), initializer=initializer(shape=(Nomegas,)), trainable=True, name="delta_omegas")
		self.delta_omegas_pre_activation = self.add_weight(shape=(Nomegas,), initializer=tf.keras.initializers.Constant(value=0.0), trainable=True, name="delta_omegas_pre_activation")

		if omegas_weights is None:
			initializer_omegas = tf.keras.initializers.RandomUniform(minval=-omega_lim, maxval=omega_lim)
			regularizer_omegas = tf.keras.regularizers.L1(l1=0.01) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			# regularizer_omegas = tf.keras.regularizers.L2(l2=100.) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			self.omegas_weights = self.add_weight(shape=(Nomegas,self.dim_in), initializer=initializer_omegas, regularizer=regularizer_omegas, trainable=True, name="omegas_weights")
		else:
			assert omegas_weights.shape[1] == self.dim_in and omegas_weights.shape[0] == Nomegas
			self.omegas_weights = omegas_weights


		# learn_dX_voxels = False
		learn_dX_voxels = True
		if learn_dX_voxels:
			self.delta_statespace_preactivation = self.add_weight(shape=(Xtrain.shape[0],1), initializer=tf.keras.initializers.Constant(value=0.0), trainable=True, name="delta_statespace_preactivation")
			# self.dX_voxel_val = (tf.reduce_max(Xtrain) - tf.reduce_min(Xtrain))**self.dim_in / Xtrain.shape[0]
			# self.dX_voxel_val = 1. / Xtrain.shape[0]
			# self.dX_voxel_val = 10. # works well for dubins car
			# self.delta_statespace_preactivation = tf.zeros((Xtrain.shape[0],1))

			# self.dX_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=10.0), trainable=True, name="dX_voxel_val")
			self.dX_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=20./4001*2), trainable=True, name="dX_voxel_val")

			"""
			Las thing to try:
			1) Just learn the voxel self.dX_voxel_val, but not dX; see if that has any impact... It's easier to explain in theory


			Last thing to try: -> doesn't work
			1) Use a discrete grid for the omegas (we can't for the state space because it comes fromm the data)
			2) LEarn the voxels; 
			3) prune the omegas whose delta voxels are close to zero
			"""
		else:
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			logger.info("[WARNING]: NOT learning dX_voxels ... (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
			# pdb.set_trace()
			# 
			self.delta_statespace_preactivation = tf.zeros((Xtrain.shape[0],1))
			# self.dX_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=10.0), trainable=True, name="dX_voxel_val")
			# self.dX_voxel_val = 10./Xtrain.shape[0]
			self.dX_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=10.0), trainable=True, name="dX_voxel_val")
			# self.dX_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=20./4001), trainable=True, name="dX_voxel_val")



		# assert tf.reduce_all(self.inverse_fourier_toolbox.spectral_density.xdata == Xtrain)
		if not tf.reduce_all(self.inverse_fourier_toolbox.spectral_density.xdata == Xtrain):
			logger.info("[WARNING]: Reconstruction loss is not using the training set, but the testing set (smaller)")

		self.Xtrain = Xtrain
		self.Ytrain = Ytrain

		# Loss history for analysis:
		self.loss_vec = None


	def get_delta_omegas(self,delta_omegas_pre_activation):
		delta_omegas = self.Dw_voxel_val*tf.keras.activations.sigmoid(delta_omegas_pre_activation) # Squeeze to (0,1)
		return delta_omegas

	def get_omegas_weights(self):
		# omegas_weights_withinlims = self.omega_lim*tf.keras.activations.tanh(self.omegas_weights)
		omegas_weights_withinlims = self.omegas_weights
		return omegas_weights_withinlims

	def get_delta_statespace(self,delta_statespace_preactivation):
		delta_statespace = self.dX_voxel_val*tf.keras.activations.sigmoid(delta_statespace_preactivation) # Squeeze to (0,1)
		return delta_statespace

	def reconstruct_function_at(self,xpred):

		delta_omegas = self.get_delta_omegas(self.delta_omegas_pre_activation)
		delta_statespace = self.get_delta_statespace(self.delta_statespace_preactivation)
		self.inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred=self.get_omegas_weights(),Dw=None,dX=delta_statespace)
		fx_integrand = self.inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred=xpred,Dw_vec=delta_omegas) # [Npoints, Nomegas]
		fx_reconstructed = tf.math.reduce_sum(fx_integrand,axis=1,keepdims=True) # Integrate wrt omegas [Npoints, 1]

		return fx_reconstructed


	def get_integrand_for_pruning(self,xpred):
		delta_omegas = self.get_delta_omegas(self.delta_omegas_pre_activation)
		self.inverse_fourier_toolbox.update_spectral_density_and_angle(omegapred=self.get_omegas_weights(),Dw=None)
		fx_integrand = self.inverse_fourier_toolbox.get_fx_integrand_variable_voxels(xpred=xpred,Dw_vec=delta_omegas) # [Npoints, Nomegas]
		return fx_integrand


	def loss_reconstruction_fun(self):

		sigma_noise_stddev = 0.5
		fx_reconstructed = self.reconstruct_function_at(xpred=self.Xtrain)
		loss_val = tf.reduce_mean(0.5*((self.Ytrain - fx_reconstructed)/sigma_noise_stddev)**2,axis=0,keepdims=True) # [1, 1]

		return loss_val

	def train(self,Nepochs,learning_rate,stop_loss_val,print_every=100):

		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

		str_banner = " << Training the delta omegas >> "

		epoch = 0
		done = False
		loss_value_best = float("Inf")
		trainable_weights_best = self.get_weights()
		plotting_dict = dict(plotting=False)
		# print_every = 100
		self.loss_vec = np.zeros(Nepochs)
		while epoch < Nepochs and not done:

			if (epoch+1) % print_every == 0:
				logger.info("="*len(str_banner))
				logger.info(str_banner)
				logger.info(" << Epoch {0:d} / {1:d} >> ".format(epoch+1, Nepochs))
				logger.info("="*len(str_banner))

			with tf.GradientTape() as tape:
				loss_value = self.loss_reconstruction_fun()

			grads = tape.gradient(loss_value, self.trainable_weights)
			optimizer.apply_gradients(zip(grads, self.trainable_weights))

			if (epoch+1) % print_every == 0:
				logger.info("    * Predictive loss (current): {0:.4f}".format(tf.squeeze(loss_value)))
				logger.info("    * Predictive loss (best): {0:.4f}".format(tf.squeeze(loss_value_best)))
				# logger.info("    * Weights (current): {0:s}".format(self._weights2str(self.trainable_weights)))
				# logger.info("    * Weights (best): {0:s}".format(self._weights2str(trainable_weights_best)))
				# logger.info("    * Gradients (current): {0:s}".format(self._weights2str(grads)))

			if loss_value <= stop_loss_val:
				done = True
			
			if loss_value < loss_value_best:
				loss_value_best = loss_value
				trainable_weights_best = self.get_weights()
			

			# Register values:
			self.loss_vec[epoch] = tf.squeeze(loss_value)
			
			epoch += 1

		if done == True:
			logger.info(" * Training finished because loss_value = {0:f} (<= {1:f})".format(float(loss_value),float(stop_loss_val)))

		self.set_weights(weights=trainable_weights_best)

		logger.info("Training finished!")

	def get_loss_history(self):
		assert self.loss_vec is not None, "Need to run .train() first"
		return self.loss_vec


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