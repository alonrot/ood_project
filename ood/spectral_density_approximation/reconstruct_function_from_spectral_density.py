import pdb
import math
import numpy as np
from ood.fourier_kernel import InverseFourierTransformKernelToolbox
from lqrker.utils.common import CommonUtils
import time
import tensorflow as tf
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class ReconstructFunctionFromSpectralDensity(tf.keras.layers.Layer):

	def __init__(self, dim_in: int, dw_voxel_init: float, dX_voxel_init: float, omega_lim: float, Nomegas: int, inverse_fourier_toolbox: InverseFourierTransformKernelToolbox, Xtest: tf.Tensor, Ytest: tf.Tensor, **kwargs):

		super().__init__(**kwargs)

		self.dim_in = dim_in
		self.inverse_fourier_toolbox = inverse_fourier_toolbox

		# Integration step omegas:
		self.Dw_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=tf.math.log(dw_voxel_init)), trainable=True, name="Dw_voxel_val")

		# initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
		# self.delta_omegas = self.add_weight(shape=(Nomegas,), initializer=initializer(shape=(Nomegas,)), trainable=True, name="delta_omegas")
		# self.delta_dw_voxels_pre_activation = self.add_weight(shape=(Nomegas,1), initializer=tf.keras.initializers.Constant(value=0.0), trainable=True, name="delta_omegas_pre_activation")



		self.dbg_flag = True
		if self.dbg_flag:

			assert dim_in == 2
			
			assert (int(np.sqrt(Nomegas) + 0.5) ** 2 == Nomegas), "Nomegas must be a power of 2"

			# # Frequencies locations:
			# initializer_omegas = tf.keras.initializers.RandomUniform(minval=-omega_lim, maxval=omega_lim)
			# regularizer_omegas = tf.keras.regularizers.L1(l1=0.01) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			# # regularizer_omegas = tf.keras.regularizers.L2(l2=100.) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			# self.omegas_weights = self.add_weight(shape=(Nomegas,self.dim_in), initializer=initializer_omegas, regularizer=regularizer_omegas, trainable=True, name="omegas_weights")
			
			self.Nomegas_for_regular_grid = Nomegas
			log_L_for_regular_grid_init = tf.zeros(self.dim_in)
			self.log_L_for_regular_grid = self.add_weight(shape=(self.dim_in), initializer=tf.keras.initializers.Constant(value=log_L_for_regular_grid_init), trainable=True, name="L_for_regular_grid")

			omega_per_dim =[]
			Nomegas_per_dim = int(np.sqrt(Nomegas))
			for ii in range(log_L_for_regular_grid_init.shape[0]):
				omega_lim_per_dim = 2.*math.pi / tf.math.exp(log_L_for_regular_grid_init[ii])
				omega_per_dim += [tf.linspace(-omega_lim_per_dim,omega_lim_per_dim,Nomegas_per_dim)]

			omega_grid = tf.meshgrid(*omega_per_dim,indexing="ij")
			self.omegas_weights = tf.concat([tf.reshape(omega_grid_el,(-1,1)) for omega_grid_el in omega_grid],axis=1)

			self.delta_dw_voxels_pre_activation = self.add_weight(shape=(1), initializer=tf.keras.initializers.Constant(value=0.0), trainable=True, name="delta_omegas_pre_activation")

		else:

			self.delta_dw_voxels_pre_activation = self.add_weight(shape=(Nomegas,1), initializer=tf.keras.initializers.Constant(value=0.0), trainable=True, name="delta_omegas_pre_activation")

			# Frequencies locations:
			initializer_omegas = tf.keras.initializers.RandomUniform(minval=-omega_lim, maxval=omega_lim)
			regularizer_omegas = tf.keras.regularizers.L1(l1=0.01) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			# regularizer_omegas = tf.keras.regularizers.L2(l2=100.) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			self.omegas_weights = self.add_weight(shape=(Nomegas,self.dim_in), initializer=initializer_omegas, regularizer=regularizer_omegas, trainable=True, name="omegas_weights")
			# self.omegas_weights = self.add_weight(shape=(Nomegas,self.dim_in), initializer=initializer_omegas, trainable=True, name="omegas_weights")


		# Integration step dX:
		self.delta_dX_voxels_preactivation = self.add_weight(shape=(Xtest.shape[0],1), initializer=tf.keras.initializers.Constant(value=0.0), trainable=True, name="delta_statespace_preactivation")
		self.dX_voxel_val = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=tf.math.log(dX_voxel_init)), trainable=True, name="dX_voxel_val")

		self.Xtest = Xtest # [Nxpoints,self.dim_in]
		self.Ytest = Ytest # [Nxpoints,1]

		assert self.Xtest.shape[0] == self.Ytest.shape[0]
		assert self.Xtest.shape[1] == self.dim_in
		assert self.Ytest.shape[1] == 1

		# Loss history for analysis:
		self.loss_vec = None

	def get_omegas_weights(self):
		# omegas_weights_withinlims = self.omega_lim*tf.keras.activations.tanh(self.omegas_weights)
		# omegas_weights_withinlims = self.omegas_weights
		# return omegas_weights_withinlims

		fac_per_dim = 1
		if self.dbg_flag:
			fac_per_dim = tf.reshape(1./tf.math.exp(self.log_L_for_regular_grid),(1,-1))

		return self.omegas_weights*fac_per_dim

	def get_delta_omegas(self):
		delta_omegas = tf.math.exp(self.Dw_voxel_val)*tf.keras.activations.sigmoid(self.delta_dw_voxels_pre_activation*tf.ones((self.Nomegas_for_regular_grid,1))) # Squeeze to (0,1)
		return delta_omegas

	def get_delta_statespace(self):
		delta_statespace = tf.math.exp(self.dX_voxel_val)*tf.keras.activations.sigmoid(self.delta_dX_voxels_preactivation) # Squeeze to (0,1)
		return delta_statespace

	def update_internal_spectral_density_parameters(self):
		omegapred = self.get_omegas_weights()
		dw_vec = self.get_delta_omegas()
		dX_vec = self.get_delta_statespace()
		self.inverse_fourier_toolbox.update_integration_parameters(	omega_locations=omegapred,
																	dw_voxel_vec=dw_vec,
																	dX_voxel_vec=dX_vec)
		Sw_vec = self.inverse_fourier_toolbox.spectral_values
		phiw_vec = self.inverse_fourier_toolbox.varphi_values
		self.inverse_fourier_toolbox.spectral_density.update_Wsamples_as(Sw_points=Sw_vec,phiw_points=phiw_vec,W_points=omegapred,dw_vec=dw_vec,dX_vec=dX_vec)
		return self.inverse_fourier_toolbox.spectral_density


	def reconstruct_function_at(self,xpred):
		self.inverse_fourier_toolbox.update_integration_parameters(	omega_locations=self.get_omegas_weights(),
																	dw_voxel_vec=self.get_delta_omegas(),
																	dX_voxel_vec=self.get_delta_statespace())
		fx_reconstructed = self.inverse_fourier_toolbox.get_fx_with_variable_integration_step(xpred) # [Npoints, 1]
		return fx_reconstructed

	def loss_reconstruction_fun(self,lengthscale_loss):

		fx_reconstructed = self.reconstruct_function_at(xpred=self.Xtest)
		loss_val = tf.reduce_mean(((self.Ytest - fx_reconstructed)/lengthscale_loss)**2,axis=0,keepdims=True) # [1, 1]

		return loss_val

	def train(self,Nepochs,learning_rate,stop_loss_val,lengthscale_loss,print_every=100):

		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

		str_banner = " << Training the delta omegas >> "

		logger.info("learning_rate = {0:f}".format(learning_rate))
		logger.info("stop_loss_val = {0:f}".format(stop_loss_val))
		logger.info("lengthscale_loss = {0:f}".format(lengthscale_loss))

		epoch = 0
		done = False
		loss_value_best = float("Inf")
		trainable_weights_best = self.get_weights()
		plotting_dict = dict(plotting=False)
		# print_every = 100
		self.loss_vec = np.zeros(Nepochs)
		time_elapsed_vec = np.zeros(Nepochs)
		while epoch < Nepochs and not done:

			time_start = time.time()

			if (epoch+1) % print_every == 0:
				logger.info("="*len(str_banner))
				logger.info(str_banner)
				logger.info(" << Epoch {0:d} / {1:d} >> ".format(epoch+1, Nepochs))
				logger.info("="*len(str_banner))

			with tf.GradientTape() as tape:
				loss_value = self.loss_reconstruction_fun(lengthscale_loss)

			grads = tape.gradient(loss_value, self.trainable_weights)
			optimizer.apply_gradients(zip(grads, self.trainable_weights))

			if (epoch+1) % print_every == 0:
				logger.info("    * Predictive loss (current): {0:.6f} [stopping loss: {1:.6f}]".format(tf.squeeze(loss_value),stop_loss_val))
				logger.info("    * Predictive loss (best):    {0:.6f} [stopping loss: {1:.6f}]".format(tf.squeeze(loss_value_best),stop_loss_val))
				# logger.info("    * Weights (current): {0:s}".format(self._weights2str(self.trainable_weights)))
				# logger.info("    * Weights (best): {0:s}".format(self._weights2str(trainable_weights_best)))
				# logger.info("    * Gradients (current): {0:s}".format(self._weights2str(grads)))

			if loss_value <= stop_loss_val:
				done = True
			
			if tf.squeeze(loss_value).numpy() < loss_value_best:
				loss_value_best = tf.squeeze(loss_value).numpy()
				trainable_weights_best = self.get_weights()
			

			# Register values:
			self.loss_vec[epoch] = tf.squeeze(loss_value)
			
			time_elapsed_vec[epoch] = time.time() - time_start

			epoch += 1

			# Report time:
			# time_per_epoch_avg = np.mean(time_elapsed_vec[0:epoch])
			time_per_epoch_avg = time_elapsed_vec[epoch-1] # Estimated remaining time based on the elapsed time of the last iteration
			if epoch % print_every == 0:
				logger.info("    * Elapsed time per epoch: {0:.2f} sec.".format(time_per_epoch_avg))
				remaining_time = time_per_epoch_avg*(Nepochs-epoch)
				if remaining_time > 60.:
					logger.info("    * Remaining time: {1:.2f} min.".format(Nepochs,remaining_time/60.))
				else:
					logger.info("    * Remaining time: {1:.2f} sec.".format(Nepochs,remaining_time))


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