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

from predictions_interface import Predictions


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

class FourierLayer(tf.keras.layers.Layer):
	
	# https://www.tensorflow.org/tutorials/customization/custom_layers
	
	def __init__(self, Nfeatures):
		assert Nfeatures % 2 == 0, "Requiring this for now"
		super(FourierLayer, self).__init__()
		self.Nfeatures = Nfeatures

	def build(self, input_shape):

		# regularizer_corr_noise_mat = tf.keras.regularizers.L1(l1=0.5) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
		# regularizer=regularizer_corr_noise_mat
		initializer_freqs = tf.keras.initializers.RandomUniform(minval=-5.0, maxval=+5.0)
		self.W = self.add_weight("freqs",shape=(self.Nfeatures//2,int(input_shape[-1])),initializer=initializer_freqs,trainable=True,name="freqs_mat")
		self.a = self.add_weight("amplis",shape=(self.Nfeatures//2,1),initializer=tf.keras.initializers.Constant(1.0),trainable=False)

	def call(self, inputs):
		"""
		inputs: [Npoints,dim]
		"""
		WX = tf.matmul(self.W,tf.transpose(inputs)) # [Nfeatures/2,Npoints]
		feat = tf.concat(values=(self.a * tf.math.cos(WX) , self.a * tf.math.sin(WX)),axis=0) # [Nfeatures,Npoints]
		return tf.transpose(feat) # [Npoints,Nfeatures]


class DeepKarhunenLoeve(tensorflow.keras.Model):

	# https://www.tensorflow.org/tutorials/quickstart/advanced

	def __init__(self):
		super(MyModel, self).__init__()
		self.embedding1 = tensorflow.keras.layers.Dense(units=128, activation='relu') # units is the output dimension
		self.embedding2 = tensorflow.keras.layers.LSTM(units=128, activation='tanh') # units is the output dimension
		self.fourier_layer = FourierLayer(Nfeatures=10) # units is the output dimension
		self.get_mean = tensorflow.keras.layers.Add() # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add

	def call(self, x):
		x = self.embedding1(x)
		x = self.embedding2(x)
		x = self.fourier_layer(x)
		x = self.get_mean(x)
		return x


@hydra.main(config_path="./config",config_name="config")
def main():

	# Train an MLP that maps from x to z
	# Fit a simple system, like a parabola

	xmin = -1.0
	xmax = +1.0
	epsi = 1e-3
	xpred = np.reshape(np.linspace(xmin+epsi,xmax-epsi,Npred),(-1,dim_in))

	ftrue_call = lambda xx: 1.1*xx**2
	ftrue = ftrue_call(xpred[:,0])

	ind_Xevals_sel_tot = [60,90,10,75,25,110,55]
	Nevals = len(ind_Xevals_sel_tot)

	Xevals = xpred[ind_Xevals_sel_tot,0:1]
	# pdb.set_trace()
	Yevals = np.reshape(np.interp(Xevals[:,0],xpred[:,0],ftrue),(-1,1))

	x_train = x_train[..., tf.newaxis].astype("float32")
	x_test = x_test[..., tf.newaxis].astype("float32")


	train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


	# Create an instance of the model
	# https://www.tensorflow.org/tutorials/keras/classification#train_the_model
	model = DeepKarhunenLoeve()
	# _ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.


	# model.fit(train_images, train_labels, epochs=10)
	# model.compile(optimizer='adam',
	#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	#               metrics=['accuracy'])

if __name__ == "__main__":

	main()