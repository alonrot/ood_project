import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from lqrker.models.rrtp import RRTPRandomFourierFeatures
import numpy as np
import numpy.random as npr
import scipy
from scipy import linalg as la
from scipy import stats
import hydra
import pickle
from ood.models.robust_feedback_gain import RobustFeedbackGain
from ood.utils.math_utils import ensure_low_condition_number
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


"""
Install tensorflow-metal
https://towardsdatascience.com/accelerated-tensorflow-model-training-on-intel-mac-gpus-aa6ee691f894
https://developer.apple.com/metal/tensorflow-plugin/

conda create -n tf-metal python=3.8
conda activate tf-metal
SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos
SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-metal

conda deactivate
conda activate tf-metal



# Looking for installation of cuda 11:
https://github.com/tensorflow/tensorflow/issues/45930#issuecomment-770342299
https://www.tensorflow.org/install/gpu

# install cuda:
https://deeplearning.lipingyang.org/2017/01/18/install-gpu-tensorflow-ubuntu-16-04/
https://www.tensorflow.org/install/gpu


"""

def test():

	# https://www.tensorflow.org/api_docs/python/tf/debugging/set_log_device_placement
	tf.debugging.set_log_device_placement(True) # Turns logging for device placement decisions on or off

	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
	print("tf.config.list_physical_devices():",tf.config.list_physical_devices())

	# https://www.tensorflow.org/guide/gpu
	print("tf.config.list_physical_devices('GPU'):",tf.config.list_physical_devices('GPU'))


	Nels = 1000

	# Create some tensors
	with tf.device('/CPU:0'):
		a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
		b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
		c = tf.matmul(a, b)
		a = tf.random.normal(shape=(Nels,Nels), dtype=tf.float32)
		b = tf.nn.relu(a)

	print(b)

	a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
	c = tf.matmul(a, b)
	a = tf.random.normal(shape=(Nels,Nels), dtype=tf.float32)
	b = tf.nn.relu(a)

if __name__ == "__main__":

	test()

