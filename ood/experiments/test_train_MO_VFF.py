import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrblr import ReducedRankBayesianLinearRegression
# from lqrker.models.rrblr import RRTPQuadraticFeatures
import numpy as np
import numpy.random as npr
import scipy

from simple_loop_2dpend import simulate_single_pend

import gpflow
from gpflow.inducing_variables import InducingVariables
from gpflow.base import TensorLike
from gpflow.utilities import to_default_float
from gpflow import covariances as cov
from gpflow import kullback_leiblers as kl
from gpflow.ci_utils import ci_niter


BlockDiag = tf.linalg.LinearOperatorBlockDiag
Diag = tf.linalg.LinearOperatorDiag
LowRank = tf.linalg.LinearOperatorLowRankUpdate


def simulate_inverted_pendulum(Nsteps,x0=None):
	
	obs_vec = simulate_single_pend(Nsteps) # [Npoints,dim]
	obs_vec_tf = tf.convert_to_tensor(value=obs_vec,dtype=np.float32)

	Nskip = 1
	X = obs_vec_tf[0:-1:Nskip,:] # [Npoints,dim_x]
	Y = obs_vec_tf[1::Nskip,:] # [Npoints,dim_x]

	return X,Y

def test():
	
	pass

if __name__ == "__main__":

	test()


