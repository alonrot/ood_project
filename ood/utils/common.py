from abc import ABC, abstractmethod
import tensorflow as tf
import math
import pdb
import numpy as np

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class CommonUtils():

	def __init__(self):
		pass

	@staticmethod
	def create_Ndim_grid(xmin,xmax,Ndiv,dim):
		"""
		
		Create a regular grid on a hypercube [xmin,xmax]**dim
		and "vectorize" it as a matrix with Ndiv**dim rows and dim columns,
		such that each point can be accessed as vectorized_grid[i,:]

		return:
			vectorized_grid: [Ndiv**dim,dim]

		"""

		xpred = tf.linspace(xmin,xmax,Ndiv)
		xpred_data = tf.meshgrid(*([xpred]*dim),indexing="ij")
		vectorized_grid = tf.concat([tf.reshape(xpred_data_el,(-1,1)) for xpred_data_el in xpred_data],axis=1)

		return vectorized_grid

	@staticmethod
	def fix_eigvals(Kmat):
		"""

		Among the negative eigenvalues, get the 'most negative one'
		and return it with flipped sign
		"""

		Kmat_sol = tf.linalg.cholesky(Kmat)
		# Kmat_sym = 0.5*(Kmat + tf.transpose(Kmat))
		# Kmat_sol = tf.linalg.cholesky(Kmat_sym)
		if tf.math.reduce_any(tf.math.is_nan(Kmat_sol)):
			logger.info("Kmat needs to be fixed...")
		else:
			logger.info("Kmat is PD; nothing to fix...")
			return Kmat

		try:
			eigvals, eigvect = tf.linalg.eigh(Kmat)
			# eigvals, eigvect = tf.linalg.eigh(Kmat_sym)
		except Exception as inst:
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("Failed to compute tf.linalg.eigh(Kmat) ...")
			pdb.set_trace()

		max_eigval = tf.reduce_max(tf.math.real(eigvals))
		min_eigval = tf.reduce_min(tf.math.real(eigvals))

		# Compte eps:
		# eps must be such that the condition number of the resulting matrix is not too large
		max_order_eigval = tf.math.ceil(tf.experimental.numpy.log10(max_eigval))
		eps = 10**(max_order_eigval-8) # We set a maximum condition number of 8

		# Fix eigenvalues:
		eigvals_fixed = eigvals + tf.abs(min_eigval) + eps

		# pdb.set_trace()
		logger.info(" Fixed by adding " + str(tf.abs(min_eigval).numpy()))
		logger.info(" and also by adding " + str(eps.numpy()))

		Kmat_fixed = eigvect @ ( tf.linalg.diag(eigvals_fixed) @ tf.transpose(eigvect) ) # tf.transpose(eigvect) is the same as tf.linalg.inv(eigvect) | checked

		# Kmat_fixed_sym = 0.5*(Kmat_fixed + tf.transpose(Kmat_fixed))

		try:
			tf.linalg.cholesky(Kmat_fixed)
		except:
			pdb.set_trace()

		# pdb.set_trace()

		return Kmat_fixed
