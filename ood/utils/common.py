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
