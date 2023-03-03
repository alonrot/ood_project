# Implementation of ESS

import numpy as np
# from ProbabilisticModels import *
# from utils.PlotBayes import PlotSamples
import tensorflow as tf
import pdb
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class EllipticalSliceSampler:
	"""
	

	1) Have a burning period
	2) Re-do the optimization for random restarts. Collect the random restarts by sampling uniformly ~1000 samples and
	then selecting the 10 best.

	"""

	def __init__(self,dim_in,target_log_lik,Nsamples,Nburning,Nrestarts,omega_lim_random_restarts,mean_w,var_w,kwargs_to_fun):

		logger.info("\nConstruct class EllipticalSliceSampler")
		logger.info("====================")

		self.dim_in = dim_in
		# Prior:

		# Construct the function:
		self.target_fun_log_lik = target_log_lik

		self.Nburning = Nburning
		self.Nsamples = Nsamples
		self.Nrestarts = Nrestarts
		assert Nsamples % Nrestarts == 0
		self.Nsamples_per_restart = Nsamples // Nrestarts
		self.omega_lim_random_restarts = omega_lim_random_restarts

		assert mean_w.shape[0] == self.dim_in
		assert var_w.shape[0] == self.dim_in
		self.mean_w = mean_w
		self.Sigma_w = np.diag(var_w)

		# self.v = np.random.multivariate_normal(self.mean_w,self.Sigma_w)

		self.kwargs_to_fun = kwargs_to_fun

	def run_ess(self):
		"""

		return:
		w_out: [self.Nrestarts*self.Nsamples,self.dim_in]
		"""

		omega0_restarts = self._get_random_restarts() # [Nrestarts,self.dim_in]
		w_after_burning = np.zeros((self.Nrestarts,self.Nsamples_per_restart,self.dim_in))
		w_after_burning[:,0,:] = omega0_restarts

		logger.info("Running Elliptical Slice Sampler!")
		logger.info("=============")
		logger.info("Number of samples: {0:d}".format(self.Nsamples))
		logger.info("Number of samples per restart: {0:d}".format(self.Nsamples_per_restart))
		logger.info("Number of restarts: {0:d}".format(self.Nrestarts))
		logger.info("Number of burning steps: {0:d}".format(self.Nburning))

		
		for ii in range(self.Nrestarts):

			logger.info("Restart {0:d} / {1:d}".format(ii+1,self.Nrestarts))

			w_cur = w_after_burning[ii,0,:]

			for jj in range(self.Nburning+self.Nsamples_per_restart-1):

				logger.info("Getting sample {0:d} / {1:d} ..".format(jj+1,self.Nburning+self.Nsamples_per_restart))

				# Generate a candidate by sampling from the current one: w_cand ~ N(w_cur,self.Cov_trans)
				w_new = self._get_sample_from_elliptical_slice(w_cur)

				# Store after burning:
				if jj >= self.Nburning:
					cc = jj - self.Nburning
					w_after_burning[ii,cc+1,:] = w_new
				
				# Update:
				# w_cur = np.copy(w_new)
				w_cur = w_new

		w_out = np.reshape(w_after_burning,(-1,self.dim_in))

		return w_out, omega0_restarts

	def _get_random_restarts(self):

		# Get more samples, then subselect:
		Nrestarts_enlarged = self.Nrestarts * 10

		# Sample uniformly:
		# omega_min = -self.omega_lim_random_restarts
		# omega_max = self.omega_lim_random_restarts
		omega_min = self.omega_lim_random_restarts[0]
		omega_max = self.omega_lim_random_restarts[1]
		omega0_restarts = omega_min + (omega_max - omega_min)*tf.math.sobol_sample(dim=self.dim_in,
																			num_results=(Nrestarts_enlarged),
																			skip=1000 + 10*np.random.randint(100))
		omega0_restarts = omega0_restarts.numpy()

		# Subselect:
		omega0_restarts = self._get_best_samples(omega0_restarts,self.Nrestarts)
		
		return omega0_restarts

	def _get_best_samples(self,omega_candidates,Nbest):
		"""
		
		omega_candidates: [Nomegas,self.dim_in]

		"""

		log_lik_vals = self.target_fun_log_lik(omega_candidates,*self.kwargs_to_fun) # [Nomegas,]
		ind_sorted = np.argsort(log_lik_vals)[::-1] # returns indices; the [::-1] reverses the vector. Hence, ind_sorted will sort the values in decreasing order
		w_sorted = omega_candidates[ind_sorted[0:Nbest],:] # [Nbest,self.dim_in]
		
		return w_sorted


	def _get_sample_from_elliptical_slice(self,w_cur):
		"""
		
		Target: maximize the likleihood


		"""

		# Choose one ellipse:
		v = np.random.multivariate_normal(self.mean_w,self.Sigma_w)

		# Log likelihood threshold:
		logy = self.target_fun_log_lik(w_cur,*self.kwargs_to_fun) + np.log(np.random.uniform())

		# Draw an initial \theta proposal, and define a bracket:
		theta = np.random.uniform(low=0,high=2*np.pi)
		theta_min = theta - 2*np.pi
		theta_max = theta

		while True:

			# Candidate, assuming non-zero mean prior:
			w_cand = (w_cur-self.mean_w)*np.cos(theta) + (v-self.mean_w)*np.sin(theta) + self.mean_w

			# Evaluate the candidate in the likelihood:
			lik_w_cand = self.target_fun_log_lik(w_cand,*self.kwargs_to_fun)

			# Compress the interval, or return candidate:
			if lik_w_cand > logy:
				# logger.info("theta: {0:f}"theta)
				# logger.info("logy:",logy)
				return w_cand
			else:
				if theta < 0:
					theta_min = theta
				elif theta > 0:
					theta_max = theta
				else:
					raise Exception("Interval reduced to zero!")

			# Sample a new theta:
			theta = np.random.uniform(low=theta_min,high=theta_max)


