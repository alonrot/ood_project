import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from matplotlib import cm
# from lqrker.models.rrblr import ReducedRankBayesianLinearRegression
from lqrker.models.rrtp import RRTPRandomFourierFeatures
from lqrker.utils.spectral_densities import MaternSpectralDensity
import numpy as np
import numpy.random as npr
import scipy
from scipy import stats
import hydra
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

ZERO_NUM = 1e-10


fontsize = 17

"""
Doesn't make any sense: We're mapping from x to g(x), where g(x)
is even more non-linear than x, and yet we expect to kind of fit a
Gaussian process on g(x)... So, what's the point of using observables
if we can't know a priori whether we'll actually obtain a linear system
in g(x) ? Or in other words, if g(x) is more non-linear than x, why not
just fitting a GP directly for x ???

So, with Koopman theory we can only expect to encode the system's behavior
in a linear-system representation. But we know that linear systems can behave
in 3 ways: (a) stable (matrix eigenvalues within the unit circle ), (b) marginally
stable, and (c) unstable. So, the underlying system can only behave in any of
these three categories. A free-falling pendulum has marginally
stable dynamics; thus, the underlying Koopman system will be marginally stable, 
i.e., matrix eigenvalues close to 1, even though it's actually linear.
So, long-story short, if our target is to design a policy for an unknown 
underlying non-linear system f(x), maybe we should directly
incldue the control input in the design of the Koopman gain K.
How far can we stretch this Koopman idea? Can we use it for predicting
long-term trajectories, or is Koopman actually restricted to work only
for MPC?

Can we actually use Koopman theory for learning the swing-up dynamics of an inverted pendulum?
A swingup passes through 2 equilibrium points; I would expect each equilibirum point
to correspond to a different Koopman matrix K. So, what do we do, then? How
do we learn 2 Koopman gainsr instead of 1?


---

(1) Connection with Koopman eigenvalues, eigenfunctions, modes
(2) Connection with KL expansion
(3) Connection with kernels
(4) How to do long-term predictions


"""

def simulate_inverted_pendulum(x0,Nsteps,deltaT,sigma_n):

	x_traj = np.zeros((x0.shape[0],Nsteps),dtype=np.float32)

	a = 1.0

	x_traj[:,0] = x0[:,0]
	for k in range(1,Nsteps):

		# pdb.set_trace()
		x_traj[:,k] = x_traj[:,k-1] + deltaT * np.array([x_traj[1,k-1] , a*np.sin(x_traj[0,k-1])])

		# Add noise:
		x_traj[:,k] += sigma_n*np.random.randn(2)

	return tf.transpose(x_traj) # [Nsteps,dim]


# def convert_trajectory_to_observables(x_traj,Nfeat):

# 	dim = x_traj.shape[0]

# 	L = 7.0

# 	rrblr = ReducedRankBayesianLinearRegression(dim=dim,Nfeat=Nfeat,L=L,sigma_n=0.0)
# 	# NOTE: The noise parameter sigma_n doesn't play a role in getting the features, so
# 	# we just pass a dummy value.


# 	x_traj_tf = tf.constant(x_traj.T,dtype=tf.float32)
# 	# pdb.set_trace()
# 	z_traj_tf = rrblr.get_features_mat(x_traj_tf) # in: [Npoints, dim]; out: [Npoints, Nfeat]

# 	z_traj = tf.cast(z_traj_tf,dtype=tf.float32).numpy()

# 	# Replace first dim elements with the trajectory itself:

# 	return z_traj.T # [Nfeat,Npoints]


def sample_eigvals_with_rejection(K_koop_mean,Sigma_inv_list):
	"""

	Assumptions:
	1) All rows of the koopman matrix follow the same covariance (not true in general)
	2) K_koop_mean has real eigenvalues
	"""

	Sigma_inv = Sigma_inv_list[0] # DEBUG/TODO: This doesn't hold true in general

	# Create the distribution using the eigenvectors of the mean:
	eigvals_mean, eigvect_mean = np.linalg.eig(K_koop_mean)






def unnormalized_log_density_eigvals_fix_covariance(eigvals,K_koop_mean,Sigma_inv):
	"""

	Assumptions:
	1) All rows of the koopman matrix follow the same covariance Sigma_inv (not true in general)
	2) K_koop_mean has real eigenvalues

	eigvals: [Npoints,dim]
	K_koop_mean: [dim,dim]
	Sigma_inv: [dim,dim]

	out: [Npoints,1]

	"""

	# Analysis on the eigenvalues distribution:
	eigvals_mean, eigvect_mean = tf.linalg.eig(tf.constant(K_koop_mean,dtype=tf.float32))

	pdb.set_trace()

	# Sort them out:
	eigvals_sorted = tf.sort(eigvals,direction="ASCENDING",axis=1) # Sort the eigenvalues in ascending order, for each instance

	eigvals_sorted_diffs_abs = tf.math.abs(eigvals_sorted[:,1::] - eigvals_sorted[:,0:-1])

	eigvals_sorted_diffs_abs_clipped = tf.clip_by_value(eigvals_sorted_diffs_abs,clip_value_min=ZERO_NUM,clip_value_max=np.inf)

	sum_log_differences = tf.reduce_sum(tf.math.log(eigvals_sorted_diffs_abs_clipped),axis=1)

	eigvect_mean_inv = tf.linalg.inv(eigvect_mean)

	f1 = tf.reduce_sum(tf.linalg.diag_part(eigvect_mean_inv @ tf.complex(Sigma_inv,0.0) @ eigvect_mean) * eigvals_sorted**2, axis=1)
	f2 = tf.reduce_sum(tf.linalg.diag_part(eigvect_mean_inv @ tf.complex(Sigma_inv,0.0) @ K_koop_mean @ eigvect_mean) * eigvals_sorted, axis=1)
	
	log_density_val = sum_log_differences - 0.5*( f1 + f2 )

	return log_density_val


def sample_eigenvalues_and_plot(th_mean,th_chol,dim_z,block=False):
	"""

	out: [Npoints,1]

	# Assumption: for each Koopman matrix sample K_l, the eigen-decomposition provides a deterministic result...

	"""

	Nsamples = 1000

	# eigvals_mean, eigvect_mean = tf.linalg.eig(tf.constant(K_koop_mean,dtype=tf.float32))

	noise_vec = np.ndarray.astype(np.random.randn(Nsamples,dim_z**2),dtype=np.float32)
	th_sample = np.reshape(th_mean,(1,-1)) + noise_vec @ th_chol # [Nsamples,dim_z**2]
	K_koop_sample = np.reshape(th_sample,(Nsamples,dim_z,dim_z),order="C") # With C, the elements are read with the last dimension changing the fastest. Then, the elements are placed in the destination matrix by filling the last dimensions faster (i.e., will fill the matrix by rows). This is exactly what we want, according to the math.

	# # Square the matrix and see the effect:
	# K_koop_sample_mod = np.zeros(K_koop_sample.shape,dtype=np.complex64)
	# for ii in range(K_koop_sample.shape[0]):
	# 	K_koop_sample_mod[ii,:,:] = K_koop_sample[ii,:,:] @ K_koop_sample[ii,:,:].T
	# K_koop_sample = K_koop_sample_mod

	# K_koop_sample_eigvals = np.linalg.eigvals(np.linalg.matrix_power(K_koop_sample,4))
	# VV, K_koop_sample_eigvals, VVH = np.linalg.svd(K_koop_sample)
	K_koop_sample_eigvals = np.linalg.eigvals(K_koop_sample)
	# K_koop_sample_eigvals = np.sqrt(K_koop_sample_eigvals)

	# pdb.set_trace()

	K_koop_sample_eigvals_real = np.real(K_koop_sample_eigvals)
	K_koop_sample_eigvals_imag = np.imag(K_koop_sample_eigvals)
	K_koop_sample_eigvals_abs = abs(K_koop_sample_eigvals)

	K_koop_sample_eigvals_abs_max = np.amax(K_koop_sample_eigvals_abs,axis=1)

	# pdb.set_trace()

	# pdb.set_trace()
	viridis = cm.get_cmap('viridis', Nsamples)
	newcolors = viridis(np.linspace(0, 1, Nsamples))[:,0:3]

	# K_koop_sample_eigvals_real = np.reshape(K_koop_sample_eigvals_real,(-1))
	# K_koop_sample_eigvals_imag = np.reshape(K_koop_sample_eigvals_imag,(-1))

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8))
	hdl_fig.suptitle("Eigenvalues from sampled Koopman matrices")

	for k in range(Nsamples):
		hdl_splots[0].plot(K_koop_sample_eigvals_real[k,:],K_koop_sample_eigvals_imag[k,:],linestyle="None",marker=".",color=newcolors[k,:])

	# hdl_splots[0].plot(K_koop_sample_eigvals_real,K_koop_sample_eigvals_imag,linestyle="None",marker=".",color="k")
	hdl_circle = plt.Circle( (0.0, 0.0 ), 1.0 , fill = False )
	hdl_splots[0].add_artist( hdl_circle )
	# hdl_splots[0].set_aspect('equal', adjustable='box')

	hdl_splots[0].set_xlim([-2,2])
	hdl_splots[0].set_ylim([-2,2])

	# K_koop_sample_eigvals_abs = K_koop_sample_eigvals_abs[K_koop_sample_eigvals_abs < 300.0]
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8))
	# K_koop_sample_eigvals_abs = K_koop_sample_eigvals_abs[K_koop_sample_eigvals_abs < 20]
	# hdl_splots.hist(K_koop_sample_eigvals_abs,density=True,bins=15)
	# hdl_splots.hist(K_koop_sample_eigvals_abs_max,density=True,bins=15)
	hdl_splots.hist(K_koop_sample_eigvals_abs_max,density=True)


	# Another way of sampling: Take 
	K_mean = np.reshape(th_mean,(dim_z,dim_z),order="C")
	K_mean_eigvals, K_mean_eigvects = np.linalg.eig(K_mean) # K_mean = K_mean_eigvects @ np.diag(K_mean_eigvals) @ 
	

	eigvals_fixed_eigvects_diag_sample = np.linalg.inv(K_mean_eigvects) @ K_koop_sample @ K_mean_eigvects
	eigvals_fixed_eigvects_diag_part_sample = np.zeros(K_koop_sample_eigvals.shape,dtype=np.complex64)
	for k in range(Nsamples):
		eigvals_fixed_eigvects_diag_part_sample[k,:] = np.linalg.eigvals(eigvals_fixed_eigvects_diag_sample[k,:,:])
		# pdb.set_trace()

	eigvals_fixed_eigvects_diag_part_sample_real = np.real(eigvals_fixed_eigvects_diag_part_sample)
	eigvals_fixed_eigvects_diag_part_sample_imag = np.imag(eigvals_fixed_eigvects_diag_part_sample)
	eigvals_fixed_eigvects_diag_part_sample_abs = abs(eigvals_fixed_eigvects_diag_part_sample)

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Eigenvalues from sampled Koopman matrices with fixed V from mean")

	for k in range(Nsamples):
		hdl_splots[0].plot(eigvals_fixed_eigvects_diag_part_sample_real[k,:],eigvals_fixed_eigvects_diag_part_sample_imag[k,:],linestyle="None",marker=".",color=newcolors[k,:])

	# hdl_splots[0].plot(K_koop_sample_eigvals_real,K_koop_sample_eigvals_imag,linestyle="None",marker=".",color="k")
	hdl_circle = plt.Circle( (0.0, 0.0 ), 1.0 , fill = False )
	hdl_splots[0].add_artist( hdl_circle )
	# hdl_splots[0].set_aspect('equal', adjustable='box')

	hdl_splots[0].set_xlim([-2,2])
	hdl_splots[0].set_ylim([-2,2])

	hdl_splots[1].hist(eigvals_fixed_eigvects_diag_part_sample_abs,density=True)

	# Fit a Gaussian on top:
	eigvals_fixed_eigvects_diag_part_sample_abs_mean = np.mean(eigvals_fixed_eigvects_diag_part_sample_abs)
	eigvals_fixed_eigvects_diag_part_sample_abs_std = np.std(eigvals_fixed_eigvects_diag_part_sample_abs)

	eigvals_fixed_eigvects_diag_part_sample_abs_min = np.amin(eigvals_fixed_eigvects_diag_part_sample_abs)
	eigvals_fixed_eigvects_diag_part_sample_abs_max = np.amax(eigvals_fixed_eigvects_diag_part_sample_abs)
	eigvals_fixed_eigvects_diag_part_sample_abs_xpred = np.linspace(eigvals_fixed_eigvects_diag_part_sample_abs_min,eigvals_fixed_eigvects_diag_part_sample_abs_max,101)

	eigvals_fixed_eigvects_diag_part_sample_abs_pdf = stats.norm.pdf(	eigvals_fixed_eigvects_diag_part_sample_abs_xpred,
																		loc=eigvals_fixed_eigvects_diag_part_sample_abs_mean,
																		scale=eigvals_fixed_eigvects_diag_part_sample_abs_std)

	hdl_splots[1].plot(eigvals_fixed_eigvects_diag_part_sample_abs_xpred,eigvals_fixed_eigvects_diag_part_sample_abs_pdf,linestyle="-",color="r")

	print("K_mean_eigvects:",K_mean_eigvects[0:5,0:5])


	# pdb.set_trace()
	plt.show(block=block)

	# # TODO: Do moment matching with the Gaussian ensemble; it's a distribution we propose... The only problem is how to deal with complex eigenvalues...
	# K_koop_sample_eigvals_mean = tf.reduce_mean(K_koop_sample_eigvals,axis=1)

	# K_koop_mean = np.reshape(th_mean,(dim_z,dim_z),order="C") # Order is C because we need to (1) read the elements and (2) place them in rows



def project_eigvals_inside_unit_circle(K_koop_sample):

	K_koop_sample_eigvals, K_koop_sample_eigvect = np.linalg.eig(K_koop_sample)

	# assert not np.all(abs(K_koop_sample_eigvals)) >= 1.0, "Check this..."

	# raise NotImplementedError("We must also modify the eigenvectors, when an eigenvalue gets rescaled... no?")

	if np.any(abs(K_koop_sample_eigvals)) >= 1.0:
		logger.info("Koopman matrix has eigenvalues outside the unit circle")
		logger.info("Fixing...")

		ind_unstable_eigvals = abs(K_koop_sample_eigvals) >= 1.0
		K_koop_sample_eigvals_fixed = K_koop_sample_eigvals
		K_koop_sample_eigvals_fixed[ind_unstable_eigvals] = K_koop_sample_eigvals[ind_unstable_eigvals] / abs(K_koop_sample_eigvals[ind_unstable_eigvals]) * 0.99999
		K_koop_sample_ret = K_koop_sample_eigvect @ np.diag(K_koop_sample_eigvals_fixed) @ np.linalg.inv(K_koop_sample_eigvect)

	else:
		K_koop_sample_ret = K_koop_sample

	return K_koop_sample_ret


@hydra.main(config_path=".",config_name="config.yaml")
def test(cfg: dict) -> None:


	"""
	1) Test RRTP with Random Fourier Features and data-driven specrtal density. Start with Matern, though.
	"""

	seed = 2
	tf.random.set_seed(seed=seed)
	np.random.seed(seed=seed)

	dim_x = 2

	# Hypercube domain:
	sigma_n_x = 0.01 # Process noise of the underlying dynamics x_{k+1} = f(x_k)
	sigma_n_z = 0.001 # Process noise of the underlying dynamics x_{k+1} = f(x_k)

	# Get dataset:
	Nevals = 1000
	Mtrajs = 10
	Nskip = 1
	deltaT = 0.01
	x0 = np.array([[np.pi/4],[0]],dtype=np.float32)
	x_traj = simulate_inverted_pendulum(x0,Nevals+1,deltaT,sigma_n_x) # [Nevals+1,dim_x]

	# Convert trajectory to feature space. We use random fourier features with Matern spectral density:
	spectral_density = MaternSpectralDensity(cfg=cfg.RRTPRandomFourierFeatures.spectral_density_pars, dim=dim_x)
	rrtp_rff = RRTPRandomFourierFeatures(dim=dim_x,cfg=cfg.RRTPRandomFourierFeatures,spectral_density=spectral_density)
	rrtp_rff.update_spectral_density(None,None)
	z_traj_feats = rrtp_rff.get_features_mat(x_traj) # [,Nfeat]

	# Concatenate with the state itself and a constant:
	z_poly_traj = tf.concat( [tf.ones((x_traj.shape[0],1)) , x_traj],axis=1)
	# z_exp_traj = tf.reshape(tf.math.exp(-0.5*tf.math.reduce_euclidean_norm(x_traj,axis=1)),(-1,1))
	# z_traj = tf.concat( [z_poly_traj , z_exp_traj, z_traj_feats], axis=1 )
	z_traj = tf.concat( [z_poly_traj , z_traj_feats], axis=1 )

	# Observables data:
	X = z_traj[0:-1:Nskip,:] # [Nevals,Nfeat]
	Y = z_traj[1::Nskip,:] # [Nevals,Nfeat]

	Nevals = X.shape[0]
	assert Nevals == Y.shape[0]

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Inverted pendulum simulation x(t)")
	hdl_splots[0].plot(np.arange(0,Nevals+1,1),x_traj[:,0],linestyle="None",marker=".",color="k")
	hdl_splots[1].plot(np.arange(0,Nevals+1,1),x_traj[:,1],linestyle="None",marker=".",color="k")

	# Selected indices for observables:
	Nfeat = z_traj.shape[1]
	dim_z = Nfeat
	ind_1 = 1
	ind_2 = Nfeat-1
	assert ind_1 < Nfeat
	assert ind_2 < Nfeat

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Inverted pendulum simulation, transformed in to z(t)")
	hdl_splots[0].plot(np.arange(0,Nevals+1,1),z_traj[:,ind_1],linestyle="None",marker=".",color="k")
	hdl_splots[1].plot(np.arange(0,Nevals+1,1),z_traj[:,ind_2],linestyle="None",marker=".",color="k")

	# plt.show(block=True)
	# import sys
	# sys.exit(0)

	# Set up Bayesian linear model:
	# z_{k+1,i} = \sum_{j=1}^dim_z w_ij z_{k,j}
	# lambda_ij = 5.0*npr.rand(dim_z,dim_z) # Need to be positive because w_ij ~ N(0,lambda_ij)
	lambda_ij = 10.0*np.ones((dim_z,dim_z)) # DO NOT CHANGE. CHANGING THIS IMPLIES CHANGING THE ASSUMPTION THAT THE COVARIANCE MATRIX IS THE SAME FOR ALL ROWS OF K
	assert np.all(lambda_ij == 10.0), "HAVING VALUES DIFFER ROWWISE IMPLIES CHANGING THE ASSUMPTION THAT THE COVARIANCE MATRIX IS THE SAME FOR ALL ROWS OF K. This will affect the function sample_eigvals_with_rejection()"

	XXT = tf.transpose(X) @ X
	Ainv_times_noise_list = [None]*dim_z
	Ainv_times_noise_list_fixed = [None]*dim_z
	Sigma_inv_list = [None]*dim_z
	th_mean_list = [None]*dim_z
	th_cov_chol_list = [None]*dim_z
	for ii in range(dim_z):
		
		Ainv_times_noise_list[ii] = sigma_n_z**2 * np.linalg.inv( XXT + sigma_n_z**2*np.diag(1./lambda_ij[ii,:]) )
		Sigma_inv_list[ii] = (1./sigma_n_z**2) * XXT + np.diag(1./lambda_ij[ii,:])

		# Compute mean:
		Yproj = tf.reshape(Y[:,ii],(-1,1)) # [Npoints,1]
		th_mean_list[ii] = ((1./sigma_n_z**2) * Ainv_times_noise_list[ii]) @ tf.transpose(X) @ Yproj

		# Compute cholesky decomposition of the covariance:
		try:
			th_cov_chol_list[ii] = np.linalg.cholesky(Ainv_times_noise_list[ii])
		except Exception as inst:
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("Failed to compute: chol( Ainv_times_noise_list[ii] ) ...")
			Ainv_times_noise_list_fixed[ii] = RRTPRandomFourierFeatures.fix_eigvals(Ainv_times_noise_list[ii])
		else:
			Ainv_times_noise_list_fixed[ii] = Ainv_times_noise_list[ii]
		th_cov_chol_list[ii] = np.linalg.cholesky(Ainv_times_noise_list_fixed[ii])

	th_mean = np.concatenate(th_mean_list,axis=0)
	th_chol = scipy.linalg.block_diag(*th_cov_chol_list)

	sample_eigenvalues_and_plot(th_mean,th_chol,dim_z,block=False)

	# pdb.set_trace()

	# Mean transition matrix (Koopman operator approximation):
	# pdb.set_trace()
	K_koop_mean = np.reshape(th_mean,(dim_z,dim_z),order="C") # Order is C because we need to (1) read the elements and (2) place them in rows

	print("K_koop_mean:",K_koop_mean[0:10,0:10])
	eigvals_K_koop = np.linalg.eigvals(K_koop_mean)
	print("eigvals_K_koop_mean:",eigvals_K_koop[0:10])
	print("abs(eigvals_K_koop_mean):",abs(eigvals_K_koop[0:10]))
	th_chol_std_diag = np.diag(th_chol)
	print("th_chol_std_diag:",th_chol_std_diag[0:10])

	# pdb.set_trace()

	# Assess prediction quality:
	xpred = X # debug
	mean_xpred_feat = np.zeros(X.shape)
	std_xpred_feat = np.zeros(X.shape)
	for ii in range(dim_z):
		mean_xpred_feat[:,ii:ii+1] = X @ th_mean_list[ii]
		cov_xpred_feat  = X @ Ainv_times_noise_list[ii] @ tf.transpose(X)
		std_xpred_feat[:,ii] = tf.sqrt(tf.linalg.diag_part(cov_xpred_feat))

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Predictions on the training set using the Koopman random gain")
	c = 0
	for kk in [ind_1,ind_2]:

		hdl_splots[c].plot(np.arange(0,Nevals,1),X[:,kk],linestyle="None",marker=".",color="k")
		hdl_splots[c].plot(np.arange(0,Nevals,1),mean_xpred_feat[:,kk],linestyle="-",lw=1,color="blue")
		hdl_splots[c].plot(np.arange(0,Nevals,1),mean_xpred_feat[:,kk] + 2.*std_xpred_feat[:,kk],linestyle="-",lw=1,color="blue")
		hdl_splots[c].plot(np.arange(0,Nevals,1),mean_xpred_feat[:,kk] - 2.*std_xpred_feat[:,kk],linestyle="-",lw=1,color="blue")
		hdl_splots[c].set_ylabel("g_"+str(kk+1)+"(x_t)")

		c += 1

	hdl_splots[1].set_xlabel("time")


	"""
	TODO:
	1) Each column/row has a different covariance matrix
	2) Rejection sampling?
	"""

	# pdb.set_trace()

	eigvals = tf.random.uniform(shape=(90,K_koop_mean.shape[0]))
	Sigma_inv = Sigma_inv_list[0]
	# unnormalized_log_density_eigvals_fix_covariance(eigvals,K_koop_mean,Sigma_inv)


	# Simulate trajectories for different realizations of K starting from z0
	z_roll_Mtrajs = np.zeros((dim_z,Nevals,Mtrajs),dtype=np.float32)
	z0 = np.reshape(z_traj[0,:],(-1,1))
	for m in range(Mtrajs):
		
		th_sample = th_mean + th_chol @ np.random.randn(dim_z**2,1)
		K_koop_sample = np.reshape(th_sample,(dim_z,dim_z),order="C") # Order is C because we need to (1) read the elements and (2) place them in rows
		K_koop_sample = project_eigvals_inside_unit_circle(K_koop_sample)

		# if np.any(abs(K_koop_sample_eigvals) >= 1.0):
		# 	print("Koopman matrix set to zero for trajectory "+str(m+1))
		# 	print("abs(K_koop_sample_eigvals):",abs(K_koop_sample_eigvals))
		# 	K_koop_sample = np.zeros((dim_z,dim_z))
		# 	# eigvals, eigvect = np.linalg.eig(K_koop_sample)
		# 	# K_koop_sample = eigvect @ np.diag(eigvals) @ eigvect.T
		# 	# eigvect_inv = np.linalg.inv(eigvect)
		# 	# vvv = eigvect @ np.diag(eigvals) @ eigvect_inv


		# Roll-out the dynamics for one trajectory, starting at the same initial condition:
		z_roll = np.zeros((dim_z,Nevals),dtype=np.float32)
		z_roll[:,0] = z0[:,0]

		for k in range(Nevals-1):
			z_roll_curr = np.reshape(z_roll[:,k],(-1,1))
			z_roll_next = K_koop_sample @ z_roll_curr + sigma_n_z*npr.randn(dim_z,1)
			z_roll[:,k+1] = z_roll_next[:,0]

		# pdb.set_trace()

		z_roll_Mtrajs[:,:,m] = z_roll


	# pdb.set_trace()

	# Obtain eDMD solution:
	z_roll_eDMD = np.zeros((dim_z,Nevals),dtype=np.float32)
	z_roll_eDMD[:,0] = z0[:,0]
	# pdb.set_trace()
	K_koop_eDMD = np.linalg.solve(tf.transpose(X) @ X, tf.transpose(X) @ z_traj[1::Nskip,:])
	# K_koop_eDMD_eigvals = np.linalg.eigvals(K_koop_eDMD)
	K_koop_eDMD_eigvals, K_koop_eDMD_eigvect = np.linalg.eig(K_koop_eDMD)
	# if np.any(abs(K_koop_eDMD_eigvals) >= 1.0):
	# 	print("EDMD Koopman matrix set to zero")
	# 	K_koop_eDMD = np.zeros((dim_z,dim_z))

	ind_stable = abs(K_koop_eDMD_eigvals) < 1.0
	assert np.any(ind_stable), "OMG"
	dim_z_proj = np.sum(ind_stable)

	assert np.sum(ind_stable) >= 2
	ind_stable_sel = np.arange(dim_z)[ind_stable]


	K_koop_eDMD_eigvals_diag = np.diag(K_koop_eDMD_eigvals[ind_stable])

	z_roll_eDMD_proj = np.zeros((np.sum(ind_stable),Nevals),dtype=np.complex64)
	K_koop_eDMD_eigvect_inv = np.linalg.inv(K_koop_eDMD_eigvect)
	z_roll_eDMD_proj_z0 = K_koop_eDMD_eigvect_inv @ z0[:,0:1]
	z_roll_eDMD_proj[:,0] = z_roll_eDMD_proj_z0[ind_stable,0]

	for k in range(Nevals-1):
		
		# Stable projection:
		z_roll_proj_curr = z_roll_eDMD_proj[:,k:k+1]
		z_roll_proj_next = K_koop_eDMD_eigvals_diag @ z_roll_proj_curr + sigma_n_z*npr.randn(dim_z_proj,1)
		z_roll_eDMD_proj[:,k+1] = z_roll_proj_next[:,0]

		# Whole vector:
		z_roll_curr = z_roll_eDMD[:,k:k+1]
		z_roll_next = K_koop_eDMD @ z_roll_curr + sigma_n_z*npr.randn(dim_z,1)
		z_roll_eDMD[:,k+1] = z_roll_next[:,0]

	# pdb.set_trace()

	# z_roll_eDMD_proj_back = 


	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	hdl_fig.suptitle("Approximate Koopman operator for a linear system with Bayesian linear models")
	for m in range(Mtrajs):
		hdl_splots[0].plot(np.arange(0,Nevals,1),z_roll_Mtrajs[ind_1,:,m],linestyle="-",lw=1,color="lightblue")
		hdl_splots[1].plot(np.arange(0,Nevals,1),z_roll_Mtrajs[ind_2,:,m],linestyle="-",lw=1,color="lightblue")

	hdl_splots[0].plot(np.arange(0,Nevals,1),X[:,ind_1],linestyle="None",marker=".",color="k")
	hdl_splots[1].plot(np.arange(0,Nevals,1),X[:,ind_2],linestyle="None",marker=".",color="k")

	hdl_splots[0].plot(np.arange(0,Nevals,1),z_roll_eDMD[ind_1,:],linestyle="-",lw=1,color="red")
	hdl_splots[1].plot(np.arange(0,Nevals,1),z_roll_eDMD[ind_2,:],linestyle="-",lw=1,color="red")

	# hdl_splots[0].plot(np.arange(0,Nevals,1),z_roll_eDMD[ind_stable_sel[0],:],linestyle="-",lw=1,color="red")
	# hdl_splots[1].plot(np.arange(0,Nevals,1),z_roll_eDMD[ind_stable_sel[1],:],linestyle="-",lw=1,color="red")


	hdl_splots[1].set_xlabel("time")
	hdl_splots[0].set_ylabel("z1")
	hdl_splots[1].set_ylabel("z2")

	hdl_splots[0].set_ylim([ np.amin(z_roll_Mtrajs[ind_1,:,:]) , np.amax(z_roll_Mtrajs[ind_1,:,:]) ])
	hdl_splots[1].set_ylim([ np.amin(z_roll_Mtrajs[ind_2,:,:]) , np.amax(z_roll_Mtrajs[ind_2,:,:]) ])

	plt.show(block=True)


if __name__ == "__main__":

	test()


