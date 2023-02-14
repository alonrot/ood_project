import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from lqrker.utils.parsing import get_logger
from lqrker.utils.common import CommonUtils
logger = get_logger(__name__)


markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


def generate_positive_definite_covariance_matrix(dim):
	"""
	Bartlett decomposition
	https://en.wikipedia.org/wiki/Wishart_distribution

	Assume V = I

	
	"""

	A = np.zeros((dim,dim))

	for ii in range(dim):
		for jj in range(ii+1,dim):
			A[ii,jj] = np.random.randn()

		A[ii,ii] = np.random.chisquare(df=dim-ii+2)

	cov_mat = A.T @ A

	return cov_mat

def features_vec(z,omegas,varphis):
	"""
	
	z: [Npoints,dim]
	omegas: [Nfeat,dim]
	varphis: [Nfeat,1]
	
	feat_vec: [Npoints,Nfeat]

	"""

	Nfeat = omegas.shape[0]

	feat_vec = (1./np.sqrt(Nfeat))*np.cos(z @ np.transpose(omegas) + np.transpose(varphis))

	return feat_vec


def main():

	"""

	Plotting the density p(z0,z1) for a Bayesian linear model with:

	z_t is scalar

	z_{t+1} = \sum_{j=1}^M beta_j * phi_j(z_t), with phi_j(z_t) = cos(w_j*z_t + varphi_j)
	beta_j \sim N(mu_beta,Sigma_beta)

	Then, the distribution p(z0,z1) = p(z1|z0)p(z0) is such that

	p(z0) = N(mu0,sigma0^2) is a prior

	p(z1 | z0) = N(z1; m, v) with m = mu_beta @ Phi(z0), v = Phi^T(z0) @ Sigma_beta @ Phi(z0)

	We plot p(z0,z1) to illustrate that it's generally non-Gaussian due to the non-linearities introduced by phi_j(z_t)
	"""


	mu0 = -2.0
	sigma0 = 2.0

	M = 300
	# mu_beta = np.random.randn(1,M)
	mu_beta = np.ones((1,M))
	logger.info("Getting PD matrix ...")
	Sigma_beta = 0.5/M*generate_positive_definite_covariance_matrix(M)

	omega_lim = 1.5
	omegas = -omega_lim + 2.*omega_lim*np.random.randn(M,1)
	varphis = math.pi*np.random.randn(M,1)
	
	# Create grid:
	Ndiv = 71
	logger.info("Creating grid for z0,z1")
	z_lim = 5.0
	dim = 2
	z_vec = CommonUtils.create_Ndim_grid(xmin=-z_lim,xmax=+z_lim,Ndiv=Ndiv,dim=dim) # [z0,z1]
	z_vec = z_vec.numpy()
	Npoints = z_vec.shape[0]

	p_z0_vec = stats.norm.pdf(z_vec[:,0],loc=mu0,scale=sigma0) # Prior on z0, independent from z1
	p_z1_given_z0 = np.zeros(Npoints)
	p_z0z1 = np.zeros(Npoints)

	for ii in range(Npoints):

		feat_at_z0 = features_vec(z_vec[ii:ii+1,0],omegas,varphis) # [Npoints=1,M], phi(z0)

		mu_z1_given_z0 = feat_at_z0 @ np.transpose(mu_beta)
		sigma2_z1_given_z0 = feat_at_z0 @ (Sigma_beta @ np.transpose(feat_at_z0))
		p_z1_given_z0[ii] = stats.norm.pdf(z_vec[ii,1],loc=mu_z1_given_z0,scale=np.sqrt(sigma2_z1_given_z0)) # Prior on z0, independent from z1

		# p_z0_vec[ii] = 1.0
		p_z0z1[ii] = p_z1_given_z0[ii] * p_z0_vec[ii]


	p_z0z1_grid = np.reshape(p_z0z1,(Ndiv,Ndiv))

	p_z0z1_grid_scaled = p_z0z1_grid / np.amax(p_z0z1_grid)

	
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=False)
	extent_plot = [z_vec[0,0],z_vec[-1,0],z_vec[0,1],z_vec[-1,1]] #  scalars (left, right, bottom, top)
	hdl_splots.imshow(p_z0z1_grid,extent=extent_plot,origin="lower")
	# plt.show(block=True)


	ax = plt.figure(figsize=(12,8)).add_subplot(projection='3d')
	zpred = np.linspace(-z_lim,z_lim,Ndiv)
	zpred_grids = np.meshgrid(*([zpred]*dim),indexing="ij")
	ax.plot_surface(zpred_grids[0], zpred_grids[1], p_z0z1_grid_scaled, edgecolor='navy', lw=0.25, rstride=15, cstride=15, alpha=0.3, color="navy")

	ax.contourf(zpred_grids[0], zpred_grids[1], p_z0z1_grid_scaled, zdir='z', offset=-0.4, cmap='coolwarm')
	# ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
	# ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

	ax.set(xlim=(-z_lim, z_lim), ylim=(-z_lim, z_lim), zlim=(-0.5, np.amax(p_z0z1_grid_scaled)))
	ax.set_xlabel(r'$z_0$', fontsize=fontsize_labels)
	ax.set_ylabel(r'$z_1$', fontsize=fontsize_labels)
	ax.set_zlabel(r'$p(z_0,z_1)$', fontsize=fontsize_labels)

	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False

	# Now set color to white (or whatever is "invisible")
	ax.xaxis.pane.set_edgecolor('w')
	ax.yaxis.pane.set_edgecolor('w')
	ax.zaxis.pane.set_edgecolor('w')

	# Bonus: To get rid of the grid as well:
	ax.grid(False)


	plt.show(block=True)



if __name__ == "__main__":

	main()
