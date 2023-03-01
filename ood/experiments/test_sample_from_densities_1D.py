import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import tensorflow as tf
import scipy
from scipy import stats
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParaboloidSpectralDensity, NoNameSpectralDensity, KinkSharpSpectralDensity
from lqrker.utils.parsing import dotdict
import hydra
import pickle
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


def get_samples_and_density(spectral_density):

	# omega_min = -7.
	# omega_max = +7.
	Nsamples = 101
	# L = 500.0
	# Ndiv = 4001

	L = 500.
	Ndiv = 2001

	# L = 10.
	# Ndiv = 41


	# S_vec_plotting, _, omegapred = spectral_density.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,normalize_density_numerically=False)
	S_vec_plotting, _, omegapred = spectral_density.get_Wpoints_discrete(L=L,Ndiv=Ndiv,normalize_density_numerically=False)
	if "part_real_dbg" in dir(spectral_density):
		part_real_dbg = spectral_density.part_real_dbg
		part_imag_dbg = spectral_density.part_imag_dbg
	else:
		part_real_dbg = part_imag_dbg = S_vec_plotting
	# _, _, W_samples_vec = spectral_density.get_Wsamples_from_Sw(Nsamples)
	_, _, W_samples_vec = spectral_density.get_Wpoints_discrete(L=L/50,Ndiv=(Ndiv-1)//50+1,normalize_density_numerically=False)



	# # Random grid using uniform/sobol randomization:
	# Ndiv = 101; L = 10.;
	# min_omega = -((Ndiv-1) //2) * (math.pi/L)
	# max_omega = +((Ndiv-1) //2) * (math.pi/L)
	# # pdb.set_trace()
	# W_samples_vec = min_omega + (max_omega - min_omega)*tf.math.sobol_sample(dim=S_vec_plotting.shape[1],num_results=(Ndiv**S_vec_plotting.shape[1]),skip=1000)
	# # omegapred = tf.random.uniform(shape=(Ndiv**self.dim_in,self.dim_in),minval=min_omega,maxval=max_omega,dtype=tf.dtypes.float32)


	# W_samples_vec = None

	return W_samples_vec, S_vec_plotting, omegapred, part_real_dbg, part_imag_dbg


@hydra.main(config_path="./config",config_name="config")
def test(cfg):

	np.random.seed(seed=0)
	dim_x = 1

	# path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubinscar_data_nominal_model_waypoints_lighter_many_trajs.pickle"
	# logger.info("Loading {0:s} ...".format(path2data))
	# file = open(path2data, 'rb')
	# data_dict = pickle.load(file)
	# file.close()
	# Xtrain = data_dict["Xtrain"]
	# Ytrain = data_dict["Ytrain"]
	# dim_x = data_dict["dim_x"]
	# dim_u = data_dict["dim_u"]
	# Nsteps = data_dict["Nsteps"]
	# Ntrajs = data_dict["Ntrajs"]
	# deltaT = data_dict["deltaT"]


	integration_method = "integrate_with_regular_grid"
	# integration_method = "integrate_with_irregular_grid"
	# integration_method = "integrate_with_bayesian_quadrature"
	# integration_method = "integrate_with_data"

	# data = data_dict["Xtrain"]
	data = None
	spectral_densities = []; labels = []
	spectral_densities += [ParaboloidSpectralDensity(cfg.spectral_density.parabola,cfg.sampler.hmc,dim_x,integration_method)]; labels += ["Parabola"]
	spectral_densities += [KinkSpectralDensity(cfg.spectral_density.kink,cfg.sampler.hmc,dim_x,integration_method)]; labels += ["Kink"]
	spectral_densities += [MaternSpectralDensity(cfg.spectral_density.matern,cfg.sampler.hmc,dim_x)]; labels += ["Matern"]
	spectral_densities += [SquaredExponentialSpectralDensity(cfg.spectral_density.squaredexp,cfg.sampler.hmc,dim_x)]; labels += ["SquaredExp"]
	# spectral_densities += [NoNameSpectralDensity(cfg.spectral_density.noname,cfg.sampler.hmc,dim_x)]; labels += ["NoName"]
	# spectral_densities += [KinkSharpSpectralDensity(cfg.spectral_density.kinksharp,cfg.sampler.hmc,dim_x)]; labels += ["KinkSharp"]
	Ndensities = len(spectral_densities)






	hdl_fig, hdl_splots = plt.subplots(Ndensities,1,figsize=(12,8),sharex=True)
	for jj in range(Ndensities):

		W_samples_vec, S_vec_plotting, omegapred, part_real_dbg, part_imag_dbg = get_samples_and_density(spectral_densities[jj])

		hdl_splots[jj].plot(omegapred,S_vec_plotting,lw=2)
		# hdl_splots[jj].plot(omegapred,part_imag_dbg,lw=2)
		hdl_splots[jj].plot(W_samples_vec,0.1*np.ones(W_samples_vec.shape[0]),marker="x",color="green",linestyle="None")
		hdl_splots[jj].set_title("Spectral density for {0:s} kernel".format(labels[jj]),fontsize=fontsize_labels)
		hdl_splots[jj].set_xlim([omegapred[0,0],omegapred[-1,0]])
		hdl_splots[jj].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	hdl_splots[-1].set_xlabel(r"$\omega$",fontsize=fontsize_labels)

	plt.show(block=True)
	plt.pause(1)





if __name__ == "__main__":

	test()


