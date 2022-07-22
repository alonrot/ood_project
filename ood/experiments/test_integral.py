import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
from scipy import integrate
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity, ParabolaSpectralDensity
from lqrker.utils.parsing import dotdict
import hydra
import tensorflow_probability as tfp


def test():


	xmin = -5.
	xmax = +5.
	Ndiv = 401
	xpred = np.linspace(xmin,xmax,Ndiv)
	Dx = xpred[1] - xpred[0]
	gaussian_pred = scipy.stats.norm.pdf(xpred)

	out = integrate.trapezoid(y=gaussian_pred,dx=Dx) # [Npoints,]
	out_tpf = tfp.math.trapz(y=gaussian_pred,dx=Dx) # [Npoints,]

	pdb.set_trace()


if __name__ == "__main__":

	test()