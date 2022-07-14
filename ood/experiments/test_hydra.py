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
from lqrker.spectral_densities import SquaredExponentialSpectralDensity, MaternSpectralDensity, KinkSpectralDensity
from lqrker.utils.parsing import dotdict
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./config",config_name="config")
def test(cfg):

	print(OmegaConf.to_yaml(cfg))

	pdb.set_trace()

if __name__ == "__main__":

	test()