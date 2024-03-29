import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ood",
    version = "1.0",
    author = "",
    author_email = "",
    description = (""),
    keywords = "Bayesian Optimization, Gaussian process, Learning representations",
    packages=[	'ood',
    			'ood.fourier_kernel',
                'ood.spectral_density_approximation'],
    long_description=read('README.md'),
)