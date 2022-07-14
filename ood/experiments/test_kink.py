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
import hydra

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 20
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)

def kink_fun(x):
	return 0.8 + (x + 0.2)*(1. - 5./(1 + np.exp(-2.*x)) )


def MVFourierTransformKink(omega_vec,get_angle=False):
	"""

	omega_vec: [Npoints,dim]
	"""

	dim = 1
	Nsteps = 401
	xdata = np.linspace(-5.,2.,Nsteps)
	dX = xdata[1] - xdata[0]

	xdata = np.reshape(xdata,(-1,dim)) # [Nsteps,dim]
	fdata = kink_fun(xdata) # [Nsteps,1] (since this is a scalar system, we only have one channel)

	omega_times_X = np.dot(omega_vec,xdata.T) # [Npoints,Nsteps]

	# pdb.set_trace()
	part_real = (np.cos(omega_times_X)@fdata)*dX # [Npoints]
	part_imag = (np.sin(omega_times_X)@fdata)*dX # [Npoints]

	Sw = np.sqrt(part_real**2 + part_imag**2) # [Npoints]

	phiw = np.arctan2(-part_imag,part_real)

	if get_angle:
		return Sw, phiw
	else:
		return Sw

def InverseMVFourierTransform(omega_vec,Sw_vec,phiw_vec):

	omega_vec = np.reshape(omega_vec,(-1))
	Sw_vec = np.reshape(Sw_vec,(-1))
	phiw_vec = np.reshape(phiw_vec,(-1))

	Nsteps = 201
	xdata = np.linspace(-5.,2.,Nsteps)
	Dw = omega_vec[1] - omega_vec[0]
	fx = np.zeros(Nsteps)
	# phiw_vec = 0.0 # DBG

	for ii in range(Nsteps):
		
		# pdb.set_trace()
		fx[ii] = integrate.trapezoid(y=np.cos(xdata[ii]*omega_vec + phiw_vec)*Sw_vec,x=omega_vec,dx=Dw)

	return fx


def spectral_matern(omega_vec):

	# Parameters:
	nu = 2.1
	ls = 0.5
	prior_var = 1.0
	dim = 1

	# Constant parameters:
	lambda_val = np.sqrt(2*nu)/ls
	const = ((2*np.sqrt(math.pi))**dim)*np.exp(math.lgamma(nu+0.5*dim))*lambda_val**(2*nu) / np.exp(math.lgamma(nu)) # tf.math.lgamma() is the same as math.lgamma()

	S_vec = const / ((lambda_val**2 + omega_vec**2)**(nu+dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)

	# Multiply dimensions (i.e., assume that matern density factorizes w.r.t omega input dimensionality). Also, pump variance into it:
	S_vec = np.prod(S_vec,axis=1)*prior_var

	return S_vec



def spectral_squared_exponential(omega_vec):
	
	ls = 0.4
	prior_var = 1.0
	dim = 1

	const = (2.*math.pi*ls**2)**(dim/2.)
	S_vec = const*np.exp(-2.*(math.pi*ls*omega_vec)**2)

	return S_vec


def feat_fun(x,omega_vec,Sw_vec,phiw_vec=0.0,square=False,add_one=False):

	const = 0.0
	if add_one:
		const = 1.0
	
	aux = np.cos(x*omega_vec + phiw_vec) + const

	if square:
		out = Sw_vec*aux**2
	else:
		out = Sw_vec*aux

	return out


def kernel_bochner(omega_vec,Sw_vec,phiw_vec=np.array([[0.0]])):

	Ndiv = 401
	Sw_vec = np.reshape(Sw_vec,(-1))
	omega_vec = np.reshape(omega_vec,(-1))
	phiw_vec = np.reshape(phiw_vec,(-1))

	dist_vec = np.linspace(0.,10.,Ndiv)
	ker_vec = np.zeros(Ndiv)
	Dw = omega_vec[1]-omega_vec[0]

	for ii in range(Ndiv):
		
		x_dist = dist_vec[ii]
		# pdb.set_trace()
		# print("ii:",ii)
		# ker_vec[ii] = integrate.trapezoid(y=np.cos(x_dist*omega_vec)*Sw_vec,x=omega_vec,dx=Dw)
		
		if phiw_vec[0] == 0.0:
			ker_vec[ii] = integrate.trapezoid(y=feat_fun(x_dist,omega_vec,Sw_vec),x=omega_vec,dx=Dw)
		else:
			aux1 = integrate.trapezoid(y=feat_fun(x_dist,omega_vec,Sw_vec,phiw_vec=phiw_vec,square=True),x=omega_vec,dx=Dw)
			# aux2 = (integrate.trapezoid(y=feat_fun(x_dist,omega_vec,Sw_vec,phiw_vec=phiw_vec),x=omega_vec,dx=Dw))**2
			aux2 = 0.0
			ker_vec[ii] = aux1 - aux2 # Var[]
			# print("aux1:",aux1)

	# pdb.set_trace()

		

		# ker_vec[ii] = integrate.trapezoid(y=feat_fun(x_dist,omega_vec,Sw_vec,phiw_vec=phiw_vec),x=omega_vec,dx=Dw)
		# ker_vec[ii] = np.sum(fun_vec)*Dw

	# pdb.set_trace()

	return dist_vec, ker_vec


def kernel2D(omega_vec,Sw_vec_kink,ker_fun,phiw_vec):

	Ndiv = 81
	dist_vec = np.linspace(-5.,5.,Ndiv)

	dist_vecXX, dist_vecYY = np.meshgrid(*([dist_vec]*2))

	dist_vecZZ = np.concatenate([dist_vecXX.reshape(-1,1),dist_vecYY.reshape(-1,1)],axis=1)

	ker_vec_aux = np.zeros(dist_vecZZ.shape[0])

	def feat_vec(x,omega_vec):
		return np.cos(x*omega_vec)


	for ii in range(dist_vecZZ.shape[0]):

		x1 = dist_vecZZ[ii,0]
		x2 = dist_vecZZ[ii,1]

		# ker_vec_aux[ii] = np.sum( feat_vec(x1,omega_vec) * feat_vec(x2,omega_vec) * Sw_vec_kink )
		ker_vec_aux[ii] = ker_fun(x1,x2,omega_vec,Sw_vec_kink,phiw_vec)
		

	ker_mat = np.reshape(ker_vec_aux,(Ndiv,Ndiv))

	return ker_mat, dist_vec


def ker_fun_kink(x1,x2,omega_vec,Sw_vec,phiw_vec):

	Dw = omega_vec[1] - omega_vec[0]

	y1 = feat_fun(x1,omega_vec,np.sqrt(Sw_vec),phiw_vec=phiw_vec,square=False,add_one=True)
	y2 = feat_fun(x2,omega_vec,np.sqrt(Sw_vec),phiw_vec=phiw_vec,square=False,add_one=True)
	ker_val = integrate.trapezoid(y=y1[:,0]*y2[:,0],x=omega_vec[:,0],dx=Dw)

	# pdb.set_trace()

	return ker_val


def ker_fun(x1,x2,omega_vec,Sw_vec_kink):
	"""

	x1: scalar
	x2: scalar
	omega_vec: [Ndiv,]
	Sw_vec_kink: [Ndiv,]

	"""

	Dw = omega_vec[1] - omega_vec[0]

	# TODO: Do this with integrate.trapezoid
	ker_val = np.sum(  (np.cos(x1*omega_vec)*np.cos(x2*omega_vec) + np.sin(x1*omega_vec)*np.sin(x2*omega_vec))*Sw_vec_kink )*Dw

	return ker_val



def sample_GP(ker_mat,dist_vec,Nsamples):

	# ker_mat, dist_vec = kernel2D(omega_vec,Sw_vec_kink)
	ker_mat_chol = np.linalg.cholesky(ker_mat + 1e-8*np.eye(ker_mat.shape[0]))

	Nels = ker_mat.shape[0]
	# fun_samples = np.zeros((len(dist_vec),Nsamples))
	fun_samples = ker_mat_chol @ np.random.randn(Nels,Nsamples)

	return fun_samples

# def sample_prior_functions(xpred,omega_vec,Sw_vec):

# 	Dw = omega_vec[1] - omega_vec[0]
# 	ker_val = np.sum(  (np.cos(x1*omega_vec)*np.cos(x2*omega_vec) + np.sin(x1*omega_vec)*np.sin(x2*omega_vec))*Sw_vec_kink )*Dw



class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


# def test_kink_spectral_density_tfp():

# 	dim_x = 1
# 	spectral_density_pars_dict = dotdict(name="kink",x_lim_min=-5.0,x_lim_max=+2.0,prior_var=1.0,Nsteps_integration=401)
# 	# pdb.set_trace()
# 	spectral_density = KinkSpectralDensity(spectral_density_pars_dict,dim=dim_x)


def test_sample_from_kink(Nsamples_per_state0,initial_states_sampling,num_burnin_steps):

	spectral_density_pars_dict = dotdict(name="kink",x_lim_min=-5.0,x_lim_max=+2.0,
										prior_var=1.0,Nsteps_integration=401,
										step_size_hmc=0.1,num_leapfrog_steps_hmc=4,
										Nsamples_per_state0=Nsamples_per_state0,
										initial_states_sampling=initial_states_sampling,
										num_burnin_steps=num_burnin_steps)
	
	spectral_density = KinkSpectralDensity(spectral_density_pars_dict,dim=1)
	samples = spectral_density.get_samples()

	return samples





def test():


	dim = 1
	Npoints = 2001
	omega_min = -20.0
	omega_max = +20.0
	omega_vec = np.linspace(omega_min,omega_max,Npoints)
	omega_vec = np.reshape(omega_vec,(-1,dim)) # [Nsteps,dim]

	Sw_vec_kink, phiw_kink = MVFourierTransformKink(omega_vec,get_angle=True)
	Sw_vec_matern = spectral_matern(omega_vec)
	Sw_vec_squa_exp = spectral_squared_exponential(omega_vec)

	"""
	TODO
	1) Can I use Bochner's theorem at all? It's meant to be for stationary kernels -> NO
	2) use here the new density class. Do not call kernel_bochner, call InverseMVFourierTransform instead
	3) Have also a 2D kernel
	4) Compare with the kernel adding +1
	5) Figure out the multiplication factors

	"""

	# Sw_vec_kink_nor = Sw_vec_kink / MVFourierTransformKink(np.array([[0.0]]))
	Sw_vec_kink_nor = Sw_vec_kink / np.sum(Sw_vec_kink)
	# pdb.set_trace()
	dist_vec, ker_vec_kink = kernel_bochner(omega_vec,Sw_vec_kink_nor)
	_, ker_vec_kink_phiw = kernel_bochner(omega_vec,Sw_vec_kink_nor,phiw_kink)
	_, ker_vec_matern = kernel_bochner(omega_vec,Sw_vec_matern)
	_, ker_vec_squa_exp = kernel_bochner(omega_vec,Sw_vec_squa_exp)


	hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(12,8),sharex=False)
	hdl_splots[0].plot(omega_vec,Sw_vec_kink / MVFourierTransformKink(np.array([[0.0]])),label="kink")
	hdl_splots[0].plot(omega_vec,Sw_vec_matern / spectral_matern(np.array([[0.0]])),label="matern")
	hdl_splots[0].plot(omega_vec,Sw_vec_squa_exp / spectral_squared_exponential(0.0),label="SE")


	# Overlay samples:
	# Nsamples = int(10e2)
	# num_burnin_steps = int(1e2)
	Nsamples_per_state0 = int(60)
	initial_states_sampling = np.array([[0.0],[0.5],[1.0]],dtype=np.float32)
	num_burnin_steps = int(50)
	samples_kink = test_sample_from_kink(Nsamples_per_state0,initial_states_sampling,num_burnin_steps)
	hdl_splots[0].plot(samples_kink[:,0],0.1*np.ones(samples_kink.shape[0]),marker="x",color="green",linestyle="None")
	# pdb.set_trace()

	# print("Sw_vec_kink[0]: "+str(Sw_vec_kink[0]))

	hdl_splots[0].set_xlim([omega_min,omega_max])
	hdl_splots[0].set_xlabel(r"$\omega$",fontsize=fontsize_labels)
	hdl_splots[0].set_ylabel(r"$S(\omega)$",fontsize=fontsize_labels)
	hdl_splots[0].set_xlim([-6.,+6.])
	hdl_splots[0].legend()

	hdl_splots[1].plot(dist_vec,ker_vec_kink / ker_vec_kink[0],label="kink")
	hdl_splots[1].plot(dist_vec,ker_vec_matern / ker_vec_matern[0],label="matern")
	hdl_splots[1].plot(dist_vec,ker_vec_squa_exp / ker_vec_squa_exp[0],label="SE")
	hdl_splots[1].plot(dist_vec,ker_vec_kink_phiw / ker_vec_kink_phiw[0],label="kink-angle")
	hdl_splots[1].set_xlim([dist_vec[0],dist_vec[-1]])
	hdl_splots[1].set_xlabel(r"$x_t-x_t^\prime$",fontsize=fontsize_labels)
	hdl_splots[1].set_ylabel(r"$k(x_t-x_t^\prime)$",fontsize=fontsize_labels)
	hdl_splots[1].legend(loc="right")


	hdl_splots[2].plot(omega_vec,phiw_kink,label="kink")

	# Recover the original function:
	Sw_vec_kink_nor = Sw_vec_kink / MVFourierTransformKink(np.array([[0.0]])) * 2.0
	raise NotImplementedError("Figure out first why the factor 2.0 is needed in the line above...!!")

	fx_kink_vec = InverseMVFourierTransform(omega_vec,Sw_vec_kink_nor,phiw_kink)
	Nsteps = 201
	xdata = np.linspace(-5.,2.,Nsteps)
	fx_kink_true_vec = kink_fun(xdata) # [Nsteps,1] (since this is a scalar system, we only have one channel)
	
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,10),sharex=False)
	hdl_splots.plot(xdata,fx_kink_true_vec,color="k")
	hdl_splots.plot(xdata,fx_kink_vec,color="r")


	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,10),sharex=False)
	ker_mat, dist_vec_samples = kernel2D(omega_vec,Sw_vec_kink_nor,ker_fun=ker_fun_kink,phiw_vec=phiw_kink)
	samples = sample_GP(ker_mat, dist_vec_samples,Nsamples=3)
	hdl_splots.plot(dist_vec_samples,samples)
	hdl_splots.set_title("Incorrect way of sampling from the prior: we need to do the recurrent thing")
	raise NotImplementedError("Incorrect way of sampling from the prior: we need to do the recurrent thing")
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=False)
	hdl_splots.imshow(ker_mat)


	_, ker_vec_kink_phiw = kernel_bochner(omega_vec,Sw_vec_kink_nor,phiw_kink)


	plt.show(block=True)





if __name__ == "__main__":

	# test()

	# test_kink_spectral_density_tfp()

	# test_sample_from_kink()

