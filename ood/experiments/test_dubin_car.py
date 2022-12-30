import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib
import numpy as np
import scipy
from scipy import stats
import hydra
import pickle
import control
from lqrker.utils.parsing import get_logger
# from lqrker.spectral_densities import MaternSpectralDensity, VanDerPolSpectralDensity
from lqrker.spectral_densities import DubinsCarSpectralDensity
# from bayeskoop.models.rrgp_model import RRGPLinearFeatures, RRGPRandomFourierFeatures
logger = get_logger(__name__)
from min_jerk_gen import min_jerk



markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


dyn_sys_true = DubinsCarSpectralDensity._controlled_dubinscar_dynamics

# def dyn_sys_true(zt,ut,deltaT,mode=1):

# 	if mode == 1:
# 		distur = 0.0
# 	elif mode == 2:
# 		distur = 2.0
# 	else:
# 		raise ValueError("Wrong mode")

# 	zt_next = np.zeros(3)

# 	zt_next[0] = deltaT*(ut[0] + distur)*np.cos(zt[2]) + zt[0]
# 	zt_next[1] = deltaT*ut[0]*np.sin(zt[2]) + zt[1]
# 	zt_next[2] = deltaT*ut[1] + zt[2]

# 	return zt_next

# def dyn_sys_model(zt,ut,deltaT,mode):

# 	if mode == 1:
# 		distur = 0.0
# 	elif mode == 2:
# 		distur = 2.0
# 	else:
# 		raise ValueError("Wrong mode")

# 	zt_next = np.zeros(3)

# 	zt_next[0] = 0.5*deltaT*ut[0]*np.cos(zt[2]) + zt[0] + distur + 1.0
# 	zt_next[1] = 0.5*deltaT*ut[0]*np.sin(zt[2]) + zt[1] + 1.0
# 	zt_next[2] = 0.5*deltaT*ut[1] + zt[2] + 1.0

# 	return zt_next


def get_sequence_of_feedback_gains_finite_horizon_LQR(deltaT,x0,ref_xt,ref_ut,mode=1):

	if mode == 1:
		distur = 0.0
		Q = np.array([[10.,0.,0.],[0.,10.,0.],[0.,0.,0.1]])
		R = 0.1*np.eye(2)
	elif mode == 2:
		distur = 2.0
		Q = np.array([[20.,0.,0.],[0.,20.,0.],[0.,0.,0.1]])
		R = 1.*np.eye(2)
	else:
		raise ValueError("Wrong mode")


	Nsteps = ref_xt.shape[0]
	Fk_all = np.zeros((Nsteps,2,3))
	uk_next_all = np.zeros((Nsteps,2))
	S = Q
	for ii in range(Nsteps-2,0,-1):

		# Linearized system matrices using the true system:
		Ak = np.array([	[1., 0., -deltaT*(ref_ut[ii,0]+distur)*np.sin(ref_xt[ii,2])],
						[0., 1.,  deltaT*ref_ut[ii,0]*np.cos(ref_xt[ii,2])],
						[0., 0.,  1.0                      ]])
		Bk = np.array([	[deltaT*np.cos(ref_xt[ii,2]), 0.],
						[deltaT*np.sin(ref_xt[ii,2]), 0.],
						[0.0                 , deltaT]])

		# Update controller:
		Fk_all[ii,...] = -np.linalg.solve( Bk.T @ S @ Bk + R , Bk.T @ S @ Ak)

		# Update matrix S:
		S = Q + Ak.T @ S @ Ak + Ak.T @ S @ Bk @ Fk_all[ii,...]

	return Fk_all


def get_feedback_control_infinite_horizon_LQR(deltaT,ref_xt,ref_ut,xt,mode=1):

	# Design matrices:
	if mode == 1:
		# Q = np.eye(3)
		# R = np.eye(2)
		Q = np.array([[10.,0.,0.],[0.,10.,0.],[0.,0.,0.1]])
		R = 0.1*np.eye(2)
	elif mode == 2:
		Q = np.eye(3)
		R = np.eye(2)
	else:
		raise ValueError("Wrong mode")


	# Linearized system matrices using the true system:
	Ak = np.array([	[1., 0., -deltaT*ref_ut[0]*np.sin(ref_xt[2])],
					[0., 1.,  deltaT*ref_ut[0]*np.cos(ref_xt[2])],
					[0., 0.,  1.0                      ]])
	Bk = np.array([	[deltaT*np.cos(ref_xt[2]), 0.],
					[deltaT*np.sin(ref_xt[2]), 0.],
					[0.0                 , deltaT]])

	# If Ak is exactly the identity, control.dare() fails below
	Ak = Ak - 1e-4*np.eye(3)

	# Infinite horizon LQR:
	try:
		P, eig, Fk = control.dare(Ak, Bk, Q, R) # closed loop: A - BK || feedabck gain: u = -Fx
	except:
		pdb.set_trace()

	# pdb.set_trace()
	uk_next = ref_ut.reshape(-1,1) + Fk @ (ref_xt-xt).reshape(-1,1)

	return uk_next.reshape(-1)

def rollout_with_INfinite_horizon_LQR(z0,deltaT,T,Nsteps,ref_xt,ref_ut,distur=0.0,sigma_n=0.002):

	assert deltaT == 0.01, "This deltaT is the one used inside _controlled_dubinscar_dynamics()"
	dim = len(z0)

	t_vec = np.linspace(0.0,T,Nsteps)
	z_vec_true = np.zeros((Nsteps,dim))
	u_vec = np.zeros((Nsteps,2))
	z_vec_true[0,:] = z0
	for tt in range(Nsteps-1):

		# Get controller:
		# pdb.set_trace()
		u_vec[tt,:] = get_feedback_control_infinite_horizon_LQR(deltaT,ref_xt[tt,:],ref_ut[tt,:],z_vec_true[tt,:],mode=1)

		# Roll-out dynamics:
		# z_vec_true[tt+1,:] = dyn_sys_true(zt=z_vec_true[tt,:],ut=u_vec[tt,:],deltaT=deltaT)
		z_vec_true[tt+1,:] = dyn_sys_true(state_vec=z_vec_true[tt,:],control_vec=u_vec[tt,:])

		# z_vec_true[tt+1,:] += sigma_n*np.random.randn(3)

	return z_vec_true, u_vec, t_vec


def rollout_with_finitie_horizon_LQR(x0,deltaT,T,Nsteps,ref_xt,ref_ut,distur=0.0,sigma_n=0.002,mode_dyn_sys=1,mode_policy=1):

	dim_x = 3
	dim_u = 2
	t_vec = np.linspace(0.0,T,Nsteps)
	x_vec_true = np.zeros((Nsteps,dim_x))
	u_vec = np.zeros((Nsteps,dim_u))
	x_vec_true[0,:] = x0

	Fk_all = get_sequence_of_feedback_gains_finite_horizon_LQR(deltaT,x0,ref_xt,ref_ut,mode=mode_policy)

	# Apply control sequence to true dynamics:
	for tt in range(Nsteps-1):

		# Feedback policy:
		uk_next = ref_ut[tt,:] - Fk_all[tt,...] @ (ref_xt[tt,:]-x_vec_true[tt,:]) # [dim_u,] - [dim_u,]
		u_vec[tt,:] = uk_next

		# Roll-out dynamics:
		# x_vec_true[tt+1,:] = dyn_sys_true(zt=x_vec_true[tt,:],ut=uk_next,deltaT=deltaT,mode=mode_dyn_sys)
		x_vec_true[tt+1,:] = dyn_sys_true(state_vec=x_vec_true[tt:tt+1,:],control_vec=uk_next.reshape(-1,dim_u))


	return x_vec_true, u_vec, t_vec



def generate_reference_trajectory(ref_pars,Nsteps,deltaT):

	sign_xT = ref_pars["sign_xT"]
	sign_Y = ref_pars["sign_Y"]
	rad = ref_pars["rad"]

	# Generate reference:
	# pos0 = np.array([[0.,0.,np.pi/4.],[2.0,2.0,np.pi/4.]])
	# pos0 = np.array([[0.],[2.]]) # Initial and final positions on the X axis
	pos0 = np.array([[0.],[sign_xT*rad]]) # Initial and final positions on the X axis
	ref_xt_vec_X,_ = min_jerk(pos=pos0, dur=Nsteps, vel=None, acc=None, psg=None) # [Nsteps, D]
	# TODO: NOTE: This generates a trajectory on a D-dimensional space from point A to point B, but FOLLOWING A STRAIGHT LINE. We can't do curves. At most, we can do a sequence of waypoints
	# that mimic a curved trajectory. 
	# In particular, for the dubin's car, the heading angle can't be included in the trajectory. It needs to be derived from X and Y. 

	# Circular reference:
	# np.seterr(all='warn')
	# with np.errstate(true_divide='raise'):
	arg_inside = rad**2 - ref_xt_vec_X**2
	if np.any(arg_inside <= 0.0):
		min_among_pos = np.amin(arg_inside[arg_inside > 0.0])
		arg_inside = np.clip(arg_inside,min_among_pos,None)
	ref_xt_vec_Y = sign_Y*np.sqrt(arg_inside)
	gradX_Y = -0.5*(2.*ref_xt_vec_X)/ref_xt_vec_Y # Gradient of the above expression (in this case, just derivative)
	ref_xt_vec_Z = np.arctan2(gradX_Y,1.)
	ref_xt_vec = np.concatenate((ref_xt_vec_X,ref_xt_vec_Y,ref_xt_vec_Z),axis=1) # Reference: [X, Y, th], where th is the heading angle

	# Compute control inputs by numerical differentiation:
	ref_xt_diff_vec = np.diff(ref_xt_vec,axis=0) / deltaT # [Nsteps-1, D]
	ref_lin_vel = np.sqrt( ref_xt_diff_vec[:,0]**2 + ref_xt_diff_vec[:,1]**2 )
	ref_ut_vec = np.concatenate((ref_lin_vel.reshape(-1,1),ref_xt_diff_vec[:,2].reshape(-1,1)),axis=1)


	# # DEBUG: See how the true dynamics react to the control input used as reference (i.e., no policy, or in other words, ut = u_ref_t)
	# z0 = ref_xt_vec[0,:]
	# z_vec_tmp = np.zeros((Nsteps,dim))
	# z_vec_tmp[0,:] = z0
	# for ii in range(Nsteps-1):
	# 	z_vec_tmp[ii+1,:] = dyn_sys_true(zt=z_vec_tmp[ii,:],ut=ref_ut_vec[ii,:],deltaT=deltaT,mode=1)
	# hdl_fig_control, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	# hdl_splots.plot(z_vec_tmp[:,0],z_vec_tmp[:,1])


	# # DEBUG: Plot everything WRT time
	# if plotting:
	# 	hdl_fig_control, hdl_splots = plt.subplots(5,1,figsize=(12,8),sharex=True)
	# 	hdl_splots[0].plot(ref_xt_vec[:,0])
	# 	hdl_splots[1].plot(ref_xt_vec[:,1])
	# 	hdl_splots[2].plot(ref_xt_vec[:,2])
	# 	hdl_splots[3].plot(ref_ut_vec[:,0])
	# 	hdl_splots[4].plot(ref_ut_vec[:,1])

	return ref_xt_vec, ref_ut_vec

def generate_trajectories(ref_pars,mode_dyn_sys=1,mode_policy=1,Nsimus=10,include_ut_in_X=False,plotting=False,x0_noise_std=1.0,batch_nr=None,path2save=None,block=False):
	"""

	Generate Nsimus trajectories of the Dubins car following a reference.
	The reference trajectory is the same for all trajectories.
	The initial condition is different for each trajectory.
	We stack row-wise all trajectories together.

	The outputs (X,Y) are the trajectories prepared to train the GP model.
	If the model is being trained in open loop, i.e., x_{t+1} = f(x_t,u_t), then X = [x_t,u_t] and Y = [x_{t+1}]
	Set include_ut_in_X = True for that to happen

	:return:
	X: [(Nsteps-1)*Nsimus, dim_x+dim_u+2]
	Y: [(Nsteps-1)*Nsimus, dim_x+dim_u+2]



	"""

	deltaT = 0.01
	T = 10.0
	Nsteps = 201
	sigma_n = 0.002
	distur = 0.0
	dim_x = 3 # State space dimensionality [X,Y,th]
	dim_u = 2

	ref_xt_vec, ref_ut_vec = generate_reference_trajectory(ref_pars,Nsteps,deltaT)

	if plotting:
		hdl_fig_control, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	x0 = ref_xt_vec[0,:]
	# x0 = np.array([0.0,2.0,np.pi/2.])
	# z_vec_many = np.zeros((Nsimus,Nsteps,dim_x))
	if include_ut_in_X:
		X = np.zeros(((Nsteps-1)*Nsimus,dim_x+dim_u))
	else:
		X = np.zeros(((Nsteps-1)*Nsimus,dim_x))
	Y = np.zeros(((Nsteps-1)*Nsimus,dim_x))
	for ii in range(Nsimus):

		# Modify initial condition:
		x0_mod = x0 + np.array([0.2,0.2,0.1])*np.random.randn(3)*x0_noise_std
		# x0_mod[2] = 2.*np.pi

		# z_vec, u_vec, t_vec = rollout_with_INfinite_horizon_LQR(x0_mod,deltaT,T,Nsteps,ref_xt_vec,ref_ut_vec,distur,sigma_n)
		z_vec, u_vec, t_vec = rollout_with_finitie_horizon_LQR(x0_mod,deltaT,T,Nsteps,ref_xt_vec,ref_ut_vec,
																distur=0.0,sigma_n=0.002,mode_dyn_sys=mode_dyn_sys,mode_policy=mode_policy)

		# z_vec_many[ii,...] = z_vec

		if include_ut_in_X:
			X[ii*(Nsteps-1):(ii+1)*(Nsteps-1),:] = np.concatenate((z_vec[0:-1,:],u_vec[0:-1,:]),axis=1)
			Y[ii*(Nsteps-1):(ii+1)*(Nsteps-1),:] = z_vec[1::,:]	
			# pdb.set_trace()
		else:
			X[ii*(Nsteps-1):(ii+1)*(Nsteps-1),:] = z_vec[0:-1,:]
			Y[ii*(Nsteps-1):(ii+1)*(Nsteps-1),:] = z_vec[1::,:]

		if plotting:
			hdl_splots.plot(z_vec[:,0],z_vec[:,1],color="gray",linestyle="-",lw=2)
			hdl_splots.plot(z_vec[0,0],z_vec[0,1],color="gray",marker="o",markersize=5) # initial
			hdl_splots.plot(x0_mod[0],x0_mod[1],color="green",marker="o",markersize=3) # x0

	# Original trajectory:
	if plotting:
		hdl_splots.plot(ref_xt_vec[:,0],ref_xt_vec[:,1],color="black",linestyle="--",lw=2)
		hdl_splots.plot(ref_xt_vec[0,0],ref_xt_vec[0,1],color="k",linestyle="None",marker="o",markersize=5) # initial
		hdl_splots.plot(ref_xt_vec[-1,0],ref_xt_vec[-1,1],color="k",linestyle="None",marker="x",markersize=10) # final
		hdl_splots.set_title(r"System 1, Policy 1",fontsize=fontsize_labels)
		hdl_splots.set_xlabel(r"$x_1$",fontsize=fontsize_labels)
		hdl_splots.set_ylabel(r"$x_2$",fontsize=fontsize_labels)
		hdl_splots.set_xticks([])
		hdl_splots.set_yticks([])

		if path2save is not None:
			if batch_nr is not None:
				path2save = path2save+"_{}_{}_{}".format(mode_dyn_sys,mode_policy,batch_nr)
			else:
				path2save = path2save+"_{}_{}".format(mode_dyn_sys,mode_policy)
			logger.info("Saving fig ...")
			hdl_fig_control.savefig(path2save,bbox_inches='tight',dpi=300,transparent=True)
			logger.info("Done saving fig!")
		else:
			# if batch_nr == 1:
			plt.show(block=block)


		# pdb.set_trace()
	# pdb.set_trace()

	return X, Y, deltaT, x0, ref_xt_vec, ref_ut_vec, Nsteps


def get_negative_log_likelihood(rrgp,ref_pars,mode_dyn_sys=1,mode_policy=1,include_ut_in_X=False,x0_noise_std=1.0,plotting=False):

	out = generate_trajectories(ref_pars,mode_dyn_sys=mode_dyn_sys,mode_policy=mode_policy,Nsimus=1,include_ut_in_X=include_ut_in_X,plotting=plotting,x0_noise_std=x0_noise_std)
	Xnew = out[0]
	Ynew = out[1]

	# Compute predictive likelihood:
	Xnew_tf = tf.convert_to_tensor(Xnew,dtype=tf.float32)
	# pdb.set_trace()
	meanpred_list, stdpred_list = rrgp.predict(Xnew_tf)

	Nsteps = Ynew.shape[0]
	dim = Ynew.shape[1]
	log_lik = np.zeros(Nsteps)
	for ii in range(dim):
		# pdb.set_trace()
		ii = 0
		tpf_normal = tfp.distributions.Normal(loc=meanpred_list[ii][:,0],scale=stdpred_list[ii])
		log_lik[:] += tpf_normal.log_prob(Ynew[:,ii])


	return log_lik


@hydra.main(config_path="./config",config_name="config.yaml")
def train_GPmodel_as_closed_loop_model(cfg):

	ref_pars = dict()
	ref_pars["rad"] = 2.0
	ref_pars["sign_Y"] = +1
	ref_pars["sign_xT"] = +1
	X, Y, deltaT, x0, ref_xt, ref_ut, Nsteps = generate_trajectories(ref_pars,mode_dyn_sys=1,mode_policy=1,Nsimus=10,include_ut_in_X=False)

	X_tf = tf.convert_to_tensor(X,dtype=tf.float32)
	Y_tf = tf.convert_to_tensor(Y,dtype=tf.float32)
	reduce_condition_number_by_adding_noise = True
	sigma_n0 = 0.1
	lambda_ij0 = 1.0**2
	learning_rate = 0.01
	training_epochs = 10
	rrgp = ReducedRankGPModel(X_tf,Y_tf,reduce_condition_number_by_adding_noise,sigma_n0,lambda_ij0,learning_rate,training_epochs)
	rrgp.train_model()

	# pdb.set_trace()
	xpred = X_tf
	# xpred = X_tf + tf.random.normal(X_tf.shape,mean=0.,stddev=0.1)
	meanpred_list, stdpred_list = rrgp.predict(xpred)

	# pdb.set_trace()
	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots.plot(Y_tf[:,0],linestyle="none",marker=".",color='k')
	hdl_splots.plot(meanpred_list[0][:,0])
	hdl_splots.plot(meanpred_list[0][:,0] + 2.*stdpred_list[0])
	hdl_splots.plot(meanpred_list[0][:,0] - 2.*stdpred_list[0])



	log_lik11 = get_negative_log_likelihood(rrgp,mode_dyn_sys=1,mode_policy=1)
	log_lik12 = get_negative_log_likelihood(rrgp,mode_dyn_sys=1,mode_policy=2)
	log_lik21 = get_negative_log_likelihood(rrgp,mode_dyn_sys=2,mode_policy=1)
	log_lik22 = get_negative_log_likelihood(rrgp,mode_dyn_sys=2,mode_policy=2)


	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots.plot(-log_lik11,label="lik11")
	hdl_splots.plot(-log_lik12,label="lik12")
	hdl_splots.plot(-log_lik21,label="lik21")
	hdl_splots.plot(-log_lik22,label="lik22")
	hdl_splots.set_ylim([np.amin(-log_lik11),np.amax(-log_lik11)*1.1])
	hdl_splots.legend()


	# Roll-out the GP model as a closed-loop model, using the predictive mean as next input:
	x_vec = np.zeros((Nsteps,X.shape[1]))
	x_vec[0,:] = x0
	for tt in range(Nsteps-1):

		x_in = tf.convert_to_tensor(x_vec[tt:tt+1,:],dtype=tf.float32)
		meanpred_list, stdpred_list = rrgp.predict(x_in)

		# Use mean:
		# x_vec[tt+1,0] = meanpred_list[0]
		# x_vec[tt+1,1] = meanpred_list[1]
		# x_vec[tt+1,2] = meanpred_list[2]


		x_vec[tt+1,0] = tf.random.normal(shape=(1,),mean=meanpred_list[0],stddev=stdpred_list[0])
		x_vec[tt+1,1] = tf.random.normal(shape=(1,),mean=meanpred_list[1],stddev=stdpred_list[1])
		x_vec[tt+1,2] = tf.random.normal(shape=(1,),mean=meanpred_list[2],stddev=stdpred_list[2])





	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots.plot(x_vec[:,0],x_vec[:,1])
	hdl_splots.plot(ref_xt[:,0],ref_xt[:,1])





	# gpflow.config.set_default_float(np.float64)
	# kernel = gpflow.kernels.SquaredExponential(variance=0.5)
	# kernel.lengthscales.assign(0.5)
	# likelihood = gpflow.likelihoods.Gaussian(variance=0.01)
	# num_features = 10
	# # inducing_variable = np.linspace(-2., 2., num_features).reshape(-1, dim)
	# inducing_variable = -2. + 4.*np.random.rand(num_features,dim)
	# # pdb.set_trace()
	# model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable)
	
	# # gpflow.set_trainable(likelihood, False)
	# # gpflow.set_trainable(kernel.variance, False)
	# gpflow.set_trainable(likelihood, False)
	# gpflow.set_trainable(kernel.variance, False)
	# gpflow.set_trainable(kernel.lengthscales, True)

	# # pdb.set_trace()
	# X_tf = tf.convert_to_tensor(X,dtype=tf.float64)
	# Y_tf = tf.convert_to_tensor(Y,dtype=tf.float64)
	# Y1_tf = tf.reshape(Y_tf[:,0],(-1,1))
	# data = (X_tf, Y1_tf)
	# model.training_loss(data)
	# optimizer = tf.optimizers.Adam()
	# training_loss = model.training_loss_closure(data)  # We save the compiled closure in a variable so as not to re-compile it each step
	# optimizer.minimize(training_loss, model.trainable_variables)  # Note that this does a single step


	# Xpred = -2. + 4.*np.random.rand(5,dim)
	# mean, var = model.predict_f(Xpred)
	# pdb.set_trace()
	# mean, var = model.predict_f(X)


	# hdl_splots.plot(z_vec[:,0],z_vec[:,1])


	# hdl_fig_control, hdl_splots = plt.subplots(5,1,figsize=(12,8),sharex=True)
	# hdl_splots[0].plot(t_vec,z_vec[:,0])
	# hdl_splots[1].plot(t_vec,z_vec[:,1])
	# hdl_splots[2].plot(t_vec,z_vec[:,2])
	# hdl_splots[3].plot(t_vec[0:-1],u_vec[:,0])
	# hdl_splots[4].plot(t_vec[0:-1],u_vec[:,1])



	plt.show(block=True)


class CFGSpectralTMP():
	def __init__(self):
		self.nu = 2.5
		self.ls = 0.5
		self.prior_var = 1.0


def train_GPmodel_as_open_loop_model(mode_nr,plotting=False):

	# We train the GP model nr="mode_nr" using dynamical system "mode_nr" and its corresponding (optimal) policy "mode_nr", where the policy was computed with access to the true system
	mode_dyn_sys = mode_nr
	mode_policy = mode_nr

	# Generate random points:
	ref_pars = dict()
	Nbatches = 10
	Nsimus = 20
	# Nbatches = 1
	# Nsimus = 2
	X_tot = [None]*Nbatches
	Y_tot = [None]*Nbatches
	for bb in range(Nbatches):

		ref_pars["rad"] = 1.5 + 1.5*np.random.rand()
		ref_pars["sign_Y"] = 2*np.random.randint(low=0,high=2) - 1
		ref_pars["sign_xT"] = 2*np.random.randint(low=0,high=2) - 1
		X, Y, deltaT, x0, ref_xt, ref_ut, Nsteps = generate_trajectories(ref_pars,mode_dyn_sys=mode_dyn_sys,mode_policy=mode_policy,Nsimus=Nsimus,plotting=plotting,include_ut_in_X=True,batch_nr=bb)

		X_tot[bb] = X
		Y_tot[bb] = Y

	X_tot = np.concatenate(X_tot)
	Y_tot = np.concatenate(Y_tot)

	# Original trajectory:
	hdl_fig_control, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
	hdl_splots.plot(X_tot[:,0],X_tot[:,1],color="b",linestyle="none",marker=".")
	# plt.show(block=True)

	# Ydelta = Y - X[:,0:3]

	X_tf = tf.convert_to_tensor(X_tot,dtype=tf.float32)
	Y_tf = tf.convert_to_tensor(Y_tot,dtype=tf.float32)
	reduce_condition_number_by_adding_noise = True
	use_prior_mean = False
	sigma_n0 = 0.1
	lambda_ij0 = 1.0**2
	learning_rate = 0.01
	training_epochs = 2
	# rrgp = ReducedRankGPModel(X_tf,Y_tf,reduce_condition_number_by_adding_noise,sigma_n0,lambda_ij0,learning_rate,training_epochs,use_prior_mean=True)
	# rrgp = RRGPLinearFeatures(X_tf,Y_tf,reduce_condition_number_by_adding_noise,sigma_n0,lambda_ij0,learning_rate,training_epochs,use_prior_mean)
	raise NotImplementedError("This function uses deprecated classes and modules from the package 'bayeskoop'. Above, we should be loading RRGPLinearFeatures()")

	# cfg_spectral = CFGSpectralTMP()
	# spectral_density = MaternSpectralDensity(cfg=cfg_spectral,dim=X.shape[1])
	# Nfeat = 20
	# rrgp = RRGPRandomFourierFeatures(X_tf,Y_tf,reduce_condition_number_by_adding_noise,sigma_n0,lambda_ij0,learning_rate,training_epochs,use_prior_mean,spectral_density,Nfeat)

	if not plotting:
		rrgp.train_model()


	if plotting:

		# pdb.set_trace()
		# xpred = X_tf
		xpred = X_tf + tf.random.normal(X_tf.shape,mean=0.,stddev=0.1)
		meanpred_list, stdpred_list = rrgp.predict(xpred)

		# pdb.set_trace()
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_splots.plot(Y_tf[:,0],linestyle="none",marker=".",color='k')
		hdl_splots.plot(meanpred_list[0][:,0])
		hdl_splots.plot(meanpred_list[0][:,0] + 2.*stdpred_list[0])
		hdl_splots.plot(meanpred_list[0][:,0] - 2.*stdpred_list[0])
		# plt.show(block=True)

		ref_pars = dict()
		ref_pars["rad"] = 2.0
		ref_pars["sign_Y"] = +1
		ref_pars["sign_xT"] = +1

		log_lik11 = get_negative_log_likelihood(rrgp,ref_pars,mode_dyn_sys=1,mode_policy=1,include_ut_in_X=True)
		log_lik12 = get_negative_log_likelihood(rrgp,ref_pars,mode_dyn_sys=1,mode_policy=2,include_ut_in_X=True)
		log_lik21 = get_negative_log_likelihood(rrgp,ref_pars,mode_dyn_sys=2,mode_policy=1,include_ut_in_X=True)
		log_lik22 = get_negative_log_likelihood(rrgp,ref_pars,mode_dyn_sys=2,mode_policy=2,include_ut_in_X=True)

		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_splots.plot(-log_lik11,label="lik11")
		hdl_splots.plot(-log_lik12,label="lik12")
		hdl_splots.plot(-log_lik21,label="lik21")
		hdl_splots.plot(-log_lik22,label="lik22")
		hdl_splots.set_ylim([np.amin(-log_lik11),np.amax(-log_lik11)*1.1])
		hdl_splots.legend()

		# plt.show(block=True)


		# Roll-out the GP model as a closed-loop model, using the predictive mean as next input:
		Nsteps_model_rollout = Nsteps
		Fk_all = get_sequence_of_feedback_gains_finite_horizon_LQR(deltaT,x0,ref_xt,ref_ut,mode=1)
		x_vec = np.zeros((Nsteps_model_rollout,X.shape[1]))
		x0_new = x0
		# x0_new = x0 + 0.1*np.random.randn(3)
		u0 = Fk_all[0,...] @ x0_new
		x_vec[0,:] = np.concatenate((x0_new,u0))
		x_vec_delta = np.zeros(3)
		for tt in range(Nsteps_model_rollout-1):

			x_in = tf.convert_to_tensor(x_vec[tt:tt+1,:],dtype=tf.float32)
			meanpred_list, stdpred_list = rrgp.predict(x_in)

			# Use mean:
			x_vec[tt+1,0] = meanpred_list[0]
			x_vec[tt+1,1] = meanpred_list[1]
			x_vec[tt+1,2] = meanpred_list[2]

			# # Use mean with deltas:
			# x_vec[tt+1,0] = meanpred_list[0] + x_vec[tt,0]
			# x_vec[tt+1,1] = meanpred_list[1] + x_vec[tt,1]
			# x_vec[tt+1,2] = meanpred_list[2] + x_vec[tt,2]

			# Using the input itself:
			# x_vec[tt+1,0] = tf.random.normal(shape=(1,),mean=meanpred_list[0],stddev=stdpred_list[0])
			# x_vec[tt+1,1] = tf.random.normal(shape=(1,),mean=meanpred_list[1],stddev=stdpred_list[1])
			# x_vec[tt+1,2] = tf.random.normal(shape=(1,),mean=meanpred_list[2],stddev=stdpred_list[2])

			# Using deltas
			# x_vec_delta[0] = tf.random.normal(shape=(1,),mean=meanpred_list[0],stddev=stdpred_list[0])
			# x_vec_delta[1] = tf.random.normal(shape=(1,),mean=meanpred_list[1],stddev=stdpred_list[1])
			# x_vec_delta[2] = tf.random.normal(shape=(1,),mean=meanpred_list[2],stddev=stdpred_list[2])
			# x_vec[tt+1,0:3] = x_vec[tt,0:3] + x_vec_delta

			# Control law:
			u_next = ref_ut[tt,:].reshape(-1,1) - Fk_all[tt,...] @ (ref_xt[tt,:]-x_vec[tt+1,0:3]).reshape(-1,1)
			x_vec[tt+1,3] = u_next[0]
			x_vec[tt+1,4] = u_next[1]



		# TODO: When sampling x_{t+1} from the model, using (xt,ut) as inputs, we get wild samples and the closed loop explodes.
		# 1) Maybe shorter horizon and do MPC?
		# 2) Maybe adjust lengthscales, etc.?
		# 3) Maybe train the model to predict long-term? See existing work.
		# 4) Maybe predict by adding each sample to the GP dataset (in closed loop fashion) as virtual observation.


		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(12,8),sharex=True)
		hdl_splots.plot(x_vec[:,0],x_vec[:,1])
		hdl_splots.plot(ref_xt[:,0],ref_xt[:,1])

		# plt.show(block=True)


	return rrgp



	# gpflow.config.set_default_float(np.float64)
	# kernel = gpflow.kernels.SquaredExponential(variance=0.5)
	# kernel.lengthscales.assign(0.5)
	# likelihood = gpflow.likelihoods.Gaussian(variance=0.01)
	# num_features = 10
	# # inducing_variable = np.linspace(-2., 2., num_features).reshape(-1, dim)
	# inducing_variable = -2. + 4.*np.random.rand(num_features,dim)
	# # pdb.set_trace()
	# model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable)
	
	# # gpflow.set_trainable(likelihood, False)
	# # gpflow.set_trainable(kernel.variance, False)
	# gpflow.set_trainable(likelihood, False)
	# gpflow.set_trainable(kernel.variance, False)
	# gpflow.set_trainable(kernel.lengthscales, True)

	# # pdb.set_trace()
	# X_tf = tf.convert_to_tensor(X,dtype=tf.float64)
	# Y_tf = tf.convert_to_tensor(Y,dtype=tf.float64)
	# Y1_tf = tf.reshape(Y_tf[:,0],(-1,1))
	# data = (X_tf, Y1_tf)
	# model.training_loss(data)
	# optimizer = tf.optimizers.Adam()
	# training_loss = model.training_loss_closure(data)  # We save the compiled closure in a variable so as not to re-compile it each step
	# optimizer.minimize(training_loss, model.trainable_variables)  # Note that this does a single step


	# Xpred = -2. + 4.*np.random.rand(5,dim)
	# mean, var = model.predict_f(Xpred)
	# pdb.set_trace()
	# mean, var = model.predict_f(X)


	# hdl_splots.plot(z_vec[:,0],z_vec[:,1])


	# hdl_fig_control, hdl_splots = plt.subplots(5,1,figsize=(12,8),sharex=True)
	# hdl_splots[0].plot(t_vec,z_vec[:,0])
	# hdl_splots[1].plot(t_vec,z_vec[:,1])
	# hdl_splots[2].plot(t_vec,z_vec[:,2])
	# hdl_splots[3].plot(t_vec[0:-1],u_vec[:,0])
	# hdl_splots[4].plot(t_vec[0:-1],u_vec[:,1])



@hydra.main(config_path="./config",config_name="config.yaml")
def test_mode_switching_ood_detection(cfg):

	rrgp1 = train_GPmodel_as_open_loop_model(mode_nr=1,plotting=False)
	rrgp2 = train_GPmodel_as_open_loop_model(mode_nr=2,plotting=False)

	ref_pars = dict()
	# ref_pars["rad"] = 1.5 + 1.5*np.random.rand()
	# ref_pars["sign_Y"] = 2*np.random.randint(low=0,high=2) - 1
	# ref_pars["sign_xT"] = 2*np.random.randint(low=0,high=2) - 1
	ref_pars["rad"] = 2.0
	ref_pars["sign_Y"] = +1
	ref_pars["sign_xT"] = +1

	# X, Y, deltaT, x0, ref_xt, ref_ut, Nsteps = generate_trajectories(ref_pars,mode_dyn_sys=1,mode_policy=1,Nsimus=1,include_ut_in_X=True)

	# pdb.set_trace()




	"""
	1) Use system 1 and policy 1 to generate an action sequence
	2) Use such action sequence on system 1 and predict with model 1
	2) Use such action sequence on system 2 and predict with model 1
	2) Use such action sequence on system 2 and predict with model 2
	"""

	Nsteps = 200
	Nstats = 10
	
	log_lik111_stats = np.zeros((Nsteps,Nstats))
	log_lik121_stats = np.zeros((Nsteps,Nstats))
	log_lik221_stats = np.zeros((Nsteps,Nstats))
	
	log_lik222_stats = np.zeros((Nsteps,Nstats))
	log_lik212_stats = np.zeros((Nsteps,Nstats))
	log_lik112_stats = np.zeros((Nsteps,Nstats))
	
	for nn in range(Nstats):
		log_lik111_stats[:,nn] = -get_negative_log_likelihood(rrgp1,ref_pars,mode_dyn_sys=1,mode_policy=1,include_ut_in_X=True,x0_noise_std=0.1,plotting=False)
		log_lik121_stats[:,nn] = -get_negative_log_likelihood(rrgp1,ref_pars,mode_dyn_sys=2,mode_policy=1,include_ut_in_X=True,x0_noise_std=0.1,plotting=False)
		log_lik221_stats[:,nn] = -get_negative_log_likelihood(rrgp2,ref_pars,mode_dyn_sys=2,mode_policy=1,include_ut_in_X=True,x0_noise_std=0.1,plotting=False)

		log_lik222_stats[:,nn] = -get_negative_log_likelihood(rrgp2,ref_pars,mode_dyn_sys=2,mode_policy=2,include_ut_in_X=True,x0_noise_std=0.1,plotting=False)
		log_lik212_stats[:,nn] = -get_negative_log_likelihood(rrgp2,ref_pars,mode_dyn_sys=1,mode_policy=2,include_ut_in_X=True,x0_noise_std=0.1,plotting=False)
		log_lik112_stats[:,nn] = -get_negative_log_likelihood(rrgp1,ref_pars,mode_dyn_sys=1,mode_policy=2,include_ut_in_X=True,x0_noise_std=0.1,plotting=False)


		# log_lik22_stats[:,nn] = -get_negative_log_likelihood(rrgp2,ref_pars,mode_dyn_sys=2,mode_policy=2,include_ut_in_X=True)


	log_lik111_stats_mean = np.mean(log_lik111_stats,axis=1)
	log_lik111_stats_std = np.std(log_lik111_stats,axis=1)

	log_lik121_stats_mean = np.mean(log_lik121_stats,axis=1)
	log_lik121_stats_std = np.std(log_lik121_stats,axis=1)

	log_lik221_stats_mean = np.mean(log_lik221_stats,axis=1)
	log_lik221_stats_std = np.std(log_lik221_stats,axis=1)

	log_lik222_stats_mean = np.mean(log_lik222_stats,axis=1)
	log_lik222_stats_std = np.std(log_lik222_stats,axis=1)

	log_lik212_stats_mean = np.mean(log_lik212_stats,axis=1)
	log_lik212_stats_std = np.std(log_lik212_stats,axis=1)

	log_lik112_stats_mean = np.mean(log_lik112_stats,axis=1)
	log_lik112_stats_std = np.std(log_lik112_stats,axis=1)


	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)

	time_vec = np.linspace(0.0,2.0-0.01,200)

	# hdl_splots[0].plot(time_vec,log_lik111_stats_mean,label="lik111",color="red")
	hdl_splots[0].plot(time_vec,log_lik111_stats_mean,label="System 1, Policy 1",color="red")
	hdl_splots[0].fill_between(time_vec,log_lik111_stats_mean - 2.*log_lik111_stats_std,
									log_lik111_stats_mean + 2.*log_lik111_stats_std,
									alpha=0.1,color="red")


	# hdl_splots[0].plot(time_vec,log_lik121_stats_mean,label="lik121",color="green")
	hdl_splots[0].plot(time_vec,log_lik121_stats_mean,label="System 2, Policy 1",color="green")
	hdl_splots[0].fill_between(time_vec,log_lik121_stats_mean - 2.*log_lik121_stats_std,
									log_lik121_stats_mean + 2.*log_lik121_stats_std,
									alpha=0.1,color="green")

	# hdl_splots[0].plot(time_vec,log_lik221_stats_mean,label="lik221",color="blue")
	# hdl_splots[0].fill_between(time_vec,log_lik221_stats_mean - 2.*log_lik221_stats_std,
	# 								log_lik221_stats_mean + 2.*log_lik221_stats_std,
	# 								alpha=0.1,color="blue")


	# hdl_splots[1].plot(time_vec,log_lik222_stats_mean,label="lik222",color="red")
	hdl_splots[1].plot(time_vec,log_lik222_stats_mean,label="System 2, Policy 2",color="red")
	hdl_splots[1].fill_between(time_vec,log_lik222_stats_mean - 2.*log_lik222_stats_std,
									log_lik222_stats_mean + 2.*log_lik222_stats_std,
									alpha=0.1,color="red")

	# hdl_splots[1].plot(time_vec,log_lik212_stats_mean,label="lik212",color="green")
	hdl_splots[1].plot(time_vec,log_lik212_stats_mean,label="System 1, Policy 2",color="green")
	hdl_splots[1].fill_between(time_vec,log_lik212_stats_mean - 2.*log_lik212_stats_std,
									log_lik212_stats_mean + 2.*log_lik212_stats_std,
									alpha=0.1,color="green")


	# hdl_splots[1].plot(time_vec,log_lik112_stats_mean,label="lik112",color="blue")
	# hdl_splots[1].fill_between(time_vec,log_lik112_stats_mean - 2.*log_lik112_stats_std,
	# 								log_lik112_stats_mean + 2.*log_lik112_stats_std,
	# 								alpha=0.1,color="blue")


	# hdl_splots.plot(-log_lik11,label="lik11")
	# hdl_splots.plot(-log_lik12,label="lik12")
	# hdl_splots.plot(-log_lik21,label="lik21")
	# hdl_splots.plot(-log_lik22,label="lik22")
	# hdl_splots.set_ylim([np.amin(-log_lik11_stats_mean),np.amax(-log_lik11_stats_mean)*1.1])


	# hdl_splots[0].set_xlabel(r"time [sec]",fontsize=fontsize_labels)
	hdl_splots[0].set_ylabel(r"$-\log p(y_{t+1}|x_t,u_t)$",fontsize=fontsize_labels)
	hdl_splots[0].legend()
	hdl_splots[0].set_xlim([0.,2.])
	hdl_splots[0].set_title(r"Evidence using GP model 1",fontsize=fontsize_labels)


	hdl_splots[1].set_xlabel(r"time [sec]",fontsize=fontsize_labels)
	hdl_splots[1].set_ylabel(r"$-\log p(y_{t+1}|x_t,u_t)$",fontsize=fontsize_labels)
	hdl_splots[1].legend()
	hdl_splots[1].set_xlim([0.,2.])
	hdl_splots[1].set_title(r"Evidence using GP model 2",fontsize=fontsize_labels)

	hdl_fig.savefig("/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/plotting/evi",bbox_inches='tight',dpi=300,transparent=True)
	logger.info("Done!")


	# plt.show(block=True)

	# pdb.set_trace()




# def test_rollout():




if __name__ == "__main__":

	# train_GPmodel_as_closed_loop_model()

	# train_GPmodel_as_open_loop_model()

	test_mode_switching_ood_detection()


