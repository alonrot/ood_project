import numpy as np
import control as ct
import control.optimal as opt
import matplotlib.pyplot as plt
import pdb

def vehicle_update(t, x, u, params):
	# Get the parameters for the model
	# l = params.get('wheelbase', 3.)         # vehicle wheelbase
	# phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

	l = 3.0
	phimax = 0.5

	pdb.set_trace()

	# Saturate the steering input
	phi = np.clip(u[1], -phimax, phimax)

	# Return the derivative of the state
	return np.array([
		np.cos(x[2]) * u[0],            # xdot = cos(theta) v
		np.sin(x[2]) * u[0],            # ydot = sin(theta) v
		(u[0] / l) * np.tan(phi)        # thdot = v/l tan(phi)
	])

def vehicle_output(t, x, u, params):
	return x # return x, y, theta (full state)





def test_mpc():


	# Define the vehicle steering dynamics as an input/output system
	vehicle = ct.NonlinearIOSystem(vehicle_update, vehicle_output, states=3, name='vehicle',inputs=('v', 'phi'), outputs=('x', 'y', 'theta'))


	Q = np.diag([0.1, 10, 10.])    # keep lateral error low
	R = np.eye(2) * 0.1
	constraints = [ opt.input_range_constraint(vehicle, [-10., -10.], [10., 10.]) ] # Constraint for the inputs (not for the states)


	x_goal = [100., 2., np.pi/2.]
	x_init = [0., -2., 0.]

	u_init = [0., 0.]
	u_goal = [0., 0.]


	for ii in range(10):

		# x0 = 

		x0 = [0., -2., 0.]
		u0 = [0., 0.]
		xf = [100., 2., np.pi/2.]
		uf = [0., 0.]
		Tf = 10

		cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)


		horizon = np.linspace(0, Tf, 20, endpoint=True)
		bend_left = [0, 0.01]        # slight left veer # Initial inputs to use as a guess for the optimal input. The inputs should either be a 2D vector of shape (ninputs, horizon) or a 1D input of shape (ninputs,) that will be broadcast by extension of the time axis.

		result = opt.solve_ocp(vehicle, horizon, x0, cost, constraints, initial_guess=bend_left, options={'eps': 0.01})    # set step size for gradient calculation

		# Extract the results
		u = result.inputs
		t, y = ct.input_output_response(vehicle, horizon, u, x0)

		# Update:
		# x0 = y[]

		# pdb.set_trace()


	hdl_fig_control, hdl_splots = plt.subplots(3,1,figsize=(12,8))
	hdl_splots[0].plot(y[0,:],y[1,:],color="r",linestyle="--")
	hdl_splots[1].plot(t,u[0,:],color="r",linestyle="--")
	hdl_splots[2].plot(t,u[1,:],color="r",linestyle="--")

	plt.show(block=True)









if __name__ == "__main__":

	test_mpc()
