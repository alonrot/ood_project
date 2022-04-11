import tensorflow_probability as tfp
import tensorflow.contrib.distributions as tfd
import pdb

# def _forward(x):

# 	y_0 = x[..., 0:1]
# 	y_1 = x[..., 1:2] - y_0**2 - 1
# 	y_tail = x[..., 2:-1]

# 	return tf.concat([y_0, y_1, y_tail], axis=-1)


# x_samples = p_x.sample(1000)
# y_samples = _forward(x_samples)



class Banana(tfp.experimental.bijectors.Bijector):

	def __init__(self, name="banana"):
		super(Banana, self).__init__(inverse_min_event_ndims=1,name=name)

	def _forward(self, x):

		y_0 = x[..., 0:1]
		y_1 = x[..., 1:2] - y_0**2 - 1
		y_tail = x[..., 2:-1]

		return tf.concat([y_0, y_1, y_tail], axis=-1)


def test():


	# Sample from Banana:
	y_samples = Banana().forward(x_samples)


	# Sample using a prior:
	rho = 0.95
	Sigma = np.float32(np.eye(N=2) + rho * np.eye(N=2)[::-1])
	p_x = tfp.distributions.MultivariateNormalTriL(scale_tril=tf.cholesky(Sigma))
	p_y = tfp.distributions.TransformedDistribution(distribution=p_x, bijector=Banana())
	y_samples = p_y.sample(1000)


	pdb.set_trace()





if __name__ == "__main__":

	test()


