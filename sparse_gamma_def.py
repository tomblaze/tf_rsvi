import os
import time, datetime

import tensorflow as tf
import numpy as np

from gamma_rejection import *

global_B = 10.0
global_int_B = int(global_B)
do_images = True
print "Running code with B=%i" %(global_int_B)

def setup_log(prefix = "SG_DEF_"):
	"""Creates a directory to store logs. One directory for date, then subdirectory for time of run."""
	run_time = datetime.datetime.now()
	run_day = "%02d" %(run_time.day)
	run_month = "%02d" %(run_time.month)
	run_hour = "%02d" %(run_time.hour)
	run_minute = "%02d" %(run_time.minute)
	log_dir = prefix + run_month + run_day + "/"
	log_subdir = run_hour + "h_" + run_minute + "m/"

	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	if not os.path.isdir(log_dir + log_subdir):
		os.mkdir(log_dir + log_subdir)

	return log_dir, log_subdir

@tf.RegisterGradient("boosted_gamma_grad")
def boosted_g_rep_grad(op, d_sample, d_alpha, d_fz):
	"""Returns gradient of sample with respect to alpha parameter."""
	gamma_draw = op.inputs[0]
	alpha_p = op.inputs[1]
	fz = op.inputs[2]

	eps = calc_epsilon(gamma_draw, alpha_p + global_B)
	h_grad = gamma_grad_h(eps, alpha_p + global_B)

	return [None, -(h_grad * d_sample + fz * get_gamma_correction(eps, alpha_p + global_B)), None]

def sample_gamma_RSVI(z, alpha, fz):
	"""Given a (frozen) Gamma sample z, and shape parameter alpha and log-prob fz, get RSVI gradient of sample.
	The key tensorflow dynamic is to overwrite the gradient of the identity_N op to work as the RSVI gradient."""
	sample, _, _ = tf.identity_n([z, alpha, fz])
	return sample

class Gamma_Layer(object):
	"""Object defining a layer that is sampled from a Gamma.
	Sparse Gamma DEFs combine a deep hierarchy of such layers."""
	def __init__(self, name, layer_dims):
		"""name: name of layer (Z_1, W_1, etc). For variable_scope.
		layer_dims: the dimensions (n_row, n_col) of layer."""

		self.sigma = 0.1
		self.l_alpha_limit = tf.log(tf.exp(1e-3) - 1.) #limit lower bound of parameters like @naesseth.
		self.l_mean_limit = tf.log(tf.exp(1e-4) - 1.)

		self.name = name
		self.layer_dims = layer_dims

		with tf.variable_scope(name):
			self._alpha = tf.get_variable("G_alpha", dtype=tf.float32, shape=layer_dims, initializer=tf.random_normal_initializer(0.0, self.sigma))
			self._alpha += 0.5
			self._mean = tf.get_variable("G_mean", dtype=tf.float32, shape=layer_dims, initializer=tf.random_normal_initializer(0.0, self.sigma))

			self._alpha = tf.maximum(self._alpha, self.l_alpha_limit)
			self._mean = tf.maximum(self._mean, self.l_mean_limit)

		self.alpha = tf.nn.softplus(self._alpha)
		self.mean = tf.nn.softplus(self._mean)

		#These are the "last draws". We set these in freeze_sample and then recover them in thaw_sample.
		#It's a bit of a hack and could lead to trouble if we don't always do freeze/thaw but it's convenient.
		self.last_gamma_draw = None
		self.last_uniform_draw = None

	def get_alpha(self):
		self._alpha = tf.maximum(self._alpha, self.l_alpha_limit)
		return tf.nn.softplus(self._alpha)

	def get_mean(self):
		#NOTE: not sure if we need to call this every time? Maybe sub-optimal.
		self._mean = tf.maximum(self._mean, self.l_mean_limit)
		return tf.nn.softplus(self._mean)

	def freeze_sample(self):
		"""This just samples from tf.random_gamma and caches the value of the relevant calculations.
		We 'freeze' the sample because we don't let it contribute to downstream gradients.
		In 'thaw' section we'll reparameterize the sample in terms of differentiable operations.
		The gradient of sample with respect to alpha will be set through sample_gamma_RSVI."""
		alpha = self.get_alpha()
		mean = self.get_mean()

		gamma_draw = tf.squeeze(tf.random_gamma([1], alpha + global_B), [0])
		gamma_draw = tf.maximum(gamma_draw, 1e-5)
		u = tf.random_uniform(self.layer_dims + [global_int_B, ])

		freeze_eps = calc_epsilon(gamma_draw, alpha + global_B)
		freeze_hval = gamma_h_boosted(freeze_eps, u, alpha, global_int_B)
		freeze_sample = freeze_hval * (mean / alpha)
		freeze_sample = tf.maximum(freeze_sample, 1e-5)

		self.last_gamma_draw = gamma_draw
		self.last_uniform_draw = u

		return freeze_sample

	def elbo_sample(self):
		"""Unconditional, unboosted sample to get estimate of ELBO.
		This is a separate function to avoid stepping on freeze/thaw cycle."""
		alpha = self.get_alpha()
		mean = self.get_mean()

		gamma_draw = tf.squeeze(tf.random_gamma([1], alpha, 1.0), [0]) * (mean / alpha)
		gamma_draw = tf.maximum(1e-300, gamma_draw)
		return gamma_draw

	def thaw_sample(self, f_z):
		"""Using the sample from the previous frozen cycle, passes sample through differentiable ops
		to get a sample with full gradients that can be used for downstream steps."""
		alpha = self.get_alpha()
		mean = self.get_mean()

		gamma_sample = sample_gamma_RSVI(self.last_gamma_draw, alpha, f_z)

		#These next lines multiply the gamma draw by uniform samples for shape augmentation.
		#Autodiff keeps track of gradients through this step.
		#If there's a better way to do this serial expand_dim let me know.
		B_tile = tf.expand_dims(tf.expand_dims(tf.range(global_B + 0.0), [1]), [1])

		a_size = self.layer_dims
		alpha_vec = tf.tile(tf.expand_dims(tf.transpose(alpha), [0]), (global_int_B, 1, 1))
		alpha_vec += tf.tile(B_tile, [1, a_size[1], a_size[0]])
		alpha_vec = tf.transpose(alpha_vec)

		u_pow = tf.pow(self.last_uniform_draw, 1. / alpha_vec)

		gamma_sample *= tf.reduce_prod(u_pow, axis=-1)

		eps = calc_epsilon(gamma_sample, alpha + global_B)
		h_val = gamma_h(eps, alpha + global_B)
		sample = h_val * (mean / alpha)
		sample = tf.maximum(sample, 1e-5)

		return sample

	def entropy(self):
		"""Entropy of gamma distribution."""
		return tf.reduce_sum(gamma_entropy(self.get_alpha(), self.get_mean()))

class SG_DEF_Model(object):
	def __init__(self, batch_size, data_n_dim, layer_sizes):
		self.latent_layers = []
		self.weight_layers = []

		self.prior_W_alpha = 0.1
		self.prior_W_beta = 0.3
		self.prior_Z_alpha = 0.1
		self.prior_Z_beta = 0.1
		self.layer_alpha = 0.1

		z_count = 0
		for layer_size in layer_sizes[::-1]:
			self.latent_layers.append(Gamma_Layer("Z_%i" %(z_count), [batch_size, layer_size]))
			z_count += 1

		w_count = 0
		layer_sizes = [data_n_dim] + layer_sizes
		for (W_out, W_in) in zip(layer_sizes[:-1], layer_sizes[1:])[::-1]:
			self.weight_layers.append(Gamma_Layer("W_%i" %(w_count), [W_in, W_out]))
			w_count += 1

	def log_prob(self, z_L, w_L, batch):
		weight_log_prob = tf.reduce_sum([tf.reduce_sum(gamma_log_prob(self.prior_W_alpha, self.prior_W_beta, w)) for w in w_L])
		z_log_prob = gamma_log_prob(self.prior_Z_alpha, self.prior_Z_beta, z_L[0])

		l_prob = weight_log_prob + tf.reduce_sum(z_log_prob)

		z_count = 0
		for (z, w) in zip(z_L[:-1], w_L[:-1]):
			z_sum_w = tf.matmul(z, w)
			g_alpha = self.layer_alpha
			g_beta = g_alpha / z_sum_w
			l_prob += tf.reduce_sum(gamma_log_prob(g_alpha, g_beta, z_L[z_count + 1]))
			z_count += 1

		obs_sum_w = tf.matmul(z_L[-1], w_L[-1])
		l_prob += tf.reduce_sum(poisson_log_prob(batch, obs_sum_w))

		#tf.summary.image("batch", tf.transpose(tf.reshape(batch, [320, 64, 64, 1]), [0, 2, 1, 3]), max_outputs = 4)
		#tf.summary.image("obs_sum_w", tf.transpose(tf.reshape(obs_sum_w, [320, 64, 64, 1]), [0, 2, 1, 3]), max_outputs=4)
		if do_images:
			tf.summary.image("sampled_reconstruction", tf.transpose(tf.reshape(tf.minimum(tf.random_poisson(obs_sum_w, [1]), 255.0), [320, 64, 64, 1]), [0, 2, 1, 3]), max_outputs = 4)

		return tf.reduce_sum(l_prob)

	def activated_on(self, batch):
		frozen_z_L = [tf.stop_gradient(z.freeze_sample()) for z in self.latent_layers]
		frozen_w_L = [tf.stop_gradient(w.freeze_sample()) for w in self.weight_layers]

		frozen_pxz = tf.stop_gradient(self.log_prob(frozen_z_L, frozen_w_L, batch))

		live_z_L = [z.thaw_sample(frozen_pxz) for z in self.latent_layers]
		live_w_L = [w.thaw_sample(frozen_pxz) for w in self.weight_layers]

		log_pxz = self.log_prob(live_z_L, live_w_L, batch)
		entropy = tf.reduce_sum([z.entropy() for z in self.latent_layers]) + tf.reduce_sum([w.entropy() for w in self.weight_layers])

		return log_pxz, entropy #report separately in case we want to monitor separately.

	def estimate_elbo(self, batch):
		elbo_z_L = [z.elbo_sample() for z in self.latent_layers]
		elbo_w_L = [w.elbo_sample() for w in self.weight_layers]
		the_prob = self.log_prob(elbo_z_L, elbo_w_L, batch)

		new_sample = tf.matmul(elbo_z_L[-1], elbo_w_L[-1])

		if do_images:
			tf.summary.image("unconditional_sample", tf.transpose(tf.reshape(tf.minimum(tf.random_poisson(new_sample, [1]), 255.0), [320, 64, 64, 1]), [0, 2, 1, 3]), max_outputs = 4)

		return the_prob + tf.reduce_sum([z.entropy() for z in self.latent_layers]) + tf.reduce_sum([w.entropy() for w in self.weight_layers])

	def loss(self, batch):
		log_pxz, entropy = self.activated_on(batch)
		tf.summary.scalar("log_pxz", log_pxz)
		tf.summary.scalar("entropy", entropy)

		estimate_elbo = self.estimate_elbo(batch)
		tf.summary.scalar('estimated_elbo', estimate_elbo)

		return log_pxz + entropy, estimate_elbo

def train(learning_rate, num_epochs = 1000):
	coord = tf.train.Coordinator()

	#Use the same Olivetti faces data as in paper.
	f_name = "faces_training.npy"
	data = np.load(f_name)

	data_N = data.shape[0]
	data_ndim = data.shape[1]

	batch_size = data_N

	data = tf.cast(data, tf.float32)

	layer_sizes = [100, 40, 15]
	with tf.variable_scope("model"):
		sg_def = SG_DEF_Model(batch_size, data_ndim, layer_sizes)

	tf_graph = tf.get_default_graph()

	with tf_graph.gradient_override_map({"IdentityN": "boosted_gamma_grad"}):
		loss, e_elbo = sg_def.loss(data)

	global_step = tf.Variable(0, trainable=False, name="global_step")
	learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.9)

	optim = tf.train.RMSPropOptimizer(learning_rate).minimize(-loss, global_step = global_step)

	log_dir, log_subdir = setup_log()

	writer = tf.summary.FileWriter(log_dir + "/" + log_subdir + "/")
	writer.add_graph(tf.get_default_graph())
	run_metadata = tf.RunMetadata()
	summaries = tf.summary.merge_all()

	sess = tf.Session(graph=tf_graph)
	init = tf.global_variables_initializer()
	sess.run(init)

	tf_graph.finalize()

	start_time = time.time()
	for epoch in range(500):
		if epoch % 10 == 0:
			summary, total_loss, est_elbo, _ = sess.run([summaries, loss, e_elbo, optim])
			print "Epoch % 4i: total loss is %.5g (est elbo: %.5g) || time elapsed: %.2f s" %(epoch, total_loss, est_elbo, time.time() - start_time)
			writer.add_summary(summary, epoch)
			writer.flush()
		else:
			_ = sess.run([optim])
	print "Done training"

	coord.request_stop()

def main():
	learning_rate = 0.1
	num_epochs = 1000
	train(learning_rate, num_epochs)

if __name__ == "__main__":
	main()
