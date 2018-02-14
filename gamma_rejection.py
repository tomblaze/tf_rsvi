import tensorflow as tf
import numpy as np

TINY = 1e-15 #small constant for logs.

def calc_epsilon(p, alpha):
	sqrt_alpha = tf.sqrt(9. * alpha - 3.)
	pow_ZA = tf.pow(p / (alpha - 1./3), 1./3)
	return sqrt_alpha * (pow_ZA - 1.)

def gamma_h(epsilon, alpha):
	b = alpha - 1./3.
	c = 1. / tf.sqrt(9. * b)
	v = 1. + epsilon * c
	return b * (v ** 3)

def gamma_grad_h(epsilon, alpha):
	b = alpha - 1.0 / 3.0
	c = 1.0 / tf.sqrt(9.0 * b)
	v = 1.0 + epsilon * c
	return v ** 3 - 13.5 * epsilon * b * (v ** 2) * (c ** 3)

def gamma_grad_logr(epsilon, alpha):
	"""Gradient of log-proposal."""
	b = alpha - 1./3.
	c = 1. / tf.sqrt(9. * b)
	v = 1. + epsilon * c
	return -0.5 / b + 9. * epsilon * ( c ** 3) / v

def gamma_grad_logq(epsilon, alpha):
	h_val = gamma_h(epsilon, alpha)
	h_der = gamma_grad_h(epsilon, alpha)

	return tf.log(h_val) + (alpha - 1.) * h_der / h_val - h_der - tf.digamma(alpha)

def gamma_h_boosted(epsilon, u, alpha, B):
	a_S = alpha.get_shape().as_list()
	alpha_vec = tf.tile(tf.expand_dims(tf.transpose(alpha), [0]), (B, 1, 1))
	B_tile = tf.expand_dims(tf.expand_dims(tf.range(B + 0.0), [1]), [1])
	alpha_vec += tf.tile(B_tile, [1, a_S[1], a_S[0]])
	alpha_vec = tf.transpose(alpha_vec)
	u_pow = tf.pow(u, 1. / alpha_vec)
	return tf.reduce_prod(u_pow, axis=-1) * gamma_h(epsilon, alpha + B)

def gamma_grad_h_boosted(epsilon, u, alpha):
	B = u.get_shape().as_list()[-1]
	a_S = alpha.get_shape().as_list()
	h_val = gamma_h(epsilon, alpha + B)
	h_der = gamma_grad_h(epsilon, alpha + B)
	B_tile = tf.expand_dims(tf.expand_dims(tf.range(B + 0.0), [1]), [1])

	alpha_vec = tf.tile(tf.expand_dims(tf.transpose(alpha), [0]), (B, 1, 1))
	alpha_vec += tf.tile(B_tile, [1, a_S[1], a_S[0]])
	alpha_vec = tf.transpose(alpha_vec)

	u_pow = tf.pow(u, 1. / alpha_vec)
	u_der = -tf.log(u) / alpha_vec ** 2
	return tf.reduce_prod(u_pow, axis=-1) * h_val * (h_der / h_val + tf.reduce_sum(u_der, axis=-1))

def get_gamma_correction(epsilon, alpha):
	return gamma_grad_logq(epsilon, alpha) - gamma_grad_logr(epsilon, alpha)

def gamma_entropy(alpha, m):
	return tf.reduce_sum(alpha + tf.log(m) - tf.log(alpha) + tf.lgamma(alpha) + (1. - alpha) * tf.digamma(alpha), 1)

def gamma_log_prob(alpha, beta, observed):
	return tf.reduce_sum(alpha * tf.log(beta) + (alpha - 1.0) * tf.log(observed + TINY) - beta * observed - tf.lgamma(alpha), 1)

def poisson_log_prob(obs, pred):
	return tf.reduce_sum(obs * tf.log(pred + TINY) - pred - tf.lgamma(obs + 1), 1)
