import time
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from .metrics import *
from ..data.pipeline import batch_generator_mono
from ..data.sampling import unit_grid2_sample_fn


def init_glorot_uniform(key, shape, fan_in, fan_out):
	"""
	# type: (int, list[int], int, int) -> jnp.array
	# shape inZ+ (*)
	# fan_in, fan_out inZ+
	# w inR shape
	"""
	limit = jnp.sqrt(6) / jnp.sqrt(fan_in+fan_out)
	w = jax.random.uniform(key, shape, minval=-limit, maxval=limit)
	return w


def init_dense_neural_network(key, layers):
	"""
	# type: (int, list[int]) -> list[tuple[jnp.array, jnp.array]]
	"""
	
	# split key
	k0, key = jax.random.split(key)
	
	# initialise input layer
	w_list = [init_glorot_uniform(k0, (layers[1], layers[0]), layers[0], layers[1])]
	b_list = [jnp.zeros((layers[1], 1))]
	
	# initialise hidden layers
	for i in range(1, len(layers)-1):
		k1, key = jax.random.split(key)
		w_list.append(init_glorot_uniform(k1, (layers[i+1], layers[i]), layers[i], layers[i+1]))
		b_list.append(jnp.zeros((layers[i+1], 1)))
	
	return list(map(tuple, zip(w_list, b_list)))


def dense_neural_network(params, x, ha=jax.nn.sigmoid):
	"""
	# type: (list[tuple[jnp.array]], jnp.array, jax.nn.[Activation functions]) -> jnp.array
	# x inR (in, 1)
	# z inR (out, 1)
	"""
	
	# shape-safe input activation
	h = x.reshape(-1, 1)
	
	# hidden activations
	for w,b in params[:-1]:
		h = ha(jnp.dot(w, h) + b)
	
	# output activation
	wn, bn = params[-1]
	z = jnp.dot(wn, h) + bn
	
	return z


def get_3d_groundwater_flow_model(
		key,
		layer_dims,
		scale_xytz,
		k=jnp.ones((1,1)),
		ss=1e-1,
		rr=1e-9,
		lam_mse=1.0,
		lam_phys=1.0,
		lam_l2=0.0,
		hidden_activation=jax.nn.relu
	):
	"""
	Docstring
	"""
	
	params = [init_dense_neural_network(key, layer_dims), (ss,rr)]
	h_fn = jax.vmap(lambda p,xyt: dense_neural_network(p, xyt, ha=hidden_activation)[0,0], in_axes=(None, 0)) # N x [0,1] x [0,1] x [0,1] -> N x [0,1]
	
	def loss_3d_groundwater_flow(params, batch_xyt, scale_xytz):
		"""
		# loss = ||R||^2
		# R = Ss * ∂h/∂t - ∇·(K ∇h) - Rr
		# ∇h = (∂h/∂x, ∂h/∂y)
		# https://en.wikipedia.org/wiki/Groundwater_flow_equation
		# https://github.com/jax-ml/jax/issues/3022#issuecomment-2733591263
		# (with scaling)
		"""
		
		h_fn_mono = lambda xyt: h_fn(params[0], xyt[jnp.newaxis, :])[0]
		h_fn_flux = lambda xyt: unit_grid2_sample_fn(k, *xyt[:2]) * jax.grad(h_fn_mono)(xyt)[:2] * (scale_xytz[3] / scale_xytz[:2])
		
		# compute 3d groundwater flow terms
		batch_dhdt = jax.vmap(lambda xyt: jax.grad(h_fn_mono)(xyt)[2] * (scale_xytz[3] / scale_xytz[2]))(batch_xyt)
		batch_div_flux = jax.vmap(lambda xyt: jnp.sum(jnp.diag(jax.jacfwd(h_fn_flux)(xyt)[:2, :2]) / scale_xytz[:2]))(batch_xyt)
		batch_ss = params[-1][0]
		batch_rr = params[-1][1]
		
		# compute l2 of PDE residual
		loss_darcyflow = batch_ss * batch_dhdt - batch_div_flux - batch_rr
		loss = jnp.mean(loss_darcyflow**2)
		
		return loss
	
	def loss_fn(params, batch_xyt, batch_z):
		
		loss_batch = lam_mse * loss_mse(h_fn(params[0], batch_xyt), batch_z)
		loss_phys = lam_phys * loss_3d_groundwater_flow(params, batch_xyt, scale_xytz)
		loss_reg = lam_l2 * lp_norm(params, order=2)
		loss = loss_batch + loss_phys + loss_reg
		
		return loss
	
	return (params, h_fn, loss_fn)


def sample_3d_model(model, param, axis_t, axis_y, axis_x, batch_size=None):
	"""
	Docstring
	"""
	
	input_points = jnp.stack(jnp.meshgrid(axis_t, axis_y, axis_x, indexing='ij')[::-1], axis=-1).reshape(-1, 3)
	if batch_size is None:
		sample = model(param, input_points)
	else:
		input_generator = batch_generator_mono(input_points, batch_size)
		input_steps = jnp.ceil(len(input_points) / batch_size)
		sample = jnp.concatenate([model(param, next(input_generator)) for _ in range(input_steps)])
	
	sample = sample.reshape(len(axis_t), len(axis_y), len(axis_x))
	
	return sample

